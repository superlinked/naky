//! Fixed Paddle-compatible DB postprocessing for PP detector maps.

use anyhow::{Context, Result, bail};
use rten_imageproc::{Point, RetrievalMode, find_contours};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

use crate::detector_db_score::{opencv_box_score_fast, validate_map};
use crate::detector_geometry::bounded_clipper_offset::bounded_rounded_offset_rectangle;
use crate::detector_geometry::{
    opencv_box_points, opencv_min_area_rect_i32, order_opencv_db_box, project_opencv_db_box,
};

const MAP_THRESHOLD: f32 = f32::from_bits(1_045_220_557);
const MAX_CONTOURS: usize = 3_000;
const FIRST_MIN_SIDE: f32 = 3.0;
const MIN_SCORE: f64 = 0.4;
const UNCLIP_RATIO: f64 = 1.4;
const SECOND_MIN_SIDE: f32 = 5.0;

/// One accepted DB row, in reverse RTen discovery order.
#[derive(Clone, Debug, PartialEq)]
pub struct DbDetection {
    pub quad_xy: [[u32; 2]; 4],
    pub score: f64,
    pub source_contour_index: usize,
}

/// Causal admission counts for the fixed DB stage sequence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DbProfile {
    pub active_pixels: usize,
    pub raw_contours: usize,
    pub bounded_contours: usize,
    pub first_side_admitted: usize,
    pub score_admitted: usize,
    pub offset_admitted: usize,
    pub second_side_admitted: usize,
}

/// Stateless fixed-profile DB postprocessor.
#[derive(Clone, Copy, Debug, Default)]
pub struct DbPostprocessor;

impl DbPostprocessor {
    pub fn process(
        &self,
        map: &[f32],
        map_shape_hw: [usize; 2],
        source_shape_hw: [u32; 2],
    ) -> Result<(Vec<DbDetection>, DbProfile)> {
        let (detections, profile, _) =
            process_with_floor(map, map_shape_hw, source_shape_hw, MIN_SCORE, false)?;
        Ok((detections, profile))
    }
}

fn process_with_floor(
    map: &[f32],
    map_shape_hw: [usize; 2],
    source_shape_hw: [u32; 2],
    score_floor: f64,
    collect_diagnostics: bool,
) -> Result<(
    Vec<DbDetection>,
    DbProfile,
    Vec<DbContourDiagnosticInternal>,
)> {
    validate_map(map, map_shape_hw)?;
    if source_shape_hw.contains(&0) {
        bail!("DB source dimensions must be positive");
    }

    let bitmap = map
        .iter()
        .map(|&value| value > MAP_THRESHOLD)
        .collect::<Vec<_>>();
    let mut profile = DbProfile {
        active_pixels: bitmap.iter().filter(|&&active| active).count(),
        ..Default::default()
    };
    let tensor = NdTensor::from_data(map_shape_hw, bitmap);
    let contours = find_contours(tensor.view(), RetrievalMode::List);
    profile.raw_contours = contours.len();
    let mut contours = contours
        .iter()
        .map(compress_direction_runs)
        .collect::<Vec<_>>();
    contours.reverse();

    let mut detections = Vec::new();
    let mut diagnostics = Vec::new();
    for (source_contour_index, contour) in contours.iter().take(MAX_CONTOURS).enumerate() {
        profile.bounded_contours += 1;
        let Some(first_rect) = opencv_min_area_rect_i32(contour).with_context(|| {
            format!("minimum-area stage failed for contour {source_contour_index}")
        })?
        else {
            push_diagnostic(
                &mut diagnostics,
                collect_diagnostics,
                source_contour_index,
                None,
                None,
                None,
                DbContourDispositionInternal::FirstRectangleEmpty,
            );
            continue;
        };
        let first_short_side = first_rect.size_wh[0].min(first_rect.size_wh[1]);
        if first_short_side < FIRST_MIN_SIDE {
            push_diagnostic(
                &mut diagnostics,
                collect_diagnostics,
                source_contour_index,
                Some(first_short_side),
                None,
                None,
                DbContourDispositionInternal::FirstSideRejected,
            );
            continue;
        }
        profile.first_side_admitted += 1;

        let first_box = order_opencv_db_box(opencv_box_points(first_rect)?);
        let score = opencv_box_score_fast(map, map_shape_hw, &first_box)?.score;
        if !score_meets_floor(score, score_floor) {
            push_diagnostic(
                &mut diagnostics,
                collect_diagnostics,
                source_contour_index,
                Some(first_short_side),
                Some(score),
                None,
                DbContourDispositionInternal::ScoreRejected,
            );
            continue;
        }
        profile.score_admitted += 1;

        let (area, perimeter) = polygon_area_perimeter(&first_box)?;
        let distance = area * UNCLIP_RATIO / perimeter;
        let offset = bounded_rounded_offset_rectangle(first_box, first_short_side, distance)
            .with_context(|| format!("rounded offset failed for contour {source_contour_index}"))?;
        profile.offset_admitted += 1;

        let Some(second_rect) = opencv_min_area_rect_i32(&offset).with_context(|| {
            format!("second minimum-area stage failed for contour {source_contour_index}")
        })?
        else {
            push_diagnostic(
                &mut diagnostics,
                collect_diagnostics,
                source_contour_index,
                Some(first_short_side),
                Some(score),
                None,
                DbContourDispositionInternal::SecondRectangleEmpty,
            );
            continue;
        };
        if second_rect.size_wh[0].min(second_rect.size_wh[1]) < SECOND_MIN_SIDE {
            push_diagnostic(
                &mut diagnostics,
                collect_diagnostics,
                source_contour_index,
                Some(first_short_side),
                Some(score),
                None,
                DbContourDispositionInternal::SecondSideRejected,
            );
            continue;
        }
        profile.second_side_admitted += 1;
        let ordered = order_opencv_db_box(opencv_box_points(second_rect)?);
        let map_shape_u32 = [
            u32::try_from(map_shape_hw[0]).context("DB map height exceeds u32")?,
            u32::try_from(map_shape_hw[1]).context("DB map width exceeds u32")?,
        ];
        let quad_xy = project_opencv_db_box(ordered, map_shape_u32, source_shape_hw)?;
        detections.push(DbDetection {
            quad_xy,
            score,
            source_contour_index,
        });
        push_diagnostic(
            &mut diagnostics,
            collect_diagnostics,
            source_contour_index,
            Some(first_short_side),
            Some(score),
            Some(quad_xy),
            DbContourDispositionInternal::Emitted,
        );
    }
    if detections.len() != profile.second_side_admitted {
        bail!("DB final detection accounting drifted");
    }
    if collect_diagnostics && diagnostics.len() != profile.bounded_contours {
        bail!("DB contour diagnostic accounting drifted");
    }
    Ok((detections, profile, diagnostics))
}

fn score_meets_floor(score: f64, floor: f64) -> bool {
    score >= floor
}

#[derive(Clone, Debug)]
struct DbContourDiagnosticInternal;

#[derive(Clone, Copy, Debug)]
enum DbContourDispositionInternal {
    FirstRectangleEmpty,
    FirstSideRejected,
    ScoreRejected,
    SecondRectangleEmpty,
    SecondSideRejected,
    Emitted,
}

#[allow(clippy::too_many_arguments)]
fn push_diagnostic(
    output: &mut Vec<DbContourDiagnosticInternal>,
    collect: bool,
    source_contour_index: usize,
    first_short_side: Option<f32>,
    score: Option<f64>,
    final_quad_xy: Option<[[u32; 2]; 4]>,
    disposition: DbContourDispositionInternal,
) {
    if !collect {
        return;
    }

    {
        let _ = (
            output,
            source_contour_index,
            first_short_side,
            score,
            final_quad_xy,
            disposition,
        );
    }
}

fn compress_direction_runs(contour: &[Point]) -> Vec<[i32; 2]> {
    if contour.len() <= 1 {
        return contour.iter().map(|point| [point.x, point.y]).collect();
    }
    let mut simple = Vec::with_capacity(contour.len());
    for index in 0..contour.len() {
        let previous = contour[(index + contour.len() - 1) % contour.len()];
        let current = contour[index];
        let next = contour[(index + 1) % contour.len()];
        let incoming = [
            (current.x - previous.x).signum(),
            (current.y - previous.y).signum(),
        ];
        let outgoing = [(next.x - current.x).signum(), (next.y - current.y).signum()];
        if incoming != outgoing {
            simple.push([current.x, current.y]);
        }
    }
    simple
}

fn polygon_area_perimeter(points: &[[f32; 2]; 4]) -> Result<(f64, f64)> {
    let mut area_twice = 0.0_f64;
    let mut perimeter = 0.0_f64;
    for index in 0..points.len() {
        let [x0, y0] = points[index];
        let [x1, y1] = points[(index + 1) % points.len()];
        area_twice += f64::from(x0) * f64::from(y1) - f64::from(y0) * f64::from(x1);
        perimeter += f64::from(x1 - x0).hypot(f64::from(y1 - y0));
    }
    let area = area_twice.abs() * 0.5;
    if !area.is_finite() || area <= 0.0 || !perimeter.is_finite() || perimeter <= 0.0 {
        bail!("DB offset polygon has invalid area or perimeter");
    }
    Ok((area, perimeter))
}
