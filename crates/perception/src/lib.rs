//! Pure-Rust perception primitives over decoded media planes.

#![forbid(unsafe_code)]

pub mod all_pp;
pub mod detector_db;
pub mod detector_geometry;
pub mod interaction_delta;
pub mod model_bundle;
pub mod motion_correspondence;
pub mod pp_detector;
// The exact DB score seam is crate-private to the fixed live DB composition.
#[allow(dead_code)]
mod detector_db_score;

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ocrs::{
    ImageSource, OcrEngine, OcrEngineParams, OcrInput, RecognitionPage, TextItem, TextLine,
};
pub use ppocrv6_tiny_preflight::{
    Identity as PpArtifactIdentity, PpOrderedParallelProfile, PpOrderedParallelRecognizer,
    PpRecognitionProfile, PpRecognizer, PpRecognizerLoadProfile, WidthPolicy as PpWidthPolicy,
};
use ppocrv6_tiny_preflight::{OCRS_SEAM_HEIGHT, Seam};
use rten::Model;
use rten_imageproc::{PointF, Rect as ImageRect, RotatedRect, bounding_rect, min_area_rect};
use rten_tensor::prelude::*;
use screenevents::{ElementObservation, Rect};
use sha2::{Digest, Sha256};

/// Whole-frame OCR that consumes a decoded luma plane directly.
pub struct LumaOcr {
    engine: OcrEngine,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OcrProfile {
    pub prepare: Duration,
    pub detect: Duration,
    pub group_lines: Duration,
    pub recognize: Duration,
    pub postprocess: Duration,
    pub detected_words: usize,
    pub line_regions: usize,
    pub recognized_lines: usize,
}

/// One selected page after OCR preparation, detection and line grouping.
///
/// The decoded luma plane is not retained.
pub struct PreparedOcrPage {
    input: OcrInput,
    line_regions: Vec<Vec<RotatedRect>>,
    width: u32,
    height: u32,
    pub profile: PreparedOcrPageProfile,
    diagnostics: Option<PreparedOcrPageDiagnostics>,
}

#[derive(Clone, Debug)]
pub struct PreparedOcrPageProfile {
    pub prepare: Duration,
    pub detect: Duration,
    pub group_lines: Duration,
    pub detected_words: usize,
    pub line_regions: usize,
}

#[derive(Clone, Debug)]
pub struct PreparedOcrPageDiagnostics {
    pub normalized_shape: [usize; 3],
    pub normalized_sha256: String,
}

/// The exact bit representation of one detector word corner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorCornerBits {
    pub x_bits: u32,
    pub y_bits: u32,
}

/// Owned diagnostic geometry for one detector word.
///
/// Corners retain the order returned by [`RotatedRect::corners`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorWordGeometry {
    pub corners: [DetectorCornerBits; 4],
}

/// Owned diagnostic geometry for one detector line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorLineGeometry {
    /// The line's zero-based index in detector output order.
    pub line_index: usize,
    /// Words in detector output order.
    pub words: Vec<DetectorWordGeometry>,
    /// The non-empty integral word union clamped to the decoded frame.
    ///
    /// This is `None` when the line is wholly removed by the frame clamp.
    pub bbox: Option<Rect>,
}

impl PreparedOcrPage {
    pub fn diagnostics(&self) -> Option<&PreparedOcrPageDiagnostics> {
        self.diagnostics.as_ref()
    }

    /// Extract exact detector line geometry without changing prepared-page state.
    ///
    /// Coordinates are returned as their raw [`f32::to_bits`] representations.
    /// Empty lines and lines with non-finite coordinates are rejected atomically.
    pub fn detector_line_geometry(&self) -> Result<Vec<DetectorLineGeometry>> {
        self.line_regions
            .iter()
            .enumerate()
            .map(|(line_index, line)| {
                let bbox = detector_line_rect(line, self.width, self.height)?;
                let words = line
                    .iter()
                    .map(|word| DetectorWordGeometry {
                        corners: word.corners().map(|corner| DetectorCornerBits {
                            x_bits: corner.x.to_bits(),
                            y_bits: corner.y.to_bits(),
                        }),
                    })
                    .collect();
                Ok(DetectorLineGeometry {
                    line_index,
                    words,
                    bbox,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecognitionLineMembership {
    pub page_index: usize,
    pub line_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecognitionBatchProfile {
    pub shape: [usize; 4],
    pub input_pixels: u64,
    pub members: Vec<RecognitionLineMembership>,
}

#[derive(Clone, Debug)]
pub struct RecognitionFlushProfile {
    pub recognize: Duration,
    pub model_calls: u64,
    pub input_pixels: u64,
    pub batches: Vec<RecognitionBatchProfile>,
}

/// The result of postprocessing one detector line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecognitionLineDisposition {
    Missing,
    DroppedAfterGeometryClamp,
    DroppedAfterTextFilter,
    Emitted,
}

/// Owned recognition diagnostics for one detector line.
///
/// Records occur in `(page_index, line_index)` order. `pp_profile` is `None`
/// for OCRS and `Some` for PP-OCRv6 Tiny. `recognizer_bbox` is meaningful only
/// for present OCRS output and is always `None` for PP. PP `raw_text` is always
/// `Some`, including for empty output. OCRS model work remains exclusively in
/// [`RecognitionFlushProfile`]; batch cost and timing are not allocated to
/// individual lines.
#[derive(Clone, Debug)]
pub struct RecognitionLineDiagnostic {
    pub page_index: usize,
    pub line_index: usize,
    pub detector_bbox: Option<Rect>,
    pub recognizer_bbox: Option<Rect>,
    pub raw_text: Option<String>,
    pub observation: Option<ElementObservation>,
    pub disposition: RecognitionLineDisposition,
    pub pp_profile: Option<PpRecognitionProfile>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PpRecognitionFlushProfile {
    pub seam_prepare: Duration,
    /// Sum of active per-call preprocessing durations.
    pub preprocess: Duration,
    /// Sum of active per-call model-forward durations.
    pub forward: Duration,
    /// Sum of active per-call decoding durations.
    pub decode: Duration,
    /// End-to-end recognition wall excluding seam preparation and postprocess.
    pub recognition_wall: Duration,
    /// Barrier wall times for ordered parallel recognition. Sequential
    /// recognition keeps this `None` and continues to report active sums.
    pub ordered_parallel_wall: Option<PpOrderedParallelProfile>,
    pub model_calls: u64,
    pub input_pixels: u64,
    pub maximum_batch_size: usize,
    pub maximum_batch_width: usize,
    pub upstream_cap_hits: u64,
    pub ctc_steps: u64,
    pub recognized_detector_line_boxes: u64,
    pub dropped_after_geometry_clamp: u64,
    pub dropped_after_text_filter: u64,
    pub total_emitted_observation_area: u64,
}

impl PpRecognitionFlushProfile {
    pub fn recognize(&self) -> Duration {
        self.seam_prepare
            .saturating_add(self.preprocess)
            .saturating_add(self.forward)
            .saturating_add(self.decode)
    }

    fn add_recognition(&mut self, profile: PpRecognitionProfile) {
        self.preprocess = self.preprocess.saturating_add(profile.preprocess);
        self.forward = self.forward.saturating_add(profile.forward);
        self.decode = self.decode.saturating_add(profile.decode);
        self.model_calls = self.model_calls.saturating_add(1);
        self.input_pixels = self.input_pixels.saturating_add(profile.input_pixels);
        self.maximum_batch_size = self.maximum_batch_size.max(profile.batch_size);
        self.maximum_batch_width = self.maximum_batch_width.max(profile.batch_width);
        self.upstream_cap_hits = self
            .upstream_cap_hits
            .saturating_add(u64::from(profile.upstream_cap_hit));
        self.ctc_steps = self
            .ctc_steps
            .saturating_add(u64::try_from(profile.ctc_steps).unwrap_or(u64::MAX));
    }
}

pub struct RecognizedOcrPage {
    pub observations: Vec<ElementObservation>,
    pub postprocess: Duration,
}

/// One detector quadrilateral in decoded-frame `[x, y]` coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorQuad {
    pub corners_xy: [[f32; 2]; 4],
}

/// Result of one ordered detector-quad recognition page.
pub struct RecognizedQuadPage {
    pub page: RecognizedOcrPage,
    pub profile: PpRecognitionFlushProfile,
    pub diagnostics: Vec<RecognitionLineDiagnostic>,
    /// Model-free H64 crop widths in detector-quad order.
    pub seam_widths: Vec<usize>,
}

/// Recognize ordered detector quadrilaterals directly from decoded luma.
///
/// Crop preparation is model-free and uses the exact OCRS polygon-fill,
/// aspect-ratio and resize path used by the recognizer. Input order is
/// retained in the returned diagnostics and observation emission order.
pub fn recognize_luma_quads_pp(
    recognizer: &PpRecognizer,
    luma: &[u8],
    width: u32,
    height: u32,
    quads: &[DetectorQuad],
) -> Result<RecognizedQuadPage> {
    let (line_regions, seams, seam_prepare) = prepare_luma_quad_lines(luma, width, height, quads)?;
    let mut profile = PpRecognitionFlushProfile {
        seam_prepare,
        ..Default::default()
    };
    let recognition_started = Instant::now();
    let mut lines = Vec::with_capacity(seams.len());
    for (line_index, seam) in seams.iter().enumerate() {
        let recognition = recognizer
            .recognize_one(seam)
            .with_context(|| format!("PP recognition failed for detector quad {line_index}"))?;
        profile.add_recognition(recognition.profile);
        lines.push(PpRecognizedLine {
            raw_text: recognition.text,
            profile: recognition.profile,
        });
    }
    profile.recognition_wall = recognition_started.elapsed();

    finish_pp_quad_page(lines, line_regions, seams, width, height, profile)
}

/// Recognize ordered detector quadrilaterals using fixed parallel PP lanes.
///
/// Seam preparation remains one serial call, and postprocessing consumes the
/// reconciled input-order results.
pub fn recognize_luma_quads_pp_ordered_parallel(
    recognizer: &mut PpOrderedParallelRecognizer,
    luma: &[u8],
    width: u32,
    height: u32,
    quads: &[DetectorQuad],
) -> Result<RecognizedQuadPage> {
    let (line_regions, seams, seam_prepare) = prepare_luma_quad_lines(luma, width, height, quads)?;
    let (recognitions, stage_wall) = recognizer
        .recognize_ordered(&seams)
        .context("ordered parallel PP recognition failed")?;
    if recognitions.len() != seams.len() {
        bail!(
            "ordered parallel PP recognition returned {} results for {} seams",
            recognitions.len(),
            seams.len()
        );
    }

    let mut profile = PpRecognitionFlushProfile {
        seam_prepare,
        recognition_wall: stage_wall.recognition_wall,
        ordered_parallel_wall: Some(stage_wall),
        ..Default::default()
    };
    let mut lines = Vec::with_capacity(recognitions.len());
    for recognition in recognitions {
        profile.add_recognition(recognition.profile);
        lines.push(PpRecognizedLine {
            raw_text: recognition.text,
            profile: recognition.profile,
        });
    }

    finish_pp_quad_page(lines, line_regions, seams, width, height, profile)
}

/// Recognize ordered detector quadrilaterals for the product all-PP path.
///
/// Detector confidences are reconciled by input position and applied directly
/// while observations are constructed. Unlike the diagnostic API above, this
/// path does not clone observations into per-line diagnostic rows or retain
/// model-free seam widths.
pub fn recognize_luma_quads_pp_ordered_parallel_scored(
    recognizer: &mut PpOrderedParallelRecognizer,
    luma: &[u8],
    width: u32,
    height: u32,
    quads: &[DetectorQuad],
    detector_scores: &[f32],
) -> Result<(RecognizedOcrPage, PpRecognitionFlushProfile)> {
    let (page, profile, diagnostics, _) = recognize_luma_quads_pp_ordered_parallel_scored_inner(
        recognizer,
        luma,
        width,
        height,
        quads,
        detector_scores,
        false,
    )?;
    debug_assert!(diagnostics.is_empty());
    Ok((page, profile))
}

#[allow(clippy::too_many_arguments)]
fn recognize_luma_quads_pp_ordered_parallel_scored_inner(
    recognizer: &mut PpOrderedParallelRecognizer,
    luma: &[u8],
    width: u32,
    height: u32,
    quads: &[DetectorQuad],
    detector_scores: &[f32],
    collect_diagnostics: bool,
) -> Result<(
    RecognizedOcrPage,
    PpRecognitionFlushProfile,
    Vec<RecognitionLineDiagnostic>,
    Vec<usize>,
)> {
    if detector_scores.len() != quads.len() {
        bail!(
            "detector score count {} differs from quad count {}",
            detector_scores.len(),
            quads.len()
        );
    }
    let (line_regions, seams, seam_prepare) = prepare_luma_quad_lines(luma, width, height, quads)?;
    let (recognitions, stage_wall) = recognizer
        .recognize_ordered(&seams)
        .context("ordered parallel PP recognition failed")?;
    if recognitions.len() != seams.len() {
        bail!(
            "ordered parallel PP recognition returned {} results for {} seams",
            recognitions.len(),
            seams.len()
        );
    }

    let mut profile = PpRecognitionFlushProfile {
        seam_prepare,
        recognition_wall: stage_wall.recognition_wall,
        ordered_parallel_wall: Some(stage_wall),
        ..Default::default()
    };
    let mut lines = Vec::with_capacity(recognitions.len());
    for recognition in recognitions {
        profile.add_recognition(recognition.profile);
        lines.push(PpRecognizedLine {
            raw_text: recognition.text,
            profile: recognition.profile,
        });
    }

    let postprocess_started = Instant::now();
    let (observations, geometry, diagnostics) = postprocess_pp_detector_lines(
        lines,
        &line_regions,
        width,
        height,
        0,
        collect_diagnostics,
        Some(detector_scores),
    )?;
    profile.recognized_detector_line_boxes = geometry.recognized_detector_line_boxes;
    profile.dropped_after_geometry_clamp = geometry.dropped_after_geometry_clamp;
    profile.dropped_after_text_filter = geometry.dropped_after_text_filter;
    profile.total_emitted_observation_area = geometry.total_emitted_observation_area;
    Ok((
        RecognizedOcrPage {
            observations,
            postprocess: postprocess_started.elapsed(),
        },
        profile,
        diagnostics,
        seams.iter().map(|seam| seam.width).collect(),
    ))
}

fn finish_pp_quad_page(
    lines: Vec<PpRecognizedLine>,
    line_regions: Vec<Vec<RotatedRect>>,
    seams: Vec<Seam>,
    width: u32,
    height: u32,
    mut profile: PpRecognitionFlushProfile,
) -> Result<RecognizedQuadPage> {
    let postprocess_started = Instant::now();
    let (observations, geometry, diagnostics) =
        postprocess_pp_detector_lines(lines, &line_regions, width, height, 0, true, None)?;
    profile.recognized_detector_line_boxes = geometry.recognized_detector_line_boxes;
    profile.dropped_after_geometry_clamp = geometry.dropped_after_geometry_clamp;
    profile.dropped_after_text_filter = geometry.dropped_after_text_filter;
    profile.total_emitted_observation_area = geometry.total_emitted_observation_area;
    Ok(RecognizedQuadPage {
        page: RecognizedOcrPage {
            observations,
            postprocess: postprocess_started.elapsed(),
        },
        profile,
        diagnostics,
        seam_widths: seams.iter().map(|seam| seam.width).collect(),
    })
}

fn prepare_luma_quad_lines(
    luma: &[u8],
    width: u32,
    height: u32,
    quads: &[DetectorQuad],
) -> Result<(Vec<Vec<RotatedRect>>, Vec<Seam>, Duration)> {
    validate_luma(luma, width, height)?;
    let line_regions = quads
        .iter()
        .enumerate()
        .map(|(index, quad)| {
            quad_to_rect(*quad, width, height)
                .map(|rect| vec![rect])
                .with_context(|| format!("invalid detector quad {index}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let prepare_started = Instant::now();
    let crop_engine = OcrEngine::new(Default::default())
        .context("failed to initialize model-free recognition crop preparer")?;
    let input = crop_engine
        .prepare_input(
            ImageSource::from_bytes(luma, (width, height))
                .context("failed to prepare luma image source")?,
        )
        .context("failed to normalize luma for recognition crops")?;
    let seams = line_regions
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let tensor =
                OcrEngine::prepare_recognition_input_with_height(&input, line, OCRS_SEAM_HEIGHT)
                    .with_context(|| format!("failed to prepare detector quad {index}"))?;
            let [height, width] = tensor.shape();
            let seam = Seam {
                height,
                width,
                values: tensor.iter().copied().collect(),
            };
            seam.validate()?;
            Ok(seam)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((line_regions, seams, prepare_started.elapsed()))
}

fn quad_to_rect(quad: DetectorQuad, width: u32, height: u32) -> Result<RotatedRect> {
    let quad_xy = quad.corners_xy;
    if quad_xy.iter().flatten().any(|value| !value.is_finite()) {
        bail!("quad contains non-finite coordinates");
    }
    if quad_xy
        .iter()
        .any(|[x, y]| *x < 0.0 || *y < 0.0 || *x > width as f32 || *y > height as f32)
    {
        bail!("quad lies outside source bounds");
    }
    let points = quad_xy.map(|[x, y]| PointF::from_yx(y, x));
    let rect = min_area_rect(&points).context("quad produced no min-area rectangle")?;
    if !rect.width().is_finite()
        || !rect.height().is_finite()
        || rect.width() <= 0.0
        || rect.height() <= 0.0
        || rect
            .corners()
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        bail!("min-area rectangle is nonfinite or degenerate");
    }
    let corners = rect.corners();
    let min_x = corners
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = corners
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min);
    let max_y = corners
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let intersection_width = max_x.min(width as f32) - min_x.max(0.0);
    let intersection_height = max_y.min(height as f32) - min_y.max(0.0);
    if intersection_width <= 0.0 || intersection_height <= 0.0 {
        bail!("min-area rectangle has no positive page intersection");
    }
    Ok(rect)
}

impl LumaOcr {
    pub fn load(detection_model: &Path, recognition_model: &Path) -> Result<Self> {
        let detection_model = Model::load_file(detection_model).with_context(|| {
            format!(
                "failed to load OCR detection model {}",
                detection_model.display()
            )
        })?;
        let recognition_model = Model::load_file(recognition_model).with_context(|| {
            format!(
                "failed to load OCR recognition model {}",
                recognition_model.display()
            )
        })?;
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            ..Default::default()
        })
        .context("failed to initialize OCR engine")?;
        Ok(Self { engine })
    }

    pub fn observe(&self, luma: &[u8], width: u32, height: u32) -> Result<Vec<ElementObservation>> {
        self.observe_profiled(luma, width, height)
            .map(|(observations, _)| observations)
    }

    pub fn observe_profiled(
        &self,
        luma: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<ElementObservation>, OcrProfile)> {
        validate_luma(luma, width, height)?;

        let prepare_started = Instant::now();
        let source = ImageSource::from_bytes(luma, (width, height))
            .context("failed to prepare luma image source")?;
        let input = self
            .engine
            .prepare_input(source)
            .context("failed to preprocess luma for OCR")?;
        let prepare = prepare_started.elapsed();
        let detect_started = Instant::now();
        let words = self
            .engine
            .detect_words(&input)
            .context("OCR text detection failed")?;
        let detect = detect_started.elapsed();
        let detected_words = words.len();
        let group_started = Instant::now();
        let line_regions = self.engine.find_text_lines(&input, &words);
        let group_lines = group_started.elapsed();
        let line_region_count = line_regions.len();
        let recognize_started = Instant::now();
        let lines = self
            .engine
            .recognize_text(&input, &line_regions)
            .context("OCR text recognition failed")?;
        let recognize = recognize_started.elapsed();

        let postprocess_started = Instant::now();
        let mut observations = Vec::new();
        for line in lines.into_iter().flatten() {
            let text = line.to_string();
            let text = text.trim();
            if text.is_empty() || !text.chars().any(char::is_alphanumeric) {
                continue;
            }
            let bbox = line.bounding_rect();
            let left = bbox.left().max(0) as u32;
            let top = bbox.top().max(0) as u32;
            let right = (bbox.right().max(0) as u32).min(width);
            let bottom = (bbox.bottom().max(0) as u32).min(height);
            if right <= left || bottom <= top {
                continue;
            }
            observations.push(ElementObservation {
                bbox: Rect {
                    x: left,
                    y: top,
                    width: right - left,
                    height: bottom - top,
                },
                text: Some(text.to_owned()),
                role: Some("text".to_owned()),
                state: BTreeMap::new(),
                // ocrs does not currently expose a calibrated line confidence.
                confidence: 0.5,
            });
        }
        let postprocess = postprocess_started.elapsed();
        let profile = OcrProfile {
            prepare,
            detect,
            group_lines,
            recognize,
            postprocess,
            detected_words,
            line_regions: line_region_count,
            recognized_lines: observations.len(),
        };
        Ok((observations, profile))
    }

    /// Prepare one selected frame without diagnostic normalized-input hashing.
    pub fn prepare_page(&self, luma: &[u8], width: u32, height: u32) -> Result<PreparedOcrPage> {
        self.prepare_page_impl(luma, width, height, false)
    }

    /// Prepare one selected frame and return its bounded profile.
    pub fn prepare_page_profiled(
        &self,
        luma: &[u8],
        width: u32,
        height: u32,
    ) -> Result<PreparedOcrPage> {
        self.prepare_page_impl(luma, width, height, true)
    }

    fn prepare_page_impl(
        &self,
        luma: &[u8],
        width: u32,
        height: u32,
        collect_diagnostics: bool,
    ) -> Result<PreparedOcrPage> {
        validate_luma(luma, width, height)?;
        let prepare_started = Instant::now();
        let source = ImageSource::from_bytes(luma, (width, height))
            .context("failed to prepare luma image source")?;
        let input = self
            .engine
            .prepare_input(source)
            .context("failed to preprocess luma for OCR")?;
        let prepare = prepare_started.elapsed();
        let detect_started = Instant::now();
        let words = self
            .engine
            .detect_words(&input)
            .context("OCR text detection failed")?;
        let detect = detect_started.elapsed();
        let detected_words = words.len();
        let group_started = Instant::now();
        let line_regions = self.engine.find_text_lines(&input, &words);
        let group_lines = group_started.elapsed();
        let diagnostics = if collect_diagnostics {
            Some(PreparedOcrPageDiagnostics {
                normalized_shape: input.normalized_shape(),
                normalized_sha256: normalized_input_sha256(&input)?,
            })
        } else {
            None
        };
        let line_region_count = line_regions.len();
        Ok(PreparedOcrPage {
            input,
            line_regions,
            width,
            height,
            diagnostics,
            profile: PreparedOcrPageProfile {
                prepare,
                detect,
                group_lines,
                detected_words,
                line_regions: line_region_count,
            },
        })
    }

    /// Recognize exactly one or two prepared pages without call diagnostics.
    pub fn recognize_prepared_pages(
        &self,
        pages: &[&PreparedOcrPage],
    ) -> Result<(Vec<RecognizedOcrPage>, Duration)> {
        validate_prepared_page_count(pages)?;
        let recognition_pages = recognition_pages(pages);
        let recognize_started = Instant::now();
        let output = self
            .engine
            .recognize_text_pages_unprofiled(&recognition_pages)
            .context("OCR text recognition failed")?;
        let recognize = recognize_started.elapsed();
        let (recognized_pages, _) = postprocess_recognized_pages(output, pages, false)?;
        Ok((recognized_pages, recognize))
    }

    /// Recognize exactly one or two prepared pages through the ordinary OCRS
    /// execution path and retain one owned diagnostic per detector line.
    pub fn recognize_prepared_pages_with_diagnostics(
        &self,
        pages: &[&PreparedOcrPage],
    ) -> Result<(
        Vec<RecognizedOcrPage>,
        Duration,
        Vec<RecognitionLineDiagnostic>,
    )> {
        validate_prepared_page_count(pages)?;
        let recognition_pages = recognition_pages(pages);
        let recognize_started = Instant::now();
        let output = self
            .engine
            .recognize_text_pages_unprofiled(&recognition_pages)
            .context("OCR text recognition failed")?;
        let recognize = recognize_started.elapsed();
        let (recognized_pages, diagnostics) = postprocess_recognized_pages(output, pages, true)?;
        Ok((recognized_pages, recognize, diagnostics))
    }

    /// Recognize exactly one or two prepared pages in shared OCRS width buckets.
    pub fn recognize_prepared_pages_profiled(
        &self,
        pages: &[&PreparedOcrPage],
    ) -> Result<(Vec<RecognizedOcrPage>, RecognitionFlushProfile)> {
        let (recognized_pages, profile, _) =
            self.recognize_prepared_pages_profiled_impl(pages, false)?;
        Ok((recognized_pages, profile))
    }

    /// Recognize prepared pages and retain one owned diagnostic per detector line.
    ///
    /// OCRS work granularity remains batch-only in the returned flush profile.
    pub fn recognize_prepared_pages_profiled_with_diagnostics(
        &self,
        pages: &[&PreparedOcrPage],
    ) -> Result<(
        Vec<RecognizedOcrPage>,
        RecognitionFlushProfile,
        Vec<RecognitionLineDiagnostic>,
    )> {
        self.recognize_prepared_pages_profiled_impl(pages, true)
    }

    fn recognize_prepared_pages_profiled_impl(
        &self,
        pages: &[&PreparedOcrPage],
        collect_diagnostics: bool,
    ) -> Result<(
        Vec<RecognizedOcrPage>,
        RecognitionFlushProfile,
        Vec<RecognitionLineDiagnostic>,
    )> {
        validate_prepared_page_count(pages)?;
        let recognition_pages = recognition_pages(pages);
        let recognize_started = Instant::now();
        let output = self
            .engine
            .recognize_text_pages(&recognition_pages)
            .context("OCR text recognition failed")?;
        let recognize = recognize_started.elapsed();
        let (recognized_pages, diagnostics) =
            postprocess_recognized_pages(output.pages, pages, collect_diagnostics)?;
        let profile = RecognitionFlushProfile {
            recognize,
            model_calls: output.profile.model_calls,
            input_pixels: output.profile.input_pixels,
            batches: output
                .profile
                .batches
                .into_iter()
                .map(|batch| RecognitionBatchProfile {
                    shape: batch.shape,
                    input_pixels: batch.input_pixels,
                    members: batch
                        .members
                        .into_iter()
                        .map(|member| RecognitionLineMembership {
                            page_index: member.page_index,
                            line_index: member.line_index,
                        })
                        .collect(),
                })
                .collect(),
        };
        Ok((recognized_pages, profile, diagnostics))
    }

    /// Recognize one or two prepared OCRS pages using PP-OCRv6 Tiny. Each line
    /// is deliberately a separate model call in page/line order.
    pub fn recognize_prepared_pages_pp(
        &self,
        recognizer: &PpRecognizer,
        pages: &[&PreparedOcrPage],
    ) -> Result<(Vec<RecognizedOcrPage>, PpRecognitionFlushProfile)> {
        let (recognized_pages, profile, _) =
            self.recognize_prepared_pages_pp_impl(recognizer, pages, false)?;
        Ok((recognized_pages, profile))
    }

    /// Recognize prepared pages with PP-OCRv6 Tiny and retain one owned
    /// diagnostic per detector line.
    pub fn recognize_prepared_pages_pp_with_diagnostics(
        &self,
        recognizer: &PpRecognizer,
        pages: &[&PreparedOcrPage],
    ) -> Result<(
        Vec<RecognizedOcrPage>,
        PpRecognitionFlushProfile,
        Vec<RecognitionLineDiagnostic>,
    )> {
        self.recognize_prepared_pages_pp_impl(recognizer, pages, true)
    }

    fn recognize_prepared_pages_pp_impl(
        &self,
        recognizer: &PpRecognizer,
        pages: &[&PreparedOcrPage],
        collect_diagnostics: bool,
    ) -> Result<(
        Vec<RecognizedOcrPage>,
        PpRecognitionFlushProfile,
        Vec<RecognitionLineDiagnostic>,
    )> {
        validate_prepared_page_count(pages)?;
        let mut profile = PpRecognitionFlushProfile::default();
        let mut page_lines = pages
            .iter()
            .map(|page| Vec::with_capacity(page.line_regions.len()))
            .collect::<Vec<_>>();

        for (page_index, page) in pages.iter().enumerate() {
            for (line_index, line) in page.line_regions.iter().enumerate() {
                validate_detector_line(line).with_context(|| {
                    format!("invalid detector line {line_index} on prepared page {page_index}")
                })?;
                let seam_started = Instant::now();
                let tensor = self
                    .engine
                    .prepare_recognition_input(&page.input, line)
                    .with_context(|| {
                        format!(
                            "failed to prepare PP recognition seam for line {line_index} on prepared page {page_index}"
                        )
                    })?;
                let [height, width] = tensor.shape();
                let seam = Seam {
                    height,
                    width,
                    values: tensor.iter().copied().collect(),
                };
                seam.validate()?;
                profile.seam_prepare = profile.seam_prepare.saturating_add(seam_started.elapsed());

                let recognition = recognizer.recognize_one(&seam).with_context(|| {
                    format!(
                        "PP recognition failed for line {line_index} on prepared page {page_index}"
                    )
                })?;
                profile.add_recognition(recognition.profile);
                page_lines[page_index].push(PpRecognizedLine {
                    raw_text: recognition.text,
                    profile: recognition.profile,
                });
            }
        }

        let mut recognized_pages = Vec::with_capacity(pages.len());
        let diagnostic_count = if collect_diagnostics {
            pages.iter().map(|page| page.line_regions.len()).sum()
        } else {
            0
        };
        let mut diagnostics = Vec::with_capacity(diagnostic_count);
        for (page_index, (lines, page)) in page_lines
            .into_iter()
            .zip(pages.iter().copied())
            .enumerate()
        {
            let postprocess_started = Instant::now();
            let (observations, geometry, page_diagnostics) = postprocess_pp_detector_lines(
                lines,
                &page.line_regions,
                page.width,
                page.height,
                page_index,
                collect_diagnostics,
                None,
            )?;
            profile.recognized_detector_line_boxes = profile
                .recognized_detector_line_boxes
                .saturating_add(geometry.recognized_detector_line_boxes);
            profile.dropped_after_geometry_clamp = profile
                .dropped_after_geometry_clamp
                .saturating_add(geometry.dropped_after_geometry_clamp);
            profile.dropped_after_text_filter = profile
                .dropped_after_text_filter
                .saturating_add(geometry.dropped_after_text_filter);
            profile.total_emitted_observation_area = profile
                .total_emitted_observation_area
                .saturating_add(geometry.total_emitted_observation_area);
            diagnostics.extend(page_diagnostics);
            recognized_pages.push(RecognizedOcrPage {
                observations,
                postprocess: postprocess_started.elapsed(),
            });
        }
        Ok((recognized_pages, profile, diagnostics))
    }
}

fn validate_prepared_page_count(pages: &[&PreparedOcrPage]) -> Result<()> {
    if !(1..=2).contains(&pages.len()) {
        bail!("recognition microbatch requires exactly one or two pages");
    }
    Ok(())
}

fn recognition_pages<'a>(pages: &[&'a PreparedOcrPage]) -> Vec<RecognitionPage<'a>> {
    pages
        .iter()
        .map(|page| RecognitionPage {
            input: &page.input,
            lines: &page.line_regions,
        })
        .collect()
}

fn postprocess_recognized_pages(
    pages_output: Vec<Vec<Option<TextLine>>>,
    pages: &[&PreparedOcrPage],
    collect_diagnostics: bool,
) -> Result<(Vec<RecognizedOcrPage>, Vec<RecognitionLineDiagnostic>)> {
    if pages_output.len() != pages.len() {
        bail!("OCR recognition returned the wrong page count");
    }
    for (lines, page) in pages_output.iter().zip(pages.iter().copied()) {
        if lines.len() != page.line_regions.len() {
            bail!("OCR recognition returned the wrong line count");
        }
    }

    let detector_bboxes = if collect_diagnostics {
        Some(
            pages
                .iter()
                .enumerate()
                .map(|(page_index, page)| {
                    page.line_regions
                        .iter()
                        .enumerate()
                        .map(|(line_index, line)| {
                            detector_line_rect(line, page.width, page.height).with_context(|| {
                                format!(
                                    "invalid detector line {line_index} on prepared page {page_index}"
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    let diagnostic_count = if collect_diagnostics {
        pages.iter().map(|page| page.line_regions.len()).sum()
    } else {
        0
    };
    let mut recognized_pages = Vec::with_capacity(pages.len());
    let mut diagnostics = Vec::with_capacity(diagnostic_count);
    for (page_index, (lines, page)) in pages_output
        .into_iter()
        .zip(pages.iter().copied())
        .enumerate()
    {
        let postprocess_started = Instant::now();
        let page_detector_bboxes = detector_bboxes
            .as_ref()
            .map(|page_bboxes| page_bboxes[page_index].as_slice());
        let (observations, page_diagnostics) = postprocess_ocrs_lines(
            lines,
            page_detector_bboxes,
            page.width,
            page.height,
            page_index,
            collect_diagnostics,
        );
        diagnostics.extend(page_diagnostics);
        recognized_pages.push(RecognizedOcrPage {
            observations,
            postprocess: postprocess_started.elapsed(),
        });
    }
    Ok((recognized_pages, diagnostics))
}

fn validate_luma(luma: &[u8], width: u32, height: u32) -> Result<()> {
    let expected_len = usize::try_from(u64::from(width) * u64::from(height))
        .context("luma dimensions exceed addressable memory")?;
    if width == 0 || height == 0 || luma.len() != expected_len {
        bail!(
            "invalid luma plane: {} bytes for {width}x{height}",
            luma.len()
        );
    }
    Ok(())
}

fn normalized_input_sha256(input: &OcrInput) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"naky.ocrs-normalized-chw-f32-bits.v0\0");
    for dimension in input.normalized_shape() {
        digest.update(
            u64::try_from(dimension)
                .context("normalized input dimension does not fit u64")?
                .to_le_bytes(),
        );
    }
    for value in input.normalized_values() {
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn postprocess_ocrs_lines(
    lines: Vec<Option<TextLine>>,
    detector_bboxes: Option<&[Option<Rect>]>,
    width: u32,
    height: u32,
    page_index: usize,
    collect_diagnostics: bool,
) -> (Vec<ElementObservation>, Vec<RecognitionLineDiagnostic>) {
    let mut observations = Vec::new();
    let mut diagnostics = Vec::with_capacity(if collect_diagnostics { lines.len() } else { 0 });
    for (line_index, line) in lines.into_iter().enumerate() {
        let raw_text = line.as_ref().map(ToString::to_string);
        let recognizer_bbox = line
            .as_ref()
            .and_then(|line| clamp_image_rect(line.bounding_rect(), width, height));
        let detector_bbox = detector_bboxes.and_then(|bboxes| bboxes[line_index]);
        let result = postprocess_line(
            raw_text,
            detector_bbox,
            recognizer_bbox,
            recognizer_bbox,
            0.5,
            None,
            collect_diagnostics.then_some((page_index, line_index)),
        );
        if let Some(observation) = result.observation {
            observations.push(observation);
        }
        if let Some(diagnostic) = result.diagnostic {
            diagnostics.push(diagnostic);
        }
    }
    (observations, diagnostics)
}

fn clamp_image_rect(bbox: ImageRect, width: u32, height: u32) -> Option<Rect> {
    let left = bbox.left().max(0) as u32;
    let top = bbox.top().max(0) as u32;
    let right = (bbox.right().max(0) as u32).min(width);
    let bottom = (bbox.bottom().max(0) as u32).min(height);
    if right <= left || bottom <= top {
        return None;
    }
    Some(Rect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

struct LinePostprocessResult {
    observation: Option<ElementObservation>,
    diagnostic: Option<RecognitionLineDiagnostic>,
    disposition: RecognitionLineDisposition,
}

fn postprocess_line(
    raw_text: Option<String>,
    detector_bbox: Option<Rect>,
    recognizer_bbox: Option<Rect>,
    emission_bbox: Option<Rect>,
    observation_confidence: f32,
    pp_profile: Option<PpRecognitionProfile>,
    diagnostic_identity: Option<(usize, usize)>,
) -> LinePostprocessResult {
    let (observation, disposition) = match raw_text.as_deref() {
        None => (None, RecognitionLineDisposition::Missing),
        Some(_) if emission_bbox.is_none() => {
            (None, RecognitionLineDisposition::DroppedAfterGeometryClamp)
        }
        Some(raw_text) => {
            let text = raw_text.trim();
            if text.is_empty() || !text.chars().any(char::is_alphanumeric) {
                (None, RecognitionLineDisposition::DroppedAfterTextFilter)
            } else {
                let observation = ElementObservation {
                    bbox: emission_bbox.expect("emission bbox was checked above"),
                    text: Some(text.to_owned()),
                    role: Some("text".to_owned()),
                    state: BTreeMap::new(),
                    confidence: observation_confidence,
                };
                (Some(observation), RecognitionLineDisposition::Emitted)
            }
        }
    };
    let diagnostic =
        diagnostic_identity.map(|(page_index, line_index)| RecognitionLineDiagnostic {
            page_index,
            line_index,
            detector_bbox,
            recognizer_bbox,
            raw_text,
            observation: observation.clone(),
            disposition,
            pp_profile,
        });
    LinePostprocessResult {
        observation,
        diagnostic,
        disposition,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DetectorLineGeometryProfile {
    recognized_detector_line_boxes: u64,
    dropped_after_geometry_clamp: u64,
    dropped_after_text_filter: u64,
    total_emitted_observation_area: u64,
}

fn validate_detector_line(line: &[RotatedRect]) -> Result<()> {
    if line.is_empty() {
        bail!("detector line contains no word boxes");
    }
    if line
        .iter()
        .flat_map(RotatedRect::corners)
        .any(|corner| !corner.x.is_finite() || !corner.y.is_finite())
    {
        bail!("detector line contains non-finite geometry");
    }
    Ok(())
}

fn detector_line_rect(line: &[RotatedRect], width: u32, height: u32) -> Result<Option<Rect>> {
    validate_detector_line(line)?;
    let bbox = bounding_rect(line.iter())
        .context("detector line contains no word boxes")?
        .integral_bounding_rect();
    let left = bbox.left().max(0) as u32;
    let top = bbox.top().max(0) as u32;
    let right = (bbox.right().max(0) as u32).min(width);
    let bottom = (bbox.bottom().max(0) as u32).min(height);
    if right <= left || bottom <= top {
        return Ok(None);
    }
    Ok(Some(Rect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }))
}

#[derive(Clone)]
struct PpRecognizedLine {
    raw_text: String,
    profile: PpRecognitionProfile,
}

fn postprocess_pp_detector_lines(
    lines: Vec<PpRecognizedLine>,
    line_regions: &[Vec<RotatedRect>],
    width: u32,
    height: u32,
    page_index: usize,
    collect_diagnostics: bool,
    detector_scores: Option<&[f32]>,
) -> Result<(
    Vec<ElementObservation>,
    DetectorLineGeometryProfile,
    Vec<RecognitionLineDiagnostic>,
)> {
    if lines.len() != line_regions.len() {
        bail!("PP recognition returned the wrong line count");
    }
    if detector_scores.is_some_and(|scores| scores.len() != lines.len()) {
        bail!("PP detector scores returned the wrong line count");
    }
    let detector_bboxes = line_regions
        .iter()
        .map(|line| detector_line_rect(line, width, height))
        .collect::<Result<Vec<_>>>()?;
    let mut observations = Vec::new();
    let mut profile = DetectorLineGeometryProfile::default();
    let mut diagnostics = Vec::with_capacity(if collect_diagnostics { lines.len() } else { 0 });
    for (line_index, (line, detector_bbox)) in lines.into_iter().zip(detector_bboxes).enumerate() {
        let observation_confidence = detector_scores.map_or(0.5, |scores| scores[line_index]);
        let result = postprocess_line(
            Some(line.raw_text),
            detector_bbox,
            None,
            detector_bbox,
            observation_confidence,
            Some(line.profile),
            collect_diagnostics.then_some((page_index, line_index)),
        );
        match result.disposition {
            RecognitionLineDisposition::Missing => {}
            RecognitionLineDisposition::DroppedAfterGeometryClamp => {
                profile.dropped_after_geometry_clamp =
                    profile.dropped_after_geometry_clamp.saturating_add(1);
            }
            RecognitionLineDisposition::DroppedAfterTextFilter => {
                profile.dropped_after_text_filter =
                    profile.dropped_after_text_filter.saturating_add(1);
            }
            RecognitionLineDisposition::Emitted => {
                profile.recognized_detector_line_boxes =
                    profile.recognized_detector_line_boxes.saturating_add(1);
                let area = result
                    .observation
                    .as_ref()
                    .map(|observation| observation.bbox.area())
                    .expect("emitted lines have observations");
                profile.total_emitted_observation_area =
                    profile.total_emitted_observation_area.saturating_add(area);
            }
        }
        if let Some(observation) = result.observation {
            observations.push(observation);
        }
        if let Some(diagnostic) = result.diagnostic {
            diagnostics.push(diagnostic);
        }
    }
    Ok((observations, profile, diagnostics))
}
