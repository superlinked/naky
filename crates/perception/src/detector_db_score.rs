//! Bounded Paddle DB `box_score_fast` compatibility.
//!
//! The polygon raster below is a modified Rust translation of OpenCV 4.12.0's
//! `fillPoly` implementation. The crop/coercion wrapper is a modified Rust
//! translation of PaddleOCR's DB postprocessor. The two upstream origins and
//! their exact applicable notices are recorded separately under `third_party/`.

use std::cmp::Ordering;

use anyhow::{Context, Result, bail};

const XY_SHIFT: u32 = 16;
const XY_ONE: i64 = 1_i64 << XY_SHIFT;

// The current detector maps are at most 704x576 and the applicable polygons
// have four points. These limits retain ample headroom while bounding every
// allocation and scanline walk independently of the caller's address space.
const MAX_MAP_DIMENSION: usize = 1 << 20;
const MAX_MAP_PIXELS: usize = 1 << 24;
const MAX_POLYGON_POINTS: usize = 64;
const MAX_ABS_COORDINATE: f32 = (1 << 20) as f32;
const MAX_POLYGON_SPAN: f32 = (1 << 15) as f32;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OpenCvBoxScore {
    pub score: f64,
    pub included_pixels: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScoreBounds {
    xmin: usize,
    ymin: usize,
    xmax: usize,
    ymax: usize,
}

impl ScoreBounds {
    fn width(self) -> Result<usize> {
        self.xmax
            .checked_sub(self.xmin)
            .and_then(|extent| extent.checked_add(1))
            .context("DB score ROI width overflow")
    }

    fn height(self) -> Result<usize> {
        self.ymax
            .checked_sub(self.ymin)
            .and_then(|extent| extent.checked_add(1))
            .context("DB score ROI height overflow")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PolyEdge {
    y0: i32,
    y1: i32,
    x: i64,
    dx: i64,
}

/// Reproduce the bounded one-contour `box_score_fast` seam used by Paddle DB.
///
/// `map` is a finite row-major `f32` probability map. `polygon_xy` is stored as
/// `f32`, as it is after OpenCV `boxPoints`. This function intentionally returns
/// `f64`: OpenCV accumulates masked `CV_32F` samples in `double` and Paddle keeps
/// the resulting Python float.
pub(crate) fn opencv_box_score_fast(
    map: &[f32],
    shape_hw: [usize; 2],
    polygon_xy: &[[f32; 2]],
) -> Result<OpenCvBoxScore> {
    let [height, width] = shape_hw;
    validate_map(map, shape_hw)?;
    let (bounds, local_polygon) = paddle_score_polygon(width, height, polygon_xy)?;
    let roi_width = bounds.width()?;
    let roi_height = bounds.height()?;
    let mask = opencv_fill_poly_mask([roi_height, roi_width], &local_polygon)?;

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (local_y, mask_row) in mask.chunks_exact(roi_width).enumerate() {
        let map_y = bounds
            .ymin
            .checked_add(local_y)
            .context("DB score map row overflow")?;
        let row_start = map_y
            .checked_mul(width)
            .and_then(|offset| offset.checked_add(bounds.xmin))
            .context("DB score map offset overflow")?;
        for (local_x, &member) in mask_row.iter().enumerate() {
            if member != 0 {
                sum += f64::from(map[row_start + local_x]);
                count += 1;
            }
        }
    }
    let score = sum * if count == 0 { 0.0 } else { 1.0 / count as f64 };
    Ok(OpenCvBoxScore {
        score,
        included_pixels: count,
    })
}

pub(crate) fn validate_map(map: &[f32], [height, width]: [usize; 2]) -> Result<()> {
    if height == 0 || width == 0 {
        bail!("DB score map dimensions must be positive");
    }
    if height > MAX_MAP_DIMENSION || width > MAX_MAP_DIMENSION {
        bail!("DB score map dimension exceeds {MAX_MAP_DIMENSION}");
    }
    let pixels = height
        .checked_mul(width)
        .context("DB score map area overflow")?;
    if pixels > MAX_MAP_PIXELS {
        bail!("DB score map area exceeds {MAX_MAP_PIXELS} pixels");
    }
    if map.len() != pixels {
        bail!(
            "DB score map length differs: expected {pixels}, got {}",
            map.len()
        );
    }
    if let Some(index) = map.iter().position(|value| !value.is_finite()) {
        bail!("DB score map value {index} is non-finite");
    }
    Ok(())
}

fn paddle_score_polygon(
    width: usize,
    height: usize,
    polygon_xy: &[[f32; 2]],
) -> Result<(ScoreBounds, Vec<[i32; 2]>)> {
    if !(3..=MAX_POLYGON_POINTS).contains(&polygon_xy.len()) {
        bail!("DB score polygon must contain 3..={MAX_POLYGON_POINTS} points");
    }
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    for (point_index, &[x, y]) in polygon_xy.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            bail!("DB score polygon point {point_index} is non-finite");
        }
        if x.abs() > MAX_ABS_COORDINATE || y.abs() > MAX_ABS_COORDINATE {
            bail!(
                "DB score polygon point {point_index} exceeds supported coordinate magnitude {MAX_ABS_COORDINATE}"
            );
        }
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    if xmax - xmin > MAX_POLYGON_SPAN || ymax - ymin > MAX_POLYGON_SPAN {
        bail!("DB score polygon span exceeds {MAX_POLYGON_SPAN}");
    }

    let clip_bound = |value: f32, upper: usize, label: &str| -> Result<usize> {
        if value < i32::MIN as f32 || value > i32::MAX as f32 {
            bail!("DB score {label} is outside int32 range");
        }
        let integer = value as i32;
        let upper = i32::try_from(upper).context("DB score map bound exceeds int32")?;
        usize::try_from(integer.clamp(0, upper)).context("clipped DB score bound is negative")
    };
    let bounds = ScoreBounds {
        xmin: clip_bound(xmin.floor(), width - 1, "xmin")?,
        ymin: clip_bound(ymin.floor(), height - 1, "ymin")?,
        xmax: clip_bound(xmax.ceil(), width - 1, "xmax")?,
        ymax: clip_bound(ymax.ceil(), height - 1, "ymax")?,
    };
    let xmin_f64 = bounds.xmin as f64;
    let ymin_f64 = bounds.ymin as f64;
    let local_polygon = polygon_xy
        .iter()
        .enumerate()
        .map(|(point_index, &[x, y])| {
            // NumPy promotes `float32 - int32` to float64, then assignment back
            // into the copied float32 box rounds once before `astype(int32)`.
            let local_x = (f64::from(x) - xmin_f64) as f32;
            let local_y = (f64::from(y) - ymin_f64) as f32;
            if local_x < i32::MIN as f32
                || local_x > i32::MAX as f32
                || local_y < i32::MIN as f32
                || local_y > i32::MAX as f32
            {
                bail!("local DB score polygon point {point_index} is outside int32 range");
            }
            Ok([local_x as i32, local_y as i32])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((bounds, local_polygon))
}

fn opencv_fill_poly_mask(shape_hw: [usize; 2], polygon_xy: &[[i32; 2]]) -> Result<Vec<u8>> {
    let [height, width] = shape_hw;
    if height == 0 || width == 0 {
        bail!("OpenCV-compatible mask dimensions must be positive");
    }
    let pixels = height
        .checked_mul(width)
        .context("OpenCV-compatible mask area overflow")?;
    if pixels > MAX_MAP_PIXELS {
        bail!("OpenCV-compatible mask area exceeds {MAX_MAP_PIXELS} pixels");
    }
    if !(3..=MAX_POLYGON_POINTS).contains(&polygon_xy.len()) {
        bail!("OpenCV-compatible polygon must contain 3..={MAX_POLYGON_POINTS} points");
    }
    let mut minimum = [i32::MAX; 2];
    let mut maximum = [i32::MIN; 2];
    for (point_index, point) in polygon_xy.iter().enumerate() {
        for axis in 0..2 {
            if i64::from(point[axis]).abs() > MAX_ABS_COORDINATE as i64 {
                bail!(
                    "OpenCV-compatible polygon point {point_index} exceeds supported coordinate magnitude {MAX_ABS_COORDINATE}"
                );
            }
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    if i64::from(maximum[0]) - i64::from(minimum[0]) > MAX_POLYGON_SPAN as i64
        || i64::from(maximum[1]) - i64::from(minimum[1]) > MAX_POLYGON_SPAN as i64
    {
        bail!("OpenCV-compatible polygon span exceeds {MAX_POLYGON_SPAN}");
    }
    let width_i32 = i32::try_from(width).context("mask width exceeds int32")?;
    let height_i32 = i32::try_from(height).context("mask height exceeds int32")?;
    let mut mask = vec![0u8; pixels];
    let mut edges = Vec::with_capacity(polygon_xy.len());
    let mut point_0 = *polygon_xy
        .last()
        .context("OpenCV-compatible polygon is empty")?;

    for &point_1 in polygon_xy {
        draw_line_8(&mut mask, width_i32, height_i32, point_0, point_1)?;

        let mut clipped_0 = point_0;
        let mut clipped_1 = point_1;
        let mut point_0c = [i64::from(point_0[0]) << XY_SHIFT, i64::from(point_0[1])];
        let mut point_1c = [i64::from(point_1[0]) << XY_SHIFT, i64::from(point_1[1])];
        if outside(clipped_0, width_i32, height_i32) || outside(clipped_1, width_i32, height_i32) {
            let _ = clip_line([width_i32, height_i32], &mut clipped_0, &mut clipped_1)?;
            if clipped_0[1] != clipped_1[1] {
                point_0c[1] = i64::from(clipped_0[1]);
                point_1c[1] = i64::from(clipped_1[1]);
            }
        }
        point_0c[0] = i64::from(clipped_0[0]) << XY_SHIFT;
        point_1c[0] = i64::from(clipped_1[0]) << XY_SHIFT;

        if point_0[1] != point_1[1] {
            let denominator = point_1c[1] - point_0c[1];
            if denominator == 0 {
                bail!("OpenCV-compatible clipped edge has zero height");
            }
            let dx = (point_1c[0] - point_0c[0]) / denominator;
            let edge = if point_0[1] < point_1[1] {
                PolyEdge {
                    y0: point_0[1],
                    y1: point_1[1],
                    x: point_0c[0] + (i64::from(point_0[1]) - point_0c[1]) * dx,
                    dx,
                }
            } else {
                PolyEdge {
                    y0: point_1[1],
                    y1: point_0[1],
                    x: point_1c[0] + (i64::from(point_1[1]) - point_1c[1]) * dx,
                    dx,
                }
            };
            edges.push(edge);
        }
        point_0 = point_1;
    }
    fill_edge_collection(&mut mask, width_i32, height_i32, &mut edges)?;
    Ok(mask)
}

fn outside(point: [i32; 2], width: i32, height: i32) -> bool {
    point[0] < 0 || point[0] >= width || point[1] < 0 || point[1] >= height
}

fn clip_code([x, y]: [i32; 2], right: i32, bottom: i32) -> u8 {
    u8::from(x < 0)
        | (u8::from(x > right) << 1)
        | (u8::from(y < 0) << 2)
        | (u8::from(y > bottom) << 3)
}

fn clipped_delta(numerator: i64, multiplier: i64, denominator: i64) -> Result<i64> {
    if denominator == 0 {
        bail!("OpenCV-compatible line clipping divided by zero");
    }
    let value = (numerator as f64) * (multiplier as f64) / (denominator as f64);
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        bail!("OpenCV-compatible line clipping overflow");
    }
    Ok(value as i64)
}

fn clip_line(shape_wh: [i32; 2], point_1: &mut [i32; 2], point_2: &mut [i32; 2]) -> Result<bool> {
    let [width, height] = shape_wh;
    if width <= 0 || height <= 0 {
        return Ok(false);
    }
    let right = width - 1;
    let bottom = height - 1;
    let (mut x1, mut y1) = (i64::from(point_1[0]), i64::from(point_1[1]));
    let (mut x2, mut y2) = (i64::from(point_2[0]), i64::from(point_2[1]));
    let mut code_1 = clip_code(*point_1, right, bottom);
    let mut code_2 = clip_code(*point_2, right, bottom);

    if code_1 & code_2 == 0 && code_1 | code_2 != 0 {
        if code_1 & 12 != 0 {
            let target = i64::from(if code_1 < 8 { 0 } else { bottom });
            x1 += clipped_delta(target - y1, x2 - x1, y2 - y1)?;
            y1 = target;
            code_1 = u8::from(x1 < 0) | (u8::from(x1 > i64::from(right)) << 1);
        }
        if code_2 & 12 != 0 {
            let target = i64::from(if code_2 < 8 { 0 } else { bottom });
            x2 += clipped_delta(target - y2, x2 - x1, y2 - y1)?;
            y2 = target;
            code_2 = u8::from(x2 < 0) | (u8::from(x2 > i64::from(right)) << 1);
        }
        if code_1 & code_2 == 0 && code_1 | code_2 != 0 {
            if code_1 != 0 {
                let target = i64::from(if code_1 == 1 { 0 } else { right });
                y1 += clipped_delta(target - x1, y2 - y1, x2 - x1)?;
                x1 = target;
                code_1 = 0;
            }
            if code_2 != 0 {
                let target = i64::from(if code_2 == 1 { 0 } else { right });
                y2 += clipped_delta(target - x2, y2 - y1, x2 - x1)?;
                x2 = target;
                code_2 = 0;
            }
        }
    }
    point_1[0] = i32::try_from(x1).context("clipped line x1 exceeds int32")?;
    point_1[1] = i32::try_from(y1).context("clipped line y1 exceeds int32")?;
    point_2[0] = i32::try_from(x2).context("clipped line x2 exceeds int32")?;
    point_2[1] = i32::try_from(y2).context("clipped line y2 exceeds int32")?;
    Ok(code_1 | code_2 == 0)
}

fn draw_line_8(
    mask: &mut [u8],
    width: i32,
    height: i32,
    mut point_1: [i32; 2],
    mut point_2: [i32; 2],
) -> Result<()> {
    if (outside(point_1, width, height) || outside(point_2, width, height))
        && !clip_line([width, height], &mut point_1, &mut point_2)?
    {
        return Ok(());
    }

    let mut delta_x = 1i32;
    let mut delta_y = 1i32;
    let mut dx = point_2[0] - point_1[0];
    let mut dy = point_2[1] - point_1[1];
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        std::mem::swap(&mut point_1, &mut point_2);
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vertical = dy > dx;
    if vertical {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut error = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    let mut minus_shift = delta_x;
    let mut plus_shift = 0;
    let mut minus_step = 0;
    let mut plus_step = delta_y;
    if vertical {
        std::mem::swap(&mut plus_step, &mut plus_shift);
        std::mem::swap(&mut minus_step, &mut minus_shift);
    }
    let mut point = point_1;
    for _ in 0..=dx {
        let index = usize::try_from(point[1])?
            .checked_mul(usize::try_from(width)?)
            .and_then(|offset| offset.checked_add(usize::try_from(point[0]).ok()?))
            .context("OpenCV-compatible line pixel offset overflow")?;
        mask[index] = 1;
        let negative = error < 0;
        error += minus_delta + if negative { plus_delta } else { 0 };
        point[0] += minus_shift + if negative { plus_shift } else { 0 };
        point[1] += minus_step + if negative { plus_step } else { 0 };
    }
    Ok(())
}

fn fill_edge_collection(
    mask: &mut [u8],
    width: i32,
    height: i32,
    edges: &mut [PolyEdge],
) -> Result<()> {
    if edges.len() < 2 {
        return Ok(());
    }
    let mut y_min = i32::MAX;
    let mut y_max = i32::MIN;
    let mut x_min = i64::MAX;
    let mut x_max = -1i64;
    for edge in edges.iter() {
        if edge.y0 >= edge.y1 {
            bail!("OpenCV-compatible polygon edge is not upward ordered");
        }
        let end_x = edge
            .x
            .checked_add(
                i64::from(edge.y1 - edge.y0)
                    .checked_mul(edge.dx)
                    .context("OpenCV-compatible polygon edge endpoint overflow")?,
            )
            .context("OpenCV-compatible polygon edge endpoint overflow")?;
        y_min = y_min.min(edge.y0);
        y_max = y_max.max(edge.y1);
        x_min = x_min.min(edge.x).min(end_x);
        x_max = x_max.max(edge.x).max(end_x);
    }
    if y_max < 0 || y_min >= height || x_max < 0 || x_min >= (i64::from(width) << XY_SHIFT) {
        return Ok(());
    }

    edges.sort_by(|left, right| {
        left.y0
            .cmp(&right.y0)
            .then(left.x.cmp(&right.x))
            .then(left.dx.cmp(&right.dx))
    });
    let mut next_edge = 0usize;
    let mut active = Vec::<PolyEdge>::new();
    let stop_y = y_max.min(height);
    let mut y = edges[0].y0;
    while y < stop_y {
        active.retain(|edge| edge.y1 != y);

        let first_new = next_edge;
        while next_edge < edges.len() && edges[next_edge].y0 == y {
            next_edge += 1;
        }
        if first_new != next_edge {
            let old = std::mem::take(&mut active);
            let new = &edges[first_new..next_edge];
            active.reserve(old.len() + new.len());
            let (mut old_index, mut new_index) = (0usize, 0usize);
            while old_index < old.len() || new_index < new.len() {
                let take_old = new_index == new.len()
                    || (old_index < old.len() && old[old_index].x < new[new_index].x);
                if take_old {
                    active.push(old[old_index]);
                    old_index += 1;
                } else {
                    active.push(new[new_index]);
                    new_index += 1;
                }
            }
        }

        let (pairs, _) = active.as_chunks_mut::<2>();
        for pair in pairs {
            if y >= 0 {
                let (left, right) = if pair[0].x <= pair[1].x {
                    (pair[0].x, pair[1].x)
                } else {
                    (pair[1].x, pair[0].x)
                };
                let mut x1 = (left + (XY_ONE - 1)) >> XY_SHIFT;
                let mut x2 = right >> XY_SHIFT;
                if x1 < i64::from(width) && x2 >= 0 {
                    x1 = x1.max(0);
                    x2 = x2.min(i64::from(width - 1));
                    let row = usize::try_from(y)?
                        .checked_mul(usize::try_from(width)?)
                        .context("OpenCV-compatible fill row overflow")?;
                    for x in x1..=x2 {
                        mask[row + usize::try_from(x)?] = 1;
                    }
                }
            }
            pair[0].x = pair[0]
                .x
                .checked_add(pair[0].dx)
                .context("OpenCV-compatible active edge overflow")?;
            pair[1].x = pair[1]
                .x
                .checked_add(pair[1].dx)
                .context("OpenCV-compatible active edge overflow")?;
        }
        active.sort_by(|left, right| {
            if left.x < right.x {
                Ordering::Less
            } else if left.x > right.x {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        y += 1;
    }
    Ok(())
}
