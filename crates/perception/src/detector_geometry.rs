//! Deterministic, model-free geometry for regional detector calls.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use anyhow::{Context, Result, bail};

pub(crate) mod bounded_clipper_offset;

/// Source-space rectangle `[left, top, right, bottom]` with exclusive right and
/// bottom edges.
pub type DetectorRect = [usize; 4];

/// Ordered source-space detector quadrilateral in `[x, y]` coordinates.
pub type IntegerDetectorQuad = [[u32; 2]; 4];

/// One terminal regional detector call from [`strict_detector_hierarchy`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorRegion {
    pub bounds_xyxy: DetectorRect,
    /// Occurrence IDs assigned after sorting proposal geometry. Exact duplicate
    /// rectangles remain distinct members even though they share one leaf call.
    pub proposal_members: Vec<usize>,
    pub detector_shape_hw: [usize; 2],
    pub detector_pixels: u64,
}

/// One accepted merge, in deterministic execution order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorMerge {
    pub operand_pixels: [u64; 2],
    pub result_rect: DetectorRect,
    pub result_members: Vec<usize>,
    pub result_pixels: u64,
    pub saving: u64,
}

/// Cost and work accounting for one hierarchy construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorHierarchyProfile {
    pub natural_proposals: usize,
    pub deduplicated_leaves: usize,
    pub terminal_regions: usize,
    pub initial_detector_pixels: u64,
    pub terminal_detector_pixels: u64,
    pub evaluated_pairs: u64,
    pub stale_heap_pops: u64,
}

/// Deterministic fixed-point result of strict detector-cost merging.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetectorHierarchy {
    pub regions: Vec<DetectorRegion>,
    pub merges: Vec<DetectorMerge>,
    pub profile: DetectorHierarchyProfile,
}

/// Decision for one suffix item passed to [`exact_fuse_quads`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuadFusionDecision {
    pub suffix_index: usize,
    pub appended: bool,
    pub fused_index: Option<usize>,
}

/// Ordered exact-identity fusion result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuadFusion {
    pub quads: Vec<IntegerDetectorQuad>,
    pub decisions: Vec<QuadFusionDecision>,
}

// The minimum-area geometry implementation below is a modified Rust
// reproduction of OpenCV 4.12.0. Source attribution and license terms are in
// `third_party/opencv-4.12.0/NOTICE.md`.

// Integer coordinates are limited well below f32's exact-integer boundary.
// The DB operands observed to date are below 2,000 in magnitude; 2^20 leaves
// more than 500x headroom while bounding every difference and cross-product.
const OPENCV_DB_MAX_ABS_COORDINATE: i32 = 1 << 20;

// Paddle's projected box remains float32 until its final integer conversion.
// Positive map/source extents through 2^24 and every rounded coordinate in
// their inclusive range are therefore exactly representable as f32 integers.
const OPENCV_DB_MAX_EXACT_FLOAT_EXTENT: u32 = 1 << 24;

/// OpenCV-compatible representation of a minimum-area rotated rectangle.
///
/// The field layout and arithmetic sequence follow OpenCV 4.12.0's
/// `cv::RotatedRect` result for an integral point set. In particular, the
/// convex hull is computed in integer coordinates before the rotating-calipers
/// input is converted to `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OpenCvRotatedRect {
    pub(crate) center_xy: [f32; 2],
    pub(crate) size_wh: [f32; 2],
    pub(crate) angle_degrees: f32,
}

/// Follow OpenCV 4.12.0 `minAreaRect` for an integral point set.
///
/// This ports the Sklansky convex hull and rotating-calipers arithmetic from
/// OpenCV without linking to OpenCV. The inner value is `None` only for an
/// empty input. One- and two-point inputs retain OpenCV's degenerate result.
pub(crate) fn opencv_min_area_rect_i32(
    points_xy: &[[i32; 2]],
) -> Result<Option<OpenCvRotatedRect>> {
    validate_opencv_db_coordinates(points_xy)?;
    let hull = opencv_convex_hull_i32(points_xy);
    let rect = match hull.as_slice() {
        [] => return Ok(None),
        [[x, y]] => OpenCvRotatedRect {
            center_xy: [*x as f32, *y as f32],
            size_wh: [0.0, 0.0],
            angle_degrees: 0.0,
        },
        [[x0, y0], [x1, y1]] => {
            let p0 = [*x0 as f32, *y0 as f32];
            let p1 = [*x1 as f32, *y1 as f32];
            let center_xy = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
            let dx = (p1[0] - p0[0]) as f64;
            let dy = (p1[1] - p0[1]) as f64;
            let width = (dx * dx + dy * dy).sqrt() as f32;
            let angle_radians = dy.atan2(dx) as f32;
            OpenCvRotatedRect {
                center_xy,
                size_wh: [width, 0.0],
                angle_degrees: (f64::from(angle_radians) * 180.0 / std::f64::consts::PI) as f32,
            }
        }
        _ => opencv_rotating_calipers_min_area(&hull)?,
    };
    validate_opencv_rect(rect)?;
    Ok(Some(rect))
}

/// Follow OpenCV 4.12.0 `RotatedRect::points` / `boxPoints` ordering.
pub(crate) fn opencv_box_points(rect: OpenCvRotatedRect) -> Result<[[f32; 2]; 4]> {
    let angle = f64::from(rect.angle_degrees) * std::f64::consts::PI / 180.0;
    let b = (angle.cos() as f32) * 0.5;
    let a = (angle.sin() as f32) * 0.5;
    let [center_x, center_y] = rect.center_xy;
    let [width, height] = rect.size_wh;
    let point_0 = [
        center_x - a * height - b * width,
        center_y + b * height - a * width,
    ];
    let point_1 = [
        center_x + a * height - b * width,
        center_y - b * height - a * width,
    ];
    let points = [
        point_0,
        point_1,
        [2.0 * center_x - point_0[0], 2.0 * center_y - point_0[1]],
        [2.0 * center_x - point_1[0], 2.0 * center_y - point_1[1]],
    ];
    if points.iter().flatten().any(|value| !value.is_finite()) {
        bail!("OpenCV-compatible box points are non-finite");
    }
    Ok(points)
}

// The ordering and projection implementation below is a modified Rust
// reproduction of PaddleOCR's DB postprocessor. Its separate source
// attribution and license terms are in
// `third_party/paddleocr-db-postprocess/NOTICE.md`.

/// Reproduce PaddleOCR's stable-x/y selection of OpenCV box points.
///
/// The result order is top-left, top-right, bottom-right, bottom-left in the
/// image-coordinate convention used by DB postprocessing.
pub(crate) fn order_opencv_db_box(points_xy: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let mut points = points_xy;
    points.sort_by(|left, right| left[0].total_cmp(&right[0]));
    let (index_1, index_4) = if points[1][1] > points[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (index_2, index_3) = if points[3][1] > points[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };
    [
        points[index_1],
        points[index_2],
        points[index_3],
        points[index_4],
    ]
}

/// Reproduce PaddleOCR's staged NumPy projection, ties-to-even rounding, and
/// inclusive source-bound clipping for one ordered DB box.
///
/// The official call builds `shape_list` with NumPy's default `float64` dtype.
/// Consequently, `box / map_extent` is evaluated as `float32` before the
/// multiplication by the `float64` source extent promotes the result.
pub(crate) fn project_opencv_db_box(
    box_xy: [[f32; 2]; 4],
    map_shape_hw: [u32; 2],
    source_shape_hw: [u32; 2],
) -> Result<IntegerDetectorQuad> {
    let [map_height, map_width] = map_shape_hw;
    let [source_height, source_width] = source_shape_hw;
    for (kind, dimensions) in [
        ("map", [map_height, map_width]),
        ("source", [source_height, source_width]),
    ] {
        if dimensions.contains(&0) {
            bail!("DB {kind} dimensions must be positive");
        }
        if dimensions
            .into_iter()
            .any(|extent| extent > OPENCV_DB_MAX_EXACT_FLOAT_EXTENT)
        {
            bail!(
                "DB {kind} dimensions exceed the exact-float-safe extent {OPENCV_DB_MAX_EXACT_FLOAT_EXTENT}"
            );
        }
    }
    let project = |value: f32, map_extent: u32, source_extent: u32| -> Result<u32> {
        if !value.is_finite() {
            bail!("DB box contains a non-finite coordinate");
        }
        let normalized = value / map_extent as f32;
        let projected = f64::from(normalized) * f64::from(source_extent);
        if !projected.is_finite() {
            bail!("DB projection produced a non-finite coordinate");
        }
        let projected = projected
            .round_ties_even()
            .clamp(0.0, f64::from(source_extent));
        Ok(projected as u32)
    };
    let mut projected = [[0_u32; 2]; 4];
    for (output, [x, y]) in projected.iter_mut().zip(box_xy) {
        *output = [
            project(x, map_width, source_width)?,
            project(y, map_height, source_height)?,
        ];
    }
    Ok(projected)
}

/// Apply the bounded OpenCV minimum-area rectangle, PaddleOCR box ordering,
/// and NumPy projection seam to an already-offset integral DB path.
///
/// Every path coordinate must have absolute magnitude at most 2^20. This is
/// more than 500 times the largest frozen operand while keeping integer inputs
/// and their differences exact in the subsequent `f32` arithmetic. Inputs
/// outside that contract fail before hull or caliper arithmetic. Map and source
/// dimensions must be positive and at most 2^24 so Paddle's intermediate and
/// final integer-valued `f32` coordinates remain exactly representable.
pub fn opencv_db_projected_quad(
    offset_path_xy: &[[i32; 2]],
    map_shape_hw: [u32; 2],
    source_shape_hw: [u32; 2],
) -> Result<IntegerDetectorQuad> {
    let rect = opencv_min_area_rect_i32(offset_path_xy)?
        .context("offset path produced no minimum-area rectangle")?;
    let ordered = order_opencv_db_box(opencv_box_points(rect)?);
    project_opencv_db_box(ordered, map_shape_hw, source_shape_hw)
}

fn validate_opencv_db_coordinates(points_xy: &[[i32; 2]]) -> Result<()> {
    for (point_index, point) in points_xy.iter().enumerate() {
        for (axis, coordinate) in ["x", "y"].into_iter().zip(point) {
            if i64::from(*coordinate).abs() > i64::from(OPENCV_DB_MAX_ABS_COORDINATE) {
                bail!(
                    "DB path point {point_index} {axis} coordinate exceeds the supported magnitude {OPENCV_DB_MAX_ABS_COORDINATE}"
                );
            }
        }
    }
    Ok(())
}

fn sign_i64(value: i64) -> i32 {
    i32::from(value > 0) - i32::from(value < 0)
}

fn sign_i128(value: i128) -> i32 {
    i32::from(value > 0) - i32::from(value < 0)
}

fn validate_opencv_rect(rect: OpenCvRotatedRect) -> Result<()> {
    if rect
        .center_xy
        .into_iter()
        .chain(rect.size_wh)
        .chain([rect.angle_degrees])
        .any(|value| !value.is_finite())
    {
        bail!("OpenCV-compatible minimum-area rectangle is non-finite");
    }
    Ok(())
}

fn sklansky_i32(
    points: &[[i32; 2]],
    sorted: &[usize],
    start: isize,
    end: isize,
    normal_sign: i32,
    convexity_sign: i32,
) -> Vec<isize> {
    let increment = if end > start { 1 } else { -1 };
    let mut previous = start;
    let mut current = previous + increment;
    let mut next = current + increment;
    if start == end || points[sorted[start as usize]] == points[sorted[end as usize]] {
        return vec![start];
    }

    let mut stack = vec![previous, current, next];
    let after_end = end + increment;
    while next != after_end {
        let current_point = points[sorted[current as usize]];
        let next_point = points[sorted[next as usize]];
        let by = i64::from(next_point[1]) - i64::from(current_point[1]);
        if sign_i64(by) != normal_sign {
            let previous_point = points[sorted[previous as usize]];
            let ax = i64::from(current_point[0]) - i64::from(previous_point[0]);
            let bx = i64::from(next_point[0]) - i64::from(current_point[0]);
            let ay = i64::from(current_point[1]) - i64::from(previous_point[1]);
            let convexity = i128::from(ay) * i128::from(bx) - i128::from(ax) * i128::from(by);
            if sign_i128(convexity) == convexity_sign && (ax != 0 || ay != 0) {
                previous = current;
                current = next;
                next += increment;
                stack.push(next);
            } else if previous == start {
                current = next;
                stack[1] = current;
                next += increment;
                stack[2] = next;
            } else {
                let len = stack.len();
                stack[len - 2] = next;
                current = previous;
                previous = stack[len - 4];
                stack.pop();
            }
        } else {
            next += increment;
            let last = stack.len() - 1;
            stack[last] = next;
        }
    }
    stack.pop();
    stack
}

fn opencv_convex_hull_i32(points: &[[i32; 2]]) -> Vec<[i32; 2]> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<_> = (0..points.len()).collect();
    sorted.sort_unstable_by_key(|&index| (points[index][0], points[index][1], index));
    let mut min_y_index = 0usize;
    let mut max_y_index = 0usize;
    for index in 1..sorted.len() {
        let y = points[sorted[index]][1];
        if points[sorted[min_y_index]][1] > y {
            min_y_index = index;
        }
        if points[sorted[max_y_index]][1] < y {
            max_y_index = index;
        }
    }
    if points[sorted[0]] == points[sorted[sorted.len() - 1]] {
        return vec![points[0]];
    }

    let last = (sorted.len() - 1) as isize;
    let mut top_left = sklansky_i32(points, &sorted, 0, max_y_index as isize, -1, 1);
    let mut top_right = sklansky_i32(points, &sorted, last, max_y_index as isize, -1, -1);
    // `clockwise=false`: OpenCV swaps the two upper stacks.
    std::mem::swap(&mut top_left, &mut top_right);
    let mut hull = Vec::with_capacity(points.len());
    hull.extend(
        top_left[..top_left.len().saturating_sub(1)]
            .iter()
            .map(|&position| sorted[position as usize]),
    );
    hull.extend(
        top_right[1..]
            .iter()
            .rev()
            .map(|&position| sorted[position as usize]),
    );
    let stop_index = if top_right.len() > 2 {
        Some(top_right[1])
    } else if top_left.len() > 2 {
        Some(top_left[top_left.len() - 2])
    } else {
        None
    };

    let mut bottom_left = sklansky_i32(points, &sorted, 0, min_y_index as isize, 1, -1);
    let mut bottom_right = sklansky_i32(points, &sorted, last, min_y_index as isize, 1, 1);
    if let Some(stop) = stop_index {
        let check = if bottom_left.len() > 2 {
            Some(bottom_left[1])
        } else if bottom_left.len() + bottom_right.len() > 2 {
            Some(bottom_right[2 - bottom_left.len()])
        } else {
            None
        };
        if check.is_some_and(|position| {
            position == stop || points[sorted[position as usize]] == points[sorted[stop as usize]]
        }) {
            bottom_left.truncate(2);
            bottom_right.truncate(2);
        }
    }
    hull.extend(
        bottom_left[..bottom_left.len().saturating_sub(1)]
            .iter()
            .map(|&position| sorted[position as usize]),
    );
    hull.extend(
        bottom_right[1..]
            .iter()
            .rev()
            .map(|&position| sorted[position as usize]),
    );

    // OpenCV cyclically shifts only when this makes original input indices a
    // fully ascending or descending sequence.
    if hull.len() >= 3 {
        let mut min_position = 0usize;
        let mut max_position = 0usize;
        let mut less_than_count = 0usize;
        let mut scanned_all = true;
        for index in 1..hull.len() {
            let value = hull[index];
            less_than_count += usize::from(hull[index - 1] < value);
            if less_than_count > 1 && less_than_count <= index.saturating_sub(2) {
                scanned_all = false;
                break;
            }
            if value < hull[min_position] {
                min_position = index;
            }
            if value > hull[max_position] {
                max_position = index;
            }
        }
        if scanned_all {
            let distance = max_position.abs_diff(min_position);
            if (distance == 1 || distance == hull.len() - 1)
                && (less_than_count <= 1 || less_than_count >= hull.len() - 2)
            {
                let ascending = (max_position + 1) % hull.len() == min_position;
                let first = if ascending {
                    min_position
                } else {
                    max_position
                };
                if first > 0 {
                    let rotated: Vec<_> = (0..hull.len())
                        .map(|offset| hull[(first + offset) % hull.len()])
                        .collect();
                    let monotonic = rotated
                        .windows(2)
                        .all(|pair| ascending == (pair[0] < pair[1]));
                    if monotonic {
                        hull = rotated;
                    }
                }
            }
        }
    }
    hull.into_iter().map(|index| points[index]).collect()
}

fn first_vector_is_right(first: [f32; 2], second: [f32; 2]) -> bool {
    let clockwise = [first[1], -first[0]];
    clockwise[0] * second[0] + clockwise[1] * second[1] < 0.0
}

fn opencv_rotating_calipers_min_area(hull_i32: &[[i32; 2]]) -> Result<OpenCvRotatedRect> {
    let points: Vec<_> = hull_i32
        .iter()
        .map(|&[x, y]| [x as f32, y as f32])
        .collect();
    let count = points.len();
    let mut vectors = vec![[0.0f32; 2]; count];
    let mut inverse_lengths = vec![0.0f32; count];
    let mut left = 0usize;
    let mut bottom = 0usize;
    let mut right = 0usize;
    let mut top = 0usize;
    let [mut left_x, mut bottom_y] = points[0];
    let mut right_x = left_x;
    let mut top_y = bottom_y;
    let mut point_0 = points[0];
    for index in 0..count {
        if point_0[0] < left_x {
            left_x = point_0[0];
            left = index;
        }
        if point_0[0] > right_x {
            right_x = point_0[0];
            right = index;
        }
        if point_0[1] > top_y {
            top_y = point_0[1];
            top = index;
        }
        if point_0[1] < bottom_y {
            bottom_y = point_0[1];
            bottom = index;
        }
        let point = points[(index + 1) % count];
        let dx = (point[0] - point_0[0]) as f64;
        let dy = (point[1] - point_0[1]) as f64;
        let squared_length = dx * dx + dy * dy;
        if !squared_length.is_finite() || squared_length <= 0.0 {
            bail!("OpenCV-compatible convex hull contains an invalid edge");
        }
        vectors[index] = [dx as f32, dy as f32];
        inverse_lengths[index] = (1.0 / squared_length.sqrt()) as f32;
        if !inverse_lengths[index].is_finite() {
            bail!("OpenCV-compatible inverse edge length is non-finite");
        }
        point_0 = point;
    }
    let mut orientation = 0.0f32;
    let [mut ax, mut ay] = vectors[count - 1].map(f64::from);
    for vector in &vectors {
        let [bx, by] = vector.map(f64::from);
        let convexity = ax * by - ay * bx;
        if convexity != 0.0 {
            orientation = if convexity > 0.0 { 1.0 } else { -1.0 };
            break;
        }
        ax = bx;
        ay = by;
    }
    if orientation == 0.0 {
        bail!("OpenCV-compatible convex hull has no finite orientation");
    }

    let mut sequence = [bottom, right, top, left];
    let mut minimum_area = f32::MAX;
    let mut best_left = 0usize;
    let mut best_bottom = 0usize;
    let mut best_width = 0.0f32;
    let mut best_height = 0.0f32;
    let mut best_a = 0.0f32;
    let mut best_b = 0.0f32;
    for _ in 0..count {
        let rotated = [
            vectors[sequence[0]],
            [vectors[sequence[1]][1], -vectors[sequence[1]][0]],
            [-vectors[sequence[2]][0], -vectors[sequence[2]][1]],
            [-vectors[sequence[3]][1], vectors[sequence[3]][0]],
        ];
        let mut main_element = 0usize;
        for index in 1..4 {
            if first_vector_is_right(rotated[index], rotated[main_element]) {
                main_element = index;
            }
        }
        let path_index = sequence[main_element];
        let lead_x = vectors[path_index][0] * inverse_lengths[path_index];
        let lead_y = vectors[path_index][1] * inverse_lengths[path_index];
        let [base_a, base_b] = match main_element {
            0 => [lead_x, lead_y],
            1 => [lead_y, -lead_x],
            2 => [-lead_x, -lead_y],
            3 => [-lead_y, lead_x],
            _ => unreachable!(),
        };
        sequence[main_element] = (sequence[main_element] + 1) % count;

        let dx = points[sequence[1]][0] - points[sequence[3]][0];
        let dy = points[sequence[1]][1] - points[sequence[3]][1];
        let width = dx * base_a + dy * base_b;
        let dx = points[sequence[2]][0] - points[sequence[0]][0];
        let dy = points[sequence[2]][1] - points[sequence[0]][1];
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if !area.is_finite() {
            bail!("OpenCV-compatible caliper area is non-finite");
        }
        if area <= minimum_area {
            minimum_area = area;
            best_left = sequence[3];
            best_bottom = sequence[0];
            best_a = base_a;
            best_b = base_b;
            best_width = width;
            best_height = height;
        }
    }

    let a_1 = best_a;
    let b_1 = best_b;
    let a_2 = -best_b;
    let b_2 = best_a;
    let c_1 = a_1 * points[best_left][0] + points[best_left][1] * b_1;
    let c_2 = a_2 * points[best_bottom][0] + points[best_bottom][1] * b_2;
    let determinant = a_1 * b_2 - a_2 * b_1;
    if !determinant.is_finite() || determinant == 0.0 {
        bail!("OpenCV-compatible caliper determinant is invalid");
    }
    let inverse_determinant = 1.0 / determinant;
    let corner_x = (c_1 * b_2 - c_2 * b_1) * inverse_determinant;
    let corner_y = (a_1 * c_2 - a_2 * c_1) * inverse_determinant;
    let vector_1 = [a_1 * best_width, b_1 * best_width];
    let vector_2 = [a_2 * best_height, b_2 * best_height];
    let center_xy = [
        corner_x + (vector_1[0] + vector_2[0]) * 0.5,
        corner_y + (vector_1[1] + vector_2[1]) * 0.5,
    ];
    let width = (f64::from(vector_1[0]) * f64::from(vector_1[0])
        + f64::from(vector_1[1]) * f64::from(vector_1[1]))
    .sqrt() as f32;
    let height = (f64::from(vector_2[0]) * f64::from(vector_2[0])
        + f64::from(vector_2[1]) * f64::from(vector_2[1]))
    .sqrt() as f32;
    let angle_radians = f64::from(vector_1[1]).atan2(f64::from(vector_1[0])) as f32;
    let rect = OpenCvRotatedRect {
        center_xy,
        size_wh: [width, height],
        angle_degrees: (f64::from(angle_radians) * 180.0 / std::f64::consts::PI) as f32,
    };
    validate_opencv_rect(rect)?;
    Ok(rect)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Node {
    node_id: usize,
    rect: DetectorRect,
    members: Vec<usize>,
    pixels: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Choice {
    // The outer heap is a min-heap. Reverse the saving so larger savings win.
    saving_key: Reverse<u64>,
    rect: DetectorRect,
    members: Vec<usize>,
    left_id: usize,
    right_id: usize,
    pixels: u64,
}

#[derive(Clone, Copy)]
struct DetectorCost {
    numerator: u128,
    denominator: u128,
}

impl DetectorCost {
    fn from_shape(source_shape_hw: [usize; 2]) -> Result<Self> {
        let [height, width] = source_shape_hw;
        if height == 0 || width == 0 {
            bail!("source dimensions must be positive");
        }
        let [full_height, full_width] = full_detector_shape(source_shape_hw)?;
        let height_product = (full_height as u128) * (width as u128);
        let width_product = (full_width as u128) * (height as u128);
        Ok(if height_product >= width_product {
            Self {
                numerator: full_height as u128,
                denominator: height as u128,
            }
        } else {
            Self {
                numerator: full_width as u128,
                denominator: width as u128,
            }
        })
    }

    fn aligned(self, extent: usize) -> Result<usize> {
        if extent == 0 {
            bail!("rectangle extents must be positive");
        }
        let stride_denominator = 32_u128
            .checked_mul(self.denominator)
            .context("detector stride denominator overflow")?;
        let scaled = (extent as u128)
            .checked_mul(self.numerator)
            .context("detector extent overflow")?;
        let strides = scaled
            .checked_add(stride_denominator - 1)
            .context("detector alignment overflow")?
            / stride_denominator;
        usize::try_from(
            strides
                .checked_mul(32)
                .context("detector aligned extent overflow")?,
        )
        .context("detector aligned extent exceeds usize")
    }

    fn shape(self, rect: DetectorRect) -> Result<[usize; 2]> {
        Ok([
            self.aligned(rect[3] - rect[1])?,
            self.aligned(rect[2] - rect[0])?,
        ])
    }

    fn pixels(self, rect: DetectorRect) -> Result<u64> {
        let [height, width] = self.shape(rect)?;
        u64::try_from(height)
            .context("detector height exceeds u64")?
            .checked_mul(u64::try_from(width).context("detector width exceeds u64")?)
            .context("detector pixel count overflow")
    }
}

/// Nominal full-frame detector shape with short side 736 and stride-32 long
/// side alignment.
pub fn full_detector_shape(source_shape_hw: [usize; 2]) -> Result<[usize; 2]> {
    let [height, width] = source_shape_hw;
    let short = height.min(width);
    let long = height.max(width);
    if short == 0 {
        bail!("source dimensions must be positive");
    }
    let scaled_long = (long as u128)
        .checked_mul(736)
        .and_then(|value| value.checked_add((short / 2) as u128))
        .context("full detector shape overflow")?
        / short as u128;
    let aligned_long = scaled_long
        .checked_add(16)
        .context("full detector alignment overflow")?
        / 32
        * 32;
    let aligned_long =
        usize::try_from(aligned_long).context("full detector extent exceeds usize")?;
    Ok(if height <= width {
        [736, aligned_long]
    } else {
        [aligned_long, 736]
    })
}

/// Return the exact stride-aligned detector shape charged for `rect`.
pub fn regional_detector_shape(
    source_shape_hw: [usize; 2],
    rect: DetectorRect,
) -> Result<[usize; 2]> {
    validate_rect(source_shape_hw, rect)?;
    DetectorCost::from_shape(source_shape_hw)?.shape(rect)
}

fn validate_rect(source_shape_hw: [usize; 2], rect: DetectorRect) -> Result<()> {
    let [height, width] = source_shape_hw;
    if height == 0 || width == 0 {
        bail!("source dimensions must be positive");
    }
    let [left, top, right, bottom] = rect;
    if left >= right || top >= bottom {
        bail!("proposal rectangle must have positive extent");
    }
    if right > width || bottom > height {
        bail!("proposal rectangle falls outside the source frame");
    }
    Ok(())
}

fn enclosure(left: DetectorRect, right: DetectorRect) -> DetectorRect {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn choice(left: &Node, right: &Node, cost: DetectorCost) -> Result<Option<Choice>> {
    let rect = enclosure(left.rect, right.rect);
    let pixels = cost.pixels(rect)?;
    let operand_pixels = left
        .pixels
        .checked_add(right.pixels)
        .context("merge operand pixel count overflow")?;
    let Some(saving) = operand_pixels.checked_sub(pixels) else {
        return Ok(None);
    };
    let mut members = Vec::with_capacity(left.members.len() + right.members.len());
    members.extend_from_slice(&left.members);
    members.extend_from_slice(&right.members);
    members.sort_unstable();
    Ok(Some(Choice {
        saving_key: Reverse(saving),
        rect,
        members,
        left_id: left.node_id.min(right.node_id),
        right_id: left.node_id.max(right.node_id),
        pixels,
    }))
}

/// Merge exact duplicate proposals into leaves, then greedily merge the pair
/// with the largest non-negative detector-pixel saving until no admissible pair
/// remains. Geometry, members, and node IDs fully break ties.
pub fn strict_detector_hierarchy(
    source_shape_hw: [usize; 2],
    proposals: &[DetectorRect],
) -> Result<DetectorHierarchy> {
    let cost = DetectorCost::from_shape(source_shape_hw)?;
    let mut counts = BTreeMap::new();
    for &rect in proposals {
        validate_rect(source_shape_hw, rect)?;
        *counts.entry(rect).or_insert(0usize) += 1;
    }

    let mut occurrence = 0usize;
    let mut leaves = Vec::with_capacity(counts.len());
    for (node_id, (rect, count)) in counts.into_iter().enumerate() {
        let end = occurrence
            .checked_add(count)
            .context("proposal occurrence count overflow")?;
        leaves.push(Node {
            node_id,
            rect,
            members: (occurrence..end).collect(),
            pixels: cost.pixels(rect)?,
        });
        occurrence = end;
    }

    let initial_detector_pixels = sum_pixels(leaves.iter().map(|node| node.pixels))?;
    let mut active: BTreeMap<usize, Node> = leaves
        .iter()
        .cloned()
        .map(|node| (node.node_id, node))
        .collect();
    let mut heap = BinaryHeap::new();
    let mut evaluated_pairs = 0u64;
    let mut add_pair =
        |left: &Node, right: &Node, heap: &mut BinaryHeap<Reverse<Choice>>| -> Result<()> {
            evaluated_pairs = evaluated_pairs.saturating_add(1);
            if let Some(candidate) = choice(left, right, cost)? {
                heap.push(Reverse(candidate));
            }
            Ok(())
        };
    for (left_index, left) in leaves.iter().enumerate() {
        for right in &leaves[left_index + 1..] {
            add_pair(left, right, &mut heap)?;
        }
    }

    let mut next_id = leaves.len();
    let mut stale_heap_pops = 0u64;
    let mut merges = Vec::new();
    while let Some(Reverse(selected)) = heap.pop() {
        let (Some(left), Some(right)) = (
            active.get(&selected.left_id).cloned(),
            active.get(&selected.right_id).cloned(),
        ) else {
            stale_heap_pops = stale_heap_pops.saturating_add(1);
            continue;
        };
        let result = Node {
            node_id: next_id,
            rect: selected.rect,
            members: selected.members,
            pixels: selected.pixels,
        };
        next_id = next_id
            .checked_add(1)
            .context("hierarchy node ID overflow")?;
        active.remove(&selected.left_id);
        active.remove(&selected.right_id);
        let mut operand_pixels = [left.pixels, right.pixels];
        operand_pixels.sort_unstable();
        merges.push(DetectorMerge {
            operand_pixels,
            result_rect: result.rect,
            result_members: result.members.clone(),
            result_pixels: result.pixels,
            saving: selected.saving_key.0,
        });
        for other in active.values() {
            add_pair(&result, other, &mut heap)?;
        }
        active.insert(result.node_id, result);
    }

    let mut terminal: Vec<_> = active.into_values().collect();
    terminal.sort_by(|left, right| {
        left.rect
            .cmp(&right.rect)
            .then(left.members.cmp(&right.members))
    });
    let terminal_detector_pixels = sum_pixels(terminal.iter().map(|node| node.pixels))?;
    let regions = terminal
        .into_iter()
        .map(|node| {
            Ok(DetectorRegion {
                bounds_xyxy: node.rect,
                proposal_members: node.members,
                detector_shape_hw: cost.shape(node.rect)?,
                detector_pixels: node.pixels,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DetectorHierarchy {
        profile: DetectorHierarchyProfile {
            natural_proposals: proposals.len(),
            deduplicated_leaves: leaves.len(),
            terminal_regions: regions.len(),
            initial_detector_pixels,
            terminal_detector_pixels,
            evaluated_pairs,
            stale_heap_pops,
        },
        regions,
        merges,
    })
}

fn sum_pixels(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .context("detector pixel sum overflow")
    })
}

/// Preserve `prefix` exactly, then append each suffix quad only when its four
/// ordered integer points have not occurred earlier. Near-overlaps are kept.
pub fn exact_fuse_quads(
    prefix: &[IntegerDetectorQuad],
    suffix: &[IntegerDetectorQuad],
) -> QuadFusion {
    let mut quads = prefix.to_vec();
    let mut seen: BTreeSet<_> = prefix.iter().copied().collect();
    let mut decisions = Vec::with_capacity(suffix.len());
    for (suffix_index, &quad) in suffix.iter().enumerate() {
        let appended = seen.insert(quad);
        let fused_index = appended.then(|| {
            let index = quads.len();
            quads.push(quad);
            index
        });
        decisions.push(QuadFusionDecision {
            suffix_index,
            appended,
            fused_index,
        });
    }
    QuadFusion { quads, decisions }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_hierarchy(
        source_shape_hw: [usize; 2],
        proposals: &[DetectorRect],
    ) -> Result<Vec<DetectorRegion>> {
        let cost = DetectorCost::from_shape(source_shape_hw)?;
        let mut counts = BTreeMap::new();
        for &rect in proposals {
            *counts.entry(rect).or_insert(0usize) += 1;
        }
        let mut occurrence = 0usize;
        let mut nodes = Vec::new();
        for (node_id, (rect, count)) in counts.into_iter().enumerate() {
            let end = occurrence + count;
            nodes.push(Node {
                node_id,
                rect,
                members: (occurrence..end).collect(),
                pixels: cost.pixels(rect)?,
            });
            occurrence = end;
        }
        let mut next_id = nodes.len();
        loop {
            let mut choices = Vec::new();
            for left_index in 0..nodes.len() {
                for right_index in left_index + 1..nodes.len() {
                    if let Some(selected) = choice(&nodes[left_index], &nodes[right_index], cost)? {
                        choices.push((selected, left_index, right_index));
                    }
                }
            }
            let Some((selected, left_index, right_index)) = choices.into_iter().min() else {
                break;
            };
            nodes = nodes
                .into_iter()
                .enumerate()
                .filter_map(|(index, node)| {
                    (index != left_index && index != right_index).then_some(node)
                })
                .chain(std::iter::once(Node {
                    node_id: next_id,
                    rect: selected.rect,
                    members: selected.members,
                    pixels: selected.pixels,
                }))
                .collect();
            next_id += 1;
            nodes.sort_by(|left, right| {
                left.rect
                    .cmp(&right.rect)
                    .then(left.members.cmp(&right.members))
            });
        }
        nodes
            .into_iter()
            .map(|node| {
                Ok(DetectorRegion {
                    bounds_xyxy: node.rect,
                    proposal_members: node.members,
                    detector_shape_hw: cost.shape(node.rect)?,
                    detector_pixels: node.pixels,
                })
            })
            .collect()
    }

    #[test]
    fn hierarchy_matches_recomputing_oracle_and_is_caller_order_independent() {
        let source = [720, 1280];
        let proposals = [
            [10, 10, 110, 40],
            [10, 10, 110, 40],
            [120, 10, 220, 40],
            [10, 100, 60, 130],
            [65, 100, 115, 130],
            [900, 600, 1000, 640],
        ];
        let expected = reference_hierarchy(source, &proposals).unwrap();
        let actual = strict_detector_hierarchy(source, &proposals).unwrap();
        assert_eq!(actual.regions, expected);
        let reversed: Vec<_> = proposals.iter().rev().copied().collect();
        assert_eq!(
            strict_detector_hierarchy(source, &reversed)
                .unwrap()
                .regions,
            expected
        );
        assert_eq!(actual.profile.natural_proposals, 6);
        assert_eq!(actual.profile.deduplicated_leaves, 5);
        assert!(actual.profile.terminal_detector_pixels <= actual.profile.initial_detector_pixels);
        assert_eq!(
            actual
                .regions
                .iter()
                .flat_map(|region| region.proposal_members.iter().copied())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1, 2, 3, 4, 5])
        );
    }

    #[test]
    fn detector_shapes_match_frozen_arithmetic() {
        assert_eq!(full_detector_shape([1920, 888]).unwrap(), [1600, 736]);
        assert_eq!(full_detector_shape([720, 1280]).unwrap(), [736, 1312]);
        assert_eq!(
            regional_detector_shape([1920, 888], [0, 233, 811, 341]).unwrap(),
            [96, 704]
        );
    }

    #[test]
    fn hierarchy_rejects_invalid_geometry() {
        assert!(strict_detector_hierarchy([0, 20], &[]).is_err());
        assert!(strict_detector_hierarchy([20, 20], &[[1, 1, 1, 5]]).is_err());
        assert!(strict_detector_hierarchy([20, 20], &[[1, 1, 21, 5]]).is_err());
    }

    #[test]
    fn exact_fusion_preserves_prefix_and_only_removes_exact_suffix_duplicates() {
        let a = [[0, 0], [4, 0], [4, 2], [0, 2]];
        let b = [[5, 0], [9, 0], [9, 2], [5, 2]];
        let near_a = [[0, 0], [4, 0], [4, 3], [0, 2]];
        let fusion = exact_fuse_quads(&[a], &[a, b, b, near_a]);
        assert_eq!(fusion.quads, vec![a, b, near_a]);
        assert_eq!(
            fusion.decisions,
            vec![
                QuadFusionDecision {
                    suffix_index: 0,
                    appended: false,
                    fused_index: None
                },
                QuadFusionDecision {
                    suffix_index: 1,
                    appended: true,
                    fused_index: Some(1)
                },
                QuadFusionDecision {
                    suffix_index: 2,
                    appended: false,
                    fused_index: None
                },
                QuadFusionDecision {
                    suffix_index: 3,
                    appended: true,
                    fused_index: Some(2)
                },
            ]
        );
    }
}
