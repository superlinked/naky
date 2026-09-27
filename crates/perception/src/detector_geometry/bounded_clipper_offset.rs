//! Bounded Clipper-style rounded offset for one DB rotated rectangle.
//!
//! This is a source-derived specialization of Clipper 6.4.2 as shipped by
//! Pyclipper 1.3.0.post6. Source attribution and license terms are in
//! `THIRD_PARTY_NOTICES.md` and `licenses/`.

use anyhow::{Context, Result, bail};

const MAX_ABS_INPUT_COORDINATE: f32 = (1_u32 << 20) as f32;
const MAX_OFFSET_DISTANCE: f64 = 4096.0;
const MAX_OUTPUT_VERTICES: usize = 1024;
const DEFAULT_ARC_TOLERANCE: f64 = 0.25;
const NEAR_ZERO: f64 = 1.0e-20;
const MIN_SHORT_SIDE: f32 = 3.0;
const BOX_POINTS_RELATIVE_TOLERANCE: f64 = 1.0 / 1024.0;
const BOX_POINTS_ABSOLUTE_TOLERANCE: f64 = 1.0 / 64.0;

type Point = [i64; 2];
type Normal = [f64; 2];

/// Expand one OpenCV DB rotated rectangle with Pyclipper 1.3.0.post6's default
/// round-join semantics.
///
/// Each finite `f32` coordinate is truncated toward zero, matching Pyclipper's
/// Cython conversion to Clipper's signed 64-bit `cInt`. Coordinates are bounded
/// to ±2^20 and `distance` must be finite, greater than Clipper's zero epsilon,
/// and at most 4096. The returned path contains at most 1024 integral vertices.
///
/// The implementation deliberately covers only a single positive offset of a
/// four-point OpenCV rotated rectangle. It rejects shapes outside that domain
/// and any rounded integer construction that is not a simple positive-area
/// path after Clipper's duplicate/collinear cleanup.
pub(crate) fn bounded_rounded_offset_rectangle(
    quad_xy: [[f32; 2]; 4],
    short_side: f32,
    distance: f64,
) -> Result<Vec<[i32; 2]>> {
    if !distance.is_finite() || distance <= NEAR_ZERO || distance > MAX_OFFSET_DISTANCE {
        bail!("bounded offset distance must be finite and in ({NEAR_ZERO}, {MAX_OFFSET_DISTANCE}]");
    }
    if !short_side.is_finite() || short_side < MIN_SHORT_SIDE {
        bail!("bounded offset rectangle did not pass the {MIN_SHORT_SIDE}-pixel short-side gate");
    }

    for [x, y] in quad_xy {
        if !x.is_finite() || !y.is_finite() {
            bail!("bounded offset rectangle contains a non-finite coordinate");
        }
        if x.abs() > MAX_ABS_INPUT_COORDINATE || y.abs() > MAX_ABS_INPUT_COORDINATE {
            bail!(
                "bounded offset rectangle exceeds the ±{MAX_ABS_INPUT_COORDINATE} coordinate bound"
            );
        }
    }
    validate_raw_rectangle(quad_xy, short_side)?;

    let mut path = Vec::with_capacity(4);
    for [x, y] in quad_xy {
        let point = [x as i64, y as i64];
        if path.last() != Some(&point) {
            path.push(point);
        }
    }
    if path.len() > 1 && path.first() == path.last() {
        path.pop();
    }
    remove_collinear_vertices(&mut path);
    if path.len() < 3 {
        bail!("bounded offset rectangle has fewer than three points after coercion");
    }

    let area_twice = signed_area_twice(&path)?;
    if area_twice == 0 {
        bail!("bounded offset rectangle is degenerate after coercion");
    }
    if area_twice < 0 {
        path.reverse();
    }
    if let Some((left, right)) = first_self_intersection(&path) {
        bail!("bounded offset input edges {left} and {right} intersect after coercion");
    }

    let normals = unit_normals(&path)?;
    let arc_tolerance = DEFAULT_ARC_TOLERANCE.min(distance * DEFAULT_ARC_TOLERANCE);
    let mut steps = std::f64::consts::PI / (1.0 - arc_tolerance / distance).acos();
    steps = steps.min(distance * std::f64::consts::PI);
    if !steps.is_finite() || steps <= 0.0 {
        bail!("bounded offset arc step calculation is invalid");
    }
    let sin_step = (std::f64::consts::TAU / steps).sin();
    let cos_step = (std::f64::consts::TAU / steps).cos();
    let steps_per_radian = steps / std::f64::consts::TAU;

    // Clipper carries `k` by mutable reference between corners. Its near-straight
    // early return deliberately leaves `k` unchanged, so the next corner may
    // start from an earlier normal rather than `j - 1`.
    let mut corners = Vec::with_capacity(path.len());
    let mut output_len = 0usize;
    let mut previous = path.len() - 1;
    for index in 0..path.len() {
        let corner_previous = previous;
        let mut sin_angle = cross_f64(normals[corner_previous], normals[index]);
        let cos_angle = dot_f64(normals[corner_previous], normals[index]);
        let early_return = (sin_angle * distance).abs() < 1.0 && cos_angle > 0.0;
        let negative_join = !early_return && sin_angle * distance < 0.0;
        let count = if early_return {
            1usize
        } else if negative_join {
            previous = index;
            3
        } else {
            sin_angle = sin_angle.clamp(-1.0, 1.0);
            let angle = sin_angle.atan2(cos_angle).abs();
            let count = usize::try_from(clipper_round(steps_per_radian * angle))
                .context("bounded offset corner step count is negative")?
                .max(1)
                .checked_add(1)
                .context("bounded offset corner step count overflow")?;
            previous = index;
            count
        };
        output_len = output_len
            .checked_add(count)
            .context("bounded offset output vertex count overflow")?;
        if output_len > MAX_OUTPUT_VERTICES {
            bail!("bounded offset exceeds the {MAX_OUTPUT_VERTICES}-vertex bound");
        }
        corners.push((corner_previous, early_return, negative_join, count));
    }

    let mut output = Vec::with_capacity(output_len);
    for index in 0..path.len() {
        let (previous, early_return, negative_join, count) = corners[index];
        if early_return {
            output.push(offset_point(path[index], normals[previous], distance)?);
            continue;
        }
        if negative_join {
            output.push(offset_point(path[index], normals[previous], distance)?);
            output.push(path[index]);
            output.push(offset_point(path[index], normals[index], distance)?);
            continue;
        }

        let mut normal = normals[previous];
        for _ in 0..count - 1 {
            output.push(offset_point(path[index], normal, distance)?);
            normal = [
                normal[0] * cos_step - sin_step * normal[1],
                normal[0] * sin_step + normal[1] * cos_step,
            ];
        }
        output.push(offset_point(path[index], normals[index], distance)?);
    }

    cleanup_simple_path(&mut output)?;
    output
        .into_iter()
        .map(|[x, y]| {
            Ok([
                i32::try_from(x).context("bounded offset x coordinate exceeds i32")?,
                i32::try_from(y).context("bounded offset y coordinate exceeds i32")?,
            ])
        })
        .collect()
}

fn validate_raw_rectangle(quad: [[f32; 2]; 4], short_side: f32) -> Result<()> {
    let edges = std::array::from_fn::<_, 4, _>(|index| {
        let next = (index + 1) % quad.len();
        [
            f64::from(quad[next][0] - quad[index][0]),
            f64::from(quad[next][1] - quad[index][1]),
        ]
    });
    let lengths = edges.map(|edge| edge[0].hypot(edge[1]));
    let observed_short_side = lengths.into_iter().fold(f64::INFINITY, f64::min);
    let short_side = f64::from(short_side);
    if !observed_short_side.is_finite()
        || (observed_short_side - short_side).abs()
            > BOX_POINTS_ABSOLUTE_TOLERANCE + short_side * BOX_POINTS_RELATIVE_TOLERANCE
    {
        bail!("bounded offset input does not match its admitted short side");
    }

    let mut orientation = 0_i8;
    for index in 0..quad.len() {
        let left = edges[(index + quad.len() - 1) % quad.len()];
        let right = edges[index];
        let cross = cross_f64(left, right);
        let length_product = lengths[(index + quad.len() - 1) % quad.len()] * lengths[index];
        if cross == 0.0
            || dot_f64(left, right).abs() > length_product * BOX_POINTS_RELATIVE_TOLERANCE
        {
            bail!("bounded offset input is not an OpenCV rotated rectangle");
        }
        let sign = if cross > 0.0 { 1 } else { -1 };
        if orientation != 0 && orientation != sign {
            bail!("bounded offset input is not an OpenCV rotated rectangle");
        }
        orientation = sign;
    }

    for (left, right) in [(0, 2), (1, 3)] {
        let mismatch = (edges[left][0] + edges[right][0]).hypot(edges[left][1] + edges[right][1]);
        let scale = lengths[left].max(lengths[right]);
        if mismatch > BOX_POINTS_ABSOLUTE_TOLERANCE + scale * BOX_POINTS_RELATIVE_TOLERANCE {
            bail!("bounded offset input is not an OpenCV boxPoints parallelogram");
        }
    }
    Ok(())
}

fn signed_area_twice(path: &[Point]) -> Result<i128> {
    let mut area = 0_i128;
    for index in 0..path.len() {
        let [x1, y1] = path[index];
        let [x2, y2] = path[(index + 1) % path.len()];
        area = area
            .checked_add(i128::from(x1) * i128::from(y2) - i128::from(y1) * i128::from(x2))
            .context("bounded offset signed area overflow")?;
    }
    Ok(area)
}

fn unit_normals(path: &[Point]) -> Result<Vec<Normal>> {
    path.iter()
        .zip(path.iter().cycle().skip(1))
        .map(|(&[x1, y1], &[x2, y2])| {
            let dx = (x2 - x1) as f64;
            let dy = (y2 - y1) as f64;
            let inverse_length = (dx * dx + dy * dy).sqrt().recip();
            if !inverse_length.is_finite() {
                bail!("bounded offset rectangle contains a zero-length edge");
            }
            Ok([dy * inverse_length, -dx * inverse_length])
        })
        .collect()
}

fn clipper_round(value: f64) -> i64 {
    if value < 0.0 {
        (value - 0.5) as i64
    } else {
        (value + 0.5) as i64
    }
}

fn offset_point(point: Point, normal: Normal, distance: f64) -> Result<Point> {
    let x = point[0] as f64 + normal[0] * distance;
    let y = point[1] as f64 + normal[1] * distance;
    if !x.is_finite() || !y.is_finite() {
        bail!("bounded offset produced a non-finite coordinate");
    }
    Ok([clipper_round(x), clipper_round(y)])
}

fn cleanup_simple_path(path: &mut Vec<Point>) -> Result<()> {
    remove_collinear_vertices(path);
    if path.len() < 3 {
        bail!("bounded offset collapsed below three output vertices");
    }
    if signed_area_twice(path)? <= 0 {
        bail!("bounded rounded path needs general polygon union to recover positive orientation");
    }
    if let Some((left, right)) = first_self_intersection(path) {
        bail!(
            "bounded rounded path needs general polygon union between output edges {left} and {right}"
        );
    }
    Ok(())
}

fn remove_collinear_vertices(path: &mut Vec<Point>) {
    remove_adjacent_duplicates(path);
    loop {
        if path.len() < 3 {
            return;
        }
        let Some(index) = (0..path.len()).find(|&index| {
            cross_i128(
                path[(index + path.len() - 1) % path.len()],
                path[index],
                path[(index + 1) % path.len()],
            ) == 0
        }) else {
            return;
        };
        path.remove(index);
        remove_adjacent_duplicates(path);
    }
}

fn remove_adjacent_duplicates(path: &mut Vec<Point>) {
    path.dedup();
    if path.len() > 1 && path.first() == path.last() {
        path.pop();
    }
}

fn first_self_intersection(path: &[Point]) -> Option<(usize, usize)> {
    for left in 0..path.len() {
        let left_next = (left + 1) % path.len();
        for right in left + 1..path.len() {
            let right_next = (right + 1) % path.len();
            if left == right_next || left_next == right {
                continue;
            }
            if segments_intersect(path[left], path[left_next], path[right], path[right_next]) {
                return Some((left, right));
            }
        }
    }
    None
}

fn cross_i128(previous: Point, current: Point, next: Point) -> i128 {
    let ax = i128::from(current[0] - previous[0]);
    let ay = i128::from(current[1] - previous[1]);
    let bx = i128::from(next[0] - current[0]);
    let by = i128::from(next[1] - current[1]);
    ax * by - ay * bx
}

fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let abc = cross_i128(a, b, c);
    let abd = cross_i128(a, b, d);
    let cda = cross_i128(c, d, a);
    let cdb = cross_i128(c, d, b);
    if ((abc > 0 && abd < 0) || (abc < 0 && abd > 0))
        && ((cda > 0 && cdb < 0) || (cda < 0 && cdb > 0))
    {
        return true;
    }
    (abc == 0 && point_on_segment(a, b, c))
        || (abd == 0 && point_on_segment(a, b, d))
        || (cda == 0 && point_on_segment(c, d, a))
        || (cdb == 0 && point_on_segment(c, d, b))
}

fn point_on_segment(a: Point, b: Point, point: Point) -> bool {
    point[0] >= a[0].min(b[0])
        && point[0] <= a[0].max(b[0])
        && point[1] >= a[1].min(b[1])
        && point[1] <= a[1].max(b[1])
}

fn cross_f64(left: Normal, right: Normal) -> f64 {
    left[0] * right[1] - right[0] * left[1]
}

fn dot_f64(left: Normal, right: Normal) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}
