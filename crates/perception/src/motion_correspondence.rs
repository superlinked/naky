//! Bounded sparse visual correspondence over decoded luma.
//!
//! The classifications in this module describe image evidence only. In
//! particular, `CoherentTranslation` is not proof of a scroll or user intent.

use std::mem::size_of;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use serde::{Serialize, Serializer};

pub const POLICY_IDENTITY: &str =
    "quarter-box4+census7+corner24-1024+fb24+unique3-exclude1+dual-residual16";
pub const COMPARISON_INTERVAL_MS: u64 = 250;
pub const DOWNSAMPLE_FACTOR: u32 = 4;
pub const DESCRIPTOR_RADIUS: i32 = 3;
pub const SEARCH_RADIUS: i32 = 24;
pub const BASE_ANCHOR_CELL_SIZE: u32 = 24;
pub const MAX_ANCHORS: usize = 1_024;
pub const MIN_CORNER_SCORE: u8 = 12;
pub const MAX_HAMMING_DISTANCE: u32 = 12;
pub const MIN_DISTANCE_MARGIN: u32 = 3;
pub const REVERSE_TOLERANCE: i32 = 1;
pub const RESIDUAL_LUMA_THRESHOLD: u8 = 16;
pub const MIN_ELIGIBLE_ANCHORS: usize = 16;
pub const MIN_ACCEPTED_MATCHES: usize = 12;
pub const MIN_SURVIVAL_PPM: u64 = 350_000;
pub const MIN_STATIONARY_SUPPORT: usize = 8;
pub const MIN_STATIONARY_CLASS_SUPPORT: usize = 12;
pub const MIN_STATIONARY_CLASS_FRACTION_PPM: u64 = 600_000;
pub const MIN_TRANSLATION_SUPPORT: usize = 12;
pub const MAX_TRANSLATION_RESIDUAL_PPM: u64 = 250_000;
pub const MAX_WIDTH: u32 = 7_680;
pub const MAX_HEIGHT: u32 = 4_320;
pub const MAX_PIXELS: u64 = MAX_WIDTH as u64 * MAX_HEIGHT as u64;
const LATENCY_BUCKETS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionDisposition {
    Initialized,
    SkippedInterval,
    InsufficientSupport,
    Stationary,
    CoherentTranslation,
    AmbiguousMotion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Correspondence {
    pub source_x: u32,
    pub source_y: u32,
    pub destination_x: u32,
    pub destination_y: u32,
    pub dx: i32,
    pub dy: i32,
    pub forward_distance: u32,
    pub forward_margin: u32,
    pub reverse_distance: u32,
    pub reverse_margin: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct CorrespondenceSeam {
    pub forward_distance_rejections: u64,
    pub forward_uniqueness_rejections: u64,
    pub reverse_distance_rejections: u64,
    pub reverse_uniqueness_rejections: u64,
    pub reverse_consistency_rejections: u64,
}

impl CorrespondenceSeam {
    fn add(&mut self, other: Self) {
        self.forward_distance_rejections = self
            .forward_distance_rejections
            .saturating_add(other.forward_distance_rejections);
        self.forward_uniqueness_rejections = self
            .forward_uniqueness_rejections
            .saturating_add(other.forward_uniqueness_rejections);
        self.reverse_distance_rejections = self
            .reverse_distance_rejections
            .saturating_add(other.reverse_distance_rejections);
        self.reverse_uniqueness_rejections = self
            .reverse_uniqueness_rejections
            .saturating_add(other.reverse_uniqueness_rejections);
        self.reverse_consistency_rejections = self
            .reverse_consistency_rejections
            .saturating_add(other.reverse_consistency_rejections);
    }

    fn total(self) -> u64 {
        self.forward_distance_rejections
            .saturating_add(self.forward_uniqueness_rejections)
            .saturating_add(self.reverse_distance_rejections)
            .saturating_add(self.reverse_uniqueness_rejections)
            .saturating_add(self.reverse_consistency_rejections)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrameObservation {
    pub timestamp_ms: u64,
    pub from_timestamp_ms: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub quarter_width: u32,
    pub quarter_height: u32,
    pub compared: bool,
    pub resize_reset: bool,
    pub disposition: MotionDisposition,
    pub eligible_anchors: u64,
    pub accepted_correspondences: u64,
    pub correspondence_seam: CorrespondenceSeam,
    pub correspondence_survival_ppm: u64,
    pub stationary_support: u64,
    pub dominant_translation_support: u64,
    pub dominant_translation_dx_quarter: Option<i32>,
    pub dominant_translation_dy_quarter: Option<i32>,
    pub dominant_translation_box: Option<MotionBox>,
    pub dual_model_residual_pixels: u64,
    pub dual_model_residual_eligible_pixels: u64,
    pub dual_model_residual_density_ppm: Option<u64>,
    pub analysis_wall_ns: u64,
    pub retained_state_bytes: u64,
    pub correspondences: Vec<Correspondence>,
}

/// Full-resolution destination-cell bounds supporting a displacement cluster.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct MotionBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LatencyHistogram([u64; LATENCY_BUCKETS]);

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self([0; LATENCY_BUCKETS])
    }
}

impl Serialize for LatencyHistogram {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.as_slice().serialize(serializer)
    }
}

impl LatencyHistogram {
    pub fn buckets(&self) -> &[u64; LATENCY_BUCKETS] {
        &self.0
    }

    fn record(&mut self, nanoseconds: u64) {
        let bucket = if nanoseconds == 0 {
            0
        } else {
            nanoseconds.ilog2() as usize
        };
        let bucket = bucket.min(LATENCY_BUCKETS - 1);
        self.0[bucket] = self.0[bucket].saturating_add(1);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RetentionProfile {
    pub quarter_luma_bytes: u64,
    pub descriptor_bytes: u64,
    pub anchor_bytes: u64,
    pub total_charged_bytes: u64,
}

impl RetentionProfile {
    fn take_max(&mut self, current: Self) {
        self.quarter_luma_bytes = self.quarter_luma_bytes.max(current.quarter_luma_bytes);
        self.descriptor_bytes = self.descriptor_bytes.max(current.descriptor_bytes);
        self.anchor_bytes = self.anchor_bytes.max(current.anchor_bytes);
        self.total_charged_bytes = self.total_charged_bytes.max(current.total_charged_bytes);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MotionCorrespondenceMetrics {
    pub frames_observed: u64,
    pub comparisons: u64,
    pub interval_skips: u64,
    pub resize_resets: u64,
    pub eligible_anchors: u64,
    pub accepted_correspondences: u64,
    pub correspondence_seam: CorrespondenceSeam,
    pub stationary_support: u64,
    pub dominant_translation_support: u64,
    pub initialized_frames: u64,
    pub insufficient_support_frames: u64,
    pub stationary_frames: u64,
    pub coherent_translation_frames: u64,
    pub ambiguous_motion_frames: u64,
    pub comparison_wall_sum_ns: u64,
    pub comparison_wall_max_ns: u64,
    pub comparison_wall_log2_ns: LatencyHistogram,
    pub retention_current: RetentionProfile,
    pub retention_high_water: RetentionProfile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Anchor {
    x: u32,
    y: u32,
}

#[derive(Debug)]
struct QuarterFrame {
    timestamp_ms: u64,
    width: u32,
    height: u32,
    luma: Vec<u8>,
    descriptors: Vec<u64>,
    anchors: Vec<Anchor>,
}

#[derive(Clone, Copy, Debug)]
struct RankedMatch {
    x: u32,
    y: u32,
    best_distance: u32,
    second_distance: u32,
}

impl RankedMatch {
    fn margin(self) -> u32 {
        self.second_distance.saturating_sub(self.best_distance)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Translation {
    dx: i32,
    dy: i32,
    support: usize,
}

/// Causal state over displayed frames. Retention is independent of duration.
#[derive(Debug)]
pub struct MotionCorrespondenceTracker {
    previous: Option<QuarterFrame>,
    last_observed_ms: Option<u64>,
    metrics: MotionCorrespondenceMetrics,
    comparison_interval_ms: u64,
}

impl Default for MotionCorrespondenceTracker {
    fn default() -> Self {
        Self {
            previous: None,
            last_observed_ms: None,
            metrics: MotionCorrespondenceMetrics::default(),
            comparison_interval_ms: COMPARISON_INTERVAL_MS,
        }
    }
}

impl MotionCorrespondenceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_comparison_interval(comparison_interval_ms: u64) -> Result<Self> {
        ensure!(
            comparison_interval_ms != 0,
            "comparison interval must be positive"
        );
        Ok(Self {
            comparison_interval_ms,
            ..Self::default()
        })
    }

    pub fn observe(
        &mut self,
        timestamp_ms: u64,
        luma: &[u8],
        width: u32,
        height: u32,
    ) -> Result<FrameObservation> {
        validate_frame(timestamp_ms, self.last_observed_ms, luma, width, height)?;
        let started = Instant::now();
        let quarter_width = width.div_ceil(DOWNSAMPLE_FACTOR);
        let quarter_height = height.div_ceil(DOWNSAMPLE_FACTOR);
        self.metrics.frames_observed = self.metrics.frames_observed.saturating_add(1);
        self.last_observed_ms = Some(timestamp_ms);

        let resize_reset = self.previous.as_ref().is_some_and(|previous| {
            previous.width != quarter_width || previous.height != quarter_height
        });
        if resize_reset {
            self.metrics.resize_resets = self.metrics.resize_resets.saturating_add(1);
        }

        if !resize_reset
            && self.previous.as_ref().is_some_and(|previous| {
                timestamp_ms.saturating_sub(previous.timestamp_ms) < self.comparison_interval_ms
            })
        {
            self.metrics.interval_skips = self.metrics.interval_skips.saturating_add(1);
            let analysis_wall_ns = duration_ns(started.elapsed())?;
            return Ok(self.noncomparison_observation(
                timestamp_ms,
                width,
                height,
                MotionDisposition::SkippedInterval,
                false,
                analysis_wall_ns,
            ));
        }

        let current = prepare_quarter_frame(timestamp_ms, luma, width, height)?;
        let Some(previous) = self.previous.take().filter(|_| !resize_reset) else {
            self.previous = Some(current);
            self.metrics.initialized_frames = self.metrics.initialized_frames.saturating_add(1);
            self.update_retention();
            let analysis_wall_ns = duration_ns(started.elapsed())?;
            return Ok(self.noncomparison_observation(
                timestamp_ms,
                width,
                height,
                MotionDisposition::Initialized,
                resize_reset,
                analysis_wall_ns,
            ));
        };

        let (correspondences, correspondence_seam) =
            forward_backward_correspondences(&previous, &current);
        let stationary_support = correspondences
            .iter()
            .filter(|item| item.dx.abs() <= 1 && item.dy.abs() <= 1)
            .count();
        let translation = dominant_translation(&correspondences);
        let translation_box = translation.and_then(|translation| {
            translation_support_box(&correspondences, translation, width, height)
        });
        let eligible_anchors = previous.anchors.len();
        let accepted_correspondences = correspondences.len();
        let survival_ppm = ratio_ppm(accepted_correspondences, eligible_anchors);
        let (residual_pixels, residual_eligible_pixels, residual_density_ppm) = translation
            .map(|translation| dual_model_residual(&previous, &current, translation))
            .unwrap_or((0, 0, None));
        let disposition = classify(
            eligible_anchors,
            accepted_correspondences,
            survival_ppm,
            stationary_support,
            translation,
            residual_density_ppm,
        );

        self.previous = Some(current);
        self.metrics.comparisons = self.metrics.comparisons.saturating_add(1);
        self.metrics.eligible_anchors = self
            .metrics
            .eligible_anchors
            .saturating_add(eligible_anchors as u64);
        self.metrics.accepted_correspondences = self
            .metrics
            .accepted_correspondences
            .saturating_add(accepted_correspondences as u64);
        self.metrics.correspondence_seam.add(correspondence_seam);
        self.metrics.stationary_support = self
            .metrics
            .stationary_support
            .saturating_add(stationary_support as u64);
        self.metrics.dominant_translation_support = self
            .metrics
            .dominant_translation_support
            .saturating_add(translation.map_or(0, |item| item.support as u64));
        match disposition {
            MotionDisposition::InsufficientSupport => {
                self.metrics.insufficient_support_frames =
                    self.metrics.insufficient_support_frames.saturating_add(1);
            }
            MotionDisposition::Stationary => {
                self.metrics.stationary_frames = self.metrics.stationary_frames.saturating_add(1);
            }
            MotionDisposition::CoherentTranslation => {
                self.metrics.coherent_translation_frames =
                    self.metrics.coherent_translation_frames.saturating_add(1);
            }
            MotionDisposition::AmbiguousMotion => {
                self.metrics.ambiguous_motion_frames =
                    self.metrics.ambiguous_motion_frames.saturating_add(1);
            }
            MotionDisposition::Initialized | MotionDisposition::SkippedInterval => unreachable!(),
        }
        self.update_retention();
        let analysis_wall_ns = duration_ns(started.elapsed())?;
        self.metrics.comparison_wall_sum_ns = self
            .metrics
            .comparison_wall_sum_ns
            .saturating_add(analysis_wall_ns);
        self.metrics.comparison_wall_max_ns =
            self.metrics.comparison_wall_max_ns.max(analysis_wall_ns);
        self.metrics
            .comparison_wall_log2_ns
            .record(analysis_wall_ns);

        Ok(FrameObservation {
            timestamp_ms,
            from_timestamp_ms: Some(previous.timestamp_ms),
            width,
            height,
            quarter_width,
            quarter_height,
            compared: true,
            resize_reset: false,
            disposition,
            eligible_anchors: eligible_anchors as u64,
            accepted_correspondences: accepted_correspondences as u64,
            correspondence_seam,
            correspondence_survival_ppm: survival_ppm,
            stationary_support: stationary_support as u64,
            dominant_translation_support: translation.map_or(0, |item| item.support as u64),
            dominant_translation_dx_quarter: translation.map(|item| item.dx),
            dominant_translation_dy_quarter: translation.map(|item| item.dy),
            dominant_translation_box: translation_box,
            dual_model_residual_pixels: residual_pixels,
            dual_model_residual_eligible_pixels: residual_eligible_pixels,
            dual_model_residual_density_ppm: residual_density_ppm,
            analysis_wall_ns,
            retained_state_bytes: self.metrics.retention_current.total_charged_bytes,
            correspondences,
        })
    }

    pub fn metrics(&self) -> &MotionCorrespondenceMetrics {
        &self.metrics
    }

    fn noncomparison_observation(
        &self,
        timestamp_ms: u64,
        width: u32,
        height: u32,
        disposition: MotionDisposition,
        resize_reset: bool,
        analysis_wall_ns: u64,
    ) -> FrameObservation {
        FrameObservation {
            timestamp_ms,
            from_timestamp_ms: None,
            width,
            height,
            quarter_width: width.div_ceil(DOWNSAMPLE_FACTOR),
            quarter_height: height.div_ceil(DOWNSAMPLE_FACTOR),
            compared: false,
            resize_reset,
            disposition,
            eligible_anchors: 0,
            accepted_correspondences: 0,
            correspondence_seam: CorrespondenceSeam::default(),
            correspondence_survival_ppm: 0,
            stationary_support: 0,
            dominant_translation_support: 0,
            dominant_translation_dx_quarter: None,
            dominant_translation_dy_quarter: None,
            dominant_translation_box: None,
            dual_model_residual_pixels: 0,
            dual_model_residual_eligible_pixels: 0,
            dual_model_residual_density_ppm: None,
            analysis_wall_ns,
            retained_state_bytes: self.metrics.retention_current.total_charged_bytes,
            correspondences: Vec::new(),
        }
    }

    fn update_retention(&mut self) {
        let current = self
            .previous
            .as_ref()
            .map_or_else(RetentionProfile::default, |frame| {
                let quarter_luma_bytes = frame.luma.capacity() as u64;
                let descriptor_bytes = (frame.descriptors.capacity() * size_of::<u64>()) as u64;
                let anchor_bytes = (frame.anchors.capacity() * size_of::<Anchor>()) as u64;
                RetentionProfile {
                    quarter_luma_bytes,
                    descriptor_bytes,
                    anchor_bytes,
                    total_charged_bytes: quarter_luma_bytes
                        .saturating_add(descriptor_bytes)
                        .saturating_add(anchor_bytes),
                }
            });
        self.metrics.retention_current = current;
        self.metrics.retention_high_water.take_max(current);
    }
}

fn translation_support_box(
    correspondences: &[Correspondence],
    translation: Translation,
    width: u32,
    height: u32,
) -> Option<MotionBox> {
    let cluster = (translation.dy.div_euclid(2), translation.dx.div_euclid(2));
    let mut bounds = None::<(u32, u32, u32, u32)>;
    for item in correspondences.iter().filter(|item| {
        (item.dy.div_euclid(2), item.dx.div_euclid(2)) == cluster
            && !(item.dx.abs() <= 1 && item.dy.abs() <= 1)
    }) {
        bounds = Some(match bounds {
            None => (
                item.destination_x,
                item.destination_y,
                item.destination_x,
                item.destination_y,
            ),
            Some((min_x, min_y, max_x, max_y)) => (
                min_x.min(item.destination_x),
                min_y.min(item.destination_y),
                max_x.max(item.destination_x),
                max_y.max(item.destination_y),
            ),
        });
    }
    bounds.map(|(min_x, min_y, max_x, max_y)| {
        let x = min_x.saturating_mul(DOWNSAMPLE_FACTOR);
        let y = min_y.saturating_mul(DOWNSAMPLE_FACTOR);
        let right = max_x
            .saturating_add(1)
            .saturating_mul(DOWNSAMPLE_FACTOR)
            .min(width);
        let bottom = max_y
            .saturating_add(1)
            .saturating_mul(DOWNSAMPLE_FACTOR)
            .min(height);
        MotionBox {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    })
}

fn validate_frame(
    timestamp_ms: u64,
    previous_timestamp_ms: Option<u64>,
    luma: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    ensure!(
        width != 0 && height != 0,
        "frame dimensions must be nonzero"
    );
    ensure!(
        width <= MAX_WIDTH && height <= MAX_HEIGHT,
        "frame dimensions exceed {MAX_WIDTH}x{MAX_HEIGHT}"
    );
    let pixels = u64::from(width) * u64::from(height);
    ensure!(pixels <= MAX_PIXELS, "frame pixel count exceeds limit");
    ensure!(
        luma.len() as u64 == pixels,
        "decoded luma length differs from dimensions"
    );
    if let Some(previous) = previous_timestamp_ms {
        ensure!(
            timestamp_ms > previous,
            "display timestamps must be strictly increasing"
        );
    }
    Ok(())
}

fn prepare_quarter_frame(
    timestamp_ms: u64,
    source: &[u8],
    source_width: u32,
    source_height: u32,
) -> Result<QuarterFrame> {
    let width = source_width.div_ceil(DOWNSAMPLE_FACTOR);
    let height = source_height.div_ceil(DOWNSAMPLE_FACTOR);
    let luma = downsample_box4(source, source_width, source_height)?;
    let descriptors = descriptor_image(&luma, width, height);
    let anchors = distributed_corner_anchors(&luma, width, height);
    Ok(QuarterFrame {
        timestamp_ms,
        width,
        height,
        luma,
        descriptors,
        anchors,
    })
}

fn downsample_box4(source: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let output_width = width.div_ceil(DOWNSAMPLE_FACTOR);
    let output_height = height.div_ceil(DOWNSAMPLE_FACTOR);
    let output_pixels = u64::from(output_width) * u64::from(output_height);
    let mut output =
        Vec::with_capacity(usize::try_from(output_pixels).context("quarter frame exceeds usize")?);
    let width_usize = width as usize;
    for output_y in 0..output_height {
        let y_start = output_y * DOWNSAMPLE_FACTOR;
        let y_end = (y_start + DOWNSAMPLE_FACTOR).min(height);
        for output_x in 0..output_width {
            let x_start = output_x * DOWNSAMPLE_FACTOR;
            let x_end = (x_start + DOWNSAMPLE_FACTOR).min(width);
            let mut sum = 0_u32;
            let mut count = 0_u32;
            for y in y_start..y_end {
                let row = y as usize * width_usize;
                for x in x_start..x_end {
                    sum += u32::from(source[row + x as usize]);
                    count += 1;
                }
            }
            output.push((sum / count) as u8);
        }
    }
    Ok(output)
}

fn descriptor_image(luma: &[u8], width: u32, height: u32) -> Vec<u64> {
    let mut output = vec![0; luma.len()];
    if width <= (DESCRIPTOR_RADIUS * 2) as u32 || height <= (DESCRIPTOR_RADIUS * 2) as u32 {
        return output;
    }
    for y in DESCRIPTOR_RADIUS as u32..height - DESCRIPTOR_RADIUS as u32 {
        for x in DESCRIPTOR_RADIUS as u32..width - DESCRIPTOR_RADIUS as u32 {
            output[index(x, y, width)] = census_descriptor(luma, width, x, y);
        }
    }
    output
}

fn census_descriptor(luma: &[u8], width: u32, x: u32, y: u32) -> u64 {
    let center = luma[index(x, y, width)];
    let mut descriptor = 0_u64;
    let mut bit = 0_u32;
    for dy in -DESCRIPTOR_RADIUS..=DESCRIPTOR_RADIUS {
        for dx in -DESCRIPTOR_RADIUS..=DESCRIPTOR_RADIUS {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbor_x = (x as i32 + dx) as u32;
            let neighbor_y = (y as i32 + dy) as u32;
            if luma[index(neighbor_x, neighbor_y, width)] < center {
                descriptor |= 1_u64 << bit;
            }
            bit += 1;
        }
    }
    descriptor
}

fn distributed_corner_anchors(luma: &[u8], width: u32, height: u32) -> Vec<Anchor> {
    let margin = (DESCRIPTOR_RADIUS + SEARCH_RADIUS) as u32;
    if width <= margin * 2 || height <= margin * 2 {
        return Vec::new();
    }
    let span_x = width - margin * 2;
    let span_y = height - margin * 2;
    let mut cell_size = BASE_ANCHOR_CELL_SIZE;
    while usize::try_from(span_x.div_ceil(cell_size))
        .unwrap_or(usize::MAX)
        .saturating_mul(usize::try_from(span_y.div_ceil(cell_size)).unwrap_or(usize::MAX))
        > MAX_ANCHORS
    {
        cell_size += 1;
    }
    let mut anchors = Vec::with_capacity(
        MAX_ANCHORS.min(span_x.div_ceil(cell_size) as usize * span_y.div_ceil(cell_size) as usize),
    );
    for cell_y in (margin..height - margin).step_by(cell_size as usize) {
        let end_y = (cell_y + cell_size).min(height - margin);
        for cell_x in (margin..width - margin).step_by(cell_size as usize) {
            let end_x = (cell_x + cell_size).min(width - margin);
            let mut best = None;
            for y in cell_y..end_y {
                for x in cell_x..end_x {
                    let horizontal =
                        luma[index(x + 1, y, width)].abs_diff(luma[index(x - 1, y, width)]);
                    let vertical =
                        luma[index(x, y + 1, width)].abs_diff(luma[index(x, y - 1, width)]);
                    let score = horizontal.min(vertical);
                    let candidate = (score, std::cmp::Reverse(y), std::cmp::Reverse(x));
                    if best.is_none_or(|(rank, _)| candidate > rank) {
                        best = Some((candidate, Anchor { x, y }));
                    }
                }
            }
            if let Some(((score, _, _), anchor)) = best
                && score >= MIN_CORNER_SCORE
            {
                anchors.push(anchor);
            }
        }
    }
    debug_assert!(anchors.len() <= MAX_ANCHORS);
    anchors
}

fn forward_backward_correspondences(
    previous: &QuarterFrame,
    current: &QuarterFrame,
) -> (Vec<Correspondence>, CorrespondenceSeam) {
    let mut correspondences = Vec::with_capacity(previous.anchors.len());
    let mut seam = CorrespondenceSeam::default();
    for anchor in &previous.anchors {
        let source_descriptor = previous.descriptors[index(anchor.x, anchor.y, previous.width)];
        let forward = ranked_match(
            source_descriptor,
            &current.descriptors,
            current.width,
            current.height,
            anchor.x,
            anchor.y,
        );
        if forward.best_distance > MAX_HAMMING_DISTANCE {
            seam.forward_distance_rejections += 1;
            continue;
        }
        if forward.margin() < MIN_DISTANCE_MARGIN {
            seam.forward_uniqueness_rejections += 1;
            continue;
        }
        let destination_descriptor =
            current.descriptors[index(forward.x, forward.y, current.width)];
        let reverse = ranked_match(
            destination_descriptor,
            &previous.descriptors,
            previous.width,
            previous.height,
            forward.x,
            forward.y,
        );
        if reverse.best_distance > MAX_HAMMING_DISTANCE {
            seam.reverse_distance_rejections += 1;
            continue;
        }
        if reverse.margin() < MIN_DISTANCE_MARGIN {
            seam.reverse_uniqueness_rejections += 1;
            continue;
        }
        if (reverse.x as i32 - anchor.x as i32).abs() > REVERSE_TOLERANCE
            || (reverse.y as i32 - anchor.y as i32).abs() > REVERSE_TOLERANCE
        {
            seam.reverse_consistency_rejections += 1;
            continue;
        }
        correspondences.push(Correspondence {
            source_x: anchor.x,
            source_y: anchor.y,
            destination_x: forward.x,
            destination_y: forward.y,
            dx: forward.x as i32 - anchor.x as i32,
            dy: forward.y as i32 - anchor.y as i32,
            forward_distance: forward.best_distance,
            forward_margin: forward.margin(),
            reverse_distance: reverse.best_distance,
            reverse_margin: reverse.margin(),
        });
    }
    debug_assert_eq!(
        previous.anchors.len() as u64,
        correspondences.len() as u64 + seam.total()
    );
    (correspondences, seam)
}

fn ranked_match(
    needle: u64,
    descriptors: &[u64],
    width: u32,
    height: u32,
    center_x: u32,
    center_y: u32,
) -> RankedMatch {
    let minimum_x = (center_x as i32 - SEARCH_RADIUS).max(DESCRIPTOR_RADIUS) as u32;
    let maximum_x =
        (center_x as i32 + SEARCH_RADIUS).min(width as i32 - DESCRIPTOR_RADIUS - 1) as u32;
    let minimum_y = (center_y as i32 - SEARCH_RADIUS).max(DESCRIPTOR_RADIUS) as u32;
    let maximum_y =
        (center_y as i32 + SEARCH_RADIUS).min(height as i32 - DESCRIPTOR_RADIUS - 1) as u32;
    let mut best = RankedMatch {
        x: minimum_x,
        y: minimum_y,
        best_distance: u32::MAX,
        second_distance: u32::MAX,
    };
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let distance = (needle ^ descriptors[index(x, y, width)]).count_ones();
            if distance < best.best_distance {
                best.best_distance = distance;
                best.x = x;
                best.y = y;
            }
        }
    }
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            if x.abs_diff(best.x) <= 1 && y.abs_diff(best.y) <= 1 {
                continue;
            }
            let distance = (needle ^ descriptors[index(x, y, width)]).count_ones();
            best.second_distance = best.second_distance.min(distance);
        }
    }
    best
}

fn dominant_translation(correspondences: &[Correspondence]) -> Option<Translation> {
    let mut displaced = correspondences
        .iter()
        .filter(|item| item.dx.abs() > 1 || item.dy.abs() > 1)
        .map(|item| {
            (
                item.dy.div_euclid(2),
                item.dx.div_euclid(2),
                item.dy,
                item.dx,
            )
        })
        .collect::<Vec<_>>();
    displaced.sort_unstable();
    let mut clusters = Vec::<(i32, i32, Vec<(i32, i32)>)>::new();
    for (cluster_dy, cluster_dx, dy, dx) in displaced {
        if let Some(last) = clusters.last_mut()
            && last.0 == cluster_dy
            && last.1 == cluster_dx
        {
            last.2.push((dy, dx));
        } else {
            clusters.push((cluster_dy, cluster_dx, vec![(dy, dx)]));
        }
    }
    clusters
        .into_iter()
        .map(|(cluster_dy, cluster_dx, mut members)| {
            members.sort_unstable();
            let mut dx_values = members.iter().map(|(_, dx)| *dx).collect::<Vec<_>>();
            let mut dy_values = members.iter().map(|(dy, _)| *dy).collect::<Vec<_>>();
            dx_values.sort_unstable();
            dy_values.sort_unstable();
            let middle = (members.len() - 1) / 2;
            (
                Translation {
                    dx: dx_values[middle],
                    dy: dy_values[middle],
                    support: members.len(),
                },
                cluster_dy,
                cluster_dx,
            )
        })
        .max_by(|left, right| {
            left.0
                .support
                .cmp(&right.0.support)
                .then_with(|| {
                    let left_magnitude = i64::from(left.0.dx).pow(2) + i64::from(left.0.dy).pow(2);
                    let right_magnitude =
                        i64::from(right.0.dx).pow(2) + i64::from(right.0.dy).pow(2);
                    left_magnitude.cmp(&right_magnitude)
                })
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| right.2.cmp(&left.2))
        })
        .map(|(translation, _, _)| translation)
}

fn dual_model_residual(
    previous: &QuarterFrame,
    current: &QuarterFrame,
    translation: Translation,
) -> (u64, u64, Option<u64>) {
    let x_start = 0.max(-translation.dx) as u32;
    let x_end = (previous.width as i32).min(current.width as i32 - translation.dx) as u32;
    let y_start = 0.max(-translation.dy) as u32;
    let y_end = (previous.height as i32).min(current.height as i32 - translation.dy) as u32;
    if x_start >= x_end || y_start >= y_end {
        return (0, 0, None);
    }
    let mut residual = 0_u64;
    let mut eligible = 0_u64;
    for y in y_start..y_end {
        for x in x_start..x_end {
            let previous_value = previous.luma[index(x, y, previous.width)];
            let stationary_value = current.luma[index(x, y, current.width)];
            let moved_value = current.luma[index(
                (x as i32 + translation.dx) as u32,
                (y as i32 + translation.dy) as u32,
                current.width,
            )];
            residual += u64::from(
                previous_value.abs_diff(stationary_value) > RESIDUAL_LUMA_THRESHOLD
                    && previous_value.abs_diff(moved_value) > RESIDUAL_LUMA_THRESHOLD,
            );
            eligible += 1;
        }
    }
    (residual, eligible, Some(residual * 1_000_000 / eligible))
}

fn classify(
    eligible: usize,
    accepted: usize,
    survival_ppm: u64,
    stationary: usize,
    translation: Option<Translation>,
    residual_ppm: Option<u64>,
) -> MotionDisposition {
    if eligible < MIN_ELIGIBLE_ANCHORS || accepted < MIN_ACCEPTED_MATCHES {
        return MotionDisposition::InsufficientSupport;
    }
    let translation_support = translation.map_or(0, |item| item.support);
    if survival_ppm >= MIN_SURVIVAL_PPM
        && stationary >= MIN_STATIONARY_CLASS_SUPPORT
        && ratio_ppm(stationary, accepted) >= MIN_STATIONARY_CLASS_FRACTION_PPM
        && translation_support < MIN_TRANSLATION_SUPPORT
    {
        return MotionDisposition::Stationary;
    }
    if let Some(translation) = translation
        && survival_ppm >= MIN_SURVIVAL_PPM
        && stationary >= MIN_STATIONARY_SUPPORT
        && translation.support >= MIN_TRANSLATION_SUPPORT
        && (translation.dx.abs() >= 2 || translation.dy.abs() >= 2)
        && residual_ppm.is_some_and(|density| density <= MAX_TRANSLATION_RESIDUAL_PPM)
    {
        return MotionDisposition::CoherentTranslation;
    }
    MotionDisposition::AmbiguousMotion
}

fn ratio_ppm(numerator: usize, denominator: usize) -> u64 {
    if denominator == 0 {
        0
    } else {
        numerator as u64 * 1_000_000 / denominator as u64
    }
}

fn index(x: u32, y: u32, width: u32) -> usize {
    y as usize * width as usize + x as usize
}

fn duration_ns(duration: Duration) -> Result<u64> {
    duration
        .as_nanos()
        .try_into()
        .context("duration exceeds u64 nanoseconds")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_WIDTH: u32 = 640;
    const SOURCE_HEIGHT: u32 = 720;

    fn texture(x: u32, y: u32, seed: u32) -> u8 {
        let value = x
            .wrapping_mul(37)
            .wrapping_add(y.wrapping_mul(73))
            .wrapping_add(x.wrapping_mul(y).wrapping_mul(11))
            .wrapping_add(seed.wrapping_mul(101));
        (value ^ (value >> 7) ^ (value >> 13)) as u8
    }

    fn source_from_quarter(quarter: &[u8], width: u32, height: u32) -> Vec<u8> {
        let mut source = vec![0; (width * 4 * height * 4) as usize];
        let source_width = width * 4;
        for y in 0..height {
            for x in 0..width {
                for dy in 0..4 {
                    for dx in 0..4 {
                        source[index(x * 4 + dx, y * 4 + dy, source_width)] =
                            quarter[index(x, y, width)];
                    }
                }
            }
        }
        source
    }

    fn base_quarter() -> Vec<u8> {
        let width = SOURCE_WIDTH / 4;
        let height = SOURCE_HEIGHT / 4;
        (0..height)
            .flat_map(|y| (0..width).map(move |x| texture(x, y, 0)))
            .collect()
    }

    fn observe_pair(previous: &[u8], current: &[u8]) -> FrameObservation {
        let mut tracker = MotionCorrespondenceTracker::new();
        tracker
            .observe(0, previous, SOURCE_WIDTH, SOURCE_HEIGHT)
            .unwrap();
        tracker
            .observe(250, current, SOURCE_WIDTH, SOURCE_HEIGHT)
            .unwrap()
    }

    #[test]
    fn stationary_texture_has_unique_stationary_matches() {
        let quarter = base_quarter();
        let source = source_from_quarter(&quarter, SOURCE_WIDTH / 4, SOURCE_HEIGHT / 4);
        let observation = observe_pair(&source, &source);
        assert_eq!(observation.disposition, MotionDisposition::Stationary);
        assert!(observation.accepted_correspondences >= 20);
        assert_eq!(
            observation.stationary_support,
            observation.accepted_correspondences
        );
        assert!(observation.correspondences.iter().all(|item| item.dx == 0
            && item.dy == 0
            && item.forward_distance == 0
            && item.reverse_distance == 0));
    }

    #[test]
    fn translated_layer_and_stationary_chrome_are_both_preserved() {
        let width = SOURCE_WIDTH / 4;
        let height = SOURCE_HEIGHT / 4;
        let previous = base_quarter();
        let mut current = previous.clone();
        // Two full anchor-cell rows make the stationary layer large enough to
        // exercise the fixed eight-anchor coexistence gate.
        let content_start = 76;
        let dy = -6_i32;
        for y in content_start..height {
            for x in 0..width {
                let source_y = y as i32 - dy;
                current[index(x, y, width)] = if source_y < height as i32 {
                    previous[index(x, source_y as u32, width)]
                } else {
                    texture(x, y, 19)
                };
            }
        }
        let observation = observe_pair(
            &source_from_quarter(&previous, width, height),
            &source_from_quarter(&current, width, height),
        );
        assert_eq!(
            observation.disposition,
            MotionDisposition::CoherentTranslation,
            "{observation:#?}"
        );
        assert!(observation.stationary_support >= MIN_STATIONARY_SUPPORT as u64);
        assert!(observation.dominant_translation_support >= MIN_TRANSLATION_SUPPORT as u64);
        assert_eq!(observation.dominant_translation_dy_quarter, Some(dy));
        assert_eq!(observation.dominant_translation_dx_quarter, Some(0));
        assert!(
            observation.dual_model_residual_density_ppm.unwrap() <= MAX_TRANSLATION_RESIDUAL_PPM
        );
    }

    #[test]
    fn periodic_corner_texture_reaches_and_fails_forward_uniqueness() {
        let width = SOURCE_WIDTH / 4;
        let height = SOURCE_HEIGHT / 4;
        // Eight-pixel periodicity supplies spatially distinct exact census
        // matches inside the fixed search window while preserving two-axis
        // gradients for the corner gate.
        const PERIOD: u32 = 8;
        let quarter = (0..height)
            .flat_map(|y| (0..width).map(move |x| texture(x % PERIOD, y % PERIOD, 41)))
            .collect::<Vec<_>>();
        let source = source_from_quarter(&quarter, width, height);
        let observation = observe_pair(&source, &source);
        assert_eq!(
            observation.disposition,
            MotionDisposition::InsufficientSupport,
            "{observation:#?}"
        );
        assert_eq!(observation.eligible_anchors, 30, "{observation:#?}");
        assert_eq!(observation.accepted_correspondences, 0, "{observation:#?}");
        assert_eq!(
            observation
                .correspondence_seam
                .forward_uniqueness_rejections,
            observation.eligible_anchors,
            "{observation:#?}"
        );
        assert_eq!(
            observation.correspondence_seam,
            CorrespondenceSeam {
                forward_uniqueness_rejections: observation.eligible_anchors,
                ..CorrespondenceSeam::default()
            },
            "{observation:#?}"
        );
    }

    #[test]
    fn incoherent_regions_without_stationary_support_remain_ambiguous() {
        let width = SOURCE_WIDTH / 4;
        let height = SOURCE_HEIGHT / 4;
        let previous = base_quarter();
        let mut current = vec![0; previous.len()];
        for y in 0..height {
            let dx = if x_region_left(y) { -5_i32 } else { 7_i32 };
            for x in 0..width {
                let source_x = x as i32 - dx;
                current[index(x, y, width)] = if (0..width as i32).contains(&source_x) {
                    previous[index(source_x as u32, y, width)]
                } else {
                    texture(x, y, 29)
                };
            }
        }
        let observation = observe_pair(
            &source_from_quarter(&previous, width, height),
            &source_from_quarter(&current, width, height),
        );
        assert_eq!(observation.disposition, MotionDisposition::AmbiguousMotion);
        assert!(observation.accepted_correspondences >= MIN_ACCEPTED_MATCHES as u64);
        assert!(observation.stationary_support < MIN_STATIONARY_SUPPORT as u64);
    }

    fn x_region_left(y: u32) -> bool {
        y < SOURCE_HEIGHT / 8
    }

    #[test]
    fn interval_and_validation_do_not_mutate_comparison_state() {
        let source = source_from_quarter(&base_quarter(), SOURCE_WIDTH / 4, SOURCE_HEIGHT / 4);
        let mut tracker = MotionCorrespondenceTracker::new();
        let first = tracker
            .observe(0, &source, SOURCE_WIDTH, SOURCE_HEIGHT)
            .unwrap();
        assert_eq!(first.disposition, MotionDisposition::Initialized);
        let skipped = tracker
            .observe(100, &source, SOURCE_WIDTH, SOURCE_HEIGHT)
            .unwrap();
        assert_eq!(skipped.disposition, MotionDisposition::SkippedInterval);
        assert!(
            tracker
                .observe(100, &source, SOURCE_WIDTH, SOURCE_HEIGHT)
                .is_err()
        );
        let compared = tracker
            .observe(250, &source, SOURCE_WIDTH, SOURCE_HEIGHT)
            .unwrap();
        assert!(compared.compared);
        assert_eq!(tracker.metrics().comparisons, 1);
        assert_eq!(tracker.metrics().frames_observed, 3);
    }

    #[test]
    fn default_and_new_share_the_validated_comparison_cadence() {
        let source = source_from_quarter(&base_quarter(), SOURCE_WIDTH / 4, SOURCE_HEIGHT / 4);
        let mut default_tracker = MotionCorrespondenceTracker::default();
        let mut new_tracker = MotionCorrespondenceTracker::new();

        for timestamp_ms in [0, COMPARISON_INTERVAL_MS - 1, COMPARISON_INTERVAL_MS] {
            let default_observation = default_tracker
                .observe(timestamp_ms, &source, SOURCE_WIDTH, SOURCE_HEIGHT)
                .unwrap();
            let new_observation = new_tracker
                .observe(timestamp_ms, &source, SOURCE_WIDTH, SOURCE_HEIGHT)
                .unwrap();
            assert_eq!(default_observation.disposition, new_observation.disposition);
            assert_eq!(default_observation.compared, new_observation.compared);
            assert_eq!(
                default_observation.from_timestamp_ms,
                new_observation.from_timestamp_ms
            );
        }

        let default_metrics = default_tracker.metrics();
        let new_metrics = new_tracker.metrics();
        assert_eq!(default_metrics.frames_observed, new_metrics.frames_observed);
        assert_eq!(default_metrics.comparisons, new_metrics.comparisons);
        assert_eq!(default_metrics.interval_skips, new_metrics.interval_skips);
        assert_eq!(default_metrics.comparisons, 1);
        assert_eq!(default_metrics.interval_skips, 1);
    }

    #[test]
    fn maximum_retained_state_is_below_gate() {
        let quarter_pixels = u64::from(MAX_WIDTH.div_ceil(4)) * u64::from(MAX_HEIGHT.div_ceil(4));
        let charged = quarter_pixels
            + quarter_pixels * size_of::<u64>() as u64
            + MAX_ANCHORS as u64 * size_of::<Anchor>() as u64;
        assert!(charged < 32 * 1024 * 1024, "charged={charged}");
    }
}
