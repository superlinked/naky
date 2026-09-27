//! Bounded decoded-luma visual-change diagnostics.
//!
//! This module deliberately reports only visual change. A [`VisualChangeEpisode`]
//! is not evidence of a cursor, click, selection, focus, scroll, object identity,
//! or user intent.

use std::cmp::Ordering;
use std::mem::size_of;
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use serde::{Serialize, Serializer};

pub const POLICY_IDENTITY: &str = "abs-luma-24+leaf-16x16-min-8+quadtree-inflation-2x+spatial-16";
pub const PIXEL_DIFFERENCE_THRESHOLD: u8 = 24;
pub const LEAF_SIZE: u32 = 16;
pub const MIN_CHANGED_PIXELS_PER_LEAF: u32 = 8;
pub const MAX_INFLATION_NUMERATOR: u64 = 2;
pub const MAX_INFLATION_DENOMINATOR: u64 = 1;
pub const EPISODE_SPATIAL_TOLERANCE: u32 = 16;
pub const MAX_WIDTH: u32 = 7_680;
pub const MAX_HEIGHT: u32 = 4_320;
pub const RETENTION_GATE_BYTES: u64 = 256 * 1024 * 1024;
const LATENCY_BUCKETS: usize = 64;
const MAX_EXPANDED_LEAF_VISITS_PER_LEAF: usize = 25;

/// One deterministically ordered region of thresholded visual change.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ChangeRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub changed_pixels: u64,
}

impl ChangeRegion {
    fn order_key(self) -> (u32, u32, u32, u32) {
        (self.y, self.x, self.height, self.width)
    }

    fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    fn right(self) -> u32 {
        self.x + self.width
    }

    fn bottom(self) -> u32 {
        self.y + self.height
    }
}

/// The lifecycle operation for a visual-change episode.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeTransitionKind {
    Started,
    Continued,
    Closed,
}

/// One transition of a spatially associated visual-change episode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EpisodeTransition {
    pub kind: EpisodeTransitionKind,
    pub episode_id: u64,
    pub timestamp_ms: u64,
    pub region: ChangeRegion,
}

/// The result of observing one displayed frame.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrameObservation {
    pub timestamp_ms: u64,
    pub width: u32,
    pub height: u32,
    pub compared: bool,
    pub resize_reset: bool,
    pub raw_changed_pixels: u64,
    pub eligible_leaves: u64,
    pub admitted_leaves: u64,
    pub admitted_charged_pixels: u64,
    pub merged_charged_pixels: u64,
    pub regions: Vec<ChangeRegion>,
    pub episode_transitions: Vec<EpisodeTransition>,
}

impl FrameObservation {
    fn reset(timestamp_ms: u64, width: u32, height: u32, resize_reset: bool) -> Self {
        Self {
            timestamp_ms,
            width,
            height,
            compared: false,
            resize_reset,
            raw_changed_pixels: 0,
            eligible_leaves: 0,
            admitted_leaves: 0,
            admitted_charged_pixels: 0,
            merged_charged_pixels: 0,
            regions: Vec::new(),
            episode_transitions: Vec::new(),
        }
    }
}

/// Closures emitted when the diagnostic stream is finalized.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FinishObservation {
    pub timestamp_ms: Option<u64>,
    pub episode_transitions: Vec<EpisodeTransition>,
}

/// A fixed 64-bin base-two histogram of analysis wall nanoseconds.
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
        self.0[bucket.min(LATENCY_BUCKETS - 1)] =
            self.0[bucket.min(LATENCY_BUCKETS - 1)].saturating_add(1);
    }
}

/// Conservative retained-capacity accounting, not an RSS estimate.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RetentionProfile {
    pub previous_luma_bytes: u64,
    pub leaf_count_slots: u64,
    pub tree_node_slots: u64,
    pub current_regions: u64,
    pub active_episodes: u64,
    pub matching_candidate_slots: u64,
    pub total_charged_bytes: u64,
}

impl RetentionProfile {
    fn take_max(&mut self, current: Self) {
        self.previous_luma_bytes = self.previous_luma_bytes.max(current.previous_luma_bytes);
        self.leaf_count_slots = self.leaf_count_slots.max(current.leaf_count_slots);
        self.tree_node_slots = self.tree_node_slots.max(current.tree_node_slots);
        self.current_regions = self.current_regions.max(current.current_regions);
        self.active_episodes = self.active_episodes.max(current.active_episodes);
        self.matching_candidate_slots = self
            .matching_candidate_slots
            .max(current.matching_candidate_slots);
        self.total_charged_bytes = self.total_charged_bytes.max(current.total_charged_bytes);
    }
}

/// Constant-space aggregate for the opt-in interaction-delta diagnostic.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct InteractionDeltaMetrics {
    pub frames_observed: u64,
    pub comparisons: u64,
    pub resizes: u64,
    pub empty_proposals: u64,
    pub raw_changed_pixels: u64,
    pub eligible_leaves: u64,
    pub admitted_leaves: u64,
    pub admitted_charged_pixels: u64,
    pub merged_charged_pixels: u64,
    pub merged_regions: u64,
    pub episode_starts: u64,
    pub episode_continuations: u64,
    pub episode_closures: u64,
    pub episode_open_at_end: u64,
    pub analysis_wall_sum_ns: u64,
    pub analysis_wall_max_ns: u64,
    pub analysis_wall_log2_ns: LatencyHistogram,
    pub retention_current: RetentionProfile,
    pub retention_high_water: RetentionProfile,
}

/// A currently open visual-change episode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisualChangeEpisode {
    id: u64,
    region: ChangeRegion,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    grid_x: u32,
    grid_y: u32,
    span_leaves: u32,
    changed_pixels: u64,
}

#[derive(Clone, Copy, Debug)]
struct MatchCandidate {
    active_index: u32,
    current_index: u32,
    overlap_area: u64,
    center_distance_squared: u64,
    episode_id: u64,
}

impl MatchCandidate {
    fn cmp_rank(&self, other: &Self, regions: &[ChangeRegion]) -> Ordering {
        other
            .overlap_area
            .cmp(&self.overlap_area)
            .then_with(|| {
                self.center_distance_squared
                    .cmp(&other.center_distance_squared)
            })
            .then_with(|| self.episode_id.cmp(&other.episode_id))
            .then_with(|| {
                regions[self.current_index as usize]
                    .order_key()
                    .cmp(&regions[other.current_index as usize].order_key())
            })
    }
}

/// Stateful, bounded visual-change tracker over consecutive displayed frames.
#[derive(Debug, Default)]
pub struct InteractionDeltaTracker {
    previous_luma: Vec<u8>,
    previous_width: u32,
    previous_height: u32,
    last_timestamp_ms: Option<u64>,
    leaf_counts: Vec<u32>,
    tree_current: Vec<Node>,
    tree_next: Vec<Node>,
    tree_final: Vec<Node>,
    current_regions: Vec<ChangeRegion>,
    active_episodes: Vec<VisualChangeEpisode>,
    next_active_episodes: Vec<VisualChangeEpisode>,
    leaf_owner: Vec<u32>,
    candidate_marks: Vec<u32>,
    match_candidates: Vec<MatchCandidate>,
    next_episode_id: u64,
    metrics: InteractionDeltaMetrics,
    finished: bool,
}

impl InteractionDeltaTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe exactly one displayed frame.
    ///
    /// Dimensions, buffer length, timestamp ordering, and lifecycle state are
    /// validated before tracker state is mutated.
    pub fn observe(
        &mut self,
        timestamp_ms: u64,
        luma: &[u8],
        width: u32,
        height: u32,
    ) -> Result<FrameObservation> {
        self.validate_input(timestamp_ms, luma, width, height)?;

        self.metrics.frames_observed = self.metrics.frames_observed.saturating_add(1);
        let first = self.last_timestamp_ms.is_none();
        let resized = !first && (width != self.previous_width || height != self.previous_height);
        self.last_timestamp_ms = Some(timestamp_ms);

        if first || resized {
            let mut observation = FrameObservation::reset(timestamp_ms, width, height, resized);
            if resized {
                self.metrics.resizes = self.metrics.resizes.saturating_add(1);
                observation.episode_transitions = self.close_all(timestamp_ms);
            }
            self.store_previous(luma, width, height);
            self.current_regions.clear();
            self.update_retention();
            return Ok(observation);
        }

        self.metrics.comparisons = self.metrics.comparisons.saturating_add(1);
        let leaf_columns = width.div_ceil(LEAF_SIZE);
        let leaf_rows = height.div_ceil(LEAF_SIZE);
        let leaf_len = usize::try_from(u64::from(leaf_columns) * u64::from(leaf_rows))?;
        self.leaf_counts.clear();
        self.leaf_counts.resize(leaf_len, 0);

        let mut raw_changed_pixels = 0_u64;
        let width_usize = width as usize;
        let leaf_columns_usize = leaf_columns as usize;
        for (index, (&previous, &current)) in self.previous_luma.iter().zip(luma.iter()).enumerate()
        {
            if previous.abs_diff(current) >= PIXEL_DIFFERENCE_THRESHOLD {
                raw_changed_pixels = raw_changed_pixels.saturating_add(1);
                let y = index / width_usize;
                let x = index % width_usize;
                let leaf_index =
                    (y / LEAF_SIZE as usize) * leaf_columns_usize + x / LEAF_SIZE as usize;
                self.leaf_counts[leaf_index] = self.leaf_counts[leaf_index].saturating_add(1);
            }
        }

        self.tree_current.clear();
        let mut eligible_leaves = 0_u64;
        let mut admitted_leaves = 0_u64;
        let mut admitted_charged_pixels = 0_u64;
        for grid_y in 0..leaf_rows {
            for grid_x in 0..leaf_columns {
                let leaf_index = grid_y as usize * leaf_columns_usize + grid_x as usize;
                let changed_pixels = self.leaf_counts[leaf_index];
                if changed_pixels != 0 {
                    eligible_leaves = eligible_leaves.saturating_add(1);
                }
                if changed_pixels >= MIN_CHANGED_PIXELS_PER_LEAF {
                    admitted_leaves = admitted_leaves.saturating_add(1);
                    admitted_charged_pixels = admitted_charged_pixels
                        .saturating_add(node_region(grid_x, grid_y, 1, width, height).area());
                    self.tree_current.push(Node {
                        grid_x,
                        grid_y,
                        span_leaves: 1,
                        changed_pixels: u64::from(changed_pixels),
                    });
                }
            }
        }

        self.coalesce(
            width,
            height,
            leaf_columns.max(leaf_rows).next_power_of_two(),
        );
        self.current_regions.clear();
        self.current_regions
            .extend(self.tree_final.iter().map(|node| {
                let mut region =
                    node_region(node.grid_x, node.grid_y, node.span_leaves, width, height);
                region.changed_pixels = node.changed_pixels;
                region
            }));
        self.current_regions
            .sort_unstable_by_key(|region| region.order_key());
        let merged_charged_pixels = self
            .current_regions
            .iter()
            .copied()
            .map(ChangeRegion::area)
            .sum::<u64>();
        let episode_transitions = self.associate_episodes(timestamp_ms, width, height);

        if self.current_regions.is_empty() {
            self.metrics.empty_proposals = self.metrics.empty_proposals.saturating_add(1);
        }
        self.metrics.raw_changed_pixels = self
            .metrics
            .raw_changed_pixels
            .saturating_add(raw_changed_pixels);
        self.metrics.eligible_leaves = self.metrics.eligible_leaves.saturating_add(eligible_leaves);
        self.metrics.admitted_leaves = self.metrics.admitted_leaves.saturating_add(admitted_leaves);
        self.metrics.admitted_charged_pixels = self
            .metrics
            .admitted_charged_pixels
            .saturating_add(admitted_charged_pixels);
        self.metrics.merged_charged_pixels = self
            .metrics
            .merged_charged_pixels
            .saturating_add(merged_charged_pixels);
        self.metrics.merged_regions = self
            .metrics
            .merged_regions
            .saturating_add(self.current_regions.len() as u64);
        self.store_previous(luma, width, height);
        self.update_retention();

        Ok(FrameObservation {
            timestamp_ms,
            width,
            height,
            compared: true,
            resize_reset: false,
            raw_changed_pixels,
            eligible_leaves,
            admitted_leaves,
            admitted_charged_pixels,
            merged_charged_pixels,
            regions: self.current_regions.clone(),
            episode_transitions,
        })
    }

    /// Record the causal wall duration surrounding [`Self::observe`].
    pub fn record_analysis_wall(&mut self, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.metrics.analysis_wall_sum_ns = self
            .metrics
            .analysis_wall_sum_ns
            .saturating_add(nanoseconds);
        self.metrics.analysis_wall_max_ns = self.metrics.analysis_wall_max_ns.max(nanoseconds);
        self.metrics.analysis_wall_log2_ns.record(nanoseconds);
    }

    /// Close all episodes once after the last displayed frame.
    pub fn finish(&mut self) -> Result<FinishObservation> {
        if self.finished {
            bail!("interaction-delta tracker was already finished");
        }
        self.finished = true;
        self.metrics.episode_open_at_end = self.active_episodes.len() as u64;
        let timestamp_ms = self.last_timestamp_ms;
        let episode_transitions = timestamp_ms
            .map(|timestamp_ms| self.close_all(timestamp_ms))
            .unwrap_or_default();
        self.current_regions.clear();
        self.update_retention();
        Ok(FinishObservation {
            timestamp_ms,
            episode_transitions,
        })
    }

    pub fn metrics(&self) -> &InteractionDeltaMetrics {
        &self.metrics
    }

    pub fn retention_profile(&self) -> RetentionProfile {
        let tree_node_slots = self
            .tree_current
            .capacity()
            .saturating_add(self.tree_next.capacity())
            .saturating_add(self.tree_final.capacity());
        let total_charged_bytes = self
            .previous_luma
            .capacity()
            .saturating_add(self.leaf_counts.capacity().saturating_mul(size_of::<u32>()))
            .saturating_add(tree_node_slots.saturating_mul(size_of::<Node>()))
            .saturating_add(
                self.current_regions
                    .capacity()
                    .saturating_mul(size_of::<ChangeRegion>()),
            )
            .saturating_add(
                self.active_episodes
                    .capacity()
                    .saturating_mul(size_of::<VisualChangeEpisode>()),
            )
            .saturating_add(
                self.next_active_episodes
                    .capacity()
                    .saturating_mul(size_of::<VisualChangeEpisode>()),
            )
            .saturating_add(self.leaf_owner.capacity().saturating_mul(size_of::<u32>()))
            .saturating_add(
                self.candidate_marks
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.match_candidates
                    .capacity()
                    .saturating_mul(size_of::<MatchCandidate>()),
            );
        RetentionProfile {
            previous_luma_bytes: self.previous_luma.capacity() as u64,
            leaf_count_slots: self.leaf_counts.capacity() as u64,
            tree_node_slots: tree_node_slots as u64,
            current_regions: self.current_regions.len() as u64,
            active_episodes: self.active_episodes.len() as u64,
            matching_candidate_slots: self.match_candidates.capacity() as u64,
            total_charged_bytes: total_charged_bytes as u64,
        }
    }

    fn validate_input(
        &self,
        timestamp_ms: u64,
        luma: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        ensure!(
            !self.finished,
            "interaction-delta tracker was already finished"
        );
        ensure!(width != 0 && height != 0, "luma dimensions must be nonzero");
        ensure!(
            width <= MAX_WIDTH && height <= MAX_HEIGHT,
            "luma dimensions exceed the decoder cap"
        );
        let expected_len = usize::try_from(
            u64::from(width)
                .checked_mul(u64::from(height))
                .ok_or_else(|| anyhow::anyhow!("luma dimensions overflow"))?,
        )?;
        ensure!(
            luma.len() == expected_len,
            "luma length does not match dimensions"
        );
        if self
            .last_timestamp_ms
            .is_some_and(|previous| timestamp_ms <= previous)
        {
            bail!("display timestamps must be strictly increasing");
        }
        Ok(())
    }

    fn store_previous(&mut self, luma: &[u8], width: u32, height: u32) {
        self.previous_luma.clear();
        self.previous_luma.extend_from_slice(luma);
        self.previous_width = width;
        self.previous_height = height;
    }

    fn coalesce(&mut self, width: u32, height: u32, maximum_span: u32) {
        self.tree_final.clear();
        let mut span = 1_u32;
        while !self.tree_current.is_empty() && span < maximum_span {
            self.tree_next.clear();
            let parent_span = span * 2;
            self.tree_current.sort_unstable_by_key(|node| {
                (
                    node.grid_y / parent_span,
                    node.grid_x / parent_span,
                    node.grid_y,
                    node.grid_x,
                )
            });
            let mut start = 0_usize;
            while start < self.tree_current.len() {
                let parent_x = (self.tree_current[start].grid_x / parent_span) * parent_span;
                let parent_y = (self.tree_current[start].grid_y / parent_span) * parent_span;
                let mut end = start + 1;
                while end < self.tree_current.len()
                    && (self.tree_current[end].grid_x / parent_span) * parent_span == parent_x
                    && (self.tree_current[end].grid_y / parent_span) * parent_span == parent_y
                {
                    end += 1;
                }
                let changed_pixels = self.tree_current[start..end]
                    .iter()
                    .map(|node| node.changed_pixels)
                    .sum::<u64>();
                let parent_area =
                    node_region(parent_x, parent_y, parent_span, width, height).area();
                let has_finalized_descendant = self.tree_current[start..end]
                    .iter()
                    .any(|node| node.changed_pixels == 0);
                if !has_finalized_descendant
                    && parent_area.saturating_mul(MAX_INFLATION_DENOMINATOR)
                        <= changed_pixels.saturating_mul(MAX_INFLATION_NUMERATOR)
                {
                    self.tree_next.push(Node {
                        grid_x: parent_x,
                        grid_y: parent_y,
                        span_leaves: parent_span,
                        changed_pixels,
                    });
                } else {
                    self.tree_final.extend(
                        self.tree_current[start..end]
                            .iter()
                            .copied()
                            .filter(|node| node.changed_pixels != 0),
                    );
                    // A zero-support sentinel propagates the fact that this
                    // subtree already has finalized descendants. No ancestor
                    // may cover it without violating the terminal partition.
                    self.tree_next.push(Node {
                        grid_x: parent_x,
                        grid_y: parent_y,
                        span_leaves: parent_span,
                        changed_pixels: 0,
                    });
                }
                start = end;
            }
            std::mem::swap(&mut self.tree_current, &mut self.tree_next);
            span = parent_span;
        }
        self.tree_final.extend(
            self.tree_current
                .iter()
                .copied()
                .filter(|node| node.changed_pixels != 0),
        );
    }

    fn associate_episodes(
        &mut self,
        timestamp_ms: u64,
        width: u32,
        height: u32,
    ) -> Vec<EpisodeTransition> {
        self.build_match_candidates(width, height);
        self.match_candidates
            .sort_unstable_by(|left, right| left.cmp_rank(right, &self.current_regions));

        let mut active_matched = vec![false; self.active_episodes.len()];
        let mut current_episode_ids = vec![None; self.current_regions.len()];
        for candidate in &self.match_candidates {
            let active_index = candidate.active_index as usize;
            let current_index = candidate.current_index as usize;
            if !active_matched[active_index] && current_episode_ids[current_index].is_none() {
                active_matched[active_index] = true;
                current_episode_ids[current_index] = Some(candidate.episode_id);
            }
        }

        let mut transitions = Vec::with_capacity(
            self.active_episodes
                .len()
                .saturating_add(self.current_regions.len()),
        );
        for (index, episode) in self.active_episodes.iter().copied().enumerate() {
            if !active_matched[index] {
                transitions.push(EpisodeTransition {
                    kind: EpisodeTransitionKind::Closed,
                    episode_id: episode.id,
                    timestamp_ms,
                    region: episode.region,
                });
                self.metrics.episode_closures = self.metrics.episode_closures.saturating_add(1);
            }
        }

        self.next_active_episodes.clear();
        for (index, region) in self.current_regions.iter().copied().enumerate() {
            let (kind, episode_id) = if let Some(episode_id) = current_episode_ids[index] {
                self.metrics.episode_continuations =
                    self.metrics.episode_continuations.saturating_add(1);
                (EpisodeTransitionKind::Continued, episode_id)
            } else {
                let episode_id = self.next_episode_id;
                self.next_episode_id = self.next_episode_id.saturating_add(1);
                self.metrics.episode_starts = self.metrics.episode_starts.saturating_add(1);
                (EpisodeTransitionKind::Started, episode_id)
            };
            self.next_active_episodes.push(VisualChangeEpisode {
                id: episode_id,
                region,
            });
            transitions.push(EpisodeTransition {
                kind,
                episode_id,
                timestamp_ms,
                region,
            });
        }
        std::mem::swap(&mut self.active_episodes, &mut self.next_active_episodes);
        self.active_episodes
            .sort_unstable_by_key(|episode| (episode.region.order_key(), episode.id));
        transitions.sort_unstable_by_key(|transition| {
            (
                transition.region.order_key(),
                transition.episode_id,
                transition.kind,
            )
        });
        transitions
    }

    fn build_match_candidates(&mut self, width: u32, height: u32) {
        self.match_candidates.clear();
        if self.active_episodes.is_empty() || self.current_regions.is_empty() {
            return;
        }
        let leaf_columns = width.div_ceil(LEAF_SIZE);
        let leaf_rows = height.div_ceil(LEAF_SIZE);
        let leaf_len = leaf_columns as usize * leaf_rows as usize;
        self.leaf_owner.clear();
        self.leaf_owner.resize(leaf_len, 0);
        for (active_index, episode) in self.active_episodes.iter().enumerate() {
            let left = episode.region.x / LEAF_SIZE;
            let top = episode.region.y / LEAF_SIZE;
            let right = episode.region.right().div_ceil(LEAF_SIZE);
            let bottom = episode.region.bottom().div_ceil(LEAF_SIZE);
            for grid_y in top..bottom {
                for grid_x in left..right {
                    self.leaf_owner[grid_y as usize * leaf_columns as usize + grid_x as usize] =
                        active_index as u32 + 1;
                }
            }
        }
        self.candidate_marks.clear();
        self.candidate_marks.resize(self.active_episodes.len(), 0);
        for (current_index, region) in self.current_regions.iter().copied().enumerate() {
            let generation = current_index as u32 + 1;
            let left = region.x.saturating_sub(EPISODE_SPATIAL_TOLERANCE * 2) / LEAF_SIZE;
            let top = region.y.saturating_sub(EPISODE_SPATIAL_TOLERANCE * 2) / LEAF_SIZE;
            let right = region
                .right()
                .saturating_add(EPISODE_SPATIAL_TOLERANCE * 2)
                .min(width)
                .div_ceil(LEAF_SIZE);
            let bottom = region
                .bottom()
                .saturating_add(EPISODE_SPATIAL_TOLERANCE * 2)
                .min(height)
                .div_ceil(LEAF_SIZE);
            for grid_y in top..bottom {
                for grid_x in left..right {
                    let owner =
                        self.leaf_owner[grid_y as usize * leaf_columns as usize + grid_x as usize];
                    if owner == 0 {
                        continue;
                    }
                    let active_index = owner as usize - 1;
                    if self.candidate_marks[active_index] == generation {
                        continue;
                    }
                    self.candidate_marks[active_index] = generation;
                    let episode = self.active_episodes[active_index];
                    if spatially_compatible(episode.region, region, width, height) {
                        self.match_candidates.push(MatchCandidate {
                            active_index: active_index as u32,
                            current_index: current_index as u32,
                            overlap_area: overlap_area(episode.region, region),
                            center_distance_squared: center_distance_squared(
                                episode.region,
                                region,
                            ),
                            episode_id: episode.id,
                        });
                    }
                }
            }
        }
    }

    fn close_all(&mut self, timestamp_ms: u64) -> Vec<EpisodeTransition> {
        let mut transitions = self
            .active_episodes
            .drain(..)
            .map(|episode| EpisodeTransition {
                kind: EpisodeTransitionKind::Closed,
                episode_id: episode.id,
                timestamp_ms,
                region: episode.region,
            })
            .collect::<Vec<_>>();
        self.metrics.episode_closures = self
            .metrics
            .episode_closures
            .saturating_add(transitions.len() as u64);
        transitions.sort_unstable_by_key(|transition| {
            (
                transition.region.order_key(),
                transition.episode_id,
                transition.kind,
            )
        });
        transitions
    }

    fn update_retention(&mut self) {
        let current = self.retention_profile();
        self.metrics.retention_current = current;
        self.metrics.retention_high_water.take_max(current);
    }
}

fn node_region(
    grid_x: u32,
    grid_y: u32,
    span_leaves: u32,
    width: u32,
    height: u32,
) -> ChangeRegion {
    let x = grid_x * LEAF_SIZE;
    let y = grid_y * LEAF_SIZE;
    ChangeRegion {
        x,
        y,
        width: (span_leaves * LEAF_SIZE).min(width - x),
        height: (span_leaves * LEAF_SIZE).min(height - y),
        changed_pixels: 0,
    }
}

fn overlap_area(left: ChangeRegion, right: ChangeRegion) -> u64 {
    let width = left
        .right()
        .min(right.right())
        .saturating_sub(left.x.max(right.x));
    let height = left
        .bottom()
        .min(right.bottom())
        .saturating_sub(left.y.max(right.y));
    u64::from(width) * u64::from(height)
}

fn center_distance_squared(left: ChangeRegion, right: ChangeRegion) -> u64 {
    let left_x = u64::from(left.x) * 2 + u64::from(left.width);
    let left_y = u64::from(left.y) * 2 + u64::from(left.height);
    let right_x = u64::from(right.x) * 2 + u64::from(right.width);
    let right_y = u64::from(right.y) * 2 + u64::from(right.height);
    left_x
        .abs_diff(right_x)
        .pow(2)
        .saturating_add(left_y.abs_diff(right_y).pow(2))
}

fn spatially_compatible(left: ChangeRegion, right: ChangeRegion, width: u32, height: u32) -> bool {
    overlap_area(
        expand(left, width, height, EPISODE_SPATIAL_TOLERANCE),
        expand(right, width, height, EPISODE_SPATIAL_TOLERANCE),
    ) != 0
}

fn expand(region: ChangeRegion, width: u32, height: u32, amount: u32) -> ChangeRegion {
    let x = region.x.saturating_sub(amount);
    let y = region.y.saturating_sub(amount);
    let right = region.right().saturating_add(amount).min(width);
    let bottom = region.bottom().saturating_add(amount).min(height);
    ChangeRegion {
        x,
        y,
        width: right - x,
        height: bottom - y,
        changed_pixels: region.changed_pixels,
    }
}

/// Maximum logical element charge for every frame-sized retained component.
///
/// This multiplies maximum logical lengths by element size. It deliberately
/// does not claim an upper bound on allocator-selected `Vec` capacity. Runtime
/// metrics separately report actual retained capacities and their high water.
pub fn maximum_logical_retained_bytes_charge() -> u64 {
    let pixels = MAX_WIDTH as usize * MAX_HEIGHT as usize;
    let leaves = MAX_WIDTH.div_ceil(LEAF_SIZE) as usize * MAX_HEIGHT.div_ceil(LEAF_SIZE) as usize;
    let matching_candidates = leaves.saturating_mul(MAX_EXPANDED_LEAF_VISITS_PER_LEAF);
    pixels
        .saturating_add(leaves.saturating_mul(size_of::<u32>()))
        .saturating_add(leaves.saturating_mul(3).saturating_mul(size_of::<Node>()))
        .saturating_add(leaves.saturating_mul(size_of::<ChangeRegion>()))
        .saturating_add(
            leaves
                .saturating_mul(2)
                .saturating_mul(size_of::<VisualChangeEpisode>()),
        )
        .saturating_add(leaves.saturating_mul(2).saturating_mul(size_of::<u32>()))
        .saturating_add(matching_candidates.saturating_mul(size_of::<MatchCandidate>())) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; width as usize * height as usize]
    }

    fn paint(luma: &mut [u8], width: u32, x: u32, y: u32, w: u32, h: u32, value: u8) {
        for row in y..y + h {
            for column in x..x + w {
                luma[row as usize * width as usize + column as usize] = value;
            }
        }
    }

    #[test]
    fn static_comparison_is_an_observed_empty_proposal() {
        let mut tracker = InteractionDeltaTracker::new();
        let luma = frame(32, 32, 20);
        assert!(!tracker.observe(0, &luma, 32, 32).unwrap().compared);
        let observation = tracker.observe(1, &luma, 32, 32).unwrap();
        assert!(observation.compared);
        assert!(observation.regions.is_empty());
        assert_eq!(tracker.metrics().empty_proposals, 1);
    }

    #[test]
    fn local_support_is_admitted_and_ordered() {
        let mut tracker = InteractionDeltaTracker::new();
        let before = frame(48, 32, 0);
        let mut after = before.clone();
        paint(&mut after, 48, 33, 17, 4, 4, 255);
        tracker.observe(0, &before, 48, 32).unwrap();
        let observation = tracker.observe(1, &after, 48, 32).unwrap();
        assert_eq!(observation.raw_changed_pixels, 16);
        assert_eq!(observation.eligible_leaves, 1);
        assert_eq!(observation.admitted_leaves, 1);
        assert_eq!(observation.regions[0].order_key(), (16, 32, 16, 16));
        assert_eq!(
            observation.episode_transitions[0].kind,
            EpisodeTransitionKind::Started
        );
    }

    #[test]
    fn dense_support_collapses_hierarchy() {
        let mut tracker = InteractionDeltaTracker::new();
        let before = frame(64, 64, 0);
        let after = frame(64, 64, 255);
        tracker.observe(0, &before, 64, 64).unwrap();
        let observation = tracker.observe(1, &after, 64, 64).unwrap();
        assert_eq!(observation.admitted_leaves, 16);
        assert_eq!(observation.regions.len(), 1);
        assert_eq!(observation.regions[0].order_key(), (0, 0, 64, 64));
        assert_eq!(observation.merged_charged_pixels, 64 * 64);
    }

    #[test]
    fn terminal_regions_have_unique_membership_across_parent_groups() {
        let mut tracker = InteractionDeltaTracker::new();
        let before = frame(64, 64, 0);
        let mut after = before.clone();
        paint(&mut after, 64, 0, 0, 48, 32, 255);
        tracker.observe(0, &before, 64, 64).unwrap();
        let observation = tracker.observe(1, &after, 64, 64).unwrap();

        assert_eq!(
            observation
                .regions
                .iter()
                .map(|region| region.changed_pixels)
                .sum::<u64>(),
            48 * 32
        );
        let mut terminal_membership = Vec::new();
        for region in &observation.regions {
            for grid_y in region.y / LEAF_SIZE..region.bottom().div_ceil(LEAF_SIZE) {
                for grid_x in region.x / LEAF_SIZE..region.right().div_ceil(LEAF_SIZE) {
                    terminal_membership.push((grid_y, grid_x));
                }
            }
        }
        let membership_count = terminal_membership.len();
        terminal_membership.sort_unstable();
        terminal_membership.dedup();
        assert_eq!(terminal_membership.len(), membership_count);
    }

    #[test]
    fn finalized_descendant_blocks_overlapping_multilevel_ancestor() {
        let mut tracker = InteractionDeltaTracker::new();
        let before = frame(64, 64, 0);
        let mut after = before.clone();
        paint(&mut after, 64, 0, 0, 8, 1, 255);
        paint(&mut after, 64, 32, 0, 32, 32, 255);
        paint(&mut after, 64, 0, 32, 32, 32, 255);
        paint(&mut after, 64, 32, 32, 32, 32, 255);
        tracker.observe(0, &before, 64, 64).unwrap();
        let observation = tracker.observe(1, &after, 64, 64).unwrap();

        assert_eq!(observation.raw_changed_pixels, 3_080);
        assert_eq!(
            observation
                .regions
                .iter()
                .map(|region| region.changed_pixels)
                .sum::<u64>(),
            observation.raw_changed_pixels
        );
        let mut terminal_membership = Vec::new();
        for region in &observation.regions {
            for grid_y in region.y / LEAF_SIZE..region.bottom().div_ceil(LEAF_SIZE) {
                for grid_x in region.x / LEAF_SIZE..region.right().div_ceil(LEAF_SIZE) {
                    terminal_membership.push((grid_y, grid_x));
                }
            }
        }
        let membership_count = terminal_membership.len();
        terminal_membership.sort_unstable();
        terminal_membership.dedup();
        assert_eq!(terminal_membership.len(), membership_count);
        for (index, left) in observation.regions.iter().enumerate() {
            assert!(
                observation.regions[index + 1..]
                    .iter()
                    .all(|right| overlap_area(*left, *right) == 0)
            );
        }
        assert!(observation.merged_charged_pixels <= 64 * 64);
    }

    #[test]
    fn sparse_regions_stay_separate() {
        let mut tracker = InteractionDeltaTracker::new();
        let before = frame(64, 64, 0);
        let mut after = before.clone();
        paint(&mut after, 64, 0, 0, 4, 4, 255);
        paint(&mut after, 64, 48, 48, 4, 4, 255);
        tracker.observe(0, &before, 64, 64).unwrap();
        let observation = tracker.observe(1, &after, 64, 64).unwrap();
        assert_eq!(observation.regions.len(), 2);
        assert!(observation.regions[0].order_key() < observation.regions[1].order_key());
    }

    #[test]
    fn spatial_match_is_deterministic_and_quiet_frame_closes() {
        let mut tracker = InteractionDeltaTracker::new();
        let base = frame(64, 32, 0);
        let mut first = base.clone();
        paint(&mut first, 64, 16, 0, 16, 16, 255);
        let mut second = first.clone();
        paint(&mut second, 64, 16, 0, 16, 16, 0);
        paint(&mut second, 64, 32, 0, 16, 16, 255);
        tracker.observe(0, &base, 64, 32).unwrap();
        let started = tracker.observe(1, &first, 64, 32).unwrap();
        let episode_id = started.episode_transitions[0].episode_id;
        let continued = tracker.observe(2, &second, 64, 32).unwrap();
        assert!(continued.episode_transitions.iter().any(|transition| {
            transition.kind == EpisodeTransitionKind::Continued
                && transition.episode_id == episode_id
        }));
        let closed = tracker.observe(3, &second, 64, 32).unwrap();
        assert!(closed.regions.is_empty());
        assert!(!closed.episode_transitions.is_empty());
        assert!(
            closed
                .episode_transitions
                .iter()
                .all(|transition| transition.kind == EpisodeTransitionKind::Closed)
        );
        assert!(
            closed
                .episode_transitions
                .iter()
                .any(|transition| transition.episode_id == episode_id)
        );
    }

    #[test]
    fn resize_resets_comparison_and_closes_episode() {
        let mut tracker = InteractionDeltaTracker::new();
        let before = frame(16, 16, 0);
        let after = frame(16, 16, 255);
        tracker.observe(0, &before, 16, 16).unwrap();
        tracker.observe(1, &after, 16, 16).unwrap();
        let resized = tracker.observe(2, &frame(32, 16, 0), 32, 16).unwrap();
        assert!(!resized.compared);
        assert!(resized.resize_reset);
        assert_eq!(
            resized.episode_transitions[0].kind,
            EpisodeTransitionKind::Closed
        );
        assert_eq!(tracker.metrics().resizes, 1);
    }

    #[test]
    fn invalid_input_does_not_mutate_state() {
        let mut tracker = InteractionDeltaTracker::new();
        let luma = frame(16, 16, 0);
        tracker.observe(10, &luma, 16, 16).unwrap();
        let metrics = tracker.metrics().clone();
        let retention = tracker.retention_profile();
        assert!(tracker.observe(10, &luma, 16, 16).is_err());
        assert!(tracker.observe(11, &luma[..255], 16, 16).is_err());
        assert_eq!(tracker.metrics(), &metrics);
        assert_eq!(tracker.retention_profile(), retention);
    }

    #[test]
    fn repeated_input_capacity_plateaus() {
        let mut tracker = InteractionDeltaTracker::new();
        let mut previous = frame(128, 64, 0);
        tracker.observe(0, &previous, 128, 64).unwrap();
        for timestamp in 1..20 {
            let mut current = previous.clone();
            paint(
                &mut current,
                128,
                (timestamp % 7 * 16) as u32,
                16,
                8,
                8,
                255,
            );
            tracker
                .observe(timestamp as u64, &current, 128, 64)
                .unwrap();
            previous = current;
        }
        let plateau = tracker.retention_profile().total_charged_bytes;
        for timestamp in 20..100 {
            let mut current = previous.clone();
            paint(&mut current, 128, (timestamp % 7 * 16) as u32, 16, 8, 8, 0);
            tracker
                .observe(timestamp as u64, &current, 128, 64)
                .unwrap();
            previous = current;
        }
        assert_eq!(tracker.retention_profile().total_charged_bytes, plateau);
    }

    #[test]
    fn maximum_dimension_logical_retention_charge_is_below_gate() {
        assert!(maximum_logical_retained_bytes_charge() < RETENTION_GATE_BYTES);
    }

    #[test]
    fn finish_records_open_at_end_and_is_single_use() {
        let mut tracker = InteractionDeltaTracker::new();
        tracker.observe(0, &frame(16, 16, 0), 16, 16).unwrap();
        tracker.observe(1, &frame(16, 16, 255), 16, 16).unwrap();
        let finish = tracker.finish().unwrap();
        assert_eq!(tracker.metrics().episode_open_at_end, 1);
        assert_eq!(finish.episode_transitions.len(), 1);
        assert!(tracker.finish().is_err());
    }

    #[test]
    fn latency_histogram_is_fixed_and_logarithmic() {
        let mut tracker = InteractionDeltaTracker::new();
        tracker.record_analysis_wall(Duration::from_nanos(0));
        tracker.record_analysis_wall(Duration::from_nanos(8));
        assert_eq!(tracker.metrics().analysis_wall_log2_ns.buckets().len(), 64);
        assert_eq!(tracker.metrics().analysis_wall_log2_ns.buckets()[0], 1);
        assert_eq!(tracker.metrics().analysis_wall_log2_ns.buckets()[3], 1);
    }
}
