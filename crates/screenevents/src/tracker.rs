use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::model::{
    CloudEvent, ElementContentChangedData, ElementData, ElementDisappearedData, ElementMovedData,
    ElementObservation, ElementState, EventBody, FrameSize, ObservationFrame, ObservedActivityData,
    ObservedActivityKind, ScreenResizedData, SnapshotData,
};

const ACTIVE_MATCH_THRESHOLD: f64 = 0.30;
const INACTIVE_MINIMUM_IOU: f64 = 0.50;
const TRANSLATION_BIN_PIXELS: i64 = 20;
const MINIMUM_TRANSLATION_ANCHORS: usize = 3;
const MAX_ACTIVE_TRACKS: usize = 3_000;
const MAX_ACTIVE_CHARGED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INACTIVE_TRACKS: usize = 65_536;
const MAX_INACTIVE_CHARGED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ASSOCIATION_CANDIDATES: usize = 1_000_000;
const TRACK_FIXED_CHARGE_BYTES: u64 = 512;
const STATE_ENTRY_FIXED_CHARGE_BYTES: u64 = 128;
const GENERATED_ID_CHARGE_BYTES: u64 = 32;
const INACTIVE_PRIORITY_DOMAIN: &[u8] = b"naky.screenevents.inactive-priority.v0\0";

#[derive(Debug, Error, Eq, PartialEq)]
pub enum EncoderError {
    #[error("stream ID must not be empty or contain whitespace")]
    InvalidStreamId,
    #[error("frame dimensions must be non-zero")]
    InvalidFrameSize,
    #[error("timestamp {current} precedes previous timestamp {previous}")]
    NonMonotonicTimestamp { previous: u64, current: u64 },
    #[error("element {index} has invalid confidence {confidence}")]
    InvalidConfidence { index: usize, confidence: String },
    #[error("element {index} box does not fit within the frame")]
    InvalidBox { index: usize },
    #[error("frame contains {count} elements; maximum is {maximum}")]
    TooManyElements { count: usize, maximum: usize },
    #[error("frame active-state charge {charged_bytes} exceeds {maximum_bytes} bytes")]
    ActiveStateTooLarge {
        charged_bytes: u64,
        maximum_bytes: u64,
    },
    #[error("frame association produced more than {maximum} eligible candidates")]
    AssociationTooLarge { maximum: usize },
    #[error("cannot emit observed activity before a structural snapshot")]
    ActivityBeforeSnapshot,
    #[error("activity interval {from}..{to} is not positive")]
    InvalidActivityInterval { from: u64, to: u64 },
    #[error("activity references unknown element {0}")]
    UnknownActivityElement(String),
    #[error("activity translation box does not fit within the current frame")]
    InvalidActivityBox,
    #[error("activity translation displacement must be nonzero")]
    ZeroActivityTranslation,
    #[error("an activity interval may contain at most one translation")]
    MultipleActivityTranslations,
}

#[derive(Clone, Debug)]
struct Track {
    element: ElementState,
    active: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RoleKey {
    Missing,
    Present(String),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExactMatchKey {
    text: String,
    role: RoleKey,
}

impl ExactMatchKey {
    fn from_parts(text: Option<&str>, role: Option<&str>) -> Option<Self> {
        let text = text.map(normalize).unwrap_or_default();
        if text.is_empty() {
            return None;
        }
        let role = match role {
            None => RoleKey::Missing,
            Some(role) => {
                let role = normalize(role);
                if role.is_empty() {
                    return None;
                }
                RoleKey::Present(role)
            }
        };
        Some(Self { text, role })
    }

    fn from_element(element: &ElementState) -> Option<Self> {
        Self::from_parts(element.text.as_deref(), element.role.as_deref())
    }

    fn from_observation(observation: &ElementObservation) -> Option<Self> {
        Self::from_parts(observation.text.as_deref(), observation.role.as_deref())
    }

    fn role_equal(&self) -> bool {
        matches!(self.role, RoleKey::Present(_))
    }

    fn variable_capacity(&self) -> u64 {
        let role = match &self.role {
            RoleKey::Missing => 0,
            RoleKey::Present(role) => saturating_u64(role.capacity()),
        };
        saturating_u64(self.text.capacity()).saturating_add(role)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RectKey {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl From<crate::model::Rect> for RectKey {
    fn from(rect: crate::model::Rect) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }
}

impl From<RectKey> for crate::model::Rect {
    fn from(rect: RectKey) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InactiveLimits {
    maximum_tracks: usize,
    maximum_charged_bytes: u64,
}

impl InactiveLimits {
    const PRODUCT: Self = Self {
        maximum_tracks: MAX_INACTIVE_TRACKS,
        maximum_charged_bytes: MAX_INACTIVE_CHARGED_BYTES,
    };
}

#[derive(Clone, Debug)]
struct InactiveLocation {
    key: ExactMatchKey,
    rect: RectKey,
    charged_bytes: u64,
    priority: [u8; 32],
}

#[derive(Debug)]
struct InactiveHistory {
    by_key: BTreeMap<ExactMatchKey, BTreeMap<RectKey, BTreeSet<u64>>>,
    locations: BTreeMap<u64, InactiveLocation>,
    eviction_order: BTreeSet<([u8; 32], u64)>,
    charged_bytes: u64,
    total_evictions: u64,
    limits: InactiveLimits,
}

impl InactiveHistory {
    fn new(limits: InactiveLimits) -> Self {
        Self {
            by_key: BTreeMap::new(),
            locations: BTreeMap::new(),
            eviction_order: BTreeSet::new(),
            charged_bytes: 0,
            total_evictions: 0,
            limits,
        }
    }

    fn insert(&mut self, track_id: u64, element: &ElementState) {
        let Some(key) = ExactMatchKey::from_element(element) else {
            return;
        };
        let rect = RectKey::from(element.bbox);
        let charged_bytes = inactive_entry_charge(&key);
        let priority = inactive_priority(&key, rect, track_id);
        let previous = self.locations.insert(
            track_id,
            InactiveLocation {
                key: key.clone(),
                rect,
                charged_bytes,
                priority,
            },
        );
        debug_assert!(previous.is_none(), "inactive track ID inserted twice");
        self.by_key
            .entry(key)
            .or_default()
            .entry(rect)
            .or_default()
            .insert(track_id);
        let inserted = self.eviction_order.insert((priority, track_id));
        debug_assert!(inserted, "inactive eviction rank inserted twice");
        self.charged_bytes = self.charged_bytes.saturating_add(charged_bytes);
    }

    fn remove(&mut self, track_id: u64) -> bool {
        let Some(location) = self.locations.remove(&track_id) else {
            return false;
        };
        let removed_rank = self.eviction_order.remove(&(location.priority, track_id));
        debug_assert!(removed_rank, "inactive eviction rank missing");

        let remove_key = {
            let rectangles = self
                .by_key
                .get_mut(&location.key)
                .expect("inactive location key exists");
            let remove_rect = {
                let ids = rectangles
                    .get_mut(&location.rect)
                    .expect("inactive location rectangle exists");
                let removed_id = ids.remove(&track_id);
                debug_assert!(removed_id, "inactive location ID missing");
                ids.is_empty()
            };
            if remove_rect {
                rectangles.remove(&location.rect);
            }
            rectangles.is_empty()
        };
        if remove_key {
            self.by_key.remove(&location.key);
        }
        self.charged_bytes = self.charged_bytes.saturating_sub(location.charged_bytes);
        true
    }

    fn enforce_limits(&mut self, mut profile: Option<&mut TrackerWorkProfile>) {
        while self.locations.len() > self.limits.maximum_tracks
            || self.charged_bytes > self.limits.maximum_charged_bytes
        {
            let count_pressure = self.locations.len() > self.limits.maximum_tracks;
            let byte_pressure = self.charged_bytes > self.limits.maximum_charged_bytes;
            let (_, track_id) = self
                .eviction_order
                .last()
                .copied()
                .expect("over-limit inactive history is nonempty");
            let removed = self.remove(track_id);
            debug_assert!(removed, "eviction target exists");
            self.total_evictions = self.total_evictions.saturating_add(1);
            if let Some(profile) = profile.as_deref_mut() {
                if count_pressure {
                    profile.inactive_count_pressure_evictions =
                        profile.inactive_count_pressure_evictions.saturating_add(1);
                }
                if byte_pressure {
                    profile.inactive_byte_pressure_evictions =
                        profile.inactive_byte_pressure_evictions.saturating_add(1);
                }
            }
        }
    }

    fn variable_bytes(&self) -> u64 {
        self.locations.values().fold(0, |total, location| {
            total.saturating_add(
                location
                    .charged_bytes
                    .saturating_sub(TRACK_FIXED_CHARGE_BYTES),
            )
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    track_id: u64,
    observation_index: usize,
    score: f64,
    ordinary_overlap: bool,
}

#[derive(Clone, Copy, Debug)]
struct MatchScore {
    value: f64,
    ordinary_overlap: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Translation {
    x: i64,
    y: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssociationPolicy {
    Exact,
}

/// Deterministic association-work counters for one tracker frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct TrackerWorkProfile {
    /// Active ID/observation pairs plus inactive exact-key ID/observation pairs
    /// scored by the ordinary association pass.
    pub candidate_match_score_pairs: u64,
    /// Active-track/observation pairs scored by the ordinary pass.
    pub active_candidate_pairs: u64,
    /// Matchable observation keys looked up in compact inactive history.
    pub inactive_key_lookups: u64,
    /// Distinct inactive rectangles inspected in exact-key buckets.
    pub inactive_rectangle_scans: u64,
    /// Inactive ID candidates admitted after exact-key and IoU checks.
    pub inactive_candidate_ids_materialized: u64,
    /// Largest exact-key inactive ID bucket consulted in this frame.
    pub maximum_inactive_key_bucket_ids: u64,
    /// Maximum eligible-candidate buffer length in this frame.
    pub maximum_candidate_buffer_len: u64,
    /// Evictions performed while the inactive count limit was exceeded.
    pub inactive_count_pressure_evictions: u64,
    /// Evictions performed while the inactive byte limit was exceeded.
    pub inactive_byte_pressure_evictions: u64,
    /// Reserved association-work counter.
    ///
    /// This remains zero under the ordinary exact association policy.
    pub fuzzy_edge_pairs: u64,
    /// Reserved association-veto counter.
    ///
    /// This remains zero under the ordinary exact association policy.
    pub fuzzy_competitor_track_scans: u64,
}

/// Deterministic post-frame retained-state counters.
///
/// `retained_variable_bytes` conservatively counts retained string capacities;
/// the charged-byte fields additionally include fixed record and state-entry
/// allowances for the product's deterministic conservative budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TrackerRetentionProfile {
    pub retained_tracks: u64,
    pub active_tracks: u64,
    pub inactive_tracks: u64,
    pub retained_variable_bytes: u64,
    pub active_charged_bytes: u64,
    pub inactive_charged_bytes: u64,
    pub inactive_evictions: u64,
}

/// Stateful encoder from perception observations to ScreenEvents.
#[derive(Debug)]
pub struct ScreenEventEncoder {
    stream_id: String,
    source: String,
    next_sequence: u64,
    next_element_id: u64,
    active_tracks: BTreeMap<u64, Track>,
    inactive: InactiveHistory,
    frame_size: Option<FrameSize>,
    last_timestamp_ms: Option<u64>,
    association_policy: AssociationPolicy,
}

impl ScreenEventEncoder {
    /// Construct an encoder with the ordinary exact-text/role/IoU policy.
    ///
    /// This constructor uses exact text, role, and overlap association.
    pub fn new(stream_id: impl Into<String>) -> Result<Self, EncoderError> {
        Self::new_with_policy(stream_id, AssociationPolicy::Exact)
    }

    fn new_with_policy(
        stream_id: impl Into<String>,
        association_policy: AssociationPolicy,
    ) -> Result<Self, EncoderError> {
        Self::new_with_policy_and_limits(stream_id, association_policy, InactiveLimits::PRODUCT)
    }

    fn new_with_policy_and_limits(
        stream_id: impl Into<String>,
        association_policy: AssociationPolicy,
        inactive_limits: InactiveLimits,
    ) -> Result<Self, EncoderError> {
        let stream_id = stream_id.into();
        if stream_id.is_empty() || stream_id.chars().any(char::is_whitespace) {
            return Err(EncoderError::InvalidStreamId);
        }

        Ok(Self {
            source: format!("urn:naky:stream:{stream_id}"),
            stream_id,
            next_sequence: 0,
            next_element_id: 1,
            active_tracks: BTreeMap::new(),
            inactive: InactiveHistory::new(inactive_limits),
            frame_size: None,
            last_timestamp_ms: None,
            association_policy,
        })
    }

    pub fn process_frame(
        &mut self,
        frame: ObservationFrame,
    ) -> Result<Vec<CloudEvent>, EncoderError> {
        self.process_frame_inner(frame, false)
            .map(|(events, _)| events)
    }

    /// Process a frame and return deterministic association-work diagnostics.
    ///
    /// This uses the same matching and event path as [`Self::process_frame`].
    /// Profiling only increments counters inside the existing association
    /// loops. Retained-state scanning is separate in [`Self::retention_profile`].
    pub fn process_frame_profiled(
        &mut self,
        frame: ObservationFrame,
    ) -> Result<(Vec<CloudEvent>, TrackerWorkProfile), EncoderError> {
        let (events, profile) = self.process_frame_inner(frame, true)?;
        Ok((events, profile.expect("profile requested")))
    }

    /// Return canonical IDs of live elements containing one half-open point.
    ///
    /// IDs follow canonical numeric order because active tracks are keyed by
    /// their generated number.
    #[must_use]
    pub fn elements_containing_point(&self, x: u32, y: u32) -> Vec<String> {
        self.active_tracks
            .values()
            .filter(|track| {
                let bbox = track.element.bbox;
                u64::from(x) >= u64::from(bbox.x)
                    && u64::from(y) >= u64::from(bbox.y)
                    && u64::from(x) < u64::from(bbox.x) + u64::from(bbox.width)
                    && u64::from(y) < u64::from(bbox.y) + u64::from(bbox.height)
            })
            .map(|track| track.element.id.clone())
            .collect()
    }

    /// Sequence a validated interval of observed activity against pre-frame
    /// structural state. This does not mutate element or activity history.
    pub fn process_observed_activity(
        &mut self,
        from_frametime: u64,
        frametime: u64,
        mut kinds: Vec<ObservedActivityKind>,
    ) -> Result<Vec<CloudEvent>, EncoderError> {
        let frame = self
            .frame_size
            .ok_or(EncoderError::ActivityBeforeSnapshot)?;
        if from_frametime >= frametime {
            return Err(EncoderError::InvalidActivityInterval {
                from: from_frametime,
                to: frametime,
            });
        }
        if let Some(previous) = self.last_timestamp_ms
            && frametime < previous
        {
            return Err(EncoderError::NonMonotonicTimestamp {
                previous,
                current: frametime,
            });
        }

        let translation_count = kinds
            .iter()
            .filter(|kind| matches!(kind, ObservedActivityKind::RegionTranslated { .. }))
            .count();
        if translation_count > 1 {
            return Err(EncoderError::MultipleActivityTranslations);
        }
        for kind in &kinds {
            match kind {
                ObservedActivityKind::ElementRegionChanged { element_id } => {
                    let Some(number) = canonical_element_number(element_id) else {
                        return Err(EncoderError::UnknownActivityElement(element_id.clone()));
                    };
                    if !self.active_tracks.contains_key(&number) {
                        return Err(EncoderError::UnknownActivityElement(element_id.clone()));
                    }
                }
                ObservedActivityKind::RegionTranslated { bbox, dx, dy } => {
                    if !bbox.fits_within(frame) {
                        return Err(EncoderError::InvalidActivityBox);
                    }
                    if *dx == 0 && *dy == 0 {
                        return Err(EncoderError::ZeroActivityTranslation);
                    }
                }
            }
        }
        kinds.sort_by_key(activity_order_key);
        kinds.dedup_by(|left, right| match (&*left, &*right) {
            (
                ObservedActivityKind::ElementRegionChanged { element_id: left },
                ObservedActivityKind::ElementRegionChanged { element_id: right },
            ) => left == right,
            _ => false,
        });
        self.last_timestamp_ms = Some(frametime);
        Ok(kinds
            .into_iter()
            .map(|kind| {
                self.wrap(
                    frametime,
                    EventBody::ObservedActivity(ObservedActivityData {
                        from_frametime,
                        kind,
                    }),
                )
            })
            .collect())
    }

    /// Scan current retained state for deterministic history-growth counters.
    #[must_use]
    pub fn retention_profile(&self) -> TrackerRetentionProfile {
        let active_tracks = self.active_tracks.len();
        let inactive_tracks = self.inactive.locations.len();
        let active_charged_bytes = self.active_tracks.values().fold(0_u64, |total, track| {
            total.saturating_add(track_charge(track))
        });
        TrackerRetentionProfile {
            retained_tracks: saturating_u64(active_tracks.saturating_add(inactive_tracks)),
            active_tracks: saturating_u64(active_tracks),
            inactive_tracks: saturating_u64(inactive_tracks),
            retained_variable_bytes: self
                .active_tracks
                .values()
                .fold(0_u64, |total, track| {
                    total.saturating_add(track_variable_bytes(track))
                })
                .saturating_add(self.inactive.variable_bytes()),
            active_charged_bytes,
            inactive_charged_bytes: self.inactive.charged_bytes,
            inactive_evictions: self.inactive.total_evictions,
        }
    }

    fn process_frame_inner(
        &mut self,
        frame: ObservationFrame,
        collect_profile: bool,
    ) -> Result<(Vec<CloudEvent>, Option<TrackerWorkProfile>), EncoderError> {
        self.validate_frame(&frame)?;

        let timestamp_ms = frame.timestamp_ms;
        let size = frame.size();
        let mut profile = collect_profile.then(TrackerWorkProfile::default);
        let bodies = if self.frame_size.is_none() {
            self.initialize(frame)
        } else {
            self.update(frame, profile.as_mut())?
        };

        self.frame_size = Some(size);
        self.last_timestamp_ms = Some(timestamp_ms);

        let events = bodies
            .into_iter()
            .map(|body| self.wrap(timestamp_ms, body))
            .collect();
        Ok((events, profile))
    }

    fn validate_frame(&self, frame: &ObservationFrame) -> Result<(), EncoderError> {
        let size = frame.size();
        if !size.is_valid() {
            return Err(EncoderError::InvalidFrameSize);
        }
        if let Some(previous) = self.last_timestamp_ms
            && frame.timestamp_ms < previous
        {
            return Err(EncoderError::NonMonotonicTimestamp {
                previous,
                current: frame.timestamp_ms,
            });
        }

        for (index, element) in frame.elements.iter().enumerate() {
            if !element.confidence.is_finite() || !(0.0..=1.0).contains(&element.confidence) {
                return Err(EncoderError::InvalidConfidence {
                    index,
                    confidence: element.confidence.to_string(),
                });
            }
            if !element.bbox.fits_within(size) {
                return Err(EncoderError::InvalidBox { index });
            }
        }
        if frame.elements.len() > MAX_ACTIVE_TRACKS {
            return Err(EncoderError::TooManyElements {
                count: frame.elements.len(),
                maximum: MAX_ACTIVE_TRACKS,
            });
        }
        let active_charge = frame.elements.iter().fold(0_u64, |total, observation| {
            total.saturating_add(observation_charge(observation))
        });
        if active_charge > MAX_ACTIVE_CHARGED_BYTES {
            return Err(EncoderError::ActiveStateTooLarge {
                charged_bytes: active_charge,
                maximum_bytes: MAX_ACTIVE_CHARGED_BYTES,
            });
        }
        Ok(())
    }

    fn initialize(&mut self, frame: ObservationFrame) -> Vec<EventBody> {
        let frame_size = frame.size();
        let mut elements = Vec::with_capacity(frame.elements.len());
        for observation in frame.elements {
            let (track_id, element) = self.create_track(observation);
            self.active_tracks.insert(
                track_id,
                Track {
                    element: element.clone(),
                    active: true,
                },
            );
            elements.push(element);
        }

        vec![EventBody::Snapshot(SnapshotData {
            frame: frame_size,
            elements,
        })]
    }

    fn update(
        &mut self,
        frame: ObservationFrame,
        mut profile: Option<&mut TrackerWorkProfile>,
    ) -> Result<Vec<EventBody>, EncoderError> {
        let mut bodies = Vec::new();
        let new_size = frame.size();
        if let Some(previous_size) = self.frame_size
            && previous_size != new_size
        {
            bodies.push(EventBody::ScreenResized(ScreenResizedData {
                from: previous_size,
                to: new_size,
            }));
        }

        let assignments = self.assign(&frame.elements, profile.as_deref_mut())?;
        let assigned_tracks: BTreeSet<u64> = assignments.values().copied().collect();

        let disappearing_ids = self
            .active_tracks
            .keys()
            .filter(|track_id| !assigned_tracks.contains(track_id))
            .copied()
            .collect::<Vec<_>>();
        let mut newly_inactive = Vec::with_capacity(disappearing_ids.len());
        for track_id in disappearing_ids {
            if let Some(track) = self.active_tracks.remove(&track_id) {
                bodies.push(EventBody::ElementDisappeared(ElementDisappearedData {
                    id: track.element.id.clone(),
                }));
                newly_inactive.push((track_id, track.element));
            }
        }

        for (index, observation) in frame.elements.into_iter().enumerate() {
            if let Some(track_id) = assignments.get(&index).copied() {
                if let Some(track) = self.active_tracks.get_mut(&track_id) {
                    let previous = track.element.clone();
                    let current = ElementState::from_observation(previous.id.clone(), observation);
                    if previous.bbox != current.bbox {
                        bodies.push(EventBody::ElementMoved(ElementMovedData {
                            id: current.id.clone(),
                            from: previous.bbox,
                            to: current.bbox,
                        }));
                    }
                    if !previous.semantic_content_eq(&current) {
                        bodies.push(EventBody::ElementContentChanged(
                            ElementContentChangedData {
                                element: current.clone(),
                            },
                        ));
                    }
                    track.element = current;
                } else {
                    let removed = self.inactive.remove(track_id);
                    debug_assert!(removed, "assignment refers to retained inactive track");
                    let current =
                        ElementState::from_observation(format!("e{track_id:06}"), observation);
                    bodies.push(EventBody::ElementAppeared(ElementData {
                        element: current.clone(),
                    }));
                    self.active_tracks.insert(
                        track_id,
                        Track {
                            element: current,
                            active: true,
                        },
                    );
                }
            } else {
                let (track_id, element) = self.create_track(observation);
                self.active_tracks.insert(
                    track_id,
                    Track {
                        element: element.clone(),
                        active: true,
                    },
                );
                bodies.push(EventBody::ElementAppeared(ElementData { element }));
            }
        }

        for (track_id, element) in newly_inactive {
            self.inactive.insert(track_id, &element);
        }
        self.inactive.enforce_limits(profile);

        Ok(bodies)
    }

    fn assign(
        &self,
        observations: &[ElementObservation],
        mut profile: Option<&mut TrackerWorkProfile>,
    ) -> Result<BTreeMap<usize, u64>, EncoderError> {
        let translation = self.dominant_translation(observations);
        let fuzzy_support: BTreeSet<(u64, usize)> = match self.association_policy {
            AssociationPolicy::Exact => BTreeSet::new(),
        };
        let mut candidates = Vec::new();
        for (track_id, track) in &self.active_tracks {
            for (observation_index, observation) in observations.iter().enumerate() {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.candidate_match_score_pairs =
                        profile.candidate_match_score_pairs.saturating_add(1);
                    profile.active_candidate_pairs =
                        profile.active_candidate_pairs.saturating_add(1);
                }
                if let Some(score) = match_score(
                    track,
                    observation,
                    translation,
                    fuzzy_support.contains(&(*track_id, observation_index)),
                ) {
                    push_candidate(
                        &mut candidates,
                        Candidate {
                            track_id: *track_id,
                            observation_index,
                            score: score.value,
                            ordinary_overlap: score.ordinary_overlap,
                        },
                        profile.as_deref_mut(),
                    )?;
                }
            }
        }

        for (observation_index, observation) in observations.iter().enumerate() {
            let Some(key) = ExactMatchKey::from_observation(observation) else {
                continue;
            };
            if let Some(profile) = profile.as_deref_mut() {
                profile.inactive_key_lookups = profile.inactive_key_lookups.saturating_add(1);
            }
            let Some(rectangles) = self.inactive.by_key.get(&key) else {
                continue;
            };
            let mut bucket_ids = 0_u64;
            for (rect, track_ids) in rectangles {
                bucket_ids = bucket_ids.saturating_add(saturating_u64(track_ids.len()));
                if let Some(profile) = profile.as_deref_mut() {
                    profile.inactive_rectangle_scans =
                        profile.inactive_rectangle_scans.saturating_add(1);
                    profile.candidate_match_score_pairs = profile
                        .candidate_match_score_pairs
                        .saturating_add(saturating_u64(track_ids.len()));
                }
                let ordinary_iou =
                    crate::model::Rect::from(*rect).intersection_over_union(observation.bbox);
                if ordinary_iou < INACTIVE_MINIMUM_IOU {
                    continue;
                }
                let score = 0.35 * ordinary_iou + 0.60 + if key.role_equal() { 0.05 } else { 0.0 };
                for track_id in track_ids {
                    push_candidate(
                        &mut candidates,
                        Candidate {
                            track_id: *track_id,
                            observation_index,
                            score,
                            ordinary_overlap: true,
                        },
                        profile.as_deref_mut(),
                    )?;
                    if let Some(profile) = profile.as_deref_mut() {
                        profile.inactive_candidate_ids_materialized = profile
                            .inactive_candidate_ids_materialized
                            .saturating_add(1);
                    }
                }
            }
            if let Some(profile) = profile.as_deref_mut() {
                profile.maximum_inactive_key_bucket_ids =
                    profile.maximum_inactive_key_bucket_ids.max(bucket_ids);
            }
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.ordinary_overlap.cmp(&left.ordinary_overlap))
                .then_with(|| left.track_id.cmp(&right.track_id))
                .then_with(|| left.observation_index.cmp(&right.observation_index))
        });

        let mut assigned_tracks = BTreeSet::new();
        let mut assignments = BTreeMap::new();
        for candidate in candidates {
            if !assigned_tracks.contains(&candidate.track_id)
                && !assignments.contains_key(&candidate.observation_index)
            {
                assigned_tracks.insert(candidate.track_id);
                assignments.insert(candidate.observation_index, candidate.track_id);
            }
        }
        Ok(assignments)
    }

    fn dominant_translation(&self, observations: &[ElementObservation]) -> Option<Translation> {
        let mut tracks_by_text: BTreeMap<String, Vec<&Track>> = BTreeMap::new();
        for track in self.active_tracks.values() {
            let text = track
                .element
                .text
                .as_deref()
                .map(normalize)
                .unwrap_or_default();
            if !text.is_empty() {
                tracks_by_text.entry(text).or_default().push(track);
            }
        }
        let mut observations_by_text: BTreeMap<String, Vec<&ElementObservation>> = BTreeMap::new();
        for observation in observations {
            let text = observation
                .text
                .as_deref()
                .map(normalize)
                .unwrap_or_default();
            if !text.is_empty() {
                observations_by_text
                    .entry(text)
                    .or_default()
                    .push(observation);
            }
        }

        let mut bins: BTreeMap<(i64, i64), Vec<Translation>> = BTreeMap::new();
        for (text, tracks) in tracks_by_text {
            let Some(matching_observations) = observations_by_text.get(&text) else {
                continue;
            };
            if tracks.len() != 1 || matching_observations.len() != 1 {
                continue;
            }
            let track = tracks[0];
            let observation = matching_observations[0];
            if !roles_compatible(track.element.role.as_deref(), observation.role.as_deref()) {
                continue;
            }
            let translation = center_translation(track.element.bbox, observation.bbox);
            let bin = (
                rounded_bin(translation.x, TRANSLATION_BIN_PIXELS),
                rounded_bin(translation.y, TRANSLATION_BIN_PIXELS),
            );
            bins.entry(bin).or_default().push(translation);
        }

        let mut strongest = bins
            .into_iter()
            .filter(|(_, translations)| translations.len() >= MINIMUM_TRANSLATION_ANCHORS)
            .max_by(|(left_bin, left), (right_bin, right)| {
                left.len()
                    .cmp(&right.len())
                    .then_with(|| right_bin.cmp(left_bin))
            })?
            .1;
        strongest.sort_by_key(|translation| (translation.x, translation.y));
        let middle = strongest.len() / 2;
        let mut x_values = strongest
            .iter()
            .map(|translation| translation.x)
            .collect::<Vec<_>>();
        let mut y_values = strongest
            .iter()
            .map(|translation| translation.y)
            .collect::<Vec<_>>();
        x_values.sort_unstable();
        y_values.sort_unstable();
        Some(Translation {
            x: x_values[middle],
            y: y_values[middle],
        })
    }

    fn create_track(&mut self, observation: ElementObservation) -> (u64, ElementState) {
        let track_id = self.next_element_id;
        self.next_element_id += 1;
        let element = ElementState::from_observation(format!("e{track_id:06}"), observation);
        (track_id, element)
    }

    fn wrap(&mut self, timestamp_ms: u64, body: EventBody) -> CloudEvent {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        CloudEvent {
            specversion: "1.0".to_owned(),
            id: format!("{}:{sequence}", self.stream_id),
            source: self.source.clone(),
            sequence,
            frametime: timestamp_ms,
            datacontenttype: "application/json".to_owned(),
            body,
        }
    }
}

fn canonical_element_number(id: &str) -> Option<u64> {
    let digits = id.strip_prefix('e')?;
    if digits.len() < 6 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let number = digits.parse::<u64>().ok()?;
    (number != 0 && format!("e{number:06}") == id).then_some(number)
}

fn activity_order_key(kind: &ObservedActivityKind) -> (u8, u64) {
    match kind {
        ObservedActivityKind::RegionTranslated { .. } => (0, 0),
        ObservedActivityKind::ElementRegionChanged { element_id } => {
            (1, canonical_element_number(element_id).unwrap_or(u64::MAX))
        }
    }
}

fn push_candidate(
    candidates: &mut Vec<Candidate>,
    candidate: Candidate,
    profile: Option<&mut TrackerWorkProfile>,
) -> Result<(), EncoderError> {
    if candidates.len() >= MAX_ASSOCIATION_CANDIDATES {
        return Err(EncoderError::AssociationTooLarge {
            maximum: MAX_ASSOCIATION_CANDIDATES,
        });
    }
    candidates.push(candidate);
    if let Some(profile) = profile {
        profile.maximum_candidate_buffer_len = profile
            .maximum_candidate_buffer_len
            .max(saturating_u64(candidates.len()));
    }
    Ok(())
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn track_variable_bytes(track: &Track) -> u64 {
    let element = &track.element;
    let direct = element
        .id
        .capacity()
        .saturating_add(element.text.as_ref().map_or(0, String::capacity))
        .saturating_add(element.role.as_ref().map_or(0, String::capacity));
    element
        .state
        .iter()
        .fold(saturating_u64(direct), |total, (key, value)| {
            total
                .saturating_add(saturating_u64(key.capacity()))
                .saturating_add(saturating_u64(value.capacity()))
        })
}

fn track_charge(track: &Track) -> u64 {
    TRACK_FIXED_CHARGE_BYTES
        .saturating_add(track_variable_bytes(track))
        .saturating_add(
            saturating_u64(track.element.state.len())
                .saturating_mul(STATE_ENTRY_FIXED_CHARGE_BYTES),
        )
}

fn observation_charge(observation: &ElementObservation) -> u64 {
    let direct = observation
        .text
        .as_ref()
        .map_or(0, String::capacity)
        .saturating_add(observation.role.as_ref().map_or(0, String::capacity));
    observation.state.iter().fold(
        TRACK_FIXED_CHARGE_BYTES
            .saturating_add(GENERATED_ID_CHARGE_BYTES)
            .saturating_add(saturating_u64(direct)),
        |total, (key, value)| {
            total
                .saturating_add(STATE_ENTRY_FIXED_CHARGE_BYTES)
                .saturating_add(saturating_u64(key.capacity()))
                .saturating_add(saturating_u64(value.capacity()))
        },
    )
}

fn inactive_entry_charge(key: &ExactMatchKey) -> u64 {
    TRACK_FIXED_CHARGE_BYTES.saturating_add(key.variable_capacity().saturating_mul(2))
}

fn inactive_priority(key: &ExactMatchKey, rect: RectKey, track_id: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(INACTIVE_PRIORITY_DOMAIN);
    digest.update(saturating_u64(key.text.len()).to_le_bytes());
    digest.update(key.text.as_bytes());
    match &key.role {
        RoleKey::Missing => digest.update([0]),
        RoleKey::Present(role) => {
            digest.update([1]);
            digest.update(saturating_u64(role.len()).to_le_bytes());
            digest.update(role.as_bytes());
        }
    }
    digest.update(rect.x.to_le_bytes());
    digest.update(rect.y.to_le_bytes());
    digest.update(rect.width.to_le_bytes());
    digest.update(rect.height.to_le_bytes());
    digest.update(track_id.to_le_bytes());
    digest.finalize().into()
}

fn match_score(
    track: &Track,
    observation: &ElementObservation,
    translation: Option<Translation>,
    fuzzy_supported: bool,
) -> Option<MatchScore> {
    let ordinary_iou = track.element.bbox.intersection_over_union(observation.bbox);
    let translated_iou = if track.active {
        translation.map_or(0.0, |translation| {
            translated_intersection_over_union(track.element.bbox, observation.bbox, translation)
        })
    } else {
        0.0
    };
    let intersection_over_union = if fuzzy_supported {
        ordinary_iou
    } else {
        ordinary_iou.max(translated_iou)
    };
    let text_equal = normalized_equal(track.element.text.as_deref(), observation.text.as_deref());
    let role_equal = normalized_equal(track.element.role.as_deref(), observation.role.as_deref());

    let score = 0.35 * intersection_over_union
        + if text_equal || fuzzy_supported {
            0.60
        } else {
            0.0
        }
        + if role_equal { 0.05 } else { 0.0 };

    let role_compatible =
        roles_compatible(track.element.role.as_deref(), observation.role.as_deref());

    let eligible = if fuzzy_supported {
        track.active && score >= ACTIVE_MATCH_THRESHOLD && ordinary_iou > 0.10
    } else if track.active {
        // A coherent translation may bridge disjoint boxes only when text and
        // role agree. Ordinary overlap retains the previous content-change
        // behavior without allowing global motion to override conflicting evidence.
        score >= ACTIVE_MATCH_THRESHOLD
            && (ordinary_iou > 0.10 || (translated_iou > 0.10 && text_equal && role_compatible))
    } else {
        // Never expire history based on time, but do not treat common text as
        // proof that an arbitrary previous element has returned.
        text_equal && role_compatible && intersection_over_union >= INACTIVE_MINIMUM_IOU
    };

    eligible.then_some(MatchScore {
        value: score,
        ordinary_overlap: ordinary_iou > 0.10,
    })
}

fn roles_compatible(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (None, None) => true,
        (left, right) => normalized_equal(left, right),
    }
}

fn center_translation(from: crate::model::Rect, to: crate::model::Rect) -> Translation {
    let from_x = i64::from(from.x) * 2 + i64::from(from.width);
    let from_y = i64::from(from.y) * 2 + i64::from(from.height);
    let to_x = i64::from(to.x) * 2 + i64::from(to.width);
    let to_y = i64::from(to.y) * 2 + i64::from(to.height);
    Translation {
        x: (to_x - from_x) / 2,
        y: (to_y - from_y) / 2,
    }
}

fn rounded_bin(value: i64, width: i64) -> i64 {
    if value >= 0 {
        (value + width / 2) / width
    } else {
        (value - width / 2) / width
    }
}

fn translated_intersection_over_union(
    from: crate::model::Rect,
    to: crate::model::Rect,
    translation: Translation,
) -> f64 {
    let from_left = i64::from(from.x) + translation.x;
    let from_top = i64::from(from.y) + translation.y;
    let from_right = from_left + i64::from(from.width);
    let from_bottom = from_top + i64::from(from.height);
    let to_left = i64::from(to.x);
    let to_top = i64::from(to.y);
    let to_right = to_left + i64::from(to.width);
    let to_bottom = to_top + i64::from(to.height);
    let intersection_width = (from_right.min(to_right) - from_left.max(to_left)).max(0) as u64;
    let intersection_height = (from_bottom.min(to_bottom) - from_top.max(to_top)).max(0) as u64;
    let intersection = intersection_width.saturating_mul(intersection_height);
    let from_area = u64::from(from.width) * u64::from(from.height);
    let to_area = u64::from(to.width) * u64::from(to.height);
    let union = from_area
        .saturating_add(to_area)
        .saturating_sub(intersection);
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn normalized_equal(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            let left = normalize(left);
            let right = normalize(right);
            !left.is_empty() && left == right
        }
        _ => false,
    }
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|character| character.is_alphanumeric())
        .collect()
}
