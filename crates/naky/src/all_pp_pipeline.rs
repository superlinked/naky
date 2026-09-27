//! Concrete canonical-AV1 to ScreenEvents product pipeline.

use anyhow::{Context, Result, bail};
use av1_decoder::{DecodedLumaFrame, Decoder, MatroskaReader};
use perception::all_pp::{AllPpOcr, AllPpProfile};
use perception::interaction_delta::{
    EpisodeTransitionKind, InteractionDeltaMetrics, InteractionDeltaTracker,
};
use perception::model_bundle::ModelBundleIdentity;
use perception::motion_correspondence::{
    MotionCorrespondenceMetrics, MotionCorrespondenceTracker, MotionDisposition,
};
use screenevents::{
    CloudEvent, ObservedActivityKind, Rect, ScreenEventEncoder, StatefulCompactRenderer,
    TrackerRetentionProfile, TrackerWorkProfile,
};
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL_MS: u64 = 1_000;
const RECOGNITION_LANES: u64 = 4;
const POLICY_IDENTITY: &str =
    "full-short-side-736-native-db-four-ordered-lanes+fixed-1000ms-distinct-terminal";
const ACTIVITY_INTERVAL_MS: u64 = 200;
const ACTIVITY_POLICY_IDENTITY: &str =
    "interval200+delta0099-onset-center-unique+motion0102-destination-bounds+translation-first";

type ProcessSampleResult = ();

pub(crate) struct Options<'a> {
    pub input: &'a Path,
    pub output: Option<&'a Path>,
    pub stateful_text_output: Option<&'a Path>,
    pub stream_id: &'a str,
    pub model_bundle: &'a Path,
    pub metrics_output: Option<&'a Path>,
}

#[derive(Debug, Serialize)]
struct Metrics {
    schema_version: &'static str,
    policy_identity: &'static str,
    sample_interval_ms: u64,
    recognition_lanes: u64,
    model_bundle: ModelBundleIdentity,
    stages_ns: StageTotals,
    total_wall_ns: u64,
    media_time_ms: u64,
    real_time_factor: Option<f64>,
    packets_read: u64,
    decoded_frames: u64,
    maximum_decoded_frame_interval_ms: u64,
    sampled_frames: u64,
    events: u64,
    detector_rows: u64,
    observations: u64,
    detector_active_pixels: u64,
    recognition_calls: u64,
    recognition_input_pixels: u64,
    tracker_work: TrackerWorkProfile,
    maximum_sample_wall_ns: u64,
    frames_over_one_second: u64,
    vm_hwm_kib: u64,
    final_tracker_retention: TrackerRetentionProfile,
    activity: ActivityMetrics,
}

#[derive(Clone, Debug, Serialize)]
struct ActivityMetrics {
    policy_identity: &'static str,
    comparison_interval_ms: u64,
    selected_boundaries: u64,
    compared_boundaries: u64,
    local_regions: u64,
    eligible_local_onsets: u64,
    associated_local_onsets: u64,
    unassociated_local_onsets: u64,
    ambiguous_local_onsets: u64,
    duplicate_local_suppressions: u64,
    latched_local_suppressions: u64,
    quiet_element_releases: u64,
    translation_precedence_suppressions: u64,
    element_region_changed_events: u64,
    region_translated_events: u64,
    observer_wall_sum_ns: u64,
    observer_wall_max_ns: u64,
    maximum_interval_ms: u64,
    retained_state_high_water_bytes: u64,
    latched_elements_current: u64,
    latched_elements_high_water: u64,
    local_observer: InteractionDeltaMetrics,
    translation_observer: MotionCorrespondenceMetrics,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct StageTotals {
    model_load: u64,
    container_open: u64,
    decoder_init: u64,
    packet_read: u64,
    decode: u64,
    detector_prepare: u64,
    detector_forward: u64,
    detector_postprocess: u64,
    recognition_seam_prepare: u64,
    recognition_preprocess: u64,
    recognition_forward: u64,
    recognition_decode: u64,
    recognition_wall: u64,
    recognition_postprocess: u64,
    tracking: u64,
    serialization: u64,
}

impl StageTotals {
    fn add_perception(&mut self, profile: AllPpProfile) -> Result<()> {
        add_duration(&mut self.detector_prepare, profile.detector.prepare)?;
        add_duration(&mut self.detector_forward, profile.detector.forward)?;
        add_duration(&mut self.detector_postprocess, profile.detector.postprocess)?;
        add_duration(
            &mut self.recognition_seam_prepare,
            profile.recognition.seam_prepare,
        )?;
        add_duration(
            &mut self.recognition_preprocess,
            profile.recognition.preprocess,
        )?;
        add_duration(&mut self.recognition_forward, profile.recognition.forward)?;
        add_duration(&mut self.recognition_decode, profile.recognition.decode)?;
        add_duration(
            &mut self.recognition_wall,
            profile.recognition.recognition_wall,
        )?;
        add_duration(
            &mut self.recognition_postprocess,
            profile.recognition_postprocess,
        )
    }
}

#[derive(Default)]
struct FixedSampler {
    last_selected_ms: Option<u64>,
}

impl FixedSampler {
    fn select(&mut self, timestamp_ms: u64, terminal: bool) -> bool {
        let selected = if terminal {
            self.last_selected_ms != Some(timestamp_ms)
        } else {
            self.last_selected_ms
                .is_none_or(|previous| timestamp_ms.saturating_sub(previous) >= SAMPLE_INTERVAL_MS)
        };
        if selected {
            self.last_selected_ms = Some(timestamp_ms);
        }
        selected
    }
}

#[derive(Default)]
struct Aggregate {
    stages: StageTotals,
    packets_read: u64,
    decoded_frames: u64,
    maximum_decoded_frame_interval_ms: u64,
    sampled_frames: u64,
    events: u64,
    detector_rows: u64,
    observations: u64,
    detector_active_pixels: u64,
    recognition_calls: u64,
    recognition_input_pixels: u64,
    tracker_work: TrackerWorkProfile,
    maximum_sample_wall_ns: u64,
    frames_over_one_second: u64,
}

impl Aggregate {
    fn add_profile(&mut self, profile: AllPpProfile) -> Result<()> {
        self.detector_rows = checked_add_usize(self.detector_rows, profile.detector_rows)?;
        self.observations = checked_add_usize(self.observations, profile.observations)?;
        self.detector_active_pixels = checked_add_usize(
            self.detector_active_pixels,
            profile.detector.db.active_pixels,
        )?;
        self.recognition_calls = self
            .recognition_calls
            .checked_add(profile.recognition.model_calls)
            .context("recognition call count overflow")?;
        self.recognition_input_pixels = self
            .recognition_input_pixels
            .checked_add(profile.recognition.input_pixels)
            .context("recognition pixel count overflow")?;
        self.stages.add_perception(profile)
    }

    fn add_tracker_work(&mut self, profile: TrackerWorkProfile) -> Result<()> {
        self.tracker_work.candidate_match_score_pairs = self
            .tracker_work
            .candidate_match_score_pairs
            .checked_add(profile.candidate_match_score_pairs)
            .context("tracker candidate count overflow")?;
        self.tracker_work.active_candidate_pairs = self
            .tracker_work
            .active_candidate_pairs
            .checked_add(profile.active_candidate_pairs)
            .context("tracker active candidate count overflow")?;
        self.tracker_work.inactive_key_lookups = self
            .tracker_work
            .inactive_key_lookups
            .checked_add(profile.inactive_key_lookups)
            .context("tracker inactive key lookup count overflow")?;
        self.tracker_work.inactive_rectangle_scans = self
            .tracker_work
            .inactive_rectangle_scans
            .checked_add(profile.inactive_rectangle_scans)
            .context("tracker inactive rectangle count overflow")?;
        self.tracker_work.inactive_candidate_ids_materialized = self
            .tracker_work
            .inactive_candidate_ids_materialized
            .checked_add(profile.inactive_candidate_ids_materialized)
            .context("tracker inactive candidate count overflow")?;
        self.tracker_work.maximum_inactive_key_bucket_ids = self
            .tracker_work
            .maximum_inactive_key_bucket_ids
            .max(profile.maximum_inactive_key_bucket_ids);
        self.tracker_work.maximum_candidate_buffer_len = self
            .tracker_work
            .maximum_candidate_buffer_len
            .max(profile.maximum_candidate_buffer_len);
        self.tracker_work.inactive_count_pressure_evictions = self
            .tracker_work
            .inactive_count_pressure_evictions
            .checked_add(profile.inactive_count_pressure_evictions)
            .context("tracker count-pressure eviction count overflow")?;
        self.tracker_work.inactive_byte_pressure_evictions = self
            .tracker_work
            .inactive_byte_pressure_evictions
            .checked_add(profile.inactive_byte_pressure_evictions)
            .context("tracker byte-pressure eviction count overflow")?;
        self.tracker_work.fuzzy_edge_pairs = self
            .tracker_work
            .fuzzy_edge_pairs
            .checked_add(profile.fuzzy_edge_pairs)
            .context("tracker fuzzy-edge count overflow")?;
        self.tracker_work.fuzzy_competitor_track_scans = self
            .tracker_work
            .fuzzy_competitor_track_scans
            .checked_add(profile.fuzzy_competitor_track_scans)
            .context("tracker competitor count overflow")?;
        Ok(())
    }
}

struct Destinations {
    input: PathBuf,
    events: Option<PathBuf>,
    stateful_text: Option<PathBuf>,
    metrics: Option<PathBuf>,
}

/// Writes the canonical event stream and, when requested, feeds those exact
/// events to the bounded stateful projection without replay or accumulation.
struct EventOutputs<W: Write, S: Write> {
    events: W,
    stateful: Option<StatefulCompactRenderer<S>>,
}

type FileEventOutputs = EventOutputs<BufWriter<Box<dyn Write>>, BufWriter<File>>;

impl<W: Write, S: Write> EventOutputs<W, S> {
    fn new(events: W, stateful: Option<S>) -> Self {
        Self {
            events,
            stateful: stateful.map(StatefulCompactRenderer::new),
        }
    }

    fn write_batch(&mut self, events: &[CloudEvent]) -> Result<u64> {
        for event in events {
            serde_json::to_writer(&mut self.events, event)?;
            self.events.write_all(b"\n")?;
            if let Some(renderer) = self.stateful.as_mut() {
                renderer
                    .write_event(event)
                    .context("failed to render ScreenEvent as bounded stateful text")?;
            }
        }
        u64::try_from(events.len()).context("event batch length exceeds u64")
    }

    fn finish(mut self) -> Result<(W, Option<S>)> {
        self.events
            .flush()
            .context("failed to flush ScreenEvents")?;
        let stateful = self
            .stateful
            .take()
            .map(|renderer| {
                renderer
                    .finish()
                    .context("failed to finish bounded stateful text")
            })
            .transpose()?;
        Ok((self.events, stateful))
    }
}

struct ActivityObserver {
    local: InteractionDeltaTracker,
    translation: MotionCorrespondenceTracker,
    last_boundary_ms: Option<u64>,
    active_elements: std::collections::BTreeSet<String>,
    metrics: ActivityMetrics,
}

impl ActivityObserver {
    fn new() -> Result<Self> {
        Ok(Self {
            local: InteractionDeltaTracker::new(),
            translation: MotionCorrespondenceTracker::with_comparison_interval(
                ACTIVITY_INTERVAL_MS,
            )?,
            last_boundary_ms: None,
            active_elements: std::collections::BTreeSet::new(),
            metrics: ActivityMetrics {
                policy_identity: ACTIVITY_POLICY_IDENTITY,
                comparison_interval_ms: ACTIVITY_INTERVAL_MS,
                selected_boundaries: 0,
                compared_boundaries: 0,
                local_regions: 0,
                eligible_local_onsets: 0,
                associated_local_onsets: 0,
                unassociated_local_onsets: 0,
                ambiguous_local_onsets: 0,
                duplicate_local_suppressions: 0,
                latched_local_suppressions: 0,
                quiet_element_releases: 0,
                translation_precedence_suppressions: 0,
                element_region_changed_events: 0,
                region_translated_events: 0,
                observer_wall_sum_ns: 0,
                observer_wall_max_ns: 0,
                maximum_interval_ms: 0,
                retained_state_high_water_bytes: 0,
                latched_elements_current: 0,
                latched_elements_high_water: 0,
                local_observer: InteractionDeltaMetrics::default(),
                translation_observer: MotionCorrespondenceMetrics::default(),
            },
        })
    }

    fn observe(
        &mut self,
        frame: &DecodedLumaFrame,
        timestamp_ms: u64,
        encoder: &mut ScreenEventEncoder,
    ) -> Result<Vec<CloudEvent>> {
        if self
            .last_boundary_ms
            .is_some_and(|previous| timestamp_ms.saturating_sub(previous) < ACTIVITY_INTERVAL_MS)
        {
            return Ok(Vec::new());
        }
        let from_frametime = self.last_boundary_ms;
        let started = Instant::now();
        let local_started = Instant::now();
        let local = self
            .local
            .observe(timestamp_ms, &frame.luma, frame.width, frame.height)?;
        self.local.record_analysis_wall(local_started.elapsed());
        let translation =
            self.translation
                .observe(timestamp_ms, &frame.luma, frame.width, frame.height)?;
        self.last_boundary_ms = Some(timestamp_ms);
        self.metrics.selected_boundaries = self.metrics.selected_boundaries.saturating_add(1);
        self.metrics.local_regions = self
            .metrics
            .local_regions
            .saturating_add(local.regions.len() as u64);

        let mut kinds = Vec::new();
        if let Some(from_frametime) = from_frametime {
            self.metrics.compared_boundaries = self.metrics.compared_boundaries.saturating_add(1);
            let interval_ms = timestamp_ms - from_frametime;
            self.metrics.maximum_interval_ms = self.metrics.maximum_interval_ms.max(interval_ms);
            let starts = local
                .episode_transitions
                .iter()
                .filter(|transition| transition.kind == EpisodeTransitionKind::Started)
                .collect::<Vec<_>>();
            self.metrics.eligible_local_onsets = self
                .metrics
                .eligible_local_onsets
                .saturating_add(starts.len() as u64);

            let current_supported = local
                .regions
                .iter()
                .flat_map(|region| {
                    let center_x = region.x.saturating_add(region.width / 2);
                    let center_y = region.y.saturating_add(region.height / 2);
                    encoder.elements_containing_point(center_x, center_y)
                })
                .collect::<std::collections::BTreeSet<_>>();
            let before_release = self.active_elements.len();
            self.active_elements
                .retain(|element_id| current_supported.contains(element_id));
            self.metrics.quiet_element_releases = self
                .metrics
                .quiet_element_releases
                .saturating_add((before_release - self.active_elements.len()) as u64);

            if translation.disposition == MotionDisposition::CoherentTranslation {
                self.metrics.translation_precedence_suppressions = self
                    .metrics
                    .translation_precedence_suppressions
                    .saturating_add(starts.len() as u64);
                let bbox = translation
                    .dominant_translation_box
                    .context("coherent translation omitted destination support bounds")?;
                let dx = translation
                    .dominant_translation_dx_quarter
                    .context("coherent translation omitted dx")?
                    .checked_mul(perception::motion_correspondence::DOWNSAMPLE_FACTOR as i32)
                    .context("translation dx overflow")?;
                let dy = translation
                    .dominant_translation_dy_quarter
                    .context("coherent translation omitted dy")?
                    .checked_mul(perception::motion_correspondence::DOWNSAMPLE_FACTOR as i32)
                    .context("translation dy overflow")?;
                kinds.push(ObservedActivityKind::RegionTranslated {
                    bbox: Rect {
                        x: bbox.x,
                        y: bbox.y,
                        width: bbox.width,
                        height: bbox.height,
                    },
                    dx,
                    dy,
                });
                self.metrics.region_translated_events =
                    self.metrics.region_translated_events.saturating_add(1);
            } else {
                let mut associated = std::collections::BTreeSet::new();
                for transition in starts {
                    let center_x = transition
                        .region
                        .x
                        .saturating_add(transition.region.width / 2);
                    let center_y = transition
                        .region
                        .y
                        .saturating_add(transition.region.height / 2);
                    let containing = encoder.elements_containing_point(center_x, center_y);
                    match containing.as_slice() {
                        [element_id] => {
                            self.metrics.associated_local_onsets =
                                self.metrics.associated_local_onsets.saturating_add(1);
                            if self.active_elements.contains(element_id) {
                                self.metrics.latched_local_suppressions =
                                    self.metrics.latched_local_suppressions.saturating_add(1);
                            } else if !associated.insert(element_id.clone()) {
                                self.metrics.duplicate_local_suppressions =
                                    self.metrics.duplicate_local_suppressions.saturating_add(1);
                            }
                        }
                        [] => {
                            self.metrics.unassociated_local_onsets =
                                self.metrics.unassociated_local_onsets.saturating_add(1);
                        }
                        _ => {
                            self.metrics.ambiguous_local_onsets =
                                self.metrics.ambiguous_local_onsets.saturating_add(1);
                        }
                    }
                }
                self.metrics.element_region_changed_events = self
                    .metrics
                    .element_region_changed_events
                    .saturating_add(associated.len() as u64);
                self.active_elements.extend(associated.iter().cloned());
                kinds.extend(
                    associated.into_iter().map(|element_id| {
                        ObservedActivityKind::ElementRegionChanged { element_id }
                    }),
                );
            }
            let events = encoder.process_observed_activity(from_frametime, timestamp_ms, kinds)?;
            self.record_wall(started.elapsed())?;
            return Ok(events);
        }
        self.record_wall(started.elapsed())?;
        Ok(kinds
            .into_iter()
            .map(|_| unreachable!("initial boundary cannot emit activity"))
            .collect())
    }

    fn record_wall(&mut self, elapsed: Duration) -> Result<()> {
        let wall_ns = duration_ns(elapsed)?;
        self.metrics.observer_wall_sum_ns =
            self.metrics.observer_wall_sum_ns.saturating_add(wall_ns);
        self.metrics.observer_wall_max_ns = self.metrics.observer_wall_max_ns.max(wall_ns);
        let observer_bytes = self
            .local
            .metrics()
            .retention_high_water
            .total_charged_bytes
            .saturating_add(
                self.translation
                    .metrics()
                    .retention_high_water
                    .total_charged_bytes,
            );
        let latch_bytes = self.active_elements.iter().fold(0_u64, |total, id| {
            total.saturating_add(64 + id.capacity() as u64)
        });
        self.metrics.retained_state_high_water_bytes = self
            .metrics
            .retained_state_high_water_bytes
            .max(observer_bytes.saturating_add(latch_bytes));
        self.metrics.latched_elements_current = self.active_elements.len() as u64;
        self.metrics.latched_elements_high_water = self
            .metrics
            .latched_elements_high_water
            .max(self.active_elements.len() as u64);
        self.metrics.local_observer = self.local.metrics().clone();
        self.metrics.translation_observer = self.translation.metrics().clone();
        Ok(())
    }

    fn metrics(&self) -> ActivityMetrics {
        self.metrics.clone()
    }
}

pub(crate) fn run(options: Options<'_>) -> Result<()> {
    let destinations = preflight_paths(
        options.input,
        options.output,
        options.stateful_text_output,
        options.metrics_output,
    )?;
    let total_started = Instant::now();
    let mut aggregate = Aggregate::default();

    let started = Instant::now();

    let (mut perception, load) = AllPpOcr::load_bundle(options.model_bundle)?;
    aggregate.stages.model_load = duration_ns(started.elapsed())?;

    let started = Instant::now();
    let file = File::open(&destinations.input)
        .with_context(|| format!("failed to open {}", destinations.input.display()))?;
    let mut reader = MatroskaReader::new(BufReader::new(file))
        .context("invalid canonical AV1 Matroska input")?;
    aggregate.stages.container_open = duration_ns(started.elapsed())?;
    let timestamp_scale_ns = reader.header().timestamp_scale_ns;

    let started = Instant::now();
    let mut decoder = Decoder::new().context("failed to initialize rav1d")?;
    aggregate.stages.decoder_init = duration_ns(started.elapsed())?;
    let mut encoder = ScreenEventEncoder::new(options.stream_id)?;
    let mut event_outputs = event_outputs(
        destinations.events.as_deref(),
        destinations.stateful_text.as_deref(),
    )?;
    let mut activity = ActivityObserver::new()?;
    let mut sampler = FixedSampler::default();
    let mut first_decoded_ms = None;
    let mut last_decoded_ms = None;
    let mut previous_decoded_ms = None;
    let mut pending_terminal = None;

    while let Some(packet) = {
        let started = Instant::now();
        let packet = reader.read_packet()?;
        add_duration(&mut aggregate.stages.packet_read, started.elapsed())?;
        packet
    } {
        aggregate.packets_read = aggregate
            .packets_read
            .checked_add(1)
            .context("packet count overflow")?;
        let started = Instant::now();
        let decoded = decoder.decode_packet(&packet.data, packet.timestamp)?;
        add_duration(&mut aggregate.stages.decode, started.elapsed())?;
        for frame in decoded {
            dispatch_decoded_frame(
                &frame,
                timestamp_scale_ns,
                &mut sampler,
                &mut previous_decoded_ms,
                &mut first_decoded_ms,
                &mut last_decoded_ms,
                &mut aggregate,
                &mut activity,
                &mut perception,
                &mut encoder,
                &mut event_outputs,
            )?;
            pending_terminal = Some(frame);
        }
    }

    let started = Instant::now();
    let drained = decoder.drain()?;
    add_duration(&mut aggregate.stages.decode, started.elapsed())?;
    for frame in drained {
        dispatch_decoded_frame(
            &frame,
            timestamp_scale_ns,
            &mut sampler,
            &mut previous_decoded_ms,
            &mut first_decoded_ms,
            &mut last_decoded_ms,
            &mut aggregate,
            &mut activity,
            &mut perception,
            &mut encoder,
            &mut event_outputs,
        )?;
        pending_terminal = Some(frame);
    }

    let used_settled_sampler = false;
    if !used_settled_sampler && let Some(frame) = pending_terminal {
        let terminal_ms = timestamp_ms(frame.timestamp, timestamp_scale_ns)?;
        if sampler.select(terminal_ms, true) {
            process_sample(
                &frame,
                timestamp_scale_ns,
                &mut perception,
                &mut encoder,
                &mut event_outputs,
                &mut aggregate,
            )?;
        }
    }
    if aggregate.sampled_frames == 0 {
        bail!("canonical AV1 produced no sampled frames");
    }
    let _ = event_outputs.finish()?;

    let first_decoded_ms = first_decoded_ms.context("canonical AV1 produced no decoded frames")?;
    let last_decoded_ms = last_decoded_ms.context("canonical AV1 produced no decoded frames")?;
    let media_time_ms = last_decoded_ms
        .checked_sub(first_decoded_ms)
        .context("decoded media time decreased")?;
    let final_tracker_retention = encoder.retention_profile();
    let total_wall_ns = duration_ns(total_started.elapsed())?;
    let metrics = Metrics {
        schema_version: "naky.av1-to-events-metrics.v0",
        policy_identity: { POLICY_IDENTITY },
        sample_interval_ms: SAMPLE_INTERVAL_MS,
        recognition_lanes: RECOGNITION_LANES,
        model_bundle: load.model_bundle,
        stages_ns: aggregate.stages,
        total_wall_ns,
        media_time_ms,
        real_time_factor: (media_time_ms != 0)
            .then_some(total_wall_ns as f64 / (media_time_ms as f64 * 1_000_000.0)),
        packets_read: aggregate.packets_read,
        decoded_frames: aggregate.decoded_frames,
        maximum_decoded_frame_interval_ms: aggregate.maximum_decoded_frame_interval_ms,
        sampled_frames: aggregate.sampled_frames,
        events: aggregate.events,
        detector_rows: aggregate.detector_rows,
        observations: aggregate.observations,
        detector_active_pixels: aggregate.detector_active_pixels,
        recognition_calls: aggregate.recognition_calls,
        recognition_input_pixels: aggregate.recognition_input_pixels,
        tracker_work: aggregate.tracker_work,
        maximum_sample_wall_ns: aggregate.maximum_sample_wall_ns,
        frames_over_one_second: aggregate.frames_over_one_second,
        vm_hwm_kib: vm_hwm_kib()?,
        final_tracker_retention,
        activity: activity.metrics(),
    };
    if let Some(path) = destinations.metrics {
        write_metrics(&path, &metrics)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_decoded_frame(
    frame: &DecodedLumaFrame,
    timestamp_scale_ns: u64,
    fixed_sampler: &mut FixedSampler,
    previous_decoded_ms: &mut Option<u64>,
    first_decoded_ms: &mut Option<u64>,
    last_decoded_ms: &mut Option<u64>,
    aggregate: &mut Aggregate,
    activity: &mut ActivityObserver,
    perception: &mut AllPpOcr,
    encoder: &mut ScreenEventEncoder,
    event_outputs: &mut EventOutputs<impl Write, impl Write>,
) -> Result<()> {
    visit_decoded_frame(
        frame,
        timestamp_scale_ns,
        previous_decoded_ms,
        first_decoded_ms,
        last_decoded_ms,
        aggregate,
        activity,
        encoder,
        event_outputs,
    )?;
    if fixed_sampler.select(timestamp_ms(frame.timestamp, timestamp_scale_ns)?, false) {
        process_sample(
            frame,
            timestamp_scale_ns,
            perception,
            encoder,
            event_outputs,
            aggregate,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn visit_decoded_frame(
    frame: &DecodedLumaFrame,
    timestamp_scale_ns: u64,
    previous_decoded_ms: &mut Option<u64>,
    first_decoded_ms: &mut Option<u64>,
    last_decoded_ms: &mut Option<u64>,
    aggregate: &mut Aggregate,
    activity: &mut ActivityObserver,
    encoder: &mut ScreenEventEncoder,
    event_outputs: &mut EventOutputs<impl Write, impl Write>,
) -> Result<()> {
    let observed_ms = timestamp_ms(frame.timestamp, timestamp_scale_ns)?;
    if previous_decoded_ms.is_some_and(|previous| observed_ms <= previous) {
        bail!("decoded display timestamps are duplicate or decreasing");
    }
    if let Some(previous) = *previous_decoded_ms {
        aggregate.maximum_decoded_frame_interval_ms = aggregate
            .maximum_decoded_frame_interval_ms
            .max(observed_ms - previous);
    }
    *previous_decoded_ms = Some(observed_ms);
    first_decoded_ms.get_or_insert(observed_ms);
    *last_decoded_ms = Some(observed_ms);
    aggregate.decoded_frames = aggregate
        .decoded_frames
        .checked_add(1)
        .context("decoded frame count overflow")?;
    let started = Instant::now();
    let events = activity.observe(frame, observed_ms, encoder)?;
    add_duration(&mut aggregate.stages.tracking, started.elapsed())?;
    let started = Instant::now();
    let event_count = event_outputs.write_batch(&events)?;
    add_duration(&mut aggregate.stages.serialization, started.elapsed())?;
    aggregate.events = aggregate
        .events
        .checked_add(event_count)
        .context("event count overflow")?;
    Ok(())
}

fn process_sample(
    frame: &DecodedLumaFrame,
    timestamp_scale_ns: u64,
    perception: &mut AllPpOcr,
    encoder: &mut ScreenEventEncoder,
    outputs: &mut EventOutputs<impl Write, impl Write>,
    aggregate: &mut Aggregate,
) -> Result<ProcessSampleResult> {
    let sample_started = Instant::now();
    let observed_ms = timestamp_ms(frame.timestamp, timestamp_scale_ns)?;
    let page = perception.observe_luma(observed_ms, &frame.luma, frame.width, frame.height)?;
    aggregate.add_profile(page.profile)?;

    let started = Instant::now();
    let (events, tracker_work) = encoder.process_frame_profiled(page.frame)?;
    add_duration(&mut aggregate.stages.tracking, started.elapsed())?;
    aggregate.add_tracker_work(tracker_work)?;

    let started = Instant::now();
    let event_count = outputs.write_batch(&events)?;
    add_duration(&mut aggregate.stages.serialization, started.elapsed())?;
    aggregate.events = aggregate
        .events
        .checked_add(event_count)
        .context("event count overflow")?;
    aggregate.sampled_frames = aggregate
        .sampled_frames
        .checked_add(1)
        .context("sampled frame count overflow")?;
    let sample_wall_ns = duration_ns(sample_started.elapsed())?;
    aggregate.maximum_sample_wall_ns = aggregate.maximum_sample_wall_ns.max(sample_wall_ns);
    aggregate.frames_over_one_second = aggregate
        .frames_over_one_second
        .checked_add(u64::from(sample_wall_ns > 1_000_000_000))
        .context("slow frame count overflow")?;

    Ok(())
}

fn event_outputs(events: Option<&Path>, stateful_text: Option<&Path>) -> Result<FileEventOutputs> {
    let events: Box<dyn Write> = match events {
        Some(path) => Box::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("refusing to overwrite {}", path.display()))?,
        ),
        None => Box::new(io::stdout()),
    };
    let stateful_text = stateful_text
        .map(|path| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map(BufWriter::new)
                .with_context(|| format!("refusing to overwrite {}", path.display()))
        })
        .transpose()?;
    Ok(EventOutputs::new(BufWriter::new(events), stateful_text))
}

fn write_metrics(path: &Path, metrics: &Metrics) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("refusing to overwrite {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, metrics)?;
    writer.write_all(b"\n")?;
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))
}

fn preflight_paths(
    input: &Path,
    events: Option<&Path>,
    stateful_text: Option<&Path>,
    metrics: Option<&Path>,
) -> Result<Destinations> {
    let input = fs::canonicalize(input)
        .with_context(|| format!("failed to resolve input {}", input.display()))?;
    let events = events.map(resolve_new_path).transpose()?;
    let stateful_text = stateful_text.map(resolve_new_path).transpose()?;
    let metrics = metrics.map(resolve_new_path).transpose()?;
    if events.as_ref() == Some(&input)
        || metrics.as_ref() == Some(&input)
        || stateful_text.as_ref() == Some(&input)
    {
        bail!("input and all output paths must be distinct");
    }
    if (events.is_some() && events == metrics)
        || (events.is_some() && events == stateful_text)
        || (metrics.is_some() && metrics == stateful_text)
    {
        bail!("input and all output paths must be distinct");
    }
    Ok(Destinations {
        input,
        events,
        stateful_text,
        metrics,
    })
}

fn resolve_new_path(path: &Path) -> Result<PathBuf> {
    if fs::symlink_metadata(path).is_ok() {
        bail!("refusing to overwrite {}", path.display());
    }
    let name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .context("output path must name a file")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let parent = fs::canonicalize(parent.unwrap_or_else(|| Path::new(".")))
        .with_context(|| format!("failed to resolve output parent for {}", path.display()))?;
    Ok(parent.join(name))
}

fn timestamp_ms(ticks: u64, scale_ns: u64) -> Result<u64> {
    ticks
        .checked_mul(scale_ns)
        .context("timestamp nanoseconds overflow")
        .map(|ns| ns / 1_000_000)
}

fn duration_ns(duration: Duration) -> Result<u64> {
    duration
        .as_nanos()
        .try_into()
        .context("duration exceeds u64 nanoseconds")
}

fn add_duration(total: &mut u64, duration: Duration) -> Result<()> {
    *total = total
        .checked_add(duration_ns(duration)?)
        .context("stage duration overflow")?;
    Ok(())
}

fn checked_add_usize(total: u64, value: usize) -> Result<u64> {
    total
        .checked_add(u64::try_from(value).context("counter exceeds u64")?)
        .context("counter overflow")
}

fn vm_hwm_kib() -> Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .context("/proc/self/status omitted VmHWM")?;
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 3 || fields[0] != "VmHWM:" || fields[2] != "kB" {
        bail!("unexpected VmHWM row");
    }
    fields[1].parse().context("invalid VmHWM value")
}
