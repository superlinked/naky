//! Authenticated PP-OCRv6 recognition runtime used by Näky.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::thread::{self, ScopedJoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rten::{FloatOperators, Model, RunOptions, ThreadPool, Value};
use rten_tensor::{NdTensor, prelude::*};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CANDIDATE_HEIGHT: usize = 48;
pub const MIN_BATCH_WIDTH: usize = 320;
pub const MAX_BATCH_WIDTH: usize = 3200;
pub const CANDIDATE_CLASSES: usize = 6906;
pub const CTC_PROBABILITY_ROW_SUM_TOLERANCE: f64 = 1e-3;
pub const MAX_BATCH_SIZE: usize = 8;
pub const OCRS_SEAM_HEIGHT: usize = 64;
pub const OCRS_SEAM_MAX_WIDTH: usize = 2400;
const MAX_ORDERED_PARALLEL_LANES: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WidthPolicy {
    OfficialMin320,
    TightStride8,
}

const SOURCE_YAML_SHA256: &str = "66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1";
const SOURCE_YAML_BYTES: u64 = 55_571;
const CANDIDATE_RTEN_SHA256: &str =
    "5c1c9a16fcdb11e9e032d289c26f111c8fcc1de061a8c1981e8ba80c2079f6bb";
const CANDIDATE_RTEN_BYTES: u64 = 4_485_096;
const DICTIONARY_SHA256: &str = "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub bytes: u64,
    pub sha256: String,
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn locked_buffer_identity(
    bytes: &[u8],
    expected_bytes: u64,
    expected_sha256: &str,
    artifact: &str,
) -> Result<Identity> {
    let actual = Identity {
        bytes: u64::try_from(bytes.len()).context("file size exceeds u64")?,
        sha256: sha256_bytes(bytes),
    };
    if actual.bytes != expected_bytes || actual.sha256 != expected_sha256 {
        bail!(
            "identity mismatch for {artifact}: expected {expected_bytes} bytes {expected_sha256}, got {} bytes {}",
            actual.bytes,
            actual.sha256
        );
    }
    Ok(actual)
}

#[derive(Deserialize)]
struct InferenceYaml {
    #[serde(rename = "PostProcess")]
    post_process: PostProcess,
}

#[derive(Deserialize)]
struct PostProcess {
    character_dict: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Dictionary {
    /// Labels 1..=6905 in model order. Class zero is blank.
    pub labels: Vec<String>,
    pub source_sha256: String,
}

impl Dictionary {
    pub fn from_yaml(bytes: &[u8]) -> Result<Self> {
        let yaml: InferenceYaml =
            serde_yaml::from_slice(bytes).context("invalid inference YAML")?;
        let source = yaml.post_process.character_dict;
        if source.len() != 6904 {
            bail!("expected 6904 dictionary entries, found {}", source.len());
        }
        if source.iter().any(String::is_empty) {
            bail!("dictionary contains an empty entry");
        }
        let unique: BTreeSet<&str> = source.iter().map(String::as_str).collect();
        if unique.len() != source.len() {
            bail!("dictionary contains duplicate entries");
        }
        if source.iter().any(|entry| entry == " ") {
            bail!("source dictionary unexpectedly already contains space");
        }
        let mut canonical = Vec::new();
        for entry in &source {
            canonical.extend_from_slice(entry.as_bytes());
            canonical.push(b'\n');
        }
        let source_sha256 = sha256_bytes(&canonical);
        if source_sha256 != DICTIONARY_SHA256 {
            bail!("ordered dictionary digest mismatch: {source_sha256}");
        }
        let mut labels = source;
        labels.push(" ".to_owned());
        if labels.len() + 1 != CANDIDATE_CLASSES {
            bail!("dictionary/model class contract mismatch");
        }
        Ok(Self {
            labels,
            source_sha256,
        })
    }
}

/// Exact, authenticated PP-OCRv6 Tiny recognizer.
///
/// It deliberately exposes only batch-one recognition.
pub struct PpRecognizer {
    model: Model,
    dictionary: Dictionary,
    policy: WidthPolicy,
    model_identity: Identity,
    inference_yaml_identity: Identity,
}

#[derive(Clone, Debug)]
pub struct PpRecognizerLoadProfile {
    pub model_load: Duration,
    pub model_identity: Identity,
    pub inference_yaml_identity: Identity,
    pub dictionary_sha256: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PpRecognitionProfile {
    pub preprocess: Duration,
    pub forward: Duration,
    pub decode: Duration,
    pub input_pixels: u64,
    pub batch_size: usize,
    pub batch_width: usize,
    pub upstream_cap_hit: bool,
    pub ctc_steps: usize,
}

#[derive(Clone, Debug)]
pub struct PpRecognition {
    pub text: String,
    pub profile: PpRecognitionProfile,
}

/// Elapsed wall time across each barrier in ordered parallel recognition.
#[derive(Clone, Copy, Debug, Default)]
pub struct PpOrderedParallelProfile {
    /// Preprocessing and validation before model execution.
    pub preprocess_and_validate_wall: Duration,
    pub forward_wall: Duration,
    pub decode_wall: Duration,
    pub recognition_wall: Duration,
}

struct PreparedParallelRecognition {
    tensor: NdTensor<f32, 4>,
    profile: PpRecognitionProfile,
}

struct ForwardParallelRecognition {
    output: Value,
    profile: PpRecognitionProfile,
}

fn reconcile_ordered<T>(lane_records: Vec<Vec<(usize, T)>>, item_count: usize) -> Result<Vec<T>> {
    let lane_count = lane_records.len();
    let mut ordered = (0..item_count).map(|_| None).collect::<Vec<_>>();
    for (lane, records) in lane_records.into_iter().enumerate() {
        for (original_index, value) in records {
            if original_index % lane_count != lane {
                bail!("parallel lane {lane} returned item {original_index} from another lane");
            }
            let slot = ordered.get_mut(original_index).with_context(|| {
                format!("parallel lane {lane} returned out-of-range item {original_index}")
            })?;
            if slot.replace(value).is_some() {
                bail!("parallel recognition returned duplicate item {original_index}");
            }
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.with_context(|| format!("parallel recognition omitted item {index}"))
        })
        .collect()
}

fn validate_parallel_lane_count(requested: NonZeroUsize, available: NonZeroUsize) -> Result<()> {
    let limit = available.get().min(MAX_ORDERED_PARALLEL_LANES);
    if requested.get() > limit {
        bail!(
            "ordered parallel recognition requested {} lanes, limit is {limit}",
            requested.get()
        );
    }
    Ok(())
}

fn join_workers<T>(handles: Vec<ScopedJoinHandle<'_, Result<T>>>, stage: &str) -> Result<Vec<T>> {
    handles
        .into_iter()
        .enumerate()
        .map(|(lane, handle)| {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("{stage} worker panicked in lane {lane}"))?
        })
        .collect()
}

/// A shared recognizer with fixed, one-thread RTen lanes.
///
/// Calls are causal within one frame. Construction permits at most four lanes
/// and never more than the process's available parallelism. Recognition takes
/// mutable access so one executor cannot oversubscribe itself across frames.
/// Item `i` always runs in lane `i % lane_count`, and results are reconciled
/// back into input order.
pub struct PpOrderedParallelRecognizer {
    recognizer: Arc<PpRecognizer>,
    lane_pools: Vec<Arc<ThreadPool>>,
}

impl PpOrderedParallelRecognizer {
    pub fn new(recognizer: Arc<PpRecognizer>, lane_count: NonZeroUsize) -> Result<Self> {
        validate_parallel_lane_count(lane_count, thread::available_parallelism()?)?;
        Ok(Self {
            recognizer,
            lane_pools: (0..lane_count.get())
                .map(|_| Arc::new(ThreadPool::with_num_threads(1)))
                .collect(),
        })
    }

    pub fn lane_count(&self) -> usize {
        self.lane_pools.len()
    }

    /// Recognize seams in deterministic strided lanes and return input-order results.
    pub fn recognize_ordered(
        &mut self,
        seams: &[Seam],
    ) -> Result<(Vec<PpRecognition>, PpOrderedParallelProfile)> {
        if seams.is_empty() {
            return Ok((Vec::new(), PpOrderedParallelProfile::default()));
        }

        let lane_count = self.lane_count();
        let recognition_started = Instant::now();
        let preprocess_started = Instant::now();
        let recognizer = Arc::clone(&self.recognizer);
        let prepared_lanes = thread::scope(|scope| -> Result<Vec<Vec<_>>> {
            let handles = (0..lane_count)
                .map(|lane| {
                    let recognizer = Arc::clone(&recognizer);
                    scope.spawn(move || -> Result<Vec<_>> {
                        (lane..seams.len())
                            .step_by(lane_count)
                            .map(|original_index| {
                                let started = Instant::now();
                                let CandidateBatch {
                                    tensor,
                                    upstream_cap_hits,
                                    ..
                                } = preprocess_seams(
                                    std::slice::from_ref(&seams[original_index]),
                                    recognizer.policy,
                                )
                                .with_context(|| {
                                    format!(
                                        "preprocess failed in lane {lane} for item {original_index}"
                                    )
                                })?;
                                let upstream_cap_hit = match upstream_cap_hits.as_slice() {
                                    [value] => *value,
                                    _ => bail!("preprocess returned invalid cap membership"),
                                };
                                let shape = tensor.shape();
                                if shape[0] != 1
                                    || shape[1] != 3
                                    || shape[2] != CANDIDATE_HEIGHT
                                    || shape[3] == 0
                                    || shape[3] > MAX_BATCH_WIDTH
                                {
                                    bail!("preprocess returned invalid PP operand shape {shape:?}");
                                }
                                if tensor.iter().any(|value| !value.is_finite()) {
                                    bail!("preprocess returned non-finite PP operand values");
                                }
                                let input_pixels = shape
                                    .into_iter()
                                    .try_fold(1_u64, |product, dimension| {
                                        product.checked_mul(u64::try_from(dimension).ok()?)
                                    })
                                    .context("PP operand shape exceeds u64 pixels")?;
                                Ok((
                                    original_index,
                                    PreparedParallelRecognition {
                                        tensor,
                                        profile: PpRecognitionProfile {
                                            preprocess: started.elapsed(),
                                            forward: Duration::ZERO,
                                            decode: Duration::ZERO,
                                            input_pixels,
                                            batch_size: shape[0],
                                            batch_width: shape[3],
                                            upstream_cap_hit,
                                            ctc_steps: 0,
                                        },
                                    },
                                ))
                            })
                            .collect()
                    })
                })
                .collect::<Vec<_>>();
            join_workers(handles, "preprocess")
        })?;
        let preprocess_and_validate_wall = preprocess_started.elapsed();

        let forward_started = Instant::now();
        let recognizer = Arc::clone(&self.recognizer);
        let forward_lanes = thread::scope(|scope| -> Result<Vec<Vec<_>>> {
            let handles = prepared_lanes
                .into_iter()
                .zip(&self.lane_pools)
                .enumerate()
                .map(|(lane, (records, pool))| {
                    let pool = Arc::clone(pool);
                    let recognizer = Arc::clone(&recognizer);
                    scope.spawn(move || -> Result<Vec<_>> {
                        records
                            .into_iter()
                            .map(|(original_index, mut prepared)| {
                                let started = Instant::now();
                                let options = RunOptions::default()
                                    .with_thread_pool(Some(Arc::clone(&pool)));
                                let output = recognizer
                                    .model
                                    .run_one(prepared.tensor.view().into(), Some(options))
                                    .with_context(|| {
                                        format!("forward failed in lane {lane} for item {original_index}")
                                    })?;
                                prepared.profile.forward = started.elapsed();
                                Ok((
                                    original_index,
                                    ForwardParallelRecognition {
                                        output,
                                        profile: prepared.profile,
                                    },
                                ))
                            })
                            .collect()
                    })
                })
                .collect::<Vec<_>>();
            join_workers(handles, "forward")
        })?;
        let forward_wall = forward_started.elapsed();

        let decode_started = Instant::now();
        let recognizer = Arc::clone(&self.recognizer);
        let decoded_lanes = thread::scope(|scope| -> Result<Vec<Vec<_>>> {
            let handles = forward_lanes
                .into_iter()
                .enumerate()
                .map(|(lane, records)| {
                    let recognizer = Arc::clone(&recognizer);
                    scope.spawn(move || -> Result<Vec<_>> {
                        records
                            .into_iter()
                            .map(|(original_index, mut forwarded)| {
                                let started = Instant::now();
                                let output: NdTensor<f32, 3> =
                                    forwarded.output.try_into().with_context(|| {
                                        format!("lane {lane} item {original_index} output must be rank-3 f32")
                                    })?;
                                let shape = output.shape();
                                validate_candidate_output_shape(shape, 1).with_context(|| {
                                    format!("lane {lane} item {original_index} output shape is invalid")
                                })?;
                                let values = output.iter().copied().collect::<Vec<_>>();
                                let [text] = decode_candidate(
                                    shape,
                                    &values,
                                    &recognizer.dictionary,
                                )?
                                .try_into()
                                .map_err(|_| anyhow::anyhow!("batch-one decode returned wrong text count"))?;
                                forwarded.profile.decode = started.elapsed();
                                forwarded.profile.ctc_steps = shape[1];
                                Ok((
                                    original_index,
                                    PpRecognition {
                                        text,
                                        profile: forwarded.profile,
                                    },
                                ))
                            })
                            .collect()
                    })
                })
                .collect::<Vec<_>>();
            join_workers(handles, "decode")
        })?;
        let decode_wall = decode_started.elapsed();
        let recognitions = reconcile_ordered(decoded_lanes, seams.len())?;
        Ok((
            recognitions,
            PpOrderedParallelProfile {
                preprocess_and_validate_wall,
                forward_wall,
                decode_wall,
                recognition_wall: recognition_started.elapsed(),
            },
        ))
    }
}

impl PpRecognizer {
    /// Authenticate both operands before loading the model or parsing the
    /// dictionary. This keeps a path typo or artifact drift out of inference.
    pub fn load(
        model_path: &Path,
        inference_yaml_path: &Path,
        policy: WidthPolicy,
    ) -> Result<(Self, PpRecognizerLoadProfile)> {
        let model_bytes = fs::read(model_path)
            .with_context(|| format!("failed to read {}", model_path.display()))?;
        let yaml_bytes = fs::read(inference_yaml_path)
            .with_context(|| format!("failed to read {}", inference_yaml_path.display()))?;
        Self::from_bytes(model_bytes, yaml_bytes, policy)
    }

    /// Load the exact recognizer and dictionary from caller-owned buffers.
    pub fn from_bytes(
        model_bytes: Vec<u8>,
        yaml_bytes: Vec<u8>,
        policy: WidthPolicy,
    ) -> Result<(Self, PpRecognizerLoadProfile)> {
        let model_identity = locked_buffer_identity(
            &model_bytes,
            CANDIDATE_RTEN_BYTES,
            CANDIDATE_RTEN_SHA256,
            "PP-OCRv6 Tiny recognizer model buffer",
        )?;
        let inference_yaml_identity = locked_buffer_identity(
            &yaml_bytes,
            SOURCE_YAML_BYTES,
            SOURCE_YAML_SHA256,
            "PP-OCRv6 Tiny recognizer inference YAML buffer",
        )?;
        let dictionary = Dictionary::from_yaml(&yaml_bytes)?;
        let model_started = Instant::now();
        let model =
            Model::load(model_bytes).context("failed to load PP-OCRv6 recognition model")?;
        let model_load = model_started.elapsed();
        let profile = PpRecognizerLoadProfile {
            model_load,
            model_identity: model_identity.clone(),
            inference_yaml_identity: inference_yaml_identity.clone(),
            dictionary_sha256: dictionary.source_sha256.clone(),
        };
        Ok((
            Self {
                model,
                dictionary,
                policy,
                model_identity,
                inference_yaml_identity,
            },
            profile,
        ))
    }

    pub fn policy(&self) -> WidthPolicy {
        self.policy
    }

    pub fn model_identity(&self) -> &Identity {
        &self.model_identity
    }

    pub fn inference_yaml_identity(&self) -> &Identity {
        &self.inference_yaml_identity
    }

    pub fn dictionary_sha256(&self) -> &str {
        &self.dictionary.source_sha256
    }

    /// Run one OCRS H64 seam. Batch composition is outside this API.
    pub fn recognize_one(&self, seam: &Seam) -> Result<PpRecognition> {
        let preprocess_started = Instant::now();
        let CandidateBatch {
            tensor,
            upstream_cap_hits,
            ..
        } = preprocess_seams(std::slice::from_ref(seam), self.policy)?;
        let preprocess = preprocess_started.elapsed();
        let mut recognition = self.recognize_packed_one(tensor, upstream_cap_hits == [true])?;
        recognition.profile.preprocess = preprocess;
        Ok(recognition)
    }

    /// Run the locked PP model and decoder on one already-packed candidate
    /// operand. This crate-private seam lets diagnostics time preparation and
    /// packing separately without duplicating inference or CTC decoding.
    pub(crate) fn recognize_packed_one(
        &self,
        tensor: NdTensor<f32, 4>,
        upstream_cap_hit: bool,
    ) -> Result<PpRecognition> {
        let input_shape = tensor.shape();
        if input_shape[0] != 1
            || input_shape[1] != 3
            || input_shape[2] != CANDIDATE_HEIGHT
            || input_shape[3] == 0
            || input_shape[3] > MAX_BATCH_WIDTH
        {
            bail!(
                "expected packed PP operand [1,3,{CANDIDATE_HEIGHT},W] with W in 1..={MAX_BATCH_WIDTH}, got {input_shape:?}"
            );
        }
        if tensor.iter().any(|value| !value.is_finite()) {
            bail!("packed PP operand contains non-finite values");
        }
        let input_pixels = input_shape
            .into_iter()
            .try_fold(1_u64, |product, dimension| {
                product.checked_mul(u64::try_from(dimension).ok()?)
            })
            .context("candidate input shape exceeds u64 pixels")?;

        let forward_started = Instant::now();
        let output = self.model.run_one(tensor.view().into(), None)?;
        let forward = forward_started.elapsed();

        let decode_started = Instant::now();
        let output: NdTensor<f32, 3> = output
            .try_into()
            .context("candidate output must be rank-3 f32")?;
        let shape = output.shape();
        validate_candidate_output_shape(shape, 1)?;
        let values = output.iter().copied().collect::<Vec<_>>();
        let mut texts = decode_candidate(shape, &values, &self.dictionary)?;
        let text = texts.pop().context("candidate omitted batch-one text")?;
        if !texts.is_empty() {
            bail!("candidate returned more than one text for a batch-one seam");
        }
        let decode = decode_started.elapsed();

        Ok(PpRecognition {
            text,
            profile: PpRecognitionProfile {
                preprocess: Duration::ZERO,
                forward,
                decode,
                input_pixels,
                batch_size: input_shape[0],
                batch_width: input_shape[3],
                upstream_cap_hit,
                ctc_steps: shape[1],
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct Seam {
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

impl Seam {
    pub fn validate(&self) -> Result<()> {
        if self.height != OCRS_SEAM_HEIGHT || self.width == 0 {
            bail!(
                "expected non-empty OCRS seam with height {OCRS_SEAM_HEIGHT}, got {}x{}",
                self.height,
                self.width
            );
        }
        if self.values.len() != self.height.saturating_mul(self.width) {
            bail!("seam shape/data length mismatch");
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            bail!("seam contains non-finite values");
        }
        if self
            .values
            .iter()
            .any(|&value| !(-0.5..=0.5).contains(&value))
        {
            bail!("seam value lies outside [-0.5, 0.5]");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CandidateBatch {
    pub tensor: NdTensor<f32, 4>,
    pub resized_widths: Vec<usize>,
    pub upstream_cap_hits: Vec<bool>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedLine {
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

impl PreparedLine {
    fn validate(&self) -> Result<()> {
        if self.height != CANDIDATE_HEIGHT || self.width == 0 {
            bail!(
                "expected non-empty prepared line with height {CANDIDATE_HEIGHT}, got {}x{}",
                self.height,
                self.width
            );
        }
        if self.values.len() != self.height.saturating_mul(self.width) {
            bail!("prepared line shape/data length mismatch");
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            bail!("prepared line contains non-finite values");
        }
        if self
            .values
            .iter()
            .any(|&value| !(-0.5..=0.5).contains(&value))
        {
            bail!("prepared line value lies outside [-0.5, 0.5]");
        }
        Ok(())
    }
}

fn resized_width(seam: &Seam) -> Result<usize> {
    seam.validate()?;
    let numerator = seam
        .width
        .checked_mul(CANDIDATE_HEIGHT)
        .context("candidate resized width overflow")?;
    Ok(numerator.div_ceil(seam.height))
}

pub fn preprocess_seams(seams: &[Seam], policy: WidthPolicy) -> Result<CandidateBatch> {
    if seams.is_empty() || seams.len() > MAX_BATCH_SIZE {
        bail!("candidate batch size must be in 1..={MAX_BATCH_SIZE}");
    }
    let resized_widths = seams
        .iter()
        .map(resized_width)
        .collect::<Result<Vec<_>>>()?;
    let mut prepared = Vec::with_capacity(seams.len());
    for (seam, &out_width) in seams.iter().zip(&resized_widths) {
        let input = NdTensor::from_data([1, 1, seam.height, seam.width], seam.values.clone());
        let resized: NdTensor<f32, 4> = input
            .resize_image([CANDIDATE_HEIGHT, out_width])?
            .try_into()
            .context("resize returned wrong rank")?;
        prepared.push(PreparedLine {
            height: CANDIDATE_HEIGHT,
            width: out_width,
            values: resized.iter().copied().collect(),
        });
    }
    let mut batch = pack_prepared_h48(
        &prepared,
        policy,
        seams
            .iter()
            .map(|seam| seam.width == OCRS_SEAM_MAX_WIDTH)
            .collect(),
    )?;
    batch.resized_widths = resized_widths;
    Ok(batch)
}

pub(crate) fn pack_prepared_h48(
    lines: &[PreparedLine],
    policy: WidthPolicy,
    upstream_cap_hits: Vec<bool>,
) -> Result<CandidateBatch> {
    if lines.is_empty() || lines.len() > MAX_BATCH_SIZE {
        bail!("candidate batch size must be in 1..={MAX_BATCH_SIZE}");
    }
    if upstream_cap_hits.len() != lines.len() {
        bail!("candidate upstream-cap accounting length mismatch");
    }
    for line in lines {
        line.validate()?;
    }
    let resized_widths = lines.iter().map(|line| line.width).collect::<Vec<_>>();
    let content_width = resized_widths.iter().copied().max().unwrap_or_default();
    let batch_width = match policy {
        WidthPolicy::OfficialMin320 => content_width.max(MIN_BATCH_WIDTH),
        WidthPolicy::TightStride8 => content_width.next_multiple_of(8),
    };
    if batch_width > MAX_BATCH_WIDTH {
        bail!(
            "candidate width {batch_width} exceeds audited maximum {MAX_BATCH_WIDTH}; refusing crop, wrap, split, or truncation"
        );
    }

    let mut data = vec![0.0_f32; lines.len() * 3 * CANDIDATE_HEIGHT * batch_width];
    for (batch, line) in lines.iter().enumerate() {
        for channel in 0..3 {
            for y in 0..CANDIDATE_HEIGHT {
                for x in 0..line.width {
                    let destination =
                        (((batch * 3 + channel) * CANDIDATE_HEIGHT + y) * batch_width) + x;
                    // OCRS seam = pixel / 255 - 0.5. The locked Paddle
                    // transform is (pixel / 255 - 0.5) / 0.5.
                    data[destination] = 2.0 * line.values[y * line.width + x];
                }
            }
        }
    }
    Ok(CandidateBatch {
        tensor: NdTensor::from_data([lines.len(), 3, CANDIDATE_HEIGHT, batch_width], data),
        resized_widths,
        upstream_cap_hits,
    })
}

pub fn canonical_f32_sha256(
    shape: &[usize],
    values: impl IntoIterator<Item = f32>,
) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"naky.canonical-f32-tensor.v0\0");
    digest.update(u64::try_from(shape.len())?.to_le_bytes());
    for &dimension in shape {
        digest.update(u64::try_from(dimension)?.to_le_bytes());
    }
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn decode_candidate(
    shape: [usize; 3],
    logits: &[f32],
    dictionary: &Dictionary,
) -> Result<Vec<String>> {
    decode_candidate_with_classes(shape, logits, dictionary, CANDIDATE_CLASSES)
}

/// Summarize only the CTC steps which emit tokens under the ordinary decoder.
fn decode_candidate_with_classes(
    shape: [usize; 3],
    logits: &[f32],
    dictionary: &Dictionary,
    expected_classes: usize,
) -> Result<Vec<String>> {
    let [batch, steps, classes] = shape;
    if batch == 0 || steps == 0 {
        bail!("candidate output batch and step dimensions must be positive");
    }
    if classes != expected_classes {
        bail!("expected candidate output [N,S,{expected_classes}], got {shape:?}");
    }
    let expected = batch
        .checked_mul(steps)
        .and_then(|value| value.checked_mul(classes))
        .context("candidate output shape overflow")?;
    if logits.len() != expected {
        bail!("candidate output shape/data length mismatch");
    }
    if logits.iter().any(|value| !value.is_finite()) {
        bail!("candidate output contains non-finite values");
    }
    if dictionary.labels.len() + 1 != classes {
        bail!("candidate output/dictionary class mismatch");
    }

    let mut output = Vec::with_capacity(batch);
    for batch_index in 0..batch {
        let mut text = String::new();
        let mut previous = None;
        for step in 0..steps {
            let offset = (batch_index * steps + step) * classes;
            let row = &logits[offset..offset + classes];
            let mut best = (0, f32::NEG_INFINITY);
            for (label, &value) in row.iter().enumerate() {
                // Strict comparison retains the first label on ties, matching
                // NumPy argmax in the locked official decoder.
                if value > best.1 {
                    best = (label, value);
                }
            }
            let label = best.0;
            if label != 0 && previous != Some(label) {
                let token = dictionary
                    .labels
                    .get(label - 1)
                    .context("candidate emitted an out-of-range label")?;
                text.push_str(token);
            }
            previous = Some(label);
        }
        output.push(text);
    }
    Ok(output)
}

pub fn validate_candidate_output_shape(shape: [usize; 3], expected_batch: usize) -> Result<()> {
    validate_candidate_output_shape_with_classes(shape, expected_batch, CANDIDATE_CLASSES)
}

fn validate_candidate_output_shape_with_classes(
    shape: [usize; 3],
    expected_batch: usize,
    expected_classes: usize,
) -> Result<()> {
    let [batch, steps, classes] = shape;
    if expected_batch == 0 || batch != expected_batch || steps == 0 || classes != expected_classes {
        bail!(
            "expected candidate output [{expected_batch},S,{expected_classes}] with positive dimensions, got {shape:?}"
        );
    }
    Ok(())
}
