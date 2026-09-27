//! Authenticated live PP-OCRv6 Tiny detector over decoded luma.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ppocrv6_tiny_preflight::{Identity, sha256_bytes};
use rten::{FloatOperators, Model};
use rten_tensor::{NdTensor, prelude::*};

use crate::detector_db::{DbDetection, DbPostprocessor, DbProfile};

const DETECTOR_MODEL_BYTES: u64 = 1_804_288;
const DETECTOR_MODEL_SHA256: &str =
    "1a1460534f3731ab88453cbf681ed2a006629937786b562c92217fcf0adf4a09";
const SCALE_BITS: u32 = 998_277_249;
const MEAN_BITS: [u32; 3] = [1_056_461_292, 1_055_488_213, 1_053_810_491];
const STD_BITS: [u32; 3] = [1_047_166_714, 1_046_831_170, 1_046_898_278];
const MAX_SOURCE_DIMENSION: usize = 16_384;
const MAX_SOURCE_PIXELS: usize = 1 << 28;
const MAX_ASPECT_RATIO: usize = 64;
const MAX_DETECTOR_PIXELS: usize = 1 << 24;

#[derive(Clone, Copy, Debug, Default)]
pub struct PpDetectorProfile {
    pub prepare: Duration,
    pub forward: Duration,
    pub postprocess: Duration,
    pub map_shape_hw: [usize; 2],
    pub db: DbProfile,
}

#[derive(Clone, Debug)]
pub struct PpDetectorPage {
    pub detections: Vec<DbDetection>,
    pub model_identity: Identity,
    pub profile: PpDetectorProfile,
}

pub struct PpDetector {
    model: Model,
    model_identity: Identity,
    postprocessor: DbPostprocessor,
}

impl PpDetector {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read detector model {}", path.display()))?;
        Self::from_bytes(bytes)
    }

    /// Load the exact detector from an owned buffer authenticated by the caller.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let identity = Identity {
            bytes: u64::try_from(bytes.len()).context("detector model size exceeds u64")?,
            sha256: sha256_bytes(&bytes),
        };
        if identity.bytes != DETECTOR_MODEL_BYTES || identity.sha256 != DETECTOR_MODEL_SHA256 {
            bail!(
                "detector model identity differs: got {} bytes {}",
                identity.bytes,
                identity.sha256
            );
        }
        let model = Model::load(bytes).context("failed to load authenticated detector model")?;
        Ok(Self {
            model,
            model_identity: identity,
            postprocessor: DbPostprocessor,
        })
    }

    pub fn model_identity(&self) -> &Identity {
        &self.model_identity
    }

    pub fn detect_luma(&self, luma: &[u8], width: u32, height: u32) -> Result<PpDetectorPage> {
        let source_shape_hw = [
            usize::try_from(height).context("source height exceeds usize")?,
            usize::try_from(width).context("source width exceeds usize")?,
        ];
        validate_source(luma, source_shape_hw)?;
        let map_shape_hw = detector_shape(source_shape_hw)?;

        let prepare_started = Instant::now();
        let input = prepare_luma(luma, source_shape_hw, map_shape_hw)?;
        let prepare = prepare_started.elapsed();

        let forward_started = Instant::now();
        let output = self
            .model
            .run_one(input.view().into(), None)
            .context("RTen detector forward failed")?;
        let forward = forward_started.elapsed();
        let output: NdTensor<f32, 4> = output
            .try_into()
            .context("detector output must be rank-4 f32")?;
        let expected_shape = [1, 1, map_shape_hw[0], map_shape_hw[1]];
        if output.shape() != expected_shape {
            bail!(
                "detector output shape differs: expected {expected_shape:?}, got {:?}",
                output.shape()
            );
        }
        let map = output.iter().copied().collect::<Vec<_>>();
        if map
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            bail!("detector output contains a nonfinite or out-of-[0,1] probability");
        }

        let postprocess_started = Instant::now();
        let processed = self
            .postprocessor
            .process(&map, map_shape_hw, [height, width]);
        let (detections, db) = processed?;
        let postprocess = postprocess_started.elapsed();
        Ok(PpDetectorPage {
            detections,
            model_identity: self.model_identity.clone(),
            profile: PpDetectorProfile {
                prepare,
                forward,
                postprocess,
                map_shape_hw,
                db,
            },
        })
    }
}

fn validate_source(luma: &[u8], [height, width]: [usize; 2]) -> Result<()> {
    if height == 0 || width == 0 {
        bail!("detector source dimensions must be positive");
    }
    if height > MAX_SOURCE_DIMENSION || width > MAX_SOURCE_DIMENSION {
        bail!("detector source dimension exceeds {MAX_SOURCE_DIMENSION}");
    }
    let pixels = height
        .checked_mul(width)
        .context("detector source area overflow")?;
    if pixels > MAX_SOURCE_PIXELS {
        bail!("detector source area exceeds {MAX_SOURCE_PIXELS}");
    }
    if luma.len() != pixels {
        bail!(
            "detector luma length differs: expected {pixels}, got {}",
            luma.len()
        );
    }
    let minimum = height.min(width);
    let maximum = height.max(width);
    if maximum.div_ceil(minimum) > MAX_ASPECT_RATIO {
        bail!("detector source aspect ratio exceeds {MAX_ASPECT_RATIO}:1");
    }
    Ok(())
}

fn detector_shape([source_height, source_width]: [usize; 2]) -> Result<[usize; 2]> {
    let minimum = source_height.min(source_width);
    let ratio = 736.0_f64 / minimum as f64;
    let mut pre_height = (source_height as f64 * ratio).trunc() as usize;
    let mut pre_width = (source_width as f64 * ratio).trunc() as usize;
    let maximum = pre_height.max(pre_width);
    if maximum > 4_000 {
        let cap_ratio = 4_000.0_f64 / maximum as f64;
        pre_height = (pre_height as f64 * cap_ratio).trunc() as usize;
        pre_width = (pre_width as f64 * cap_ratio).trunc() as usize;
    }
    let round_stride = |dimension: usize| -> Result<usize> {
        let units = (dimension as f64 / 32.0).round_ties_even();
        if !units.is_finite() || units < 0.0 || units > usize::MAX as f64 / 32.0 {
            bail!("detector stride rounding overflow");
        }
        Ok((units as usize)
            .checked_mul(32)
            .context("detector stride overflow")?
            .max(32))
    };
    let shape = [round_stride(pre_height)?, round_stride(pre_width)?];
    let pixels = shape[0]
        .checked_mul(shape[1])
        .context("detector operand area overflow")?;
    if pixels > MAX_DETECTOR_PIXELS {
        bail!("detector operand area exceeds {MAX_DETECTOR_PIXELS}");
    }
    Ok(shape)
}

fn prepare_luma(
    luma: &[u8],
    source_shape_hw: [usize; 2],
    detector_shape_hw: [usize; 2],
) -> Result<NdTensor<f32, 4>> {
    let promoted = luma
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let source = NdTensor::from_data([1, 1, source_shape_hw[0], source_shape_hw[1]], promoted);
    let resized: NdTensor<f32, 4> = source
        .resize_image(detector_shape_hw)
        .context("RTen f32 luma resize failed")?
        .try_into()
        .context("detector resize returned wrong rank")?;
    let spatial = detector_shape_hw[0]
        .checked_mul(detector_shape_hw[1])
        .context("detector input area overflow")?;
    if resized.shape() != [1, 1, detector_shape_hw[0], detector_shape_hw[1]]
        || resized.len() != spatial
    {
        bail!("detector resize returned unexpected shape");
    }

    let mut normalized = vec![
        0.0_f32;
        3_usize
            .checked_mul(spatial)
            .context("detector normalized input overflow")?
    ];
    let scale = f32::from_bits(SCALE_BITS);
    for channel in 0..3 {
        let mean = f32::from_bits(MEAN_BITS[channel]);
        let standard_deviation = f32::from_bits(STD_BITS[channel]);
        let offset = channel * spatial;
        for (index, &pixel) in resized.iter().enumerate() {
            normalized[offset + index] = (pixel * scale - mean) / standard_deviation;
        }
    }
    Ok(NdTensor::from_data(
        [1, 3, detector_shape_hw[0], detector_shape_hw[1]],
        normalized,
    ))
}
