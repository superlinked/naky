//! Live full-frame PP detection and ordered PP recognition composition.

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use screenevents::ObservationFrame;

use crate::model_bundle::{ModelBundleIdentity, load_product_bundle};
use crate::pp_detector::{PpDetector, PpDetectorProfile};
use crate::{
    DetectorQuad, PpOrderedParallelRecognizer, PpRecognitionFlushProfile, PpRecognizer,
    PpWidthPolicy, recognize_luma_quads_pp_ordered_parallel_scored,
};

const RECOGNITION_LANES: usize = 4;

pub struct AllPpLoadProfile {
    pub model_bundle: ModelBundleIdentity,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllPpProfile {
    pub detector: PpDetectorProfile,
    pub recognition: PpRecognitionFlushProfile,
    pub recognition_postprocess: Duration,
    pub detector_rows: usize,
    pub observations: usize,
}

pub struct AllPpPage {
    pub frame: ObservationFrame,
    pub profile: AllPpProfile,
}

pub struct AllPpOcr {
    detector: PpDetector,
    recognizer: PpOrderedParallelRecognizer,
}

impl AllPpOcr {
    pub fn load_bundle(model_bundle: &Path) -> Result<(Self, AllPpLoadProfile)> {
        let bundle = load_product_bundle(model_bundle)?;
        let detector = PpDetector::from_bytes(bundle.detector_model)?;
        let (recognizer, _) = PpRecognizer::from_bytes(
            bundle.recognizer_model,
            bundle.inference_yaml,
            PpWidthPolicy::OfficialMin320,
        )?;
        let recognizer = PpOrderedParallelRecognizer::new(
            Arc::new(recognizer),
            NonZeroUsize::new(RECOGNITION_LANES).expect("fixed lane count is nonzero"),
        )?;
        Ok((
            Self {
                detector,
                recognizer,
            },
            AllPpLoadProfile {
                model_bundle: bundle.identity,
            },
        ))
    }

    pub fn observe_luma(
        &mut self,
        timestamp_ms: u64,
        luma: &[u8],
        width: u32,
        height: u32,
    ) -> Result<AllPpPage> {
        let detected = self
            .detector
            .detect_luma(luma, width, height)
            .with_context(|| format!("PP detection failed at {timestamp_ms} ms"))?;
        let quads = detected
            .detections
            .iter()
            .map(|detection| DetectorQuad {
                corners_xy: detection.quad_xy.map(|[x, y]| [x as f32, y as f32]),
            })
            .collect::<Vec<_>>();
        let detector_scores = detected
            .detections
            .iter()
            .map(|detection| detection.score as f32)
            .collect::<Vec<_>>();
        let (recognized, recognition_profile) = recognize_luma_quads_pp_ordered_parallel_scored(
            &mut self.recognizer,
            luma,
            width,
            height,
            &quads,
            &detector_scores,
        )
        .with_context(|| format!("PP recognition failed at {timestamp_ms} ms"))?;
        let elements = recognized.observations;
        let profile = AllPpProfile {
            detector: detected.profile,
            recognition: recognition_profile,
            recognition_postprocess: recognized.postprocess,
            detector_rows: detected.detections.len(),
            observations: elements.len(),
        };
        Ok(AllPpPage {
            frame: ObservationFrame {
                timestamp_ms,
                width,
                height,
                elements,
            },
            profile,
        })
    }
}
