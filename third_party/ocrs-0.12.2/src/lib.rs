use anyhow::anyhow;
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

mod detection;
mod errors;
mod geom_util;
mod layout_analysis;
mod log;
mod model;
mod preprocess;
mod recognition;

#[cfg(test)]
mod test_util;

mod text_items;

use detection::{TextDetector, TextDetectorParams};
use layout_analysis::find_text_lines;
use model::Model;
use preprocess::prepare_image;
use recognition::{prepare_input_with_height, RecognitionOpt, TextRecognizer};

pub use preprocess::{DimOrder, ImagePixels, ImageSource, ImageSourceError};
pub use recognition::{
    DecodeMethod, RecognitionBatchProfile, RecognitionLineId, RecognitionPage,
    RecognitionPagesOutput, RecognitionRunProfile,
};
pub use text_items::{TextChar, TextItem, TextLine, TextWord};

// nb. The "E" before "ABCDE" should be the EUR symbol.
const DEFAULT_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Configuration for an [OcrEngine] instance.
#[derive(Default)]
pub struct OcrEngineParams {
    /// Model used to detect text words in the image.
    pub detection_model: Option<rten::Model>,

    /// Model used to recognize lines of text in the image.
    ///
    /// If using a custom model, you may need to adjust the
    /// [`alphabet`](Self::alphabet) to match.
    pub recognition_model: Option<rten::Model>,

    /// Enable debug logging.
    pub debug: bool,

    /// Method used to decode outputs of text recognition model.
    pub decode_method: DecodeMethod,

    /// Alphabet used for text recognition.
    ///
    /// This is useful if you are using a custom recognition model with a
    /// modified alphabet. If not specified a default alphabet will be used
    /// which matches the one used to train the [original
    /// models](https://github.com/robertknight/ocrs-models).
    pub alphabet: Option<String>,

    /// Set of characters that may be produced by text recognition.
    ///
    /// This is useful when you need the text recognition model to
    /// produce text that only includes a predefined set of characters, for
    /// example only numbers or lower-case letters.
    ///
    /// If this option is not set, text recognition may produce any character
    /// from the recognition model's alphabet.
    pub allowed_chars: Option<String>,
}

/// Internal counterpart to [`OcrEngineParams`] which is generic over the
/// inference engine.
#[derive(Default)]
struct OcrEngineParamsImpl<M: Model> {
    detection_model: Option<M>,
    recognition_model: Option<M>,
    debug: bool,
    decode_method: DecodeMethod,
    alphabet: Option<String>,
    allowed_chars: Option<String>,
}

impl From<OcrEngineParams> for OcrEngineParamsImpl<rten::Model> {
    fn from(params: OcrEngineParams) -> Self {
        let OcrEngineParams {
            detection_model,
            recognition_model,
            debug,
            decode_method,
            alphabet,
            allowed_chars,
        } = params;

        Self {
            detection_model,
            recognition_model,
            debug,
            decode_method,
            alphabet,
            allowed_chars,
        }
    }
}

/// Detects and recognizes text in images.
///
/// OcrEngine uses machine learning models to detect text, analyze layout
/// and recognize text in an image.
pub struct OcrEngine {
    detector: Option<TextDetector>,
    recognizer: Option<TextRecognizer>,
    debug: bool,
    decode_method: DecodeMethod,
    alphabet: String,

    /// Indices of characters in `alphabet` that are excluded from recognition
    /// output. See [`OcrEngineParams::allowed_chars`].
    excluded_char_labels: Option<Vec<usize>>,
}

/// Input image for OCR analysis. Instances are created using
/// [OcrEngine::prepare_input]
pub struct OcrInput {
    /// CHW tensor with normalized pixel values in [BLACK_VALUE, BLACK_VALUE + 1.].
    pub(crate) image: NdTensor<f32, 3>,
}

impl OcrInput {
    /// Return the shape of the normalized CHW input tensor.
    pub fn normalized_shape(&self) -> [usize; 3] {
        self.image.shape()
    }

    /// Iterate over normalized CHW values in canonical tensor order.
    ///
    /// This read-only view exists for diagnostics which bind prepared inputs.
    pub fn normalized_values(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.image.iter().copied()
    }
}

impl OcrEngine {
    /// Construct a new engine from a given configuration.
    pub fn new(params: OcrEngineParams) -> anyhow::Result<OcrEngine> {
        Self::new_impl(params.into())
    }

    /// Internal constructor which allows using a dummy inference engine in tests.
    fn new_impl<M: Model + Send + Sync + 'static>(
        params: OcrEngineParamsImpl<M>,
    ) -> anyhow::Result<OcrEngine> {
        let detector = params
            .detection_model
            .map(|model| TextDetector::from_model(model, Default::default()))
            .transpose()?;
        let recognizer = params
            .recognition_model
            .map(TextRecognizer::from_model)
            .transpose()?;

        let alphabet = params
            .alphabet
            .unwrap_or_else(|| DEFAULT_ALPHABET.to_string());

        let excluded_char_labels = params.allowed_chars.map(|allowed_characters| {
            alphabet
                .chars()
                .enumerate()
                .filter_map(|(index, char)| {
                    if !allowed_characters.contains(char) {
                        // Index `0` is reserved for the CTC blank character and
                        // `i + 1` is used as training label for character at
                        // index `i` of `alphabet` string.
                        //
                        // See https://github.com/robertknight/ocrs-models/blob/3d98fc655d6fd4acddc06e7f5d60a55b55748a48/ocrs_models/datasets/util.py#L113
                        Some(index + 1)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        });

        Ok(OcrEngine {
            detector,
            recognizer,
            alphabet,
            excluded_char_labels,
            debug: params.debug,
            decode_method: params.decode_method,
        })
    }

    /// Preprocess an image for use with other methods of the engine.
    pub fn prepare_input(&self, image: ImageSource) -> anyhow::Result<OcrInput> {
        Ok(OcrInput {
            image: prepare_image(image),
        })
    }

    /// Detect text words in an image.
    ///
    /// Returns an unordered list of the oriented bounding rectangles of each
    /// word found.
    pub fn detect_words(&self, input: &OcrInput) -> anyhow::Result<Vec<RotatedRect>> {
        if let Some(detector) = self.detector.as_ref() {
            detector.detect_words(input.image.view(), self.debug)
        } else {
            Err(anyhow!("Detection model not loaded"))
        }
    }

    /// Detect text pixels in an image.
    ///
    /// Returns an (H, W) tensor indicating the probability of each pixel in the
    /// input being part of a text word. This is a low-level API that is useful
    /// for debugging purposes. Use [detect_words](OcrEngine::detect_words) for
    /// a higher-level API that returns oriented bounding boxes of words.
    pub fn detect_text_pixels(&self, input: &OcrInput) -> anyhow::Result<NdTensor<f32, 2>> {
        if let Some(detector) = self.detector.as_ref() {
            detector.detect_text_pixels(input.image.view(), self.debug)
        } else {
            Err(anyhow!("Detection model not loaded"))
        }
    }

    /// Perform layout analysis to group words into lines and sort them in
    /// reading order.
    ///
    /// `words` is an unordered list of text word rectangles found by
    /// [OcrEngine::detect_words]. The result is a list of lines, in reading
    /// order. Each line is a sequence of word bounding rectangles, in reading
    /// order.
    pub fn find_text_lines(
        &self,
        _input: &OcrInput,
        words: &[RotatedRect],
    ) -> Vec<Vec<RotatedRect>> {
        find_text_lines(words)
    }

    /// Recognize lines of text in an image.
    ///
    /// `lines` is an ordered list of the text line boxes in an image,
    /// produced by [OcrEngine::find_text_lines].
    ///
    /// The output is a list of [TextLine]s corresponding to the input image
    /// regions. Entries can be `None` if no text was found in a given line.
    pub fn recognize_text(
        &self,
        input: &OcrInput,
        lines: &[Vec<RotatedRect>],
    ) -> anyhow::Result<Vec<Option<TextLine>>> {
        if let Some(recognizer) = self.recognizer.as_ref() {
            let pages = [RecognitionPage { input, lines }];
            recognizer
                .recognize_text_pages(
                    &pages,
                    RecognitionOpt {
                        debug: self.debug,
                        decode_method: self.decode_method,
                        alphabet: &self.alphabet,
                        excluded_char_labels: self.excluded_char_labels.as_deref(),
                    },
                    false,
                )
                .map(|output| output.pages.into_iter().next().unwrap_or_default())
        } else {
            Err(anyhow!("Recognition model not loaded"))
        }
    }

    /// Recognize lines from one or two pages in shared width buckets.
    ///
    /// Results retain page-local line order. The profile is collected at the
    /// actual recognition model invocation boundary.
    pub fn recognize_text_pages(
        &self,
        pages: &[RecognitionPage<'_>],
    ) -> anyhow::Result<RecognitionPagesOutput> {
        if let Some(recognizer) = self.recognizer.as_ref() {
            recognizer.recognize_text_pages(
                pages,
                RecognitionOpt {
                    debug: self.debug,
                    decode_method: self.decode_method,
                    alphabet: &self.alphabet,
                    excluded_char_labels: self.excluded_char_labels.as_deref(),
                },
                true,
            )
        } else {
            Err(anyhow!("Recognition model not loaded"))
        }
    }

    /// Recognize lines from one or two pages without retaining call diagnostics.
    pub fn recognize_text_pages_unprofiled(
        &self,
        pages: &[RecognitionPage<'_>],
    ) -> anyhow::Result<Vec<Vec<Option<TextLine>>>> {
        if let Some(recognizer) = self.recognizer.as_ref() {
            recognizer
                .recognize_text_pages(
                    pages,
                    RecognitionOpt {
                        debug: self.debug,
                        decode_method: self.decode_method,
                        alphabet: &self.alphabet,
                        excluded_char_labels: self.excluded_char_labels.as_deref(),
                    },
                    false,
                )
                .map(|output| output.pages)
        } else {
            Err(anyhow!("Recognition model not loaded"))
        }
    }

    /// Prepare an image for input into the text line recognition model.
    ///
    /// This method exists to help with debugging recognition issues by exposing
    /// the preprocessing that [OcrEngine::recognize_text] does before it feeds
    /// an image into the recognition model. Use [OcrEngine::recognize_text] to
    /// recognize text.
    ///
    /// `line` is a sequence of [RotatedRect]s that make up a line of text.
    ///
    /// Returns a greyscale (H, W) image with values in [-0.5, 0.5].
    pub fn prepare_recognition_input(
        &self,
        input: &OcrInput,
        line: &[RotatedRect],
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let Some(recognizer) = self.recognizer.as_ref() else {
            return Err(anyhow!("Recognition model not loaded"));
        };
        let line_image = recognizer.prepare_input(input.image.view(), line);
        Ok(line_image)
    }

    /// Prepare a recognition crop at an explicit height without loading a
    /// recognition model.
    ///
    /// This delegates to the same polygon fill, crop, aspect-ratio width clamp
    /// and interpolation used by [`Self::prepare_recognition_input`].
    pub fn prepare_recognition_input_with_height(
        input: &OcrInput,
        line: &[RotatedRect],
        output_height: usize,
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        prepare_input_with_height(input.image.view(), line, output_height)
    }

    /// Prepare a recognition crop at an explicit `[height, width]` without
    /// invoking the recognition model.
    ///
    /// This narrow debugging API uses the same polygon fill and interpolation
    /// as [`Self::prepare_recognition_input`]. Ordinary OCRS preparation keeps
    /// its existing model-height and width-cap behavior.
    pub fn prepare_recognition_input_with_shape(
        &self,
        input: &OcrInput,
        line: &[RotatedRect],
        output_shape: [usize; 2],
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let Some(recognizer) = self.recognizer.as_ref() else {
            return Err(anyhow!("Recognition model not loaded"));
        };
        recognizer.prepare_input_with_shape(input.image.view(), line, output_shape)
    }

    /// Return the confidence threshold applied to the output of the text
    /// detection model to determine whether a pixel is text or not.
    pub fn detection_threshold(&self) -> f32 {
        self.detector
            .as_ref()
            .map(|detector| detector.threshold())
            .unwrap_or(TextDetectorParams::default().text_threshold)
    }

    /// Convenience API that extracts all text from an image as a single string.
    pub fn get_text(&self, input: &OcrInput) -> anyhow::Result<String> {
        let word_rects = self.detect_words(input)?;
        let line_rects = self.find_text_lines(input, &word_rects);
        let text = self
            .recognize_text(input, &line_rects)?
            .into_iter()
            .filter_map(|line| line.map(|l| l.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::{Arc, Mutex};

    use rten::Dimension;
    use rten_imageproc::{fill_rect, BoundingRect, Rect, RectF, RotatedRect};
    use rten_tensor::prelude::*;
    use rten_tensor::TensorView;
    use rten_tensor::{NdTensor, NdTensorView, Tensor};

    use super::{
        DimOrder, ImageSource, Model, OcrEngine, OcrEngineParamsImpl, RecognitionLineId,
        RecognitionPage, DEFAULT_ALPHABET,
    };

    /// Generate a dummy CHW input image for OCR processing.
    ///
    /// The result is an RGB image which is black except for one line containing
    /// `n_words` white-filled rects.
    fn gen_test_image(n_words: usize) -> NdTensor<f32, 3> {
        let mut image = NdTensor::zeros([3, 100, 200]);

        for word_idx in 0..n_words {
            for chan_idx in 0..3 {
                fill_rect(
                    image.slice_mut([chan_idx]),
                    Rect::from_tlhw(30, (word_idx * 70) as i32, 20, 50),
                    1.,
                );
            }
        }

        image
    }

    /// Fake text detection model.
    ///
    /// Takes a CHW input tensor with values in `[-0.5, 0.5]` and adds a +0.5
    /// bias to produce an output "probability map".
    #[derive(Default)]
    struct FakeDetectionModel {}

    impl Model for FakeDetectionModel {
        fn input_shape(&self) -> anyhow::Result<Vec<Dimension>> {
            Ok([
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                // The real model uses larger inputs (800x600). The fake uses
                // smaller inputs to make tests run faster.
                Dimension::Fixed(200),
                Dimension::Fixed(100),
            ]
            .into())
        }

        fn run(
            &self,
            input: TensorView<f32>,
            _opts: Option<rten::RunOptions>,
        ) -> anyhow::Result<Tensor<f32>> {
            Ok(input.map(|v| v + 0.5))
        }
    }

    /// Fake text recognition model.
    ///
    /// This takes an NCHW input with C=1, H=64 and returns an output with
    /// shape `[W / 4, N, C]`. In the real model the last dimension is the
    /// log-probability of each class label. In this fake we just re-interpret
    /// each column of the input as a vector of probabilities.
    ///
    /// Returns a `(model, alphabet)` tuple.
    #[derive(Default)]
    struct FakeRecognitionModel {
        calls: Option<Arc<Mutex<Vec<Vec<usize>>>>>,
    }

    impl Model for FakeRecognitionModel {
        fn input_shape(&self) -> anyhow::Result<Vec<Dimension>> {
            let output_columns = 64;
            Ok([
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                Dimension::Fixed(output_columns),
                Dimension::Symbolic("seq".to_string()),
            ]
            .into())
        }

        fn run(
            &self,
            input: TensorView<f32>,
            _opts: Option<rten::RunOptions>,
        ) -> anyhow::Result<Tensor<f32>> {
            if let Some(calls) = &self.calls {
                calls.lock().unwrap().push(input.shape().to_vec());
            }
            let nchw: NdTensorView<f32, 4> = input.try_into()?;
            assert_eq!(nchw.size(1), 1);

            let nhw = nchw.slice((.., 0)); // Remove channel axis
            let [batch, height, width] = nhw.shape();
            assert_eq!(height, 64);

            // Width downsampling factor.
            const W_SCALE: usize = 4;

            // Max-pool to reduce width scale.
            let mut output = NdTensor::zeros([batch, height, width / W_SCALE]);
            for n in 0..batch {
                for h in 0..height {
                    for w_block in 0..width / W_SCALE {
                        let mut max = f32::MIN;
                        for i in 0..W_SCALE {
                            max = max.max(nhw[[n, h, w_block * W_SCALE + i]]);
                        }
                        output[[n, h, w_block]] = max;
                    }
                }
            }

            // Transpose NHW/4 => W/4NH
            output.permute([2, 0, 1]);
            output.make_contiguous();

            Ok(output.into())
        }
    }

    fn make_alphabet() -> String {
        let output_columns = 64;
        DEFAULT_ALPHABET.chars().take(output_columns - 1).collect()
    }

    fn recognition_image(width: usize, alphabet_row: usize) -> NdTensor<f32, 3> {
        let mut image = NdTensor::zeros([1, 64, width]);
        image.slice_mut((.., alphabet_row, ..)).fill(1.);
        image
    }

    fn full_line(width: usize) -> Vec<RotatedRect> {
        vec![RotatedRect::from_rect(
            Rect::from_tlhw(0, 0, 64, width as i32).to_f32(),
        )]
    }

    /// Return expected word locations for an image generated by
    /// `gen_test_image(3)`.
    ///
    /// The output boxes are slightly larger than in input image. This is
    /// because the real detection model is trained to predict boxes that are
    /// slightly smaller than the ground truth, in order to create a gap between
    /// adjacent boxes. The connected components in model outputs are then
    /// expanded in post-processing to recover the correct boxes.
    fn expected_word_boxes() -> Vec<RectF> {
        let [top, height] = [27, 25];
        [
            Rect::from_tlhw(top, -3, height, 56).to_f32(),
            Rect::from_tlhw(top, 66, height, 57).to_f32(),
            Rect::from_tlhw(top, 136, height, 57).to_f32(),
        ]
        .into()
    }

    #[test]
    fn test_ocr_engine_prepare_input() -> Result<(), Box<dyn Error>> {
        let image = gen_test_image(3 /* n_words */);
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: Some(FakeDetectionModel {}),
            recognition_model: None,
            ..Default::default()
        })?;
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;

        let [chans, height, width] = input.image.shape();
        assert_eq!(chans, 1);
        assert_eq!(width, image.size(2));
        assert_eq!(height, image.size(1));

        Ok(())
    }

    #[test]
    fn explicit_recognition_shape_preserves_legacy_preparation() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 4000]);
        for y in 0..64 {
            for x in 0..4000 {
                image[[0, y, x]] = f32::from(((y * 17 + x * 29) % 256) as u8) / 255.0;
            }
        }
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            recognition_model: Some(FakeRecognitionModel::default()),
            ..Default::default()
        })?;
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
        let line = full_line(4000);
        let legacy = engine.prepare_recognition_input(&input, &line)?;
        assert_eq!(legacy.shape(), [64, 2400]);
        let model_free_engine = OcrEngine::new_impl(OcrEngineParamsImpl::<FakeRecognitionModel> {
            recognition_model: None,
            ..Default::default()
        })?;
        let model_free_input = model_free_engine
            .prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
        let model_free =
            OcrEngine::prepare_recognition_input_with_height(&model_free_input, &line, 64)?;
        assert_eq!(
            legacy
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            model_free
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let explicit = engine.prepare_recognition_input_with_shape(&input, &line, [64, 2400])?;
        assert_eq!(
            legacy
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            explicit
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        // Simple fixed bit identity catches changes to the legacy crop/fill/
        // interpolation bytes without adding a hashing dependency to OCRS.
        let golden = legacy.iter().fold(0xcbf29ce484222325_u64, |hash, value| {
            value
                .to_bits()
                .to_le_bytes()
                .into_iter()
                .fold(hash, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                })
        });
        assert_eq!(golden, 7_705_393_120_550_646_672);

        let direct = engine.prepare_recognition_input_with_shape(&input, &line, [48, 3000])?;
        assert_eq!(direct.shape(), [48, 3000]);
        assert!(engine
            .prepare_recognition_input_with_shape(&input, &line, [0, 1])
            .is_err());
        assert!(engine
            .prepare_recognition_input_with_shape(&input, &[], [48, 1])
            .is_err());
        assert!(OcrEngine::prepare_recognition_input_with_height(&input, &line, 0).is_err());
        assert!(OcrEngine::prepare_recognition_input_with_height(&input, &[], 64).is_err());
        Ok(())
    }

    #[test]
    fn test_ocr_engine_detect_words() -> Result<(), Box<dyn Error>> {
        let n_words = 3;
        let image = gen_test_image(n_words);
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: Some(FakeDetectionModel {}),
            recognition_model: None,
            ..Default::default()
        })?;
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
        let words = engine.detect_words(&input)?;

        assert_eq!(words.len(), n_words);

        let mut boxes: Vec<RectF> = words
            .into_iter()
            .map(|rotated_rect| rotated_rect.bounding_rect())
            .collect();
        boxes.sort_by_key(|b| [b.top() as i32, b.left() as i32]);

        assert_eq!(boxes, expected_word_boxes());

        Ok(())
    }

    // Test recognition using a dummy recognition model.
    //
    // The dummy model treats each column of the input image as a vector of
    // character class probabilities. Pre-processing of the input will shift
    // values from [0, 1] to [-0.5, 0.5]. CTC decoding of the output will ignore
    // class 0 (as it represents a CTC blank) and repeated characters.
    //
    // Filling a single input row with "1"s will produce a single char output
    // where the char's index in the alphabet is the row index - 1.  ie. Filling
    // the first row produces " ", the second row "0" and so on, using the
    // default alphabet.
    fn test_recognition(
        engine: OcrEngine,
        image: NdTensorView<f32, 3>,
        expected_text: &str,
    ) -> Result<(), Box<dyn Error>> {
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;

        // Create a dummy input line with a single word which fills the image.
        let line_regions: Vec<Vec<RotatedRect>> =
            vec![[
                Rect::from_tlhw(0, 0, image.shape()[1] as i32, image.shape()[2] as i32).to_f32(),
            ]
            .map(RotatedRect::from_rect)
            .into()];

        let lines = engine.recognize_text(&input, &line_regions)?;
        assert_eq!(lines.len(), line_regions.len());

        assert!(!lines.is_empty());
        let line = lines[0].as_ref().unwrap();
        assert_eq!(line.to_string(), expected_text);

        Ok(())
    }

    #[test]
    fn test_ocr_engine_recognize_lines() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Set the probability of character 1 in the alphabet ('0') to 1 and
        // leave all other characters with a probability of zero.
        image.slice_mut((.., 2, ..)).fill(1.);

        let rec_model = FakeRecognitionModel::default();
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(rec_model),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        test_recognition(engine, image.view(), "0")?;

        Ok(())
    }

    #[test]
    fn test_ocr_engine_filter_chars() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Set the probability of "0" to 0.7 and "1" to 0.3.
        image.slice_mut((.., 2, ..)).fill(0.7);
        image.slice_mut((.., 3, ..)).fill(0.3);

        let alphabet = make_alphabet();

        let rec_model = FakeRecognitionModel::default();
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(rec_model),
            alphabet: Some(alphabet.clone()),
            ..Default::default()
        })?;
        test_recognition(engine, image.view(), "0")?;

        // Run recognition again but exclude "0" from the output.
        let rec_model = FakeRecognitionModel::default();
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(rec_model),
            alphabet: Some(alphabet),
            allowed_chars: Some("123456789".into()),
            ..Default::default()
        })?;
        test_recognition(engine, image.view(), "1")?;

        Ok(())
    }

    #[test]
    fn profiled_singleton_matches_compatibility_api() -> Result<(), Box<dyn Error>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image = recognition_image(32, 2);
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
        let lines = vec![full_line(32)];

        let ordinary = engine.recognize_text(&input, &lines)?;
        let profiled = engine.recognize_text_pages(&[RecognitionPage {
            input: &input,
            lines: &lines,
        }])?;

        assert_eq!(ordinary.len(), 1);
        assert_eq!(profiled.pages.len(), 1);
        assert_eq!(
            ordinary[0].as_ref().map(ToString::to_string),
            profiled.pages[0][0].as_ref().map(ToString::to_string)
        );
        assert_eq!(profiled.profile.model_calls, 1);
        assert_eq!(profiled.profile.batches[0].shape, [1, 1, 64, 50]);
        assert_eq!(profiled.profile.input_pixels, 3_200);
        assert_eq!(calls.lock().unwrap().as_slice(), &[[1, 1, 64, 50]; 2]);
        Ok(())
    }

    #[test]
    fn two_pages_pool_and_reconstruct_page_local_results() -> Result<(), Box<dyn Error>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image_a = recognition_image(32, 2);
        let image_b = recognition_image(32, 3);
        let input_a =
            engine.prepare_input(ImageSource::from_tensor(image_a.view(), DimOrder::Chw)?)?;
        let input_b =
            engine.prepare_input(ImageSource::from_tensor(image_b.view(), DimOrder::Chw)?)?;
        let lines_a = vec![full_line(32)];
        let lines_b = vec![full_line(32)];

        let output = engine.recognize_text_pages(&[
            RecognitionPage {
                input: &input_a,
                lines: &lines_a,
            },
            RecognitionPage {
                input: &input_b,
                lines: &lines_b,
            },
        ])?;

        assert_eq!(output.pages.len(), 2);
        assert_eq!(output.pages[0][0].as_ref().unwrap().to_string(), "0");
        assert_eq!(output.pages[1][0].as_ref().unwrap().to_string(), "1");
        assert_eq!(output.profile.model_calls, 1);
        assert_eq!(output.profile.input_pixels, 6_400);
        assert_eq!(output.profile.batches[0].shape, [2, 1, 64, 50]);
        assert_eq!(
            output.profile.batches[0].members,
            vec![
                RecognitionLineId {
                    page_index: 0,
                    line_index: 0,
                },
                RecognitionLineId {
                    page_index: 1,
                    line_index: 0,
                },
            ]
        );
        assert_eq!(calls.lock().unwrap().as_slice(), &[[2, 1, 64, 50]]);
        Ok(())
    }

    #[test]
    fn unprofiled_two_page_api_preserves_results_without_profile_output(
    ) -> Result<(), Box<dyn Error>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image_a = recognition_image(32, 2);
        let image_b = recognition_image(32, 3);
        let input_a =
            engine.prepare_input(ImageSource::from_tensor(image_a.view(), DimOrder::Chw)?)?;
        let input_b =
            engine.prepare_input(ImageSource::from_tensor(image_b.view(), DimOrder::Chw)?)?;
        let lines_a = vec![full_line(32)];
        let lines_b = vec![full_line(32)];

        let pages = engine.recognize_text_pages_unprofiled(&[
            RecognitionPage {
                input: &input_a,
                lines: &lines_a,
            },
            RecognitionPage {
                input: &input_b,
                lines: &lines_b,
            },
        ])?;

        assert_eq!(pages[0][0].as_ref().unwrap().to_string(), "0");
        assert_eq!(pages[1][0].as_ref().unwrap().to_string(), "1");
        assert_eq!(calls.lock().unwrap().as_slice(), &[[2, 1, 64, 50]]);
        Ok(())
    }

    #[test]
    fn pooled_same_width_lines_use_their_differently_sized_page_tensors(
    ) -> Result<(), Box<dyn Error>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image_a = recognition_image(32, 2);
        let image_b = recognition_image(96, 3);
        let input_a =
            engine.prepare_input(ImageSource::from_tensor(image_a.view(), DimOrder::Chw)?)?;
        let input_b =
            engine.prepare_input(ImageSource::from_tensor(image_b.view(), DimOrder::Chw)?)?;
        let lines_a = vec![full_line(32)];
        let lines_b = vec![full_line(32)];

        let output = engine.recognize_text_pages(&[
            RecognitionPage {
                input: &input_a,
                lines: &lines_a,
            },
            RecognitionPage {
                input: &input_b,
                lines: &lines_b,
            },
        ])?;

        assert_eq!(input_a.normalized_shape(), [1, 64, 32]);
        assert_eq!(input_b.normalized_shape(), [1, 64, 96]);
        assert_eq!(output.pages[0][0].as_ref().unwrap().to_string(), "0");
        assert_eq!(output.pages[1][0].as_ref().unwrap().to_string(), "1");
        assert_eq!(output.profile.batches[0].shape, [2, 1, 64, 50]);
        assert_eq!(calls.lock().unwrap().as_slice(), &[[2, 1, 64, 50]]);
        Ok(())
    }

    #[test]
    fn mixed_empty_and_nonempty_pages_preserve_page_local_membership() -> Result<(), Box<dyn Error>>
    {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image_a = recognition_image(32, 2);
        let image_b = recognition_image(32, 3);
        let input_a =
            engine.prepare_input(ImageSource::from_tensor(image_a.view(), DimOrder::Chw)?)?;
        let input_b =
            engine.prepare_input(ImageSource::from_tensor(image_b.view(), DimOrder::Chw)?)?;
        let no_lines = Vec::new();
        let lines_b = vec![full_line(32)];

        let output = engine.recognize_text_pages(&[
            RecognitionPage {
                input: &input_a,
                lines: &no_lines,
            },
            RecognitionPage {
                input: &input_b,
                lines: &lines_b,
            },
        ])?;

        assert!(output.pages[0].is_empty());
        assert_eq!(output.pages[1][0].as_ref().unwrap().to_string(), "1");
        assert_eq!(output.profile.model_calls, 1);
        assert_eq!(
            output.profile.batches[0].members,
            vec![RecognitionLineId {
                page_index: 1,
                line_index: 0,
            }]
        );
        assert_eq!(calls.lock().unwrap().as_slice(), &[[1, 1, 64, 50]]);
        Ok(())
    }

    #[test]
    fn batching_separates_widths_and_splits_twenty_one_lines() -> Result<(), Box<dyn Error>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image_a = recognition_image(32, 2);
        let image_b = recognition_image(64, 3);
        let input_a =
            engine.prepare_input(ImageSource::from_tensor(image_a.view(), DimOrder::Chw)?)?;
        let input_b =
            engine.prepare_input(ImageSource::from_tensor(image_b.view(), DimOrder::Chw)?)?;
        let lines_a = vec![full_line(32); 21];
        let lines_b = vec![full_line(64)];

        let output = engine.recognize_text_pages(&[
            RecognitionPage {
                input: &input_a,
                lines: &lines_a,
            },
            RecognitionPage {
                input: &input_b,
                lines: &lines_b,
            },
        ])?;

        assert_eq!(output.pages[0].len(), 21);
        assert_eq!(output.pages[1].len(), 1);
        assert_eq!(output.profile.model_calls, 3);
        assert_eq!(
            output
                .profile
                .batches
                .iter()
                .map(|batch| batch.shape)
                .collect::<Vec<_>>(),
            vec![[20, 1, 64, 50], [1, 1, 64, 50], [1, 1, 64, 100]]
        );
        assert_eq!(output.profile.batches[1].members[0].line_index, 20);
        assert_eq!(output.profile.batches[2].members[0].page_index, 1);
        assert_eq!(calls.lock().unwrap().len(), 3);
        Ok(())
    }

    #[test]
    fn empty_pages_have_no_calls_and_page_count_is_closed() -> Result<(), Box<dyn Error>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(FakeRecognitionModel {
                calls: Some(calls.clone()),
            }),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        let image = recognition_image(32, 2);
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
        let no_lines = Vec::new();
        let page = RecognitionPage {
            input: &input,
            lines: &no_lines,
        };

        let output = engine.recognize_text_pages(&[page, page])?;
        assert_eq!(output.pages.len(), 2);
        assert!(output.pages.iter().all(Vec::is_empty));
        assert_eq!(output.profile, Default::default());
        assert!(calls.lock().unwrap().is_empty());
        assert!(engine.recognize_text_pages(&[]).is_err());
        assert!(engine.recognize_text_pages(&[page, page, page]).is_err());
        Ok(())
    }
}
