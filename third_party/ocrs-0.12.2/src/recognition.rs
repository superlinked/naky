use core::f32;
use std::collections::HashMap;

use rayon::prelude::*;
use rten::ctc::{CtcDecoder, CtcHypothesis};
use rten::{thread_pool, Dimension, FloatOperators};
use rten_imageproc::{
    bounding_rect, BoundingRect, Line, Point, PointF, Polygon, Rect, RotatedRect,
};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor};

use crate::OcrInput;

use crate::errors::ModelRunError;
use crate::geom_util::{downwards_line, leftmost_edge, rightmost_edge};
use crate::model::Model;
use crate::preprocess::BLACK_VALUE;
use crate::text_items::{TextChar, TextLine};

/// Return a polygon which contains all the rects in `words`.
///
/// `words` is assumed to be a series of disjoint rectangles ordered from left
/// to right. The returned points are arranged in clockwise order starting from
/// the top-left point.
///
/// There are several ways to compute a polygon for a line. The simplest is
/// to use [min_area_rect] on the union of the line's points. However the result
/// will not tightly fit curved lines. This function returns a polygon which
/// closely follows the edges of individual words.
fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut polygon = Vec::new();

    let floor_point = |p: PointF| Point::from_yx(p.y as i32, p.x as i32);

    // Add points from top edges, in left-to-right order.
    for word_rect in words.iter() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(left.start));
        polygon.push(floor_point(right.start));
    }

    // Add points from bottom edges, in right-to-left order.
    for word_rect in words.iter().rev() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(right.end));
        polygon.push(floor_point(left.end));
    }

    polygon
}

/// Compute width to resize a text line image to, for a given height.
fn resized_line_width(orig_width: i32, orig_height: i32, height: i32) -> u32 {
    let min_width = 10.;

    // A larger maximum width avoids horizontally squashing long input lines,
    // affecting accuracy. However it also increases the processing time.
    //
    // The current value was chosen to be large enough to produce good results
    // on screenshots taken from the longest lines in English Wikipedia articles
    // (image size approx 1860x30, 150 characters).
    //
    // The widest image seen during training may be constrained to a shorter
    // value than this, but we rely on the model's ability to generalize to
    // longer sequences.
    let max_width = 2400.;

    let aspect_ratio = orig_width as f32 / orig_height as f32;
    (height as f32 * aspect_ratio).clamp(min_width, max_width) as u32
}

/// Details about a text line needed to prepare the input to the text
/// recognition model.
#[derive(Clone)]
struct TextRecLine {
    /// Page-local identity of this line.
    id: RecognitionLineId,

    /// Region of the image containing this line.
    region: Polygon,

    /// Width to resize this line to.
    resized_width: u32,
}

type ProfiledLineResults =
    Result<Vec<(Vec<LineRecResult>, Option<RecognitionBatchProfile>)>, ModelRunError>;

/// A page and its ordered line regions for recognition.
#[derive(Clone, Copy)]
pub struct RecognitionPage<'a> {
    pub input: &'a OcrInput,
    pub lines: &'a [Vec<RotatedRect>],
}

/// Stable identity of a line within a recognition operation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RecognitionLineId {
    pub page_index: usize,
    pub line_index: usize,
}

/// Actual model input and membership for one recognition call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecognitionBatchProfile {
    pub shape: [usize; 4],
    pub input_pixels: u64,
    pub members: Vec<RecognitionLineId>,
}

/// Counters captured at actual recognition model calls.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RecognitionRunProfile {
    pub model_calls: u64,
    pub input_pixels: u64,
    pub batches: Vec<RecognitionBatchProfile>,
}

/// Page-local recognition outputs and actual call counters.
pub struct RecognitionPagesOutput {
    pub pages: Vec<Vec<Option<TextLine>>>,
    pub profile: RecognitionRunProfile,
}

fn prepare_text_line(
    image: NdTensorView<f32, 3>,
    page_rect: Rect,
    line_region: &Polygon,
    resized_width: u32,
    output_height: usize,
) -> NdTensor<f32, 2> {
    // Page rect adjusted to only contain coordinates that are valid for
    // indexing into the input image.
    let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

    let grey_chan = image.slice([0]);

    let line_rect = line_region.bounding_rect();
    let mut line_img = NdTensor::full(
        [line_rect.height() as usize, line_rect.width() as usize],
        BLACK_VALUE,
    );

    for in_p in line_region.fill_iter() {
        let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());
        if !page_index_rect.contains_point(in_p) || !page_index_rect.contains_point(out_p) {
            continue;
        }
        line_img[[out_p.y as usize, out_p.x as usize]] =
            grey_chan[[in_p.y as usize, in_p.x as usize]];
    }

    let resized_line_img = line_img
        .reshaped([1, 1, line_img.size(0), line_img.size(1)])
        .resize_image([output_height, resized_width as usize])
        .unwrap();

    let out_shape = [resized_line_img.size(2), resized_line_img.size(3)];
    resized_line_img.into_shape(out_shape)
}

/// Prepare a line using the same crop and aspect-ratio policy as recognition,
/// without requiring a recognition model.
pub(crate) fn prepare_input_with_height(
    image: NdTensorView<f32, 3>,
    line: &[RotatedRect],
    output_height: usize,
) -> anyhow::Result<NdTensor<f32, 2>> {
    if output_height == 0 {
        return Err(anyhow::anyhow!(
            "recognition preparation output height must be positive"
        ));
    }
    let [_, img_height, img_width] = image.shape();
    let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
    let line_rect = bounding_rect(line.iter())
        .ok_or_else(|| anyhow::anyhow!("line has no words"))?
        .integral_bounding_rect();
    if line_rect.width() <= 0 || line_rect.height() <= 0 {
        return Err(anyhow::anyhow!("line has empty integral geometry"));
    }
    let output_height_i32 = i32::try_from(output_height)
        .map_err(|_| anyhow::anyhow!("recognition preparation height exceeds i32"))?;
    let resized_width =
        resized_line_width(line_rect.width(), line_rect.height(), output_height_i32);
    let line_poly = Polygon::new(line_polygon(line));
    Ok(prepare_text_line(
        image,
        page_rect,
        &line_poly,
        resized_width,
        output_height,
    ))
}

/// Prepare an NCHW tensor containing a batch of text line images, for input
/// into the text recognition model.
///
/// For each line in `lines`, the line region is extracted from `image`, resized
/// to a fixed `output_height` and a line-specific width, then copied to the
/// output tensor. Lines in the batch can have different widths, so the output
/// is padded on the right side to a common width of `output_width`.
fn prepare_text_line_batch(
    pages: &[RecognitionPage<'_>],
    lines: &[TextRecLine],
    output_height: usize,
    output_width: usize,
) -> NdTensor<f32, 4> {
    let mut output = NdTensor::full([lines.len(), 1, output_height, output_width], BLACK_VALUE);

    for (group_line_index, line) in lines.iter().enumerate() {
        let image = pages[line.id.page_index].input.image.view();
        let [_, image_height, image_width] = image.shape();
        let page_rect = Rect::from_hw(image_height as i32, image_width as i32);
        let resized_line_img = prepare_text_line(
            image,
            page_rect,
            &line.region,
            line.resized_width,
            output_height,
        );
        output
            .slice_mut((group_line_index, 0, .., ..(line.resized_width as usize)))
            .copy_from(&resized_line_img);
    }

    output
}

/// Return the bounding rectangle of the slice of a polygon with X coordinates
/// between `min_x` and `max_x` inclusive.
fn polygon_slice_bounding_rect(
    poly: Polygon<i32, &[Point]>,
    min_x: i32,
    max_x: i32,
) -> Option<Rect> {
    poly.edges()
        .filter_map(|e| {
            let e = e.rightwards();

            // Filter out edges that don't overlap [min_x, max_x].
            if (e.start.x < min_x && e.end.x < min_x) || (e.start.x > max_x && e.end.x > max_x) {
                return None;
            }

            // Truncate edge to [min_x, max_x].
            let trunc_edge_start = e
                .to_f32()
                .y_for_x(min_x as f32)
                .map_or(e.start, |y| Point::from_yx(y.round() as i32, min_x));

            let trunc_edge_end = e
                .to_f32()
                .y_for_x(max_x as f32)
                .map_or(e.end, |y| Point::from_yx(y.round() as i32, max_x));

            Some(Line::from_endpoints(trunc_edge_start, trunc_edge_end))
        })
        .fold(None, |bounding_rect, e| {
            let edge_br = e.bounding_rect();
            bounding_rect.map(|br| br.union(edge_br)).or(Some(edge_br))
        })
}

/// Method used to decode sequence model outputs to a sequence of labels.
///
/// See [CtcDecoder] for more details.
#[derive(Copy, Clone, Default)]
pub enum DecodeMethod {
    #[default]
    Greedy,
    BeamSearch {
        width: u32,
    },
}

#[derive(Clone, Default)]
pub struct RecognitionOpt<'a> {
    pub debug: bool,

    /// Method used to decode character sequence outputs to character values.
    pub decode_method: DecodeMethod,

    pub alphabet: &'a str,

    pub excluded_char_labels: Option<&'a [usize]>,
}

/// Input and output from recognition for a single text line.
struct LineRecResult {
    /// Input to the recognition model.
    line: TextRecLine,

    /// Length of input sequences to recognition model, padded so that all
    /// lines in batch have the same length.
    rec_input_len: usize,

    /// Length of output sequences from recognition model, used as input to
    /// CTC decoding.
    ctc_input_len: usize,

    /// Output label sequence produced by CTC decoding.
    ctc_output: CtcHypothesis,
}

/// Combine information from the input and output of text line recognition
/// to produce [TextLine]s containing character sequences and bounding boxes
/// for each line.
///
/// Entries in the result may be `None` if no text was recognized for a line.
fn text_lines_from_recognition_results(
    results: &[LineRecResult],
    alphabet: &str,
) -> Vec<Option<TextLine>> {
    results
        .iter()
        .map(|result| {
            let line_rect = result.line.region.bounding_rect();
            let x_scale_factor = (line_rect.width() as f32) / (result.line.resized_width as f32);
            // Calculate how much the recognition model downscales the image
            // width. We assume this will be an integer factor, or close to it
            // if the input width is not an exact multiple of the downscaling
            // factor.
            let downsample_factor =
                (result.rec_input_len as f32 / result.ctc_input_len as f32).round() as u32;

            let steps = result.ctc_output.steps();
            let text_line: Vec<TextChar> = steps
                .iter()
                .enumerate()
                .filter_map(|(i, step)| {
                    // X coord range of character in line recognition input image.
                    let start_x = step.pos * downsample_factor;
                    let end_x = if let Some(next_step) = steps.get(i + 1) {
                        next_step.pos * downsample_factor
                    } else {
                        result.line.resized_width
                    };

                    // Map X coords to those of the input image.
                    let [start_x, end_x] = [start_x, end_x]
                        .map(|x| line_rect.left() + (x as f32 * x_scale_factor) as i32);

                    // Since the recognition input is padded, it is possible to
                    // get predicted characters in the output with positions
                    // that correspond to the padding region, and thus are
                    // outside the bounds of the original line. Ignore these.
                    if start_x >= line_rect.right() {
                        return None;
                    }

                    let char = alphabet
                        .chars()
                        // Index `0` is reserved for blank character and `i + 1` is used as training
                        // label for character at index `i` of `alphabet` string.  Here we're
                        // subtracting 1 to get the actual index from the output label
                        //
                        // See https://github.com/robertknight/ocrs-models/blob/3d98fc655d6fd4acddc06e7f5d60a55b55748a48/ocrs_models/datasets/util.py#L113
                        .nth((step.label - 1) as usize)
                        .unwrap_or('?');

                    Some(TextChar {
                        char,
                        rect: polygon_slice_bounding_rect(
                            result.line.region.borrow(),
                            start_x,
                            end_x,
                        )
                        .expect("invalid X coords"),
                    })
                })
                .collect();

            if text_line.is_empty() {
                None
            } else {
                Some(TextLine::new(text_line))
            }
        })
        .collect()
}

/// Extracts character sequences and coordinates from text lines detected in
/// an image.
pub struct TextRecognizer {
    model: Box<dyn Model + Send + Sync>,
    input_shape: Vec<Dimension>,
}

impl TextRecognizer {
    /// Initialize a text recognizer from a trained RTen model. Fails if the
    /// model does not have the expected inputs or outputs.
    pub fn from_model(model: impl Model + Send + Sync + 'static) -> anyhow::Result<TextRecognizer> {
        let input_shape = model.input_shape()?;
        Ok(TextRecognizer {
            model: Box::new(model),
            input_shape,
        })
    }

    /// Return the expected height of input line images.
    fn input_height(&self) -> u32 {
        match self.input_shape[2] {
            Dimension::Fixed(size) => size.try_into().unwrap(),
            Dimension::Symbolic(_) => 50,
        }
    }

    /// Run text recognition on an NCHW batch of text line images, and return
    /// a `[batch, seq, label]` tensor of class probabilities.
    fn run(&self, input: NdTensor<f32, 4>) -> Result<NdTensor<f32, 3>, ModelRunError> {
        let input: Tensor<f32> = input.into();
        let output = self
            .model
            .run(input.view(), None)
            .map_err(|err| ModelRunError::RunFailed(err.into()))?;

        let output_ndim = output.ndim();
        let mut rec_sequence: NdTensor<f32, 3> = output.try_into().map_err(|_| {
            ModelRunError::WrongOutput(format!(
                "expected recognition output to have 3 dims but it has {}",
                output_ndim
            ))
        })?;

        // Transpose from [seq, batch, class] => [batch, seq, class]
        rec_sequence.permute([1, 0, 2]);

        Ok(rec_sequence)
    }

    /// Prepare a text line for input into the recognition model.
    ///
    /// This method exists for model debugging purposes to expose the
    /// preprocessing that [TextRecognizer::recognize_text_lines] does.
    pub fn prepare_input(
        &self,
        image: NdTensorView<f32, 3>,
        line: &[RotatedRect],
    ) -> NdTensor<f32, 2> {
        let rec_img_height = self.input_height();
        prepare_input_with_height(image, line, rec_img_height as usize)
            .expect("recognition line preparation failed")
    }

    /// Prepare a text line at an explicit output shape.
    ///
    /// This is an experiment/debugging seam. It deliberately shares the crop,
    /// polygon-fill and resize implementation used by [`Self::prepare_input`]
    /// without invoking the recognition model.
    pub fn prepare_input_with_shape(
        &self,
        image: NdTensorView<f32, 3>,
        line: &[RotatedRect],
        output_shape: [usize; 2],
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let [output_height, output_width] = output_shape;
        if output_height == 0 || output_width == 0 {
            return Err(anyhow::anyhow!(
                "recognition preparation output dimensions must be positive"
            ));
        }
        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
        let line_rect = bounding_rect(line.iter())
            .ok_or_else(|| anyhow::anyhow!("line has no words"))?
            .integral_bounding_rect();
        if line_rect.width() <= 0 || line_rect.height() <= 0 {
            return Err(anyhow::anyhow!("line has empty integral geometry"));
        }
        let resized_width = u32::try_from(output_width)
            .map_err(|_| anyhow::anyhow!("requested recognition width exceeds u32"))?;
        let line_poly = Polygon::new(line_polygon(line));
        Ok(prepare_text_line(
            image,
            page_rect,
            &line_poly,
            resized_width,
            output_height,
        ))
    }

    pub fn recognize_text_pages(
        &self,
        pages: &[RecognitionPage<'_>],
        opts: RecognitionOpt,
        collect_profile: bool,
    ) -> anyhow::Result<RecognitionPagesOutput> {
        if !(1..=2).contains(&pages.len()) {
            return Err(anyhow::anyhow!(
                "recognition requires exactly one or two pages"
            ));
        }
        let RecognitionOpt {
            debug,
            decode_method,
            alphabet,
            excluded_char_labels,
        } = opts;

        // Group lines into batches which will have similar widths after resizing
        // to a fixed height.
        //
        // It is more efficient to run recognition on multiple lines at once, but
        // all line images in a batch must be padded to an equal length. Some
        // computation is wasted on shorter lines in the batch. Choosing batches
        // such that all line images have a similar width reduces this wastage.
        // There is a trade-off between maximizing the batch size and minimizing
        // the variance in width of images in the batch.
        let rec_img_height = self.input_height();
        let mut line_groups: HashMap<i32, Vec<TextRecLine>> = HashMap::new();
        for (page_index, page) in pages.iter().enumerate() {
            for (line_index, word_rects) in page.lines.iter().enumerate() {
                let line_rect = bounding_rect(word_rects.iter())
                    .ok_or_else(|| anyhow::anyhow!("line has no words"))?
                    .integral_bounding_rect();
                let resized_width = resized_line_width(
                    line_rect.width(),
                    line_rect.height(),
                    rec_img_height as i32,
                );
                let group_width = resized_width.next_multiple_of(50);
                line_groups
                    .entry(group_width as i32)
                    .or_default()
                    .push(TextRecLine {
                        id: RecognitionLineId {
                            page_index,
                            line_index,
                        },
                        region: Polygon::new(line_polygon(word_rects)),
                        resized_width,
                    });
            }
        }

        // Split large line groups up into smaller batches that can be processed
        // in parallel.
        let max_lines_per_group = 20;
        let mut line_groups: Vec<(i32, Vec<TextRecLine>)> = line_groups
            .into_iter()
            .flat_map(|(group_width, lines)| {
                lines
                    .chunks(max_lines_per_group)
                    .map(|chunk| (group_width, chunk.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect();
        line_groups
            .sort_by_key(|(group_width, lines)| (*group_width, lines.first().map(|line| line.id)));

        let alphabet_len = alphabet.chars().count();

        // Run text recognition on batches of lines.
        let batch_rec_results: ProfiledLineResults = thread_pool().run(|| {
            line_groups
                .into_par_iter()
                .map(|(group_width, lines)| {
                    if debug {
                        println!(
                            "Processing group of {} lines of width {}",
                            lines.len(),
                            group_width,
                        );
                    }

                    let rec_input = prepare_text_line_batch(
                        pages,
                        &lines,
                        rec_img_height as usize,
                        group_width as usize,
                    );

                    let profile = if collect_profile {
                        let shape = rec_input.shape();
                        let input_pixels = shape
                            .into_iter()
                            .try_fold(1_u64, |product, dim| {
                                u64::try_from(dim)
                                    .ok()
                                    .and_then(|dim| product.checked_mul(dim))
                            })
                            .ok_or_else(|| {
                                ModelRunError::WrongOutput(
                                    "recognition input pixel count overflow".to_string(),
                                )
                            })?;
                        Some(RecognitionBatchProfile {
                            shape,
                            input_pixels,
                            members: lines.iter().map(|line| line.id).collect(),
                        })
                    } else {
                        None
                    };

                    let mut rec_output = self.run(rec_input)?;

                    if alphabet_len + 1 != rec_output.size(2) {
                        return Err(ModelRunError::WrongOutput(format!(
                            "output column count ({}) does not match alphabet size ({})",
                            rec_output.size(2),
                            alphabet_len + 1
                        )));
                    }

                    let ctc_input_len = rec_output.shape()[1];

                    // Apply CTC decoding to get the label sequence for each line.
                    let line_rec_results = lines
                        .into_iter()
                        .enumerate()
                        .map(|(group_line_index, line)| {
                            let decoder = CtcDecoder::new();

                            let mut input_seq_slice = rec_output.slice_mut([group_line_index]);
                            let input_seq = Self::filter_excluded_char_labels(
                                excluded_char_labels,
                                &mut input_seq_slice,
                            );

                            let ctc_output = match decode_method {
                                DecodeMethod::Greedy => decoder.decode_greedy(input_seq),
                                DecodeMethod::BeamSearch { width } => {
                                    decoder.decode_beam(input_seq, width)
                                }
                            };
                            LineRecResult {
                                line,
                                rec_input_len: group_width as usize,
                                ctc_input_len,
                                ctc_output,
                            }
                        })
                        .collect::<Vec<_>>();

                    Ok((line_rec_results, profile))
                })
                .collect()
        });

        let batch_rec_results = batch_rec_results?;
        let mut profile = RecognitionRunProfile::default();
        let mut line_rec_results = Vec::new();
        for (results, batch_profile) in batch_rec_results {
            line_rec_results.extend(results);
            if let Some(batch_profile) = batch_profile {
                profile.model_calls = profile
                    .model_calls
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("recognition call count overflow"))?;
                profile.input_pixels = profile
                    .input_pixels
                    .checked_add(batch_profile.input_pixels)
                    .ok_or_else(|| anyhow::anyhow!("recognition input pixel count overflow"))?;
                profile.batches.push(batch_profile);
            }
        }

        // The recognition outputs are in a different order than the inputs due to
        // batching and parallel processing. Re-sort them into input order.
        line_rec_results.sort_by_key(|result| result.line.id);
        profile
            .batches
            .sort_by_key(|batch| (batch.shape[3], batch.members.first().copied()));

        let mut page_results = pages
            .iter()
            .map(|page| {
                std::iter::repeat_with(|| None)
                    .take(page.lines.len())
                    .collect::<Vec<Option<TextLine>>>()
            })
            .collect::<Vec<_>>();
        let text_lines = text_lines_from_recognition_results(&line_rec_results, alphabet);
        for (result, text_line) in line_rec_results.into_iter().zip(text_lines) {
            page_results[result.line.id.page_index][result.line.id.line_index] = text_line;
        }

        Ok(RecognitionPagesOutput {
            pages: page_results,
            profile,
        })
    }

    /// Post-process recognition model outputs to filter excluded characters.
    ///
    /// `input_seq_slice` is a (seq, char_prob) matrix of log probabilities for
    /// characters. `excluded_char_labels` specifies indices of characters that
    /// should be excluded, by setting the log probability to -Inf.
    fn filter_excluded_char_labels<'a>(
        excluded_char_labels: Option<&[usize]>,
        input_seq_slice: &'a mut NdTensorViewMut<'_, f32, 2>,
    ) -> NdTensorView<'a, f32, 2> {
        if let Some(excluded_char_labels) = excluded_char_labels {
            for row in 0..input_seq_slice.size(0) {
                for &excluded_char_label in excluded_char_labels.iter() {
                    // Setting the output value of excluded char to -Inf causes the
                    // `decode_method` to favour chars other than the excluded char.
                    (*input_seq_slice)[[row, excluded_char_label]] = f32::NEG_INFINITY;
                }
            }
        }
        input_seq_slice.view()
    }
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{BoundingRect, Point, Polygon, RotatedRect, Vec2};

    use super::line_polygon;

    #[test]
    fn test_line_polygon() {
        let words: Vec<RotatedRect> = (0..5)
            .map(|i| {
                let center = Point::from_yx(10., i as f32 * 20.);
                let width = 10.;
                let height = 5.;

                // Vary the orientation of words. The output of `line_polygon`
                // should be invariant to different orientations of a RotatedRect
                // that cover the same pixels.
                let up = Vec2::from_yx(if i % 2 == 0 { -1. } else { 1. }, 0.);
                RotatedRect::new(center, up, width, height)
            })
            .collect();
        let poly = Polygon::new(line_polygon(&words));

        assert!(poly.is_simple());
        for word in words {
            let center = word.bounding_rect().center();
            assert!(poly.contains_pixel(Point::from_yx(
                center.y.round() as i32,
                center.x.round() as i32
            )));
        }
    }
}
