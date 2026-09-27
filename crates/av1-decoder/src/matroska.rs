use std::io::{Read, Seek};

use matroska_demuxer::{DemuxError, Frame, MatroskaFile, TrackType};
use thiserror::Error;

const AV1_CODEC_ID: &str = "V_AV1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatroskaHeader {
    pub width: u32,
    pub height: u32,
    /// Nanoseconds represented by one packet timestamp tick.
    pub timestamp_scale_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatroskaPacket {
    pub timestamp: u64,
    pub data: Vec<u8>,
    /// Whether the Matroska block marks this packet as a keyframe candidate.
    ///
    /// This is `None` for block forms that do not carry the flag. Callers must
    /// not infer random access from a missing flag or use a present flag
    /// without validating fresh-decoder output.
    pub is_keyframe: Option<bool>,
    /// Whether the packet must be decoded without displaying its picture.
    pub is_invisible: bool,
}

#[derive(Debug, Error)]
pub enum MatroskaError {
    #[error("failed to demux Matroska: {0}")]
    Demux(#[from] DemuxError),
    #[error("canonical Matroska must contain exactly one track, found {0}")]
    TrackCount(usize),
    #[error("canonical Matroska track must be video, found {0:?}")]
    TrackType(TrackType),
    #[error("Matroska video codec is {0}, expected V_AV1")]
    Codec(String),
    #[error("encoded or encrypted Matroska tracks are unsupported")]
    ContentEncoding,
    #[error("Matroska AV1 track has unsupported codec delay or seek preroll")]
    CodecDelay,
    #[error("Matroska video track has no video metadata")]
    MissingVideo,
    #[error("Matroska dimensions exceed the supported range: {width}x{height}")]
    Dimensions { width: u64, height: u64 },
}

pub struct MatroskaReader<R: Read + Seek> {
    inner: MatroskaFile<R>,
    video_track: u64,
    header: MatroskaHeader,
    frame: Frame,
}

impl<R: Read + Seek> MatroskaReader<R> {
    pub fn new(inner: R) -> Result<Self, MatroskaError> {
        let inner = MatroskaFile::open(inner)?;
        if inner.tracks().len() != 1 {
            return Err(MatroskaError::TrackCount(inner.tracks().len()));
        }
        let track = &inner.tracks()[0];
        if track.track_type() != TrackType::Video {
            return Err(MatroskaError::TrackType(track.track_type()));
        }
        if track.codec_id() != AV1_CODEC_ID {
            return Err(MatroskaError::Codec(track.codec_id().to_owned()));
        }
        if track
            .content_encodings()
            .is_some_and(|items| !items.is_empty())
        {
            return Err(MatroskaError::ContentEncoding);
        }
        if track.codec_delay().is_some_and(|delay| delay != 0)
            || track.seek_pre_roll().is_some_and(|delay| delay != 0)
        {
            return Err(MatroskaError::CodecDelay);
        }
        let video = track.video().ok_or(MatroskaError::MissingVideo)?;
        let width_u64 = video.pixel_width().get();
        let height_u64 = video.pixel_height().get();
        let width = u32::try_from(width_u64).map_err(|_| MatroskaError::Dimensions {
            width: width_u64,
            height: height_u64,
        })?;
        let height = u32::try_from(height_u64).map_err(|_| MatroskaError::Dimensions {
            width: width_u64,
            height: height_u64,
        })?;

        Ok(Self {
            video_track: track.track_number().get(),
            header: MatroskaHeader {
                width,
                height,
                timestamp_scale_ns: inner.info().timestamp_scale().get(),
            },
            inner,
            frame: Frame::default(),
        })
    }

    pub fn header(&self) -> MatroskaHeader {
        self.header
    }

    pub fn read_packet(&mut self) -> Result<Option<MatroskaPacket>, MatroskaError> {
        while self.inner.next_frame(&mut self.frame)? {
            if self.frame.track != self.video_track {
                continue;
            }
            return Ok(Some(MatroskaPacket {
                timestamp: self.frame.timestamp,
                data: std::mem::take(&mut self.frame.data),
                is_keyframe: self.frame.is_keyframe,
                is_invisible: self.frame.is_invisible,
            }));
        }
        Ok(None)
    }

    /// Positions the demuxer at the first block whose timestamp is at least
    /// `timestamp`.
    ///
    /// This establishes only container position. A caller seeking to an
    /// inter-predicted target must pass a known preceding random-access point,
    /// create a fresh decoder, and decode forward from it.
    pub fn seek(&mut self, timestamp: u64) -> Result<(), MatroskaError> {
        self.inner.seek(timestamp)?;
        self.frame = Frame::default();
        Ok(())
    }
}
