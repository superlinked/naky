use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};
use std::slice;

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};
use thiserror::Error;

const MAX_FRAME_PIXELS: u32 = 7_680 * 4_320;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedLumaFrame {
    pub timestamp: u64,
    pub width: u32,
    pub height: u32,
    /// Contiguous 8-bit luma in row-major order.
    pub luma: Vec<u8>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum DecodeError {
    #[error("rav1d {operation} failed with code {code}")]
    Rav1d { operation: &'static str, code: i32 },
    #[error("compressed AV1 packet must not be empty")]
    EmptyPacket,
    #[error("packet timestamp {0} exceeds rav1d's signed timestamp range")]
    TimestampOutOfRange(u64),
    #[error("decoder produced invalid picture dimensions {width}x{height}")]
    InvalidDimensions { width: i32, height: i32 },
    #[error("decoder produced unsupported {0}-bit luma")]
    UnsupportedBitDepth(i32),
    #[error("decoder produced a picture without a luma plane")]
    MissingLuma,
    #[error("decoder luma stride {stride} is smaller than width {width}")]
    InvalidStride { stride: isize, width: usize },
    #[error("decoder produced an invalid timestamp {0}")]
    InvalidTimestamp(i64),
    #[error("decoder asked for input retry without making a picture available")]
    Stalled,
}

/// Single-threaded deterministic rav1d context.
///
/// All unsafe calls and raw plane access are contained in this module. Returned
/// frames own their luma bytes and never borrow rav1d-managed memory.
pub struct Decoder {
    context: Option<Dav1dContext>,
}

impl Decoder {
    pub fn new() -> Result<Self, DecodeError> {
        Self::open(1, 1)
    }

    fn open(n_threads: i32, max_frame_delay: i32) -> Result<Self, DecodeError> {
        let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
        // SAFETY: settings is valid uninitialized storage which rav1d writes
        // exactly once before it is assumed initialized below.
        // MaybeUninit::as_mut_ptr is non-null by construction.
        unsafe { dav1d_default_settings(NonNull::new_unchecked(settings.as_mut_ptr())) };
        // SAFETY: dav1d_default_settings initialized the complete value.
        let mut settings = unsafe { settings.assume_init() };
        settings.n_threads = n_threads;
        settings.max_frame_delay = max_frame_delay;
        settings.apply_grain = 0;
        settings.frame_size_limit = MAX_FRAME_PIXELS;
        settings.strict_std_compliance = 1;

        let mut context = None;
        // SAFETY: both pointers refer to live writable/readable values for the
        // duration of the call. rav1d initializes context on success.
        let result = unsafe { dav1d_open(NonNull::new(&mut context), NonNull::new(&mut settings)) };
        check("open", result.0)?;
        Ok(Self { context })
    }

    pub fn decode_packet(
        &mut self,
        packet: &[u8],
        timestamp: u64,
    ) -> Result<Vec<DecodedLumaFrame>, DecodeError> {
        if packet.is_empty() {
            return Err(DecodeError::EmptyPacket);
        }
        let timestamp =
            i64::try_from(timestamp).map_err(|_| DecodeError::TimestampOutOfRange(timestamp))?;

        let mut data = Dav1dData::default();
        // SAFETY: data is live writable storage. On success rav1d owns an
        // allocation of exactly packet.len() bytes until send or unref.
        let destination = unsafe { dav1d_data_create(NonNull::new(&mut data), packet.len()) };
        let Some(destination) = NonNull::new(destination) else {
            return Err(DecodeError::Rav1d {
                operation: "allocate packet",
                code: -libc::ENOMEM,
            });
        };
        // SAFETY: source and destination are valid for packet.len() bytes,
        // and the new rav1d allocation cannot overlap the caller's slice.
        unsafe {
            ptr::copy_nonoverlapping(packet.as_ptr(), destination.as_ptr(), packet.len());
        }
        data.m.timestamp = timestamp;

        let mut frames = Vec::new();
        loop {
            // SAFETY: the context came from dav1d_open and remains live;
            // data is a valid rav1d packet value.
            let result = unsafe { dav1d_send_data(self.context, NonNull::new(&mut data)) };
            if result.0 == 0 {
                break;
            }
            if result.0 != -libc::EAGAIN {
                // SAFETY: data has not been consumed on failure.
                unsafe { dav1d_data_unref(NonNull::new(&mut data)) };
                return Err(DecodeError::Rav1d {
                    operation: "send packet",
                    code: result.0,
                });
            }
            let Some(frame) = self.receive_one()? else {
                // SAFETY: data remains owned by this stack value.
                unsafe { dav1d_data_unref(NonNull::new(&mut data)) };
                return Err(DecodeError::Stalled);
            };
            frames.push(frame);
        }

        while let Some(frame) = self.receive_one()? {
            frames.push(frame);
        }
        Ok(frames)
    }

    pub fn drain(&mut self) -> Result<Vec<DecodedLumaFrame>, DecodeError> {
        let mut frames = Vec::new();
        while let Some(frame) = self.receive_one()? {
            frames.push(frame);
        }
        Ok(frames)
    }

    fn receive_one(&mut self) -> Result<Option<DecodedLumaFrame>, DecodeError> {
        let mut picture = Dav1dPicture::default();
        // SAFETY: the context is live and picture is writable. A successful
        // call transfers one picture reference into picture.
        let result = unsafe { dav1d_get_picture(self.context, NonNull::new(&mut picture)) };
        if result.0 == -libc::EAGAIN {
            return Ok(None);
        }
        check("receive picture", result.0)?;

        let frame = copy_luma(&picture);
        // SAFETY: a successful dav1d_get_picture initialized this picture;
        // unref releases its rav1d-managed references exactly once.
        unsafe { dav1d_picture_unref(NonNull::new(&mut picture)) };
        frame.map(Some)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the optional context came from dav1d_open, has not been
        // closed, and this is its sole application owner.
        unsafe { dav1d_close(NonNull::new(&mut self.context)) };
    }
}

fn copy_luma(picture: &Dav1dPicture) -> Result<DecodedLumaFrame, DecodeError> {
    let width = u32::try_from(picture.p.w).map_err(|_| DecodeError::InvalidDimensions {
        width: picture.p.w,
        height: picture.p.h,
    })?;
    let height = u32::try_from(picture.p.h).map_err(|_| DecodeError::InvalidDimensions {
        width: picture.p.w,
        height: picture.p.h,
    })?;
    if width == 0 || height == 0 {
        return Err(DecodeError::InvalidDimensions {
            width: picture.p.w,
            height: picture.p.h,
        });
    }
    if picture.p.bpc != 8 {
        return Err(DecodeError::UnsupportedBitDepth(picture.p.bpc));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let stride = picture.stride[0];
    if stride.unsigned_abs() < width_usize {
        return Err(DecodeError::InvalidStride {
            stride,
            width: width_usize,
        });
    }
    let base = picture.data[0].ok_or(DecodeError::MissingLuma)?;
    let mut luma = Vec::with_capacity(width_usize * height_usize);
    for row in 0..height_usize {
        let offset = stride
            .checked_mul(row as isize)
            .ok_or(DecodeError::InvalidStride {
                stride,
                width: width_usize,
            })?;
        // SAFETY: rav1d guarantees data[0] points at the first displayed
        // luma row and each signed-stride row contains at least width bytes.
        let row = unsafe {
            slice::from_raw_parts(base.as_ptr().cast::<u8>().offset(offset), width_usize)
        };
        luma.extend_from_slice(row);
    }

    let timestamp = u64::try_from(picture.m.timestamp)
        .map_err(|_| DecodeError::InvalidTimestamp(picture.m.timestamp))?;
    Ok(DecodedLumaFrame {
        timestamp,
        width,
        height,
        luma,
    })
}

fn check(operation: &'static str, code: i32) -> Result<(), DecodeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(DecodeError::Rav1d { operation, code })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_packets_before_entering_rav1d() {
        let mut decoder = Decoder::new().unwrap();
        assert_eq!(decoder.decode_packet(&[], 0), Err(DecodeError::EmptyPacket));
    }

    #[test]
    fn rejects_timestamps_outside_rav1d_range() {
        let mut decoder = Decoder::new().unwrap();
        assert_eq!(
            decoder.decode_packet(&[0], u64::MAX),
            Err(DecodeError::TimestampOutOfRange(u64::MAX))
        );
    }
}
