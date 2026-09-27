//! Safe application-facing boundary around the upstream rav1d decoder.

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

mod ivf;
mod matroska;
#[allow(unsafe_code)]
mod rav1d_adapter;

pub use ivf::{IvfError, IvfHeader, IvfPacket, IvfReader};
pub use matroska::{MatroskaError, MatroskaHeader, MatroskaPacket, MatroskaReader};
pub use rav1d_adapter::{DecodeError, DecodedLumaFrame, Decoder};
