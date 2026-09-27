use std::io::{self, Read};

use thiserror::Error;

const HEADER_LEN: usize = 32;
const FRAME_HEADER_LEN: usize = 12;
const MAX_PACKET_LEN: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IvfHeader {
    pub width: u16,
    pub height: u16,
    /// IVF clock ticks per second.
    pub time_base_denominator: u32,
    /// Seconds per IVF clock tick numerator.
    pub time_base_numerator: u32,
    /// Declared frame count; zero means unknown.
    pub frame_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IvfPacket {
    pub timestamp: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum IvfError {
    #[error("failed to read IVF {part}: {source}")]
    Read {
        part: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("input is not an IVF stream (missing DKIF signature)")]
    Signature,
    #[error("unsupported IVF version {0}")]
    Version(u16),
    #[error("unsupported IVF header length {0}")]
    HeaderLength(u16),
    #[error("IVF stream is not AV1 (FourCC is {0:?}, expected AV01)")]
    Codec([u8; 4]),
    #[error("invalid IVF dimensions {width}x{height}")]
    Dimensions { width: u16, height: u16 },
    #[error("invalid IVF time base {numerator}/{denominator}")]
    TimeBase { numerator: u32, denominator: u32 },
    #[error("truncated IVF frame header")]
    TruncatedFrameHeader,
    #[error("invalid IVF packet length {0}")]
    PacketLength(u32),
}

pub struct IvfReader<R> {
    inner: R,
    header: IvfHeader,
}

impl<R: Read> IvfReader<R> {
    pub fn new(mut inner: R) -> Result<Self, IvfError> {
        let mut bytes = [0_u8; HEADER_LEN];
        inner
            .read_exact(&mut bytes)
            .map_err(|source| IvfError::Read {
                part: "header",
                source,
            })?;

        if &bytes[0..4] != b"DKIF" {
            return Err(IvfError::Signature);
        }
        let version = le_u16(&bytes[4..6]);
        if version != 0 {
            return Err(IvfError::Version(version));
        }
        let header_length = le_u16(&bytes[6..8]);
        if usize::from(header_length) != HEADER_LEN {
            return Err(IvfError::HeaderLength(header_length));
        }
        let codec = bytes[8..12].try_into().expect("fixed-size slice");
        if &codec != b"AV01" {
            return Err(IvfError::Codec(codec));
        }

        let width = le_u16(&bytes[12..14]);
        let height = le_u16(&bytes[14..16]);
        if width == 0 || height == 0 {
            return Err(IvfError::Dimensions { width, height });
        }
        let time_base_denominator = le_u32(&bytes[16..20]);
        let time_base_numerator = le_u32(&bytes[20..24]);
        if time_base_denominator == 0 || time_base_numerator == 0 {
            return Err(IvfError::TimeBase {
                numerator: time_base_numerator,
                denominator: time_base_denominator,
            });
        }

        Ok(Self {
            inner,
            header: IvfHeader {
                width,
                height,
                time_base_denominator,
                time_base_numerator,
                frame_count: le_u32(&bytes[24..28]),
            },
        })
    }

    pub fn header(&self) -> IvfHeader {
        self.header
    }

    pub fn read_packet(&mut self) -> Result<Option<IvfPacket>, IvfError> {
        let mut header = [0_u8; FRAME_HEADER_LEN];
        match self.inner.read(&mut header[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => {}
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(source) => {
                return Err(IvfError::Read {
                    part: "frame header",
                    source,
                });
            }
        }
        if let Err(source) = self.inner.read_exact(&mut header[1..]) {
            return if source.kind() == io::ErrorKind::UnexpectedEof {
                Err(IvfError::TruncatedFrameHeader)
            } else {
                Err(IvfError::Read {
                    part: "frame header",
                    source,
                })
            };
        }

        let packet_len_u32 = le_u32(&header[0..4]);
        let packet_len = packet_len_u32 as usize;
        if packet_len == 0 || packet_len > MAX_PACKET_LEN {
            return Err(IvfError::PacketLength(packet_len_u32));
        }
        let mut data = vec![0_u8; packet_len];
        self.inner
            .read_exact(&mut data)
            .map_err(|source| IvfError::Read {
                part: "frame payload",
                source,
            })?;
        Ok(Some(IvfPacket {
            timestamp: le_u64(&header[4..12]),
            data,
        }))
    }
}

fn le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes.try_into().expect("two-byte slice"))
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte slice"))
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight-byte slice"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn header() -> [u8; HEADER_LEN] {
        let mut bytes = [0_u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(b"DKIF");
        bytes[6..8].copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        bytes[8..12].copy_from_slice(b"AV01");
        bytes[12..14].copy_from_slice(&64_u16.to_le_bytes());
        bytes[14..16].copy_from_slice(&32_u16.to_le_bytes());
        bytes[16..20].copy_from_slice(&30_u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&1_u32.to_le_bytes());
        bytes
    }

    #[test]
    fn reads_header_and_packet() {
        let mut bytes = header().to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);

        let mut reader = IvfReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(
            reader.header(),
            IvfHeader {
                width: 64,
                height: 32,
                time_base_denominator: 30,
                time_base_numerator: 1,
                frame_count: 1,
            }
        );
        assert_eq!(
            reader.read_packet().unwrap(),
            Some(IvfPacket {
                timestamp: 7,
                data: vec![1, 2, 3],
            })
        );
        assert_eq!(reader.read_packet().unwrap(), None);
    }

    #[test]
    fn rejects_non_av1_and_malformed_frames() {
        let mut bad_codec = header();
        bad_codec[8..12].copy_from_slice(b"VP90");
        assert!(matches!(
            IvfReader::new(Cursor::new(bad_codec)),
            Err(IvfError::Codec(_))
        ));

        let mut partial = header().to_vec();
        partial.extend_from_slice(&[1, 2]);
        let mut reader = IvfReader::new(Cursor::new(partial)).unwrap();
        assert!(matches!(
            reader.read_packet(),
            Err(IvfError::TruncatedFrameHeader)
        ));
    }
}
