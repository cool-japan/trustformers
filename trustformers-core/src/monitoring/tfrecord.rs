//! TFRecord framing and protobuf wire encoding for TensorBoard event files.
//!
//! TensorBoard reads `events.out.tfevents.*` files as a stream of TFRecords.
//! Each record is framed as:
//!
//! ```text
//! uint64  length                       (little endian)
//! uint32  masked_crc32c(length bytes)  (little endian)
//! bytes   data[length]
//! uint32  masked_crc32c(data)          (little endian)
//! ```
//!
//! where the checksum is CRC-32C (Castagnoli, reflected polynomial
//! `0x82F63B78`) put through TensorFlow's mask:
//!
//! ```text
//! masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8
//! ```
//!
//! The payload is a serialised `tensorflow.Event` protobuf message. This module
//! hand-rolls just enough of the protobuf wire format (varints, fixed32,
//! fixed64 and length-delimited fields) to emit `Event`, `Summary`,
//! `Summary.Value` and `HistogramProto` — no code generator and no C/C++
//! dependency.
//!
//! Relevant field numbers (from `tensorflow/core/util/event.proto` and
//! `tensorflow/core/framework/summary.proto`):
//!
//! | message           | field           | number | wire type        |
//! |-------------------|-----------------|--------|------------------|
//! | `Event`           | `wall_time`     | 1      | fixed64 (double) |
//! | `Event`           | `step`          | 2      | varint (int64)   |
//! | `Event`           | `file_version`  | 3      | length-delimited |
//! | `Event`           | `summary`       | 5      | length-delimited |
//! | `Summary`         | `value`         | 1      | length-delimited |
//! | `Summary.Value`   | `tag`           | 1      | length-delimited |
//! | `Summary.Value`   | `simple_value`  | 2      | fixed32 (float)  |
//! | `Summary.Value`   | `histo`         | 5      | length-delimited |
//! | `HistogramProto`  | `min`           | 1      | fixed64 (double) |
//! | `HistogramProto`  | `max`           | 2      | fixed64 (double) |
//! | `HistogramProto`  | `num`           | 3      | fixed64 (double) |
//! | `HistogramProto`  | `sum`           | 4      | fixed64 (double) |
//! | `HistogramProto`  | `sum_squares`   | 5      | fixed64 (double) |
//! | `HistogramProto`  | `bucket_limit`  | 6      | packed doubles   |
//! | `HistogramProto`  | `bucket`        | 7      | packed doubles   |

use std::io::{self, Read, Write};

// ── Protobuf wire types ──────────────────────────────────────────────────────

/// Varint-encoded field (`int32`, `int64`, `bool`, enums).
pub const WIRE_TYPE_VARINT: u32 = 0;
/// 64-bit fixed field (`double`, `fixed64`).
pub const WIRE_TYPE_FIXED64: u32 = 1;
/// Length-delimited field (`string`, `bytes`, embedded messages, packed repeats).
pub const WIRE_TYPE_LENGTH_DELIMITED: u32 = 2;
/// 32-bit fixed field (`float`, `fixed32`).
pub const WIRE_TYPE_FIXED32: u32 = 5;

// ── CRC-32C (Castagnoli) ─────────────────────────────────────────────────────

/// Reflected CRC-32C polynomial used by TFRecord framing.
const CRC32C_POLYNOMIAL: u32 = 0x82f6_3b78;

/// TensorFlow's CRC mask rotation constant.
const CRC_MASK_DELTA: u32 = 0xa282_ead8;

const fn build_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut crc = index as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ CRC32C_POLYNOMIAL } else { crc >> 1 };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
}

static CRC32C_TABLE: [u32; 256] = build_crc32c_table();

/// CRC-32C (Castagnoli) checksum.
///
/// Verified against the standard check value: `crc32c(b"123456789")` is
/// `0xE3069283`.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xff) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[index];
    }
    !crc
}

/// TensorFlow's masked CRC-32C, as required by the TFRecord framing.
pub fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    crc.rotate_right(15).wrapping_add(CRC_MASK_DELTA)
}

// ── TFRecord framing ─────────────────────────────────────────────────────────

/// Write one TFRecord: `[len][masked_crc(len)][data][masked_crc(data)]`.
pub fn write_record<W: Write>(writer: &mut W, payload: &[u8]) -> io::Result<()> {
    let length = payload.len() as u64;
    let length_bytes = length.to_le_bytes();

    writer.write_all(&length_bytes)?;
    writer.write_all(&masked_crc32c(&length_bytes).to_le_bytes())?;
    writer.write_all(payload)?;
    writer.write_all(&masked_crc32c(payload).to_le_bytes())?;
    Ok(())
}

/// Read one TFRecord, verifying both masked CRC-32C checksums.
///
/// Returns `Ok(None)` at a clean end of stream.
pub fn read_record<R: Read>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut length_bytes = [0u8; 8];
    match reader.read_exact(&mut length_bytes) {
        Ok(()) => {},
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }

    let mut length_crc = [0u8; 4];
    reader.read_exact(&mut length_crc)?;
    if u32::from_le_bytes(length_crc) != masked_crc32c(&length_bytes) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TFRecord length checksum mismatch",
        ));
    }

    let length = u64::from_le_bytes(length_bytes);
    let length = usize::try_from(length)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "TFRecord length exceeds usize"))?;

    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload)?;

    let mut payload_crc = [0u8; 4];
    reader.read_exact(&mut payload_crc)?;
    if u32::from_le_bytes(payload_crc) != masked_crc32c(&payload) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TFRecord payload checksum mismatch",
        ));
    }

    Ok(Some(payload))
}

// ── Protobuf wire encoding primitives ────────────────────────────────────────

/// Append a base-128 varint.
pub fn encode_varint(buffer: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buffer.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Append a field tag (`field_number << 3 | wire_type`).
pub fn encode_tag(buffer: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    encode_varint(buffer, ((field_number as u64) << 3) | wire_type as u64);
}

/// Append a `double` field (fixed64).
pub fn encode_double(buffer: &mut Vec<u8>, field_number: u32, value: f64) {
    encode_tag(buffer, field_number, WIRE_TYPE_FIXED64);
    buffer.extend_from_slice(&value.to_le_bytes());
}

/// Append a `float` field (fixed32).
pub fn encode_float(buffer: &mut Vec<u8>, field_number: u32, value: f32) {
    encode_tag(buffer, field_number, WIRE_TYPE_FIXED32);
    buffer.extend_from_slice(&value.to_le_bytes());
}

/// Append an `int64` field (varint, two's-complement for negatives).
pub fn encode_int64(buffer: &mut Vec<u8>, field_number: u32, value: i64) {
    encode_tag(buffer, field_number, WIRE_TYPE_VARINT);
    encode_varint(buffer, value as u64);
}

/// Append a length-delimited field (`string`, `bytes` or an embedded message).
pub fn encode_bytes(buffer: &mut Vec<u8>, field_number: u32, value: &[u8]) {
    encode_tag(buffer, field_number, WIRE_TYPE_LENGTH_DELIMITED);
    encode_varint(buffer, value.len() as u64);
    buffer.extend_from_slice(value);
}

/// Append a packed `repeated double` field.
pub fn encode_packed_doubles(buffer: &mut Vec<u8>, field_number: u32, values: &[f64]) {
    if values.is_empty() {
        return;
    }
    encode_tag(buffer, field_number, WIRE_TYPE_LENGTH_DELIMITED);
    encode_varint(buffer, (values.len() * 8) as u64);
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
}

// ── TensorBoard message builders ─────────────────────────────────────────────

/// A `tensorflow.HistogramProto` payload.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramProto {
    /// Smallest observed value.
    pub min: f64,
    /// Largest observed value.
    pub max: f64,
    /// Number of observations (a `double` in the proto, not an integer).
    pub num: f64,
    /// Sum of the observations.
    pub sum: f64,
    /// Sum of the squared observations.
    pub sum_squares: f64,
    /// Upper edge (inclusive) of each bucket.
    pub bucket_limit: Vec<f64>,
    /// Count in each bucket; must be the same length as `bucket_limit`.
    pub bucket: Vec<f64>,
}

impl HistogramProto {
    /// Serialise to the protobuf wire format.
    pub fn encode(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(64 + self.bucket.len() * 16);
        encode_double(&mut buffer, 1, self.min);
        encode_double(&mut buffer, 2, self.max);
        encode_double(&mut buffer, 3, self.num);
        encode_double(&mut buffer, 4, self.sum);
        encode_double(&mut buffer, 5, self.sum_squares);
        encode_packed_doubles(&mut buffer, 6, &self.bucket_limit);
        encode_packed_doubles(&mut buffer, 7, &self.bucket);
        buffer
    }
}

/// The value carried by one `Summary.Value`.
#[derive(Debug, Clone, PartialEq)]
pub enum SummaryValue {
    /// `Summary.Value.simple_value` (field 2).
    Simple(f32),
    /// `Summary.Value.histo` (field 5).
    Histogram(HistogramProto),
}

/// Encode a `tensorflow.Summary` containing a single tagged value.
pub fn encode_summary(tag: &str, value: &SummaryValue) -> Vec<u8> {
    let mut summary_value = Vec::new();
    encode_bytes(&mut summary_value, 1, tag.as_bytes());
    match value {
        SummaryValue::Simple(scalar) => encode_float(&mut summary_value, 2, *scalar),
        SummaryValue::Histogram(histogram) => {
            encode_bytes(&mut summary_value, 5, &histogram.encode())
        },
    }

    let mut summary = Vec::with_capacity(summary_value.len() + 8);
    encode_bytes(&mut summary, 1, &summary_value);
    summary
}

/// Encode a `tensorflow.Event` carrying a `Summary`.
pub fn encode_summary_event(wall_time: f64, step: i64, tag: &str, value: &SummaryValue) -> Vec<u8> {
    let summary = encode_summary(tag, value);
    let mut event = Vec::with_capacity(summary.len() + 24);
    encode_double(&mut event, 1, wall_time);
    encode_int64(&mut event, 2, step);
    encode_bytes(&mut event, 5, &summary);
    event
}

/// The `file_version` string TensorBoard expects as the first record of a run.
pub const FILE_VERSION: &str = "brain.Event:2";

/// Encode the leading `tensorflow.Event` carrying `file_version`.
pub fn encode_file_version_event(wall_time: f64) -> Vec<u8> {
    let mut event = Vec::with_capacity(32);
    encode_double(&mut event, 1, wall_time);
    encode_int64(&mut event, 2, 0);
    encode_bytes(&mut event, 3, FILE_VERSION.as_bytes());
    event
}

// ── Minimal protobuf reader (used by tests and by tooling) ───────────────────

/// One decoded protobuf field.
#[derive(Debug, Clone, PartialEq)]
pub enum WireField {
    /// Varint payload.
    Varint(u64),
    /// 64-bit payload.
    Fixed64(u64),
    /// 32-bit payload.
    Fixed32(u32),
    /// Length-delimited payload.
    Bytes(Vec<u8>),
}

impl WireField {
    /// Interpret a `Fixed64` field as a `double`.
    pub fn as_double(&self) -> Option<f64> {
        match self {
            WireField::Fixed64(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Interpret a `Fixed32` field as a `float`.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            WireField::Fixed32(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    /// Borrow a length-delimited payload.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            WireField::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }
}

/// Decode a protobuf message into `(field_number, field)` pairs, preserving order.
pub fn decode_message(mut data: &[u8]) -> io::Result<Vec<(u32, WireField)>> {
    fn read_varint(data: &mut &[u8]) -> io::Result<u64> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *data.first().ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "truncated protobuf varint")
            })?;
            *data = &data[1..];
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "protobuf varint exceeds 64 bits",
                ));
            }
        }
    }

    fn take<'a>(data: &mut &'a [u8], count: usize) -> io::Result<&'a [u8]> {
        if data.len() < count {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated protobuf field",
            ));
        }
        let (head, tail) = data.split_at(count);
        *data = tail;
        Ok(head)
    }

    let mut fields = Vec::new();
    while !data.is_empty() {
        let tag = read_varint(&mut data)?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u32;
        let field = match wire_type {
            WIRE_TYPE_VARINT => WireField::Varint(read_varint(&mut data)?),
            WIRE_TYPE_FIXED64 => {
                let bytes = take(&mut data, 8)?;
                let mut buffer = [0u8; 8];
                buffer.copy_from_slice(bytes);
                WireField::Fixed64(u64::from_le_bytes(buffer))
            },
            WIRE_TYPE_FIXED32 => {
                let bytes = take(&mut data, 4)?;
                let mut buffer = [0u8; 4];
                buffer.copy_from_slice(bytes);
                WireField::Fixed32(u32::from_le_bytes(buffer))
            },
            WIRE_TYPE_LENGTH_DELIMITED => {
                let length = read_varint(&mut data)? as usize;
                WireField::Bytes(take(&mut data, length)?.to_vec())
            },
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported protobuf wire type {}", other),
                ))
            },
        };
        fields.push((field_number, field));
    }

    Ok(fields)
}

/// Decode a packed `repeated double` payload.
pub fn decode_packed_doubles(data: &[u8]) -> io::Result<Vec<f64>> {
    if !data.len().is_multiple_of(8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "packed double payload is not a multiple of 8 bytes",
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| {
            let mut buffer = [0u8; 8];
            buffer.copy_from_slice(chunk);
            f64::from_le_bytes(buffer)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard CRC-32C check value. A CRC that round-trips against itself
    /// but uses the wrong polynomial (for example CRC-32/IEEE `0xedb88320`,
    /// which the previous implementation used) fails here.
    #[test]
    fn test_crc32c_standard_check_vector() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn test_crc32c_is_not_crc32_ieee() {
        // CRC-32/IEEE of "123456789" is 0xCBF43926; CRC-32C must differ.
        assert_ne!(crc32c(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_masked_crc_matches_documented_formula() {
        let payload = b"trustformers";
        let raw = crc32c(payload);
        // ((crc >> 15) | (crc << 17)) is exactly a 15-bit right rotation.
        let expected = raw.rotate_right(15).wrapping_add(0xa282_ead8);
        assert_eq!(masked_crc32c(payload), expected);
        // The mask must actually change the value.
        assert_ne!(masked_crc32c(payload), raw);
    }

    #[test]
    fn test_tfrecord_framing_is_byte_exact() {
        let payload = b"hello tfrecord".to_vec();
        let mut buffer = Vec::new();
        write_record(&mut buffer, &payload).expect("write_record failed");

        // 8 length bytes + 4 crc + payload + 4 crc.
        assert_eq!(buffer.len(), 8 + 4 + payload.len() + 4);

        let length_bytes = &buffer[0..8];
        assert_eq!(
            u64::from_le_bytes(length_bytes.try_into().expect("8 bytes")),
            payload.len() as u64
        );

        let length_crc = u32::from_le_bytes(buffer[8..12].try_into().expect("4 bytes"));
        assert_eq!(length_crc, masked_crc32c(length_bytes));

        let data_start = 12;
        let data_end = data_start + payload.len();
        assert_eq!(&buffer[data_start..data_end], &payload[..]);

        let data_crc =
            u32::from_le_bytes(buffer[data_end..data_end + 4].try_into().expect("4 bytes"));
        assert_eq!(data_crc, masked_crc32c(&payload));
    }

    #[test]
    fn test_tfrecord_round_trip() {
        let records: Vec<Vec<u8>> = vec![b"first".to_vec(), b"second record".to_vec(), Vec::new()];
        let mut buffer = Vec::new();
        for record in &records {
            write_record(&mut buffer, record).expect("write_record failed");
        }

        let mut cursor = std::io::Cursor::new(buffer);
        for expected in &records {
            let decoded = read_record(&mut cursor).expect("read_record failed").expect("record");
            assert_eq!(&decoded, expected);
        }
        assert!(read_record(&mut cursor).expect("read_record failed").is_none());
    }

    #[test]
    fn test_tfrecord_detects_payload_corruption() {
        let mut buffer = Vec::new();
        write_record(&mut buffer, b"payload").expect("write_record failed");
        // Flip a bit inside the payload.
        buffer[13] ^= 0x01;

        let mut cursor = std::io::Cursor::new(buffer);
        let error = read_record(&mut cursor).expect_err("corruption must be detected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_varint_encoding() {
        let mut buffer = Vec::new();
        encode_varint(&mut buffer, 0);
        assert_eq!(buffer, vec![0x00]);

        buffer.clear();
        encode_varint(&mut buffer, 300);
        assert_eq!(buffer, vec![0xac, 0x02]);

        buffer.clear();
        encode_varint(&mut buffer, 1);
        assert_eq!(buffer, vec![0x01]);
    }

    #[test]
    fn test_scalar_event_decodes_as_tensorflow_event() {
        let event = encode_summary_event(1234.5, 7, "loss/train", &SummaryValue::Simple(0.25));
        let fields = decode_message(&event).expect("decode event");

        let wall_time = fields
            .iter()
            .find(|(number, _)| *number == 1)
            .and_then(|(_, field)| field.as_double())
            .expect("wall_time");
        assert!((wall_time - 1234.5).abs() < 1e-9);

        let step = fields
            .iter()
            .find(|(number, _)| *number == 2)
            .map(|(_, field)| field.clone())
            .expect("step");
        assert_eq!(step, WireField::Varint(7));

        let summary_bytes = fields
            .iter()
            .find(|(number, _)| *number == 5)
            .and_then(|(_, field)| field.as_bytes())
            .expect("summary")
            .to_vec();

        let summary_fields = decode_message(&summary_bytes).expect("decode summary");
        let value_bytes = summary_fields
            .iter()
            .find(|(number, _)| *number == 1)
            .and_then(|(_, field)| field.as_bytes())
            .expect("summary value")
            .to_vec();

        let value_fields = decode_message(&value_bytes).expect("decode summary value");
        let tag = value_fields
            .iter()
            .find(|(number, _)| *number == 1)
            .and_then(|(_, field)| field.as_bytes())
            .expect("tag");
        assert_eq!(std::str::from_utf8(tag).expect("utf8"), "loss/train");

        let simple_value = value_fields
            .iter()
            .find(|(number, _)| *number == 2)
            .and_then(|(_, field)| field.as_float())
            .expect("simple_value");
        assert!((simple_value - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_event_decodes_with_packed_buckets() {
        let histogram = HistogramProto {
            min: -1.0,
            max: 3.0,
            num: 4.0,
            sum: 4.0,
            sum_squares: 12.0,
            bucket_limit: vec![0.0, 2.0, 4.0],
            bucket: vec![1.0, 2.0, 1.0],
        };
        let encoded = histogram.encode();
        let fields = decode_message(&encoded).expect("decode histogram");

        assert_eq!(
            fields
                .iter()
                .find(|(number, _)| *number == 3)
                .and_then(|(_, field)| field.as_double()),
            Some(4.0),
            "HistogramProto.num must be encoded as a double"
        );

        let limits = fields
            .iter()
            .find(|(number, _)| *number == 6)
            .and_then(|(_, field)| field.as_bytes())
            .map(decode_packed_doubles)
            .expect("bucket_limit")
            .expect("packed doubles");
        assert_eq!(limits, vec![0.0, 2.0, 4.0]);

        let counts = fields
            .iter()
            .find(|(number, _)| *number == 7)
            .and_then(|(_, field)| field.as_bytes())
            .map(decode_packed_doubles)
            .expect("bucket")
            .expect("packed doubles");
        assert_eq!(counts, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_file_version_event() {
        let event = encode_file_version_event(1.0);
        let fields = decode_message(&event).expect("decode event");
        let version = fields
            .iter()
            .find(|(number, _)| *number == 3)
            .and_then(|(_, field)| field.as_bytes())
            .expect("file_version");
        assert_eq!(std::str::from_utf8(version).expect("utf8"), "brain.Event:2");
    }
}
