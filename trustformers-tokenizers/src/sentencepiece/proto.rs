//! Minimal, bounds-checked reader for the SentencePiece `ModelProto` wire format.
//!
//! Field numbers follow `sentencepiece_model.proto`:
//!
//! ```text
//! message ModelProto {
//!   repeated SentencePiece pieces          = 1;
//!   optional TrainerSpec    trainer_spec    = 2;
//!   optional NormalizerSpec normalizer_spec = 3;
//! }
//! message SentencePiece { string piece = 1; float score = 2; Type type = 3; }
//! message TrainerSpec    { ModelType model_type = 3; bool treat_whitespace_as_suffix = 24;
//!                          bool byte_fallback = 35; }
//! message NormalizerSpec { string name = 1; bytes precompiled_charsmap = 2;
//!                          bool add_dummy_prefix = 3; bool remove_extra_whitespaces = 4;
//!                          bool escape_whitespaces = 5; }
//! ```
//!
//! Every read is bounds-checked and every skip is wire-type aware, so a
//! truncated or hostile `.model` file produces an error instead of a panic or a
//! desynchronized parse.

use trustformers_core::errors::{Result, TrustformersError};

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LENGTH_DELIMITED: u8 = 2;
const WIRE_FIXED32: u8 = 5;

/// Type of a sentence piece (`SentencePiece.Type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PieceType {
    #[default]
    Normal = 0,
    Unknown = 1,
    Control = 2,
    UserDefined = 3,
    Unused = 4,
    Byte = 5,
}

impl PieceType {
    fn from_wire(value: u64) -> Self {
        match value {
            1 => PieceType::Unknown,
            2 => PieceType::Control,
            3 => PieceType::UserDefined,
            4 => PieceType::Unused,
            5 => PieceType::Byte,
            _ => PieceType::Normal,
        }
    }
}

/// One vocabulary entry of a SentencePiece model.
#[derive(Debug, Default, Clone)]
pub struct SentencePiece {
    pub piece: String,
    pub score: f32,
    pub piece_type: PieceType,
}

/// The subset of `TrainerSpec` this crate consumes.
#[derive(Debug, Clone)]
pub struct TrainerSpec {
    /// 1 = UNIGRAM, 2 = BPE, 3 = WORD, 4 = CHAR.
    pub model_type: u64,
    pub byte_fallback: bool,
    pub treat_whitespace_as_suffix: bool,
}

impl Default for TrainerSpec {
    fn default() -> Self {
        // proto2 defaults from sentencepiece_model.proto.
        Self {
            model_type: 1,
            byte_fallback: false,
            treat_whitespace_as_suffix: false,
        }
    }
}

/// The subset of `NormalizerSpec` this crate consumes.
#[derive(Debug, Clone)]
pub struct NormalizerSpec {
    pub name: String,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        // proto2 defaults from sentencepiece_model.proto: all three are `true`.
        Self {
            name: String::new(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

/// A parsed SentencePiece `ModelProto`.
#[derive(Debug, Default, Clone)]
pub struct ModelProto {
    pub pieces: Vec<SentencePiece>,
    pub trainer_spec: TrainerSpec,
    pub normalizer_spec: NormalizerSpec,
}

/// Cursor over a protobuf message with bounds-checked primitives.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn has_remaining(&self) -> bool {
        self.pos < self.data.len()
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;

        while self.pos < self.data.len() {
            let byte = self.data[self.pos];
            self.pos += 1;

            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
            if shift >= 64 {
                return Err(TrustformersError::invalid_config(
                    "SentencePiece model: varint exceeds 64 bits".to_string(),
                ));
            }
        }

        Err(TrustformersError::invalid_config(
            "SentencePiece model: truncated varint".to_string(),
        ))
    }

    /// Read a field tag, returning `(field_number, wire_type)`.
    fn read_tag(&mut self) -> Result<(u64, u8)> {
        let key = self.read_varint()?;
        let field_number = key >> 3;
        let wire_type = (key & 0x7) as u8;

        if field_number == 0 {
            return Err(TrustformersError::invalid_config(
                "SentencePiece model: field number 0 is not valid protobuf".to_string(),
            ));
        }

        Ok((field_number, wire_type))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or_else(|| {
            TrustformersError::invalid_config(
                "SentencePiece model: length overflows the buffer".to_string(),
            )
        })?;
        if end > self.data.len() {
            return Err(TrustformersError::invalid_config(format!(
                "SentencePiece model: truncated field (need {} bytes, {} available)",
                len,
                self.data.len().saturating_sub(self.pos)
            )));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_length_delimited(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.read_varint()?).map_err(|_| {
            TrustformersError::invalid_config(
                "SentencePiece model: field length does not fit in usize".to_string(),
            )
        })?;
        self.take(len)
    }

    fn read_fixed32(&mut self) -> Result<[u8; 4]> {
        let bytes = self.take(4)?;
        let mut out = [0u8; 4];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    fn expect_wire(&self, field_number: u64, wire_type: u8, expected: u8) -> Result<()> {
        if wire_type == expected {
            Ok(())
        } else {
            Err(TrustformersError::invalid_config(format!(
                "SentencePiece model: field {} has wire type {}, expected {}",
                field_number, wire_type, expected
            )))
        }
    }

    /// Skip a field of any wire type without desynchronizing the stream.
    fn skip_field(&mut self, wire_type: u8) -> Result<()> {
        match wire_type {
            WIRE_VARINT => {
                self.read_varint()?;
                Ok(())
            },
            WIRE_FIXED64 => {
                self.take(8)?;
                Ok(())
            },
            WIRE_LENGTH_DELIMITED => {
                self.read_length_delimited()?;
                Ok(())
            },
            WIRE_FIXED32 => {
                self.take(4)?;
                Ok(())
            },
            other => Err(TrustformersError::invalid_config(format!(
                "SentencePiece model: unsupported wire type {}",
                other
            ))),
        }
    }
}

/// Parse a full `ModelProto`.
pub fn parse_model_proto(data: &[u8]) -> Result<ModelProto> {
    let mut model = ModelProto::default();
    let mut reader = Reader::new(data);

    while reader.has_remaining() {
        let (field_number, wire_type) = reader.read_tag()?;
        match field_number {
            1 => {
                reader.expect_wire(field_number, wire_type, WIRE_LENGTH_DELIMITED)?;
                let bytes = reader.read_length_delimited()?;
                model.pieces.push(parse_piece(bytes)?);
            },
            2 => {
                reader.expect_wire(field_number, wire_type, WIRE_LENGTH_DELIMITED)?;
                let bytes = reader.read_length_delimited()?;
                model.trainer_spec = parse_trainer_spec(bytes)?;
            },
            3 => {
                reader.expect_wire(field_number, wire_type, WIRE_LENGTH_DELIMITED)?;
                let bytes = reader.read_length_delimited()?;
                model.normalizer_spec = parse_normalizer_spec(bytes)?;
            },
            _ => reader.skip_field(wire_type)?,
        }
    }

    if model.pieces.is_empty() {
        return Err(TrustformersError::invalid_config(
            "SentencePiece model: no vocabulary pieces found".to_string(),
        ));
    }

    Ok(model)
}

fn parse_piece(data: &[u8]) -> Result<SentencePiece> {
    let mut piece = SentencePiece::default();
    let mut reader = Reader::new(data);

    while reader.has_remaining() {
        let (field_number, wire_type) = reader.read_tag()?;
        match field_number {
            1 => {
                reader.expect_wire(field_number, wire_type, WIRE_LENGTH_DELIMITED)?;
                let bytes = reader.read_length_delimited()?;
                piece.piece = String::from_utf8_lossy(bytes).into_owned();
            },
            2 => {
                reader.expect_wire(field_number, wire_type, WIRE_FIXED32)?;
                piece.score = f32::from_le_bytes(reader.read_fixed32()?);
            },
            3 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                piece.piece_type = PieceType::from_wire(reader.read_varint()?);
            },
            _ => reader.skip_field(wire_type)?,
        }
    }

    Ok(piece)
}

fn parse_trainer_spec(data: &[u8]) -> Result<TrainerSpec> {
    let mut spec = TrainerSpec::default();
    let mut reader = Reader::new(data);

    while reader.has_remaining() {
        let (field_number, wire_type) = reader.read_tag()?;
        match field_number {
            3 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                spec.model_type = reader.read_varint()?;
            },
            24 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                spec.treat_whitespace_as_suffix = reader.read_varint()? != 0;
            },
            35 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                spec.byte_fallback = reader.read_varint()? != 0;
            },
            _ => reader.skip_field(wire_type)?,
        }
    }

    Ok(spec)
}

fn parse_normalizer_spec(data: &[u8]) -> Result<NormalizerSpec> {
    let mut spec = NormalizerSpec::default();
    let mut reader = Reader::new(data);

    while reader.has_remaining() {
        let (field_number, wire_type) = reader.read_tag()?;
        match field_number {
            1 => {
                reader.expect_wire(field_number, wire_type, WIRE_LENGTH_DELIMITED)?;
                let bytes = reader.read_length_delimited()?;
                spec.name = String::from_utf8_lossy(bytes).into_owned();
            },
            3 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                spec.add_dummy_prefix = reader.read_varint()? != 0;
            },
            4 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                spec.remove_extra_whitespaces = reader.read_varint()? != 0;
            },
            5 => {
                reader.expect_wire(field_number, wire_type, WIRE_VARINT)?;
                spec.escape_whitespaces = reader.read_varint()? != 0;
            },
            _ => reader.skip_field(wire_type)?,
        }
    }

    Ok(spec)
}

#[cfg(test)]
pub(crate) mod encode {
    //! Tiny protobuf *writer* used by the tests to build reference fixtures.

    pub fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                return out;
            }
        }
    }

    pub fn tag(field_number: u64, wire_type: u8) -> Vec<u8> {
        varint((field_number << 3) | u64::from(wire_type))
    }

    pub fn length_delimited(field_number: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = tag(field_number, 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    pub fn varint_field(field_number: u64, value: u64) -> Vec<u8> {
        let mut out = tag(field_number, 0);
        out.extend(varint(value));
        out
    }

    pub fn fixed32_field(field_number: u64, value: f32) -> Vec<u8> {
        let mut out = tag(field_number, 5);
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    /// Build a `SentencePiece` submessage.
    pub fn piece(text: &str, score: f32, piece_type: u64) -> Vec<u8> {
        let mut out = length_delimited(1, text.as_bytes());
        out.extend(fixed32_field(2, score));
        out.extend(varint_field(3, piece_type));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::encode::*;
    use super::*;

    fn reference_model() -> Vec<u8> {
        let mut out = Vec::new();
        // pieces = 1
        out.extend(length_delimited(1, &piece("<unk>", 0.0, 1)));
        out.extend(length_delimited(1, &piece("▁hello", -1.5, 0)));
        out.extend(length_delimited(1, &piece("<s>", 0.0, 2)));

        // trainer_spec = 2 (model_type = BPE(2), byte_fallback = true), plus a
        // length-delimited field the parser must skip without desyncing.
        let mut trainer = Vec::new();
        trainer.extend(length_delimited(1, b"corpus.txt")); // TrainerSpec.input
        trainer.extend(varint_field(3, 2)); // model_type = BPE
        trainer.extend(varint_field(35, 1)); // byte_fallback = true
        out.extend(length_delimited(2, &trainer));

        // normalizer_spec = 3 (add_dummy_prefix = false), again with a
        // length-delimited field (precompiled_charsmap) in front of it.
        let mut normalizer = Vec::new();
        normalizer.extend(length_delimited(1, b"nmt_nfkc"));
        normalizer.extend(length_delimited(2, &[0xDE, 0xAD, 0xBE, 0xEF]));
        normalizer.extend(varint_field(3, 0)); // add_dummy_prefix = false
        out.extend(length_delimited(3, &normalizer));

        out
    }

    #[test]
    fn test_parses_reference_model_proto() {
        let model = parse_model_proto(&reference_model()).expect("reference model must parse");

        assert_eq!(model.pieces.len(), 3);
        assert_eq!(model.pieces[0].piece, "<unk>");
        assert_eq!(model.pieces[0].piece_type, PieceType::Unknown);
        assert_eq!(model.pieces[1].piece, "▁hello");
        assert!((model.pieces[1].score - (-1.5)).abs() < 1e-6);
        assert_eq!(model.pieces[2].piece_type, PieceType::Control);

        assert_eq!(model.trainer_spec.model_type, 2);
        assert!(model.trainer_spec.byte_fallback);

        assert_eq!(model.normalizer_spec.name, "nmt_nfkc");
        assert!(!model.normalizer_spec.add_dummy_prefix);
        // proto2 defaults survive for fields the file does not carry.
        assert!(model.normalizer_spec.remove_extra_whitespaces);
        assert!(model.normalizer_spec.escape_whitespaces);
    }

    /// A length-delimited field inside a skipped sub-message must consume its
    /// payload; otherwise the stream desyncs and later fields are garbage.
    #[test]
    fn test_skipped_length_delimited_fields_do_not_desync() {
        let mut trainer = Vec::new();
        trainer.extend(length_delimited(1, b"a rather long input path value"));
        trainer.extend(length_delimited(2, b"model_prefix"));
        trainer.extend(varint_field(3, 4)); // model_type = CHAR
        let bytes = {
            let mut out = length_delimited(1, &piece("x", 1.0, 0));
            out.extend(length_delimited(2, &trainer));
            out
        };

        let model = parse_model_proto(&bytes).expect("model must parse");
        assert_eq!(model.trainer_spec.model_type, 4);
    }

    #[test]
    fn test_truncated_input_errors_instead_of_panicking() {
        let bytes = reference_model();
        for cut in 1..bytes.len() {
            // Must never panic; either it parses a prefix or reports an error.
            let _ = parse_model_proto(&bytes[..cut]);
        }

        // A length header that claims more bytes than exist is an error.
        let mut hostile = tag(1, 2);
        hostile.extend(varint(4096));
        hostile.extend_from_slice(b"short");
        assert!(parse_model_proto(&hostile).is_err());
    }

    #[test]
    fn test_field_number_zero_is_rejected() {
        // 0x01 => field number 0, wire type 1: invalid protobuf.
        assert!(parse_model_proto(&[0x01, 0x00]).is_err());
    }

    #[test]
    fn test_empty_model_is_an_error() {
        assert!(parse_model_proto(&[]).is_err());
    }
}
