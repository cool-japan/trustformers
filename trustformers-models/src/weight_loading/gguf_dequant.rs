//! Real ggml block dequantization for the GGUF loader.
//!
//! Every routine here mirrors the reference layout in `ggml-common.h` /
//! `ggml-quants.c`, super-block for super-block. Nothing is approximated: a type
//! is either dequantized from its actual packed scales, mins and quants, or
//! [`dequantize`] returns an error naming the type. A previous revision routed
//! every unhandled type through a "generic" path that emitted `(byte - 128) / 128`
//! noise and reported it as a successfully loaded weight; that path is gone.
//!
//! # Block geometry
//!
//! ggml packs `elements_per_block` values into `bytes_per_block` bytes. Both
//! numbers come from [`BlockGeometry`], which is the single source of truth for
//! the loader's read sizes, the reported tensor byte sizes and the dequantizers,
//! so a size can never drift between them.

use trustformers_core::errors::{Result, TrustformersError};

use super::gguf::GGMLType;

/// Elements per ggml super-block for the K-quant families.
const QK_K: usize = 256;

/// How ggml packs a tensor of a given type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGeometry {
    /// Number of tensor elements encoded by one block.
    pub elements_per_block: usize,
    /// Number of bytes one block occupies on disk.
    pub bytes_per_block: usize,
}

impl BlockGeometry {
    /// Exact byte size of `elements` values stored in this layout.
    ///
    /// ggml always stores whole blocks, so a partially-filled trailing block
    /// still costs a full block on disk.
    pub fn bytes_for(&self, elements: usize) -> usize {
        elements.div_ceil(self.elements_per_block) * self.bytes_per_block
    }
}

/// The on-disk geometry of a ggml type.
///
/// The values are the `blck_size` / `type_size` pairs from ggml's type traits
/// table. They are exact: `Q4_K` is 144 bytes per 256 elements, not the 128 that
/// a naive "4 bits per weight" estimate would give, because the super-block also
/// carries `d`, `dmin` and twelve packed 6-bit scale bytes.
pub fn block_geometry(ggml_type: &GGMLType) -> BlockGeometry {
    let (elements_per_block, bytes_per_block) = match ggml_type {
        GGMLType::F32 => (1, 4),
        GGMLType::F16 => (1, 2),
        GGMLType::Q4_0 => (32, 18),
        GGMLType::Q4_1 => (32, 20),
        GGMLType::Q5_0 => (32, 22),
        GGMLType::Q5_1 => (32, 24),
        GGMLType::Q8_0 => (32, 34),
        GGMLType::Q8_1 => (32, 36),
        GGMLType::Q2K => (QK_K, 84),
        GGMLType::Q3K => (QK_K, 110),
        GGMLType::Q4K => (QK_K, 144),
        GGMLType::Q5K => (QK_K, 176),
        GGMLType::Q6K => (QK_K, 210),
        GGMLType::Q8K => (QK_K, 292),
        GGMLType::Iq2Xxs => (QK_K, 66),
        GGMLType::Iq2Xs => (QK_K, 74),
        GGMLType::Iq3Xxs => (QK_K, 98),
        GGMLType::Iq1S => (QK_K, 50),
        GGMLType::Iq4Nl => (32, 18),
        GGMLType::Iq3S => (QK_K, 110),
        GGMLType::Iq2S => (QK_K, 82),
        GGMLType::Iq4Xs => (QK_K, 136),
    };
    BlockGeometry {
        elements_per_block,
        bytes_per_block,
    }
}

/// Read a little-endian IEEE-754 half at `data[offset..offset + 2]`.
fn read_f16(data: &[u8], offset: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
}

/// Split `data` into whole blocks, erroring when the payload is short.
fn blocks<'a>(
    data: &'a [u8],
    elements: usize,
    geometry: BlockGeometry,
    type_name: &str,
) -> Result<impl Iterator<Item = &'a [u8]> + 'a> {
    let needed = geometry.bytes_for(elements);
    if data.len() < needed {
        return Err(TrustformersError::weight_load_error(format!(
            "insufficient data for {type_name}: {elements} elements need {needed} bytes but only \
             {} are available",
            data.len()
        )));
    }
    Ok(data[..needed].chunks_exact(geometry.bytes_per_block))
}

/// Dequantize `elements` values of `ggml_type` from `data`.
///
/// # Errors
///
/// Fails when the payload is shorter than the block layout requires, or when the
/// type has no implemented dequantizer. It never invents values for a type it
/// cannot decode.
pub fn dequantize(ggml_type: &GGMLType, data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut out = match ggml_type {
        GGMLType::F32 => dequantize_f32(data, elements)?,
        GGMLType::F16 => dequantize_f16(data, elements)?,
        GGMLType::Q4_0 => dequantize_q4_0(data, elements)?,
        GGMLType::Q4_1 => dequantize_q4_1(data, elements)?,
        GGMLType::Q5_0 => dequantize_q5_0(data, elements)?,
        GGMLType::Q5_1 => dequantize_q5_1(data, elements)?,
        GGMLType::Q8_0 => dequantize_q8_0(data, elements)?,
        GGMLType::Q8_1 => dequantize_q8_1(data, elements)?,
        GGMLType::Q2K => dequantize_q2_k(data, elements)?,
        GGMLType::Q3K => dequantize_q3_k(data, elements)?,
        GGMLType::Q4K => dequantize_q4_k(data, elements)?,
        GGMLType::Q5K => dequantize_q5_k(data, elements)?,
        GGMLType::Q6K => dequantize_q6_k(data, elements)?,
        GGMLType::Q8K => dequantize_q8_k(data, elements)?,
        unsupported => {
            return Err(TrustformersError::not_implemented(format!(
                "GGUF dequantization for ggml type {unsupported:?}: this loader implements F32, \
                 F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 and the K-quant families (Q2_K, Q3_K, \
                 Q4_K, Q5_K, Q6_K, Q8_K). Re-quantize the file to one of those types."
            )))
        },
    };
    // The trailing block may encode more values than the tensor holds.
    out.truncate(elements);
    if out.len() != elements {
        return Err(TrustformersError::weight_load_error(format!(
            "dequantization of {ggml_type:?} produced {} values but the tensor declares {elements}",
            out.len()
        )));
    }
    Ok(out)
}

fn dequantize_f32(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let needed = elements * 4;
    if data.len() < needed {
        return Err(TrustformersError::weight_load_error(format!(
            "insufficient data for F32: need {needed} bytes, have {}",
            data.len()
        )));
    }
    Ok(data[..needed]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dequantize_f16(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let needed = elements * 2;
    if data.len() < needed {
        return Err(TrustformersError::weight_load_error(format!(
            "insufficient data for F16: need {needed} bytes, have {}",
            data.len()
        )));
    }
    Ok(data[..needed]
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// `block_q4_0 { half d; uint8 qs[16]; }`
///
/// The low nibble of `qs[j]` is element `j`, the high nibble is element `j + 16`
/// — the halves are **not** interleaved.
fn dequantize_q4_0(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q4_0);
    let mut out = vec![0.0f32; elements.div_ceil(32) * 32];
    for (block_idx, block) in blocks(data, elements, geometry, "Q4_0")?.enumerate() {
        let d = read_f16(block, 0);
        let qs = &block[2..18];
        let base = block_idx * 32;
        for (j, &byte) in qs.iter().enumerate() {
            out[base + j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            out[base + j + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
    }
    Ok(out)
}

/// `block_q4_1 { half d; half m; uint8 qs[16]; }`
fn dequantize_q4_1(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q4_1);
    let mut out = vec![0.0f32; elements.div_ceil(32) * 32];
    for (block_idx, block) in blocks(data, elements, geometry, "Q4_1")?.enumerate() {
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qs = &block[4..20];
        let base = block_idx * 32;
        for (j, &byte) in qs.iter().enumerate() {
            out[base + j] = (byte & 0x0F) as f32 * d + m;
            out[base + j + 16] = (byte >> 4) as f32 * d + m;
        }
    }
    Ok(out)
}

/// `block_q5_0 { half d; uint8 qh[4]; uint8 qs[16]; }`
///
/// `qh` is a 32-bit little-endian word holding the fifth bit of every element:
/// bit `j` for element `j` and bit `j + 16` for element `j + 16`.
fn dequantize_q5_0(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q5_0);
    let mut out = vec![0.0f32; elements.div_ceil(32) * 32];
    for (block_idx, block) in blocks(data, elements, geometry, "Q5_0")?.enumerate() {
        let d = read_f16(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let base = block_idx * 32;
        for (j, &byte) in qs.iter().enumerate() {
            let hi_low = ((qh >> j) & 1) as u8;
            let hi_high = ((qh >> (j + 16)) & 1) as u8;
            let low = ((byte & 0x0F) | (hi_low << 4)) as i32 - 16;
            let high = ((byte >> 4) | (hi_high << 4)) as i32 - 16;
            out[base + j] = low as f32 * d;
            out[base + j + 16] = high as f32 * d;
        }
    }
    Ok(out)
}

/// `block_q5_1 { half d; half m; uint8 qh[4]; uint8 qs[16]; }`
fn dequantize_q5_1(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q5_1);
    let mut out = vec![0.0f32; elements.div_ceil(32) * 32];
    for (block_idx, block) in blocks(data, elements, geometry, "Q5_1")?.enumerate() {
        let d = read_f16(block, 0);
        let m = read_f16(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let base = block_idx * 32;
        for (j, &byte) in qs.iter().enumerate() {
            let hi_low = ((qh >> j) & 1) as u8;
            let hi_high = ((qh >> (j + 16)) & 1) as u8;
            let low = ((byte & 0x0F) | (hi_low << 4)) as f32;
            let high = ((byte >> 4) | (hi_high << 4)) as f32;
            out[base + j] = low * d + m;
            out[base + j + 16] = high * d + m;
        }
    }
    Ok(out)
}

/// `block_q8_0 { half d; int8 qs[32]; }`
fn dequantize_q8_0(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q8_0);
    let mut out = vec![0.0f32; elements.div_ceil(32) * 32];
    for (block_idx, block) in blocks(data, elements, geometry, "Q8_0")?.enumerate() {
        let d = read_f16(block, 0);
        let base = block_idx * 32;
        for (j, &byte) in block[2..34].iter().enumerate() {
            out[base + j] = (byte as i8) as f32 * d;
        }
    }
    Ok(out)
}

/// `block_q8_1 { half d; half s; int8 qs[32]; }`
///
/// `s` caches `d * sum(qs)` for dot products and carries no extra information,
/// so dequantization only needs `d`.
fn dequantize_q8_1(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q8_1);
    let mut out = vec![0.0f32; elements.div_ceil(32) * 32];
    for (block_idx, block) in blocks(data, elements, geometry, "Q8_1")?.enumerate() {
        let d = read_f16(block, 0);
        let base = block_idx * 32;
        for (j, &byte) in block[4..36].iter().enumerate() {
            out[base + j] = (byte as i8) as f32 * d;
        }
    }
    Ok(out)
}

/// `block_q2_K { uint8 scales[16]; uint8 qs[64]; half d; half dmin; }`
///
/// Each `scales[i]` packs a 4-bit scale in its low nibble and a 4-bit min in its
/// high nibble. `qs` holds 2-bit quants, four sub-groups of 16 per 32-byte
/// stride, selected by a rolling shift.
fn dequantize_q2_k(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q2K);
    let mut out = vec![0.0f32; elements.div_ceil(QK_K) * QK_K];
    for (block_idx, block) in blocks(data, elements, geometry, "Q2_K")?.enumerate() {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16(block, 80);
        let dmin = read_f16(block, 82);

        let mut y = block_idx * QK_K;
        let mut is = 0usize;
        for half_idx in 0..2 {
            let q = &qs[half_idx * 32..half_idx * 32 + 32];
            let mut shift = 0u32;
            for _ in 0..4 {
                for group in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0F) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for l in 0..16 {
                        let quant = ((q[group * 16 + l] >> shift) & 3) as f32;
                        out[y] = dl * quant - ml;
                        y += 1;
                    }
                }
                shift += 2;
            }
        }
    }
    Ok(out)
}

/// Unpack Q3_K's twelve 6-bit scale bytes into sixteen signed scales.
///
/// The reference implementation does this with four `uint32` words and the masks
/// `0x03030303` / `0x0f0f0f0f`; the byte-wise form below is the same shuffle.
fn unpack_q3_k_scales(packed: &[u8]) -> [i8; 16] {
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;

    let word = |i: usize| -> u32 {
        u32::from_le_bytes([
            packed[i * 4],
            packed[i * 4 + 1],
            packed[i * 4 + 2],
            packed[i * 4 + 3],
        ])
    };

    let a0 = word(0);
    let a1 = word(1);
    let tmp = word(2);

    let aux = [
        (a0 & kmask2) | (((tmp) & kmask1) << 4),
        (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
        ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
        ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
    ];

    let mut scales = [0i8; 16];
    for (i, value) in aux.iter().enumerate() {
        for (j, byte) in value.to_le_bytes().iter().enumerate() {
            scales[i * 4 + j] = *byte as i8;
        }
    }
    scales
}

/// `block_q3_K { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d; }`
///
/// `qs` holds the low two bits of each quant and `hmask` the third bit, inverted:
/// a clear mask bit means "subtract 4".
fn dequantize_q3_k(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q3K);
    let mut out = vec![0.0f32; elements.div_ceil(QK_K) * QK_K];
    for (block_idx, block) in blocks(data, elements, geometry, "Q3_K")?.enumerate() {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales = unpack_q3_k_scales(&block[96..108]);
        let d_all = read_f16(block, 108);

        let mut y = block_idx * QK_K;
        let mut is = 0usize;
        let mut m = 1u8;
        for half_idx in 0..2 {
            let q = &qs[half_idx * 32..half_idx * 32 + 32];
            let mut shift = 0u32;
            for _ in 0..4 {
                for group in 0..2 {
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let idx = group * 16 + l;
                        let low = ((q[idx] >> shift) & 3) as i32;
                        let high = if hmask[idx] & m != 0 { 0 } else { 4 };
                        out[y] = dl * (low - high) as f32;
                        y += 1;
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(out)
}

/// ggml's `get_scale_min_k4`: unpack the `j`-th 6-bit scale/min pair.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `block_q4_K { half d; half dmin; uint8 scales[12]; uint8 qs[128]; }`
fn dequantize_q4_k(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q4K);
    let mut out = vec![0.0f32; elements.div_ceil(QK_K) * QK_K];
    for (block_idx, block) in blocks(data, elements, geometry, "Q4_K")?.enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut y = block_idx * QK_K;
        let mut is = 0usize;
        for chunk in 0..4 {
            let q = &qs[chunk * 32..chunk * 32 + 32];
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            is += 2;
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for &byte in q.iter() {
                out[y] = d1 * (byte & 0x0F) as f32 - min1;
                y += 1;
            }
            for &byte in q.iter() {
                out[y] = d2 * (byte >> 4) as f32 - min2;
                y += 1;
            }
        }
    }
    Ok(out)
}

/// `block_q5_K { half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; }`
///
/// `qh[l]` carries the fifth bit of the two elements written from `qs[l]`; the
/// selecting masks walk `1, 2`, then `4, 8`, `16, 32`, `64, 128`.
fn dequantize_q5_k(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q5K);
    let mut out = vec![0.0f32; elements.div_ceil(QK_K) * QK_K];
    for (block_idx, block) in blocks(data, elements, geometry, "Q5_K")?.enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut y = block_idx * QK_K;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in 0..4 {
            let ql = &qs[chunk * 32..chunk * 32 + 32];
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            is += 2;
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for (l, &byte) in ql.iter().enumerate() {
                let extra = if qh[l] & u1 != 0 { 16 } else { 0 };
                out[y] = d1 * ((byte & 0x0F) as i32 + extra) as f32 - min1;
                y += 1;
            }
            for (l, &byte) in ql.iter().enumerate() {
                let extra = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[y] = d2 * ((byte >> 4) as i32 + extra) as f32 - min2;
                y += 1;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

/// `block_q6_K { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; }`
///
/// Each 128-element half consumes 64 `ql` bytes, 32 `qh` bytes and 8 signed
/// scales; the six-bit quants are biased by -32.
fn dequantize_q6_k(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q6K);
    let mut out = vec![0.0f32; elements.div_ceil(QK_K) * QK_K];
    for (block_idx, block) in blocks(data, elements, geometry, "Q6_K")?.enumerate() {
        let ql_all = &block[0..128];
        let qh_all = &block[128..192];
        let scales_all = &block[192..208];
        let d = read_f16(block, 208);

        for half_idx in 0..2 {
            let ql = &ql_all[half_idx * 64..half_idx * 64 + 64];
            let qh = &qh_all[half_idx * 32..half_idx * 32 + 32];
            let sc = &scales_all[half_idx * 8..half_idx * 8 + 8];
            let base = block_idx * QK_K + half_idx * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                out[base + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                out[base + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                out[base + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                out[base + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// `block_q8_K { float d; int8 qs[256]; int16 bsums[16]; }`
///
/// `bsums` caches per-group sums for dot products and is not needed to recover
/// the values.
fn dequantize_q8_k(data: &[u8], elements: usize) -> Result<Vec<f32>> {
    let geometry = block_geometry(&GGMLType::Q8K);
    let mut out = vec![0.0f32; elements.div_ceil(QK_K) * QK_K];
    for (block_idx, block) in blocks(data, elements, geometry, "Q8_K")?.enumerate() {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let base = block_idx * QK_K;
        for (j, &byte) in block[4..260].iter().enumerate() {
            out[base + j] = (byte as i8) as f32 * d;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(value: f32) -> [u8; 2] {
        half::f16::from_f32(value).to_bits().to_le_bytes()
    }

    #[test]
    fn geometry_matches_ggml_type_sizes() {
        // These are ggml's `blck_size` / `type_size` pairs. Getting them wrong
        // makes the loader read the wrong number of bytes from a real file, so
        // they are asserted explicitly rather than derived.
        let cases: &[(GGMLType, usize, usize)] = &[
            (GGMLType::F32, 1, 4),
            (GGMLType::F16, 1, 2),
            (GGMLType::Q4_0, 32, 18),
            (GGMLType::Q4_1, 32, 20),
            (GGMLType::Q5_0, 32, 22),
            (GGMLType::Q5_1, 32, 24),
            (GGMLType::Q8_0, 32, 34),
            (GGMLType::Q2K, 256, 84),
            (GGMLType::Q3K, 256, 110),
            (GGMLType::Q4K, 256, 144),
            (GGMLType::Q5K, 256, 176),
            (GGMLType::Q6K, 256, 210),
            (GGMLType::Q8K, 256, 292),
        ];
        for (ty, elements, bytes) in cases {
            let geometry = block_geometry(ty);
            assert_eq!(
                (geometry.elements_per_block, geometry.bytes_per_block),
                (*elements, *bytes),
                "wrong geometry for {ty:?}"
            );
        }
    }

    #[test]
    fn q4_0_places_low_nibbles_in_the_first_half_of_the_block() {
        // Regression: the previous implementation interleaved the nibbles as
        // (2j, 2j+1), which scrambles every Q4_0 tensor. ggml writes the low
        // nibble of qs[j] to element j and the high nibble to element j + 16.
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(2.0));
        // qs[0] = 0x9A -> low 0xA (=10) at element 0, high 0x9 (=9) at element 16.
        block.push(0x9A);
        block.extend_from_slice(&[0x88u8; 15]); // 8 -> 0 after the -8 bias

        let values = dequantize(&GGMLType::Q4_0, &block, 32).expect("Q4_0 must dequantize");
        assert_eq!(values[0], (10 - 8) as f32 * 2.0);
        assert_eq!(values[16], (9 - 8) as f32 * 2.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[17], 0.0);
    }

    #[test]
    fn q4_1_applies_scale_and_min_without_bias() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(0.5)); // d
        block.extend_from_slice(&f16_bytes(-1.0)); // m
        block.push(0x31); // low 1 -> element 0, high 3 -> element 16
        block.extend_from_slice(&[0x00u8; 15]);

        let values = dequantize(&GGMLType::Q4_1, &block, 32).expect("Q4_1 must dequantize");
        assert_eq!(values[0], 1.0 * 0.5 - 1.0);
        assert_eq!(values[16], 3.0 * 0.5 - 1.0);
        assert_eq!(values[1], -1.0);
    }

    #[test]
    fn q5_0_uses_the_fifth_bit_from_qh() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0)); // d
                                                  // qh bit 0 -> element 0; qh bit 16 -> element 16.
        block.extend_from_slice(&0x0001_0001u32.to_le_bytes());
        block.push(0x21); // low 1 -> element 0, high 2 -> element 16
        block.extend_from_slice(&[0x00u8; 15]);

        let values = dequantize(&GGMLType::Q5_0, &block, 32).expect("Q5_0 must dequantize");
        // element 0: (1 | 16) - 16 = 1
        assert_eq!(values[0], 1.0);
        // element 16: (2 | 16) - 16 = 2
        assert_eq!(values[16], 2.0);
        // element 1: (0 | 0) - 16 = -16
        assert_eq!(values[1], -16.0);
    }

    #[test]
    fn q5_1_adds_the_min_after_scaling() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(0.25)); // d
        block.extend_from_slice(&f16_bytes(2.0)); // m
        block.extend_from_slice(&0x0000_0001u32.to_le_bytes()); // fifth bit of element 0
        block.push(0x05); // low 5 -> element 0
        block.extend_from_slice(&[0x00u8; 15]);

        let values = dequantize(&GGMLType::Q5_1, &block, 32).expect("Q5_1 must dequantize");
        assert_eq!(values[0], (5 + 16) as f32 * 0.25 + 2.0);
        assert_eq!(values[1], 2.0);
    }

    #[test]
    fn q8_0_scales_signed_bytes() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(0.5));
        block.push(0x7F); // +127
        block.push(0x80); // -128
        block.extend_from_slice(&[0u8; 30]);
        let values = dequantize(&GGMLType::Q8_0, &block, 32).expect("Q8_0 must dequantize");
        assert_eq!(values[0], 127.0 * 0.5);
        assert_eq!(values[1], -128.0 * 0.5);
    }

    #[test]
    fn q2_k_recovers_hand_computed_values() {
        // scales[0] = 0x21 -> scale 1, min 2. scales[1] = 0x30 -> scale 0, min 3.
        let mut block = vec![0u8; 84];
        block[0] = 0x21;
        block[1] = 0x30;
        // qs[0] holds four 2-bit quants; the first sub-group uses shift 0 -> value 3.
        block[16] = 0b0000_0011;
        // The second sub-group of the first 32-byte stride is qs[16].
        block[16 + 16] = 0b0000_0010;
        block[80..82].copy_from_slice(&f16_bytes(2.0)); // d
        block[82..84].copy_from_slice(&f16_bytes(0.5)); // dmin

        let values = dequantize(&GGMLType::Q2K, &block, 256).expect("Q2_K must dequantize");
        // element 0: dl = 2 * 1 = 2, ml = 0.5 * 2 = 1 -> 2*3 - 1 = 5
        assert_eq!(values[0], 5.0);
        // element 1: quant 0 -> -1
        assert_eq!(values[1], -1.0);
        // element 16 belongs to the second sub-group: dl = 0, ml = 0.5*3 = 1.5
        assert_eq!(values[16], -1.5);
    }

    #[test]
    fn q3_k_applies_the_inverted_high_mask() {
        let mut block = vec![0u8; 110];
        // scales: 12 packed 6-bit values, all zero -> unpacked scale 0 -> (0 - 32).
        block[108..110].copy_from_slice(&f16_bytes(1.0)); // d
                                                          // qs[0] low 2 bits = 3, hmask bit 0 set -> no -4 correction.
        block[32] = 0b0000_0011;
        block[0] = 0b0000_0001;
        // element 1: quant 0, hmask bit clear -> (0 - 4)
        let values = dequantize(&GGMLType::Q3K, &block, 256).expect("Q3_K must dequantize");
        // dl = 1 * (0 - 32) = -32
        assert_eq!(values[0], -32.0 * 3.0);
        assert_eq!(values[1], -32.0 * -4.0);
    }

    #[test]
    fn q3_k_scale_unpacking_matches_the_reference_shuffle() {
        // Six-bit scales packed as ggml does: low four bytes hold scales 0..3 and
        // 8..11's low bits, byte 8..11 hold the high 2-bit parts.
        let packed: [u8; 12] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00, 0x00, 0x00, 0x00,
        ];
        let scales = unpack_q3_k_scales(&packed);
        // With the high-bit word zero, scales 0..7 are the low nibbles of the
        // first eight bytes and scales 8..15 are their high nibbles.
        assert_eq!(&scales[0..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&scales[8..16], &[0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn q4_k_recovers_hand_computed_values() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_bytes(1.0)); // d
        block[2..4].copy_from_slice(&f16_bytes(0.5)); // dmin
                                                      // scales[0] = 3 -> sc for group 0; scales[4] = 2 -> min for group 0.
        block[4] = 3;
        block[8] = 2;
        // scales[1] = 5 -> sc for group 1; scales[5] = 1 -> min for group 1.
        block[5] = 5;
        block[9] = 1;
        // qs[0]: low nibble 7 (group 0, element 0), high nibble 4 (group 1, element 32)
        block[16] = 0x47;

        let values = dequantize(&GGMLType::Q4K, &block, 256).expect("Q4_K must dequantize");
        assert_eq!(values[0], 1.0 * 3.0 * 7.0 - 0.5 * 2.0);
        assert_eq!(values[32], 1.0 * 5.0 * 4.0 - 0.5 * 1.0);
        assert_eq!(values[1], -1.0); // quant 0 in group 0 -> -min1
    }

    #[test]
    fn q5_k_adds_sixteen_when_the_high_bit_is_set() {
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16_bytes(1.0)); // d
        block[2..4].copy_from_slice(&f16_bytes(0.0)); // dmin -> mins vanish
        block[4] = 1; // scales[0] -> sc = 1 for group 0
        block[5] = 1; // scales[1] -> sc = 1 for group 1
        block[16] = 0b0000_0011; // qh[0]: bit0 (group 0) and bit1 (group 1) set
        block[48] = 0x21; // qs[0]: low 1 -> element 0, high 2 -> element 32

        let values = dequantize(&GGMLType::Q5K, &block, 256).expect("Q5_K must dequantize");
        assert_eq!(values[0], (1 + 16) as f32);
        assert_eq!(values[32], (2 + 16) as f32);
    }

    #[test]
    fn q6_k_recovers_hand_computed_values() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f16_bytes(1.0)); // d
        block[192] = 1; // scales[0] -> used by elements 0..15
        block[194] = 2; // scales[2] -> used by elements 32..47
        block[196] = 3; // scales[4] -> used by elements 64..79
        block[198] = 4; // scales[6] -> used by elements 96..111
        block[0] = 0x35; // ql[0]: low 5 -> element 0, high 3 -> element 64
        block[32] = 0x12; // ql[32]: low 2 -> element 32, high 1 -> element 96
        block[128] = 0b0100_0001; // qh[0]: bits 0..1 = 1, bits 6..7 = 1

        let values = dequantize(&GGMLType::Q6K, &block, 256).expect("Q6_K must dequantize");
        // element 0: (5 | (1 << 4)) - 32 = 21 - 32 = -11, scale 1
        assert_eq!(values[0], -11.0);
        // element 32: (2 | (0 << 4)) - 32 = -30, scale 2
        assert_eq!(values[32], 2.0 * -30.0);
        // element 64: (3 | (0 << 4)) - 32 = -29, scale 3
        assert_eq!(values[64], 3.0 * -29.0);
        // element 96: (1 | (1 << 4)) - 32 = 17 - 32 = -15, scale 4
        assert_eq!(values[96], 4.0 * -15.0);
    }

    #[test]
    fn q8_k_scales_signed_bytes_by_an_f32_delta() {
        let mut block = vec![0u8; 292];
        block[0..4].copy_from_slice(&0.25f32.to_le_bytes());
        block[4] = 0x10; // +16
        block[5] = 0xF0; // -16
        let values = dequantize(&GGMLType::Q8K, &block, 256).expect("Q8_K must dequantize");
        assert_eq!(values[0], 4.0);
        assert_eq!(values[1], -4.0);
    }

    #[test]
    fn unsupported_types_error_instead_of_returning_noise() {
        // The old `dequantize_generic_quantized` mapped every byte to
        // (byte - 128) / 128 and reported success. Anything it cannot decode must
        // now fail loudly.
        for ty in [
            GGMLType::Iq2Xxs,
            GGMLType::Iq2Xs,
            GGMLType::Iq3Xxs,
            GGMLType::Iq1S,
            GGMLType::Iq4Nl,
            GGMLType::Iq3S,
            GGMLType::Iq2S,
            GGMLType::Iq4Xs,
        ] {
            let data = vec![0x55u8; 4096];
            let err = dequantize(&ty, &data, 256)
                .expect_err("unimplemented quantization must not fabricate values");
            assert!(
                err.to_string().contains(&format!("{ty:?}")),
                "error should name the type, got: {err}"
            );
        }
    }

    #[test]
    fn short_payloads_error_instead_of_being_padded() {
        let data = vec![0u8; 10];
        let err = dequantize(&GGMLType::Q4K, &data, 256)
            .expect_err("a truncated block must not be silently zero-filled");
        assert!(err.to_string().contains("insufficient data"), "got: {err}");
    }

    #[test]
    fn trailing_partial_block_is_truncated_to_the_element_count() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16_bytes(1.0));
        block.extend_from_slice(&[0x88u8; 16]);
        let values = dequantize(&GGMLType::Q4_0, &block, 5).expect("Q4_0 must dequantize");
        assert_eq!(values.len(), 5);
    }
}
