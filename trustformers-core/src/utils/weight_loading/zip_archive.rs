//! A minimal, read-only ZIP reader for PyTorch checkpoint archives.
//!
//! `torch.save` writes a ZIP archive whose members are **stored** (compression
//! method 0) and padded so that each record's payload begins on a 64-byte
//! boundary. That is all this module needs to handle, so it implements exactly
//! that: locate the end-of-central-directory record, walk the central directory,
//! and resolve each entry's payload through its local file header.
//!
//! A deflate-compressed member produces a clear error rather than a silent wrong
//! answer — adding a decompressor here would mean taking on a banned dependency.
//!
//! ZIP64 is supported for the fields PyTorch actually uses (entry count, central
//! directory offset, and per-entry sizes/offsets), because checkpoints routinely
//! exceed 4 GiB.

use anyhow::{anyhow, Result};

const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4b50;
const CENTRAL_FILE_HEADER: u32 = 0x0201_4b50;
const LOCAL_FILE_HEADER: u32 = 0x0403_4b50;
const ZIP64_END_OF_CENTRAL_DIRECTORY: u32 = 0x0606_4b50;
const ZIP64_EOCD_LOCATOR: u32 = 0x0706_4b50;
const ZIP64_EXTRA_FIELD_ID: u16 = 0x0001;

/// Compression method 0: the member is stored verbatim.
const METHOD_STORED: u16 = 0;

/// One member of the archive.
#[derive(Debug, Clone)]
pub struct ZipEntry {
    /// Path as recorded in the archive, e.g. `archive/data/0`.
    pub name: String,
    /// Byte offset of the member's payload within the archive.
    pub data_offset: usize,
    /// Payload length in bytes.
    pub size: usize,
}

/// A parsed ZIP archive held in memory.
#[derive(Debug)]
pub struct ZipArchive<'a> {
    bytes: &'a [u8],
    entries: Vec<ZipEntry>,
}

impl<'a> ZipArchive<'a> {
    /// Parse the central directory of `bytes`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let eocd_offset = find_end_of_central_directory(bytes)?;
        let mut entry_count = read_u16(bytes, eocd_offset + 10)? as u64;
        let mut directory_offset = read_u32(bytes, eocd_offset + 16)? as u64;

        // ZIP64: the 32-bit fields are saturated and the real values live in the
        // ZIP64 end-of-central-directory record.
        if entry_count == 0xFFFF || directory_offset == 0xFFFF_FFFF {
            let (zip64_count, zip64_offset) = read_zip64_locator(bytes, eocd_offset)?;
            entry_count = zip64_count;
            directory_offset = zip64_offset;
        }

        let mut entries = Vec::with_capacity(entry_count.min(4096) as usize);
        let mut cursor = usize::try_from(directory_offset)
            .map_err(|_| anyhow!("zip: central directory offset does not fit in memory"))?;

        for index in 0..entry_count {
            if read_u32(bytes, cursor)? != CENTRAL_FILE_HEADER {
                return Err(anyhow!(
                    "zip: central directory entry {index} has a bad signature"
                ));
            }
            let method = read_u16(bytes, cursor + 10)?;
            let mut compressed_size = read_u32(bytes, cursor + 20)? as u64;
            let mut uncompressed_size = read_u32(bytes, cursor + 24)? as u64;
            let name_len = read_u16(bytes, cursor + 28)? as usize;
            let extra_len = read_u16(bytes, cursor + 30)? as usize;
            let comment_len = read_u16(bytes, cursor + 32)? as usize;
            let mut local_offset = read_u32(bytes, cursor + 42)? as u64;

            let name_start = cursor + 46;
            let name = std::str::from_utf8(slice(bytes, name_start, name_len)?)
                .map_err(|e| anyhow!("zip: entry name is not valid UTF-8: {e}"))?
                .to_string();

            let extra = slice(bytes, name_start + name_len, extra_len)?;
            read_zip64_extra(
                extra,
                &mut uncompressed_size,
                &mut compressed_size,
                &mut local_offset,
            )?;

            cursor = name_start + name_len + extra_len + comment_len;

            // Directory members are recorded with a trailing slash and no payload.
            if name.ends_with('/') {
                continue;
            }

            if method != METHOD_STORED {
                return Err(anyhow!(
                    "zip: member '{name}' uses compression method {method}, but this reader \
                     handles stored (uncompressed) members only. PyTorch writes uncompressed \
                     archives; re-save the checkpoint with `torch.save(..., _use_new_zipfile_\
                     serialization=True)` or convert it to safetensors."
                ));
            }

            let local_offset = usize::try_from(local_offset)
                .map_err(|_| anyhow!("zip: local header offset does not fit in memory"))?;
            if read_u32(bytes, local_offset)? != LOCAL_FILE_HEADER {
                return Err(anyhow!("zip: member '{name}' has a bad local header"));
            }
            let local_name_len = read_u16(bytes, local_offset + 26)? as usize;
            let local_extra_len = read_u16(bytes, local_offset + 28)? as usize;
            let data_offset = local_offset + 30 + local_name_len + local_extra_len;

            let size = usize::try_from(uncompressed_size)
                .map_err(|_| anyhow!("zip: member '{name}' is too large for this platform"))?;
            if data_offset + size > bytes.len() {
                return Err(anyhow!(
                    "zip: member '{name}' claims {size} bytes at offset {data_offset}, past the \
                     end of a {}-byte archive",
                    bytes.len()
                ));
            }

            entries.push(ZipEntry {
                name,
                data_offset,
                size,
            });
        }

        Ok(Self { bytes, entries })
    }

    /// All members, in central-directory order.
    pub fn entries(&self) -> &[ZipEntry] {
        &self.entries
    }

    /// The payload of a member selected by exact name.
    pub fn read(&self, name: &str) -> Result<&'a [u8]> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| anyhow!("zip: no member named '{name}'"))?;
        Ok(&self.bytes[entry.data_offset..entry.data_offset + entry.size])
    }

    /// The payload of the single member whose name ends with `suffix`.
    ///
    /// PyTorch prefixes every member with an archive name that varies with how the
    /// checkpoint was written, so members are located by suffix.
    pub fn read_suffix(&self, suffix: &str) -> Result<&'a [u8]> {
        let mut matches = self.entries.iter().filter(|entry| entry.name.ends_with(suffix));
        let entry = matches
            .next()
            .ok_or_else(|| anyhow!("zip: no member whose name ends with '{suffix}'"))?;
        if matches.next().is_some() {
            return Err(anyhow!(
                "zip: more than one member ends with '{suffix}'; the archive layout is not one \
                 this reader understands"
            ));
        }
        Ok(&self.bytes[entry.data_offset..entry.data_offset + entry.size])
    }

    /// Whether any member's name ends with `suffix`.
    pub fn has_suffix(&self, suffix: &str) -> bool {
        self.entries.iter().any(|entry| entry.name.ends_with(suffix))
    }
}

/// Whether `bytes` starts with a local file header, i.e. looks like a ZIP archive.
pub fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && read_u32(bytes, 0).map(|magic| magic == LOCAL_FILE_HEADER).unwrap_or(false)
}

fn find_end_of_central_directory(bytes: &[u8]) -> Result<usize> {
    if bytes.len() < 22 {
        return Err(anyhow!("zip: file is too short to be an archive"));
    }
    // The EOCD is at most 22 + 65535 bytes from the end (comment field).
    let search_start = bytes.len().saturating_sub(22 + 0xFFFF);
    for offset in (search_start..=bytes.len() - 22).rev() {
        if read_u32(bytes, offset)? == END_OF_CENTRAL_DIRECTORY {
            return Ok(offset);
        }
    }
    Err(anyhow!(
        "zip: no end-of-central-directory record; this is not a ZIP archive"
    ))
}

fn read_zip64_locator(bytes: &[u8], eocd_offset: usize) -> Result<(u64, u64)> {
    if eocd_offset < 20 {
        return Err(anyhow!("zip: ZIP64 locator is missing"));
    }
    let locator = eocd_offset - 20;
    if read_u32(bytes, locator)? != ZIP64_EOCD_LOCATOR {
        return Err(anyhow!("zip: ZIP64 locator has a bad signature"));
    }
    let zip64_eocd = usize::try_from(read_u64(bytes, locator + 8)?)
        .map_err(|_| anyhow!("zip: ZIP64 record offset does not fit in memory"))?;
    if read_u32(bytes, zip64_eocd)? != ZIP64_END_OF_CENTRAL_DIRECTORY {
        return Err(anyhow!(
            "zip: ZIP64 end-of-central-directory has a bad signature"
        ));
    }
    let entry_count = read_u64(bytes, zip64_eocd + 32)?;
    let directory_offset = read_u64(bytes, zip64_eocd + 48)?;
    Ok((entry_count, directory_offset))
}

/// Fill in whichever of the three fields were saturated, from the ZIP64 extra field.
fn read_zip64_extra(
    extra: &[u8],
    uncompressed_size: &mut u64,
    compressed_size: &mut u64,
    local_offset: &mut u64,
) -> Result<()> {
    let needs_uncompressed = *uncompressed_size == 0xFFFF_FFFF;
    let needs_compressed = *compressed_size == 0xFFFF_FFFF;
    let needs_offset = *local_offset == 0xFFFF_FFFF;
    if !(needs_uncompressed || needs_compressed || needs_offset) {
        return Ok(());
    }

    let mut cursor = 0usize;
    while cursor + 4 <= extra.len() {
        let field_id = read_u16(extra, cursor)?;
        let field_len = read_u16(extra, cursor + 2)? as usize;
        let payload_start = cursor + 4;
        if payload_start + field_len > extra.len() {
            return Err(anyhow!("zip: truncated extra field"));
        }
        if field_id == ZIP64_EXTRA_FIELD_ID {
            let mut position = payload_start;
            if needs_uncompressed {
                *uncompressed_size = read_u64(extra, position)?;
                position += 8;
            }
            if needs_compressed {
                *compressed_size = read_u64(extra, position)?;
                position += 8;
            }
            if needs_offset {
                *local_offset = read_u64(extra, position)?;
            }
            return Ok(());
        }
        cursor = payload_start + field_len;
    }

    Err(anyhow!(
        "zip: entry needs a ZIP64 extra field but none is present"
    ))
}

fn slice(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8]> {
    bytes
        .get(offset..offset + len)
        .ok_or_else(|| anyhow!("zip: read of {len} bytes at {offset} is out of bounds"))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let raw = slice(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = slice(bytes, offset, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw = slice(bytes, offset, 8)?;
    Ok(u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

// ---------------------------------------------------------------------------
// Writer used by tests to build fixtures
// ---------------------------------------------------------------------------

/// Build a stored-only ZIP archive.
///
/// This exists so the reader can be tested against archives built byte by byte,
/// rather than against archives produced by the reader's own assumptions.
#[derive(Debug, Default)]
pub struct StoredZipBuilder {
    entries: Vec<(String, Vec<u8>)>,
}

impl StoredZipBuilder {
    /// Start an empty archive.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a member.
    pub fn add(mut self, name: &str, data: Vec<u8>) -> Self {
        self.entries.push((name.to_string(), data));
        self
    }

    /// Serialise the archive.
    pub fn build(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut directory: Vec<u8> = Vec::new();

        for (name, data) in &self.entries {
            let local_offset = out.len() as u32;
            let crc = crc32(data);

            out.extend_from_slice(&LOCAL_FILE_HEADER.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes()); // version needed
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
            out.extend_from_slice(&METHOD_STORED.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // mod time
            out.extend_from_slice(&0u16.to_le_bytes()); // mod date
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra length
            out.extend_from_slice(name.as_bytes());
            let data_offset = out.len();
            out.extend_from_slice(data);

            directory.extend_from_slice(&CENTRAL_FILE_HEADER.to_le_bytes());
            directory.extend_from_slice(&20u16.to_le_bytes()); // version made by
            directory.extend_from_slice(&20u16.to_le_bytes()); // version needed
            directory.extend_from_slice(&0u16.to_le_bytes()); // flags
            directory.extend_from_slice(&METHOD_STORED.to_le_bytes());
            directory.extend_from_slice(&0u16.to_le_bytes());
            directory.extend_from_slice(&0u16.to_le_bytes());
            directory.extend_from_slice(&crc.to_le_bytes());
            directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
            directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
            directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
            directory.extend_from_slice(&0u16.to_le_bytes()); // extra
            directory.extend_from_slice(&0u16.to_le_bytes()); // comment
            directory.extend_from_slice(&0u16.to_le_bytes()); // disk number
            directory.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
            directory.extend_from_slice(&0u32.to_le_bytes()); // external attrs
            directory.extend_from_slice(&local_offset.to_le_bytes());
            directory.extend_from_slice(name.as_bytes());

            debug_assert!(data_offset <= out.len());
        }

        let directory_offset = out.len() as u32;
        let directory_size = directory.len() as u32;
        out.extend_from_slice(&directory);

        out.extend_from_slice(&END_OF_CENTRAL_DIRECTORY.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0u16.to_le_bytes()); // start disk
        out.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&directory_size.to_le_bytes());
        out.extend_from_slice(&directory_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment length

        out
    }

    /// Serialise the archive, marking `deflated_member` as deflate-compressed.
    ///
    /// The payload is still stored verbatim; the point is to produce an archive the
    /// reader must refuse.
    pub fn build_claiming_deflate(&self, deflated_member: &str) -> Vec<u8> {
        let mut bytes = self.build();
        // Patch the compression method in both the local and the central header.
        let target = deflated_member.as_bytes();
        let mut index = 0usize;
        while index + 30 <= bytes.len() {
            if bytes[index..index + 4] == LOCAL_FILE_HEADER.to_le_bytes() {
                let name_len = u16::from_le_bytes([bytes[index + 26], bytes[index + 27]]) as usize;
                if bytes[index + 30..index + 30 + name_len] == *target {
                    bytes[index + 8] = 8;
                    bytes[index + 9] = 0;
                }
            } else if bytes[index..index + 4] == CENTRAL_FILE_HEADER.to_le_bytes() {
                let name_len = u16::from_le_bytes([bytes[index + 28], bytes[index + 29]]) as usize;
                if bytes[index + 46..index + 46 + name_len] == *target {
                    bytes[index + 10] = 8;
                    bytes[index + 11] = 0;
                }
            }
            index += 1;
        }
        bytes
    }
}

/// CRC-32 (IEEE) over `data`.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_hand_built_archive() {
        let bytes = StoredZipBuilder::new()
            .add("archive/data.pkl", b"pickle bytes".to_vec())
            .add("archive/data/0", vec![1, 2, 3, 4])
            .add("archive/version", b"3\n".to_vec())
            .build();

        assert!(looks_like_zip(&bytes));
        let archive = ZipArchive::parse(&bytes).expect("parse");
        assert_eq!(archive.entries().len(), 3);
        assert_eq!(
            archive.read("archive/data/0").expect("member"),
            &[1, 2, 3, 4]
        );
        assert_eq!(
            archive.read_suffix("data.pkl").expect("suffix"),
            b"pickle bytes"
        );
        assert!(archive.has_suffix("version"));
        assert!(archive.read("missing").is_err());
    }

    #[test]
    fn deflated_members_are_refused_with_a_clear_message() {
        let bytes = StoredZipBuilder::new()
            .add("archive/data.pkl", b"pickle".to_vec())
            .build_claiming_deflate("archive/data.pkl");

        let err = ZipArchive::parse(&bytes).expect_err("deflate must be refused");
        let message = err.to_string();
        assert!(message.contains("compression method 8"), "{message}");
        assert!(message.contains("stored"), "{message}");
    }

    #[test]
    fn non_zip_bytes_are_rejected() {
        let err = ZipArchive::parse(b"\x80\x02}q\x00.").expect_err("not a zip");
        assert!(err.to_string().contains("too short") || err.to_string().contains("not a ZIP"));
        assert!(!looks_like_zip(b"\x80\x02}q\x00."));
    }

    #[test]
    fn truncated_archives_are_rejected() {
        let mut bytes = StoredZipBuilder::new().add("archive/data/0", vec![7u8; 128]).build();
        // Corrupt the recorded uncompressed size so the payload runs off the end.
        let directory_offset = bytes.len() - 22 + 16;
        let directory_start = u32::from_le_bytes([
            bytes[directory_offset],
            bytes[directory_offset + 1],
            bytes[directory_offset + 2],
            bytes[directory_offset + 3],
        ]) as usize;
        bytes[directory_start + 24..directory_start + 28]
            .copy_from_slice(&(1_000_000u32).to_le_bytes());

        let err = ZipArchive::parse(&bytes).expect_err("truncated");
        assert!(err.to_string().contains("past the end"), "{err}");
    }

    #[test]
    fn crc32_matches_the_known_check_value() {
        // The IEEE CRC-32 of "123456789" is 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
