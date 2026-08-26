//! # Real audio DSP for the speech / audio pipelines
//!
//! Pure-Rust, dependency-honest implementations of the signal processing the
//! speech and audio pipelines need:
//!
//! 1. [`decode_wav`] — a real RIFF/WAVE container parser (PCM 8/16/24/32-bit
//!    integer and IEEE 32/64-bit float, any channel count, downmixed to mono).
//! 2. [`resample_linear`] — sample-rate conversion by linear interpolation.
//! 3. [`stft_power`] — Hann-windowed short-time Fourier transform, power
//!    spectrum, computed with the pure-Rust `oxifft` FFT.
//! 4. [`mel_filterbank`] — a Slaney-scale triangular mel filterbank
//!    (`librosa.filters.mel(htk=False, norm="slaney")`).
//! 5. [`log_mel_spectrogram`] — the Whisper feature front-end: power STFT →
//!    mel projection → `log10` → 8 dB dynamic-range clamp → `(x + 4) / 4`.
//!
//! ## Conventions (pinned deliberately)
//!
//! - **Mel scale**: Slaney (`f_sp = 200/3` Hz per mel below 1 kHz, logarithmic
//!   above with `logstep = ln(6.4) / 27`). This is what `librosa` — and hence
//!   Whisper's reference feature extractor — uses. HTK's `2595·log10(1+f/700)`
//!   is *not* used; the two disagree materially about filter centres.
//! - **Filter normalisation**: Slaney (`2 / (f[i+2] - f[i])`), so each triangle
//!   has unit area rather than unit peak.
//! - **Window**: periodic Hann, `w[i] = 0.5 - 0.5·cos(2πi/N)` — matching
//!   `torch.hann_window(N, periodic=True)`.
//! - **Framing**: `center = true` reflect-pads the signal by `n_fft / 2` on
//!   both sides, so frame `t` is centred on sample `t · hop`. The number of
//!   frames is then `1 + len / hop`; Whisper drops the final frame, which
//!   [`log_mel_spectrogram`] does too, yielding exactly `len / hop` frames.
//!
//! Every function here either computes the real quantity or returns a
//! structured error. Nothing returns zeros, silence, or synthesised audio.

use crate::error::{Result, TrustformersError};

/// Decoded, mono-mixed audio.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Mono samples in `[-1, 1]` (integer formats are scaled by their full range).
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels present in the source container (before downmixing).
    pub source_channels: u16,
}

impl DecodedAudio {
    /// Duration of the decoded audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / f64::from(self.sample_rate)
        }
    }
}

// ---------------------------------------------------------------------------
// WAV (RIFF/WAVE) decoding
// ---------------------------------------------------------------------------

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16> {
    bytes
        .get(offset..offset + 2)
        .and_then(|s| <[u8; 2]>::try_from(s).ok())
        .map(u16::from_le_bytes)
        .ok_or_else(|| {
            TrustformersError::invalid_input_simple(format!(
                "WAV: truncated while reading u16 at offset {offset}"
            ))
        })
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| {
            TrustformersError::invalid_input_simple(format!(
                "WAV: truncated while reading u32 at offset {offset}"
            ))
        })
}

/// Returns `true` when `bytes` starts with a RIFF/WAVE container header.
///
/// Used to sniff base64 / raw byte payloads instead of *assuming* a format.
pub fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// Decode a RIFF/WAVE byte stream into mono `f32` samples.
///
/// Supported sample formats: `WAVE_FORMAT_PCM` with 8, 16, 24 or 32 bits per
/// sample, and `WAVE_FORMAT_IEEE_FLOAT` with 32 or 64 bits per sample.
/// `WAVE_FORMAT_EXTENSIBLE` is accepted and resolved through its sub-format
/// GUID's first two bytes.
///
/// Multi-channel audio is downmixed to mono by averaging the channels, which is
/// what every speech front-end in this crate expects.
///
/// # Errors
///
/// Returns [`TrustformersError::InvalidInput`] if the container is malformed,
/// truncated, or uses an unsupported codec (e.g. an ADPCM or MP3-in-WAV
/// payload). It never falls back to silence.
pub fn decode_wav(bytes: &[u8]) -> Result<DecodedAudio> {
    if !is_wav(bytes) {
        return Err(TrustformersError::invalid_input_simple(
            "not a RIFF/WAVE stream: missing 'RIFF'…'WAVE' magic".to_string(),
        ));
    }

    let mut offset = 12usize;
    let mut format_tag: Option<u16> = None;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data: Option<&[u8]> = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = read_u32_le(bytes, offset + 4)? as usize;
        let body_start = offset + 8;
        let body_end = body_start.saturating_add(chunk_size).min(bytes.len());
        let body = &bytes[body_start..body_end];

        if chunk_id == b"fmt " {
            if body.len() < 16 {
                return Err(TrustformersError::invalid_input_simple(
                    "WAV: 'fmt ' chunk shorter than 16 bytes".to_string(),
                ));
            }
            let tag = read_u16_le(body, 0)?;
            channels = read_u16_le(body, 2)?;
            sample_rate = read_u32_le(body, 4)?;
            bits_per_sample = read_u16_le(body, 14)?;
            format_tag = Some(if tag == WAVE_FORMAT_EXTENSIBLE {
                // The sub-format GUID starts at byte 24 of the extended fmt chunk;
                // its first two bytes carry the real format tag.
                if body.len() >= 26 {
                    read_u16_le(body, 24)?
                } else {
                    return Err(TrustformersError::invalid_input_simple(
                        "WAV: WAVE_FORMAT_EXTENSIBLE without a sub-format GUID".to_string(),
                    ));
                }
            } else {
                tag
            });
        } else if chunk_id == b"data" {
            data = Some(body);
        }

        // Chunks are word-aligned: an odd size is followed by one pad byte.
        offset = body_start + chunk_size + usize::from(chunk_size % 2 == 1);
    }

    let format_tag = format_tag.ok_or_else(|| {
        TrustformersError::invalid_input_simple("WAV: no 'fmt ' chunk found".to_string())
    })?;
    let data = data.ok_or_else(|| {
        TrustformersError::invalid_input_simple("WAV: no 'data' chunk found".to_string())
    })?;

    if channels == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "WAV: channel count is zero".to_string(),
        ));
    }
    if sample_rate == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "WAV: sample rate is zero".to_string(),
        ));
    }

    let interleaved: Vec<f32> = match (format_tag, bits_per_sample) {
        (WAVE_FORMAT_PCM, 8) => data.iter().map(|&b| (f32::from(b) - 128.0) / 128.0).collect(),
        (WAVE_FORMAT_PCM, 16) => data
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect(),
        (WAVE_FORMAT_PCM, 24) => data
            .chunks_exact(3)
            .map(|c| {
                // Sign-extend the 24-bit little-endian sample into an i32.
                let raw = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                raw as f32 / 8_388_608.0
            })
            .collect(),
        (WAVE_FORMAT_PCM, 32) => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        (WAVE_FORMAT_IEEE_FLOAT, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (WAVE_FORMAT_IEEE_FLOAT, 64) => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        (tag, bits) => {
            return Err(TrustformersError::invalid_input_simple(format!(
                "WAV: unsupported sample format (format tag {tag:#06x}, {bits} bits per sample); \
                 supported: PCM 8/16/24/32-bit and IEEE float 32/64-bit"
            )))
        },
    };

    let channels_usize = usize::from(channels);
    let samples: Vec<f32> = if channels_usize == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels_usize)
            .map(|frame| frame.iter().sum::<f32>() / channels_usize as f32)
            .collect()
    };

    Ok(DecodedAudio {
        samples,
        sample_rate,
        source_channels: channels,
    })
}

/// Encode mono `f32` samples as a 16-bit PCM RIFF/WAVE byte stream.
///
/// This is the exact inverse of the 16-bit branch of [`decode_wav`] (up to
/// quantisation), and is used both by callers that need to hand a WAV buffer to
/// another process and by the round-trip tests in this module.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let q = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Resample `samples` from `from_rate` to `to_rate` by linear interpolation.
///
/// Output length is `round(len · to_rate / from_rate)`. Linear interpolation is
/// a real (if modest) resampler: it preserves low-frequency content exactly
/// enough for speech front-ends, but it does not band-limit, so downsampling
/// content above the new Nyquist frequency will alias. That limitation is
/// documented rather than hidden.
///
/// # Errors
///
/// Returns an error when either rate is zero.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == 0 || to_rate == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "resample: sample rates must be non-zero".to_string(),
        ));
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if samples.len() == 1 {
        return Ok(samples.to_vec());
    }

    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = samples.len() - 1;

    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        if idx >= last {
            out.push(samples[last]);
        } else {
            let frac = (src - idx as f64) as f32;
            out.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Windowing / STFT
// ---------------------------------------------------------------------------

/// Periodic Hann window of length `n`: `w[i] = 0.5 - 0.5·cos(2πi/n)`.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad a signal by `pad` samples on each side (`numpy` "reflect" mode).
///
/// Requires `pad < signal.len()`; shorter signals are zero-extended first so the
/// reflection is always well defined.
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return signal.to_vec();
    }
    let mut base = signal.to_vec();
    if base.len() <= pad {
        base.resize(pad + 1, 0.0);
    }
    let n = base.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(base[i]);
    }
    out.extend_from_slice(&base);
    for i in 1..=pad {
        out.push(base[n - 1 - i]);
    }
    out
}

/// Hann-windowed STFT power spectrum.
///
/// Returns `frames × (n_fft/2 + 1)` power values (`|X|²`). When `center` is
/// true the signal is reflect-padded by `n_fft / 2`, so frame `t` is centred on
/// input sample `t · hop_length` and the frame count is `1 + len / hop_length`.
///
/// # Errors
///
/// Returns an error if `n_fft` or `hop_length` is zero.
pub fn stft_power(
    samples: &[f32],
    n_fft: usize,
    hop_length: usize,
    center: bool,
) -> Result<Vec<Vec<f32>>> {
    if n_fft == 0 || hop_length == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "stft: n_fft and hop_length must be > 0".to_string(),
        ));
    }

    let padded = if center { reflect_pad(samples, n_fft / 2) } else { samples.to_vec() };
    if padded.len() < n_fft {
        return Ok(Vec::new());
    }

    let window = hann_window(n_fft);
    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let n_bins = n_fft / 2 + 1;
    let mut frames = Vec::with_capacity(n_frames);
    let mut buffer = vec![0.0f32; n_fft];

    for t in 0..n_frames {
        let start = t * hop_length;
        for (i, slot) in buffer.iter_mut().enumerate() {
            *slot = padded[start + i] * window[i];
        }
        let spectrum = oxifft::rfft(&buffer);
        debug_assert_eq!(spectrum.len(), n_bins);
        let mut power = Vec::with_capacity(n_bins);
        for bin in spectrum.iter().take(n_bins) {
            power.push(bin.norm_sqr());
        }
        frames.push(power);
    }

    Ok(frames)
}

// ---------------------------------------------------------------------------
// Mel scale / filterbank
// ---------------------------------------------------------------------------

/// Slaney mel-scale linear region: Hz per mel below 1 kHz.
const SLANEY_F_SP: f64 = 200.0 / 3.0;
/// Slaney mel-scale break point in Hz.
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
/// Mel value at the break point (`1000 / f_sp`).
const SLANEY_MIN_LOG_MEL: f64 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;

fn slaney_logstep() -> f64 {
    (6.4f64).ln() / 27.0
}

/// Convert a frequency in Hz to the Slaney mel scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    if hz < SLANEY_MIN_LOG_HZ {
        hz / SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / slaney_logstep()
    }
}

/// Convert a Slaney mel value back to Hz.
pub fn mel_to_hz(mel: f64) -> f64 {
    if mel < SLANEY_MIN_LOG_MEL {
        mel * SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_HZ * ((mel - SLANEY_MIN_LOG_MEL) * slaney_logstep()).exp()
    }
}

/// Build a Slaney-normalised triangular mel filterbank.
///
/// Returns `n_mels × (n_fft/2 + 1)` non-negative weights. Filter `m` is a
/// triangle rising from `mel_f[m]` to its peak at `mel_f[m+1]` and falling to
/// `mel_f[m+2]`, scaled by `2 / (mel_f[m+2] - mel_f[m])` so each filter has
/// unit area (Slaney normalisation, `librosa`'s `norm="slaney"`).
///
/// # Errors
///
/// Returns an error when `n_mels` or `n_fft` is zero, or when
/// `f_min >= f_max`.
pub fn mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
) -> Result<Vec<Vec<f32>>> {
    if n_mels == 0 || n_fft == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "mel_filterbank: n_mels and n_fft must be > 0".to_string(),
        ));
    }
    if f_min >= f_max {
        return Err(TrustformersError::invalid_input_simple(format!(
            "mel_filterbank: f_min ({f_min}) must be < f_max ({f_max})"
        )));
    }

    let n_bins = n_fft / 2 + 1;
    let nyquist = f64::from(sample_rate) / 2.0;
    let fft_freqs: Vec<f64> =
        (0..n_bins).map(|k| nyquist * k as f64 / (n_bins - 1).max(1) as f64).collect();

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let mut bank = vec![vec![0.0f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let left_width = (center - left).max(f64::EPSILON);
        let right_width = (right - center).max(f64::EPSILON);
        let enorm = 2.0 / (right - left).max(f64::EPSILON);

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / left_width;
            let upper = (right - freq) / right_width;
            let weight = lower.min(upper).max(0.0) * enorm;
            bank[m][k] = weight as f32;
        }
    }

    Ok(bank)
}

/// Apply a mel filterbank to a power spectrogram.
///
/// `power` is `frames × bins`; `bank` is `n_mels × bins`. Returns
/// `frames × n_mels`.
///
/// # Errors
///
/// Returns an error when the bin counts disagree.
pub fn apply_mel_filterbank(power: &[Vec<f32>], bank: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    if bank.is_empty() {
        return Err(TrustformersError::invalid_input_simple(
            "apply_mel_filterbank: empty filterbank".to_string(),
        ));
    }
    let n_bins = bank[0].len();
    let mut out = Vec::with_capacity(power.len());
    for frame in power {
        if frame.len() != n_bins {
            return Err(TrustformersError::invalid_input_simple(format!(
                "apply_mel_filterbank: spectrogram has {} bins but the filterbank expects {}",
                frame.len(),
                n_bins
            )));
        }
        let mut mels = Vec::with_capacity(bank.len());
        for filter in bank {
            let mut acc = 0.0f32;
            for (w, p) in filter.iter().zip(frame.iter()) {
                acc += w * p;
            }
            mels.push(acc);
        }
        out.push(mels);
    }
    Ok(out)
}

/// Configuration for [`log_mel_spectrogram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelConfig {
    /// FFT size (Whisper: 400).
    pub n_fft: usize,
    /// Hop between frames in samples (Whisper: 160).
    pub hop_length: usize,
    /// Number of mel bands (Whisper: 80).
    pub n_mels: usize,
}

impl Default for MelConfig {
    /// Whisper's front-end configuration: 400-point FFT, 160-sample hop, 80 mels.
    fn default() -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
        }
    }
}

/// Compute a Whisper-style log-mel spectrogram.
///
/// Pipeline: reflect-padded Hann STFT → power spectrum → Slaney mel filterbank
/// (0 Hz … Nyquist) → `log10(max(mel, 1e-10))` → clamp to 8 units below the
/// maximum → affine rescale `(x + 4) / 4`.
///
/// Returns `frames × n_mels` where `frames = samples.len() / hop_length`
/// (Whisper discards the final STFT frame).
///
/// # Errors
///
/// Propagates errors from [`stft_power`] and [`mel_filterbank`]; returns an
/// error when `samples` is empty.
pub fn log_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    config: MelConfig,
) -> Result<Vec<Vec<f32>>> {
    if samples.is_empty() {
        return Err(TrustformersError::invalid_input_simple(
            "log_mel_spectrogram: empty audio buffer".to_string(),
        ));
    }

    let mut power = stft_power(samples, config.n_fft, config.hop_length, true)?;
    // Whisper: `magnitudes = stft[..., :-1].abs() ** 2`
    if !power.is_empty() {
        power.pop();
    }
    if power.is_empty() {
        return Err(TrustformersError::invalid_input_simple(format!(
            "log_mel_spectrogram: audio too short ({} samples) for n_fft={} / hop={}",
            samples.len(),
            config.n_fft,
            config.hop_length
        )));
    }

    let bank = mel_filterbank(
        sample_rate,
        config.n_fft,
        config.n_mels,
        0.0,
        f64::from(sample_rate) / 2.0,
    )?;
    let mel = apply_mel_filterbank(&power, &bank)?;

    let mut log_spec: Vec<Vec<f32>> = mel
        .into_iter()
        .map(|frame| frame.into_iter().map(|v| v.max(1e-10).log10()).collect())
        .collect();

    let max_val = log_spec
        .iter()
        .flat_map(|f| f.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let floor = max_val - 8.0;
    for frame in &mut log_spec {
        for v in frame.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
    }

    Ok(log_spec)
}

/// Root-mean-square energy of a signal.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    // ---- WAV decoding ----

    #[test]
    fn decode_wav_rejects_non_riff_bytes() {
        // Regression: the old pipelines "decoded" `vec![0u8; 512]` into silence
        // and reported success. A real parser must reject it.
        let err = decode_wav(&[0u8; 512]).expect_err("512 zero bytes are not a WAV file");
        assert!(err.to_string().contains("RIFF"), "err: {err}");
    }

    #[test]
    fn decode_wav_round_trips_pcm16() {
        let samples = sine(440.0, 16_000, 512);
        let bytes = encode_wav_pcm16(&samples, 16_000);
        let decoded = decode_wav(&bytes).expect("round-trip decode");
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.source_channels, 1);
        assert_eq!(decoded.samples.len(), samples.len());
        for (a, b) in decoded.samples.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-3, "sample mismatch: {a} vs {b}");
        }
        assert!(
            decoded.samples.iter().any(|&s| s.abs() > 0.5),
            "decoded audio must not be silence"
        );
    }

    #[test]
    fn decode_wav_downmixes_stereo() {
        // Hand-built stereo WAV: L = +1.0, R = -1.0 → mono average 0.0.
        let mut bytes = Vec::new();
        let data: Vec<u8> = [i16::MAX, i16::MIN, i16::MAX, i16::MIN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // stereo
        bytes.extend_from_slice(&8000u32.to_le_bytes());
        bytes.extend_from_slice(&32000u32.to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&data);

        let decoded = decode_wav(&bytes).expect("stereo decode");
        assert_eq!(decoded.source_channels, 2);
        assert_eq!(decoded.samples.len(), 2);
        for s in &decoded.samples {
            assert!(s.abs() < 1e-4, "L/R cancellation expected, got {s}");
        }
    }

    #[test]
    fn decode_wav_rejects_unsupported_codec() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&36u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&0x0011u16.to_le_bytes()); // IMA ADPCM
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8000u32.to_le_bytes());
        bytes.extend_from_slice(&4000u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let err = decode_wav(&bytes).expect_err("ADPCM is not supported");
        assert!(err.to_string().contains("unsupported"), "err: {err}");
    }

    #[test]
    fn is_wav_sniffs_magic() {
        assert!(is_wav(&encode_wav_pcm16(&[0.1, -0.1], 8000)));
        assert!(!is_wav(b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00"));
        assert!(!is_wav(&[]));
    }

    // ---- Resampling ----

    #[test]
    fn resample_preserves_length_ratio() {
        let input = sine(100.0, 8_000, 800);
        let out = resample_linear(&input, 8_000, 16_000).expect("resample");
        assert_eq!(out.len(), 1600);
    }

    #[test]
    fn resample_interpolates_linearly() {
        // Hand-computed: doubling the rate of [0, 1, 2, 3] samples the source at
        // 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 → 0, 0.5, 1, 1.5, 2, 2.5, 3, 3 (clamped).
        let out = resample_linear(&[0.0, 1.0, 2.0, 3.0], 1000, 2000).expect("resample");
        let expected = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0];
        assert_eq!(out.len(), expected.len());
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn resample_preserves_tone_frequency() {
        // A 200 Hz tone resampled 8 k → 16 k must still peak at 200 Hz.
        let input = sine(200.0, 8_000, 4_096);
        let out = resample_linear(&input, 8_000, 16_000).expect("resample");
        let spectrum = oxifft::rfft(&out[..4096]);
        let peak = spectrum
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| {
                a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .expect("non-empty spectrum");
        // bin width = 16000 / 4096 = 3.90625 Hz → 200 Hz ≈ bin 51.2
        assert!(
            (51i64 - peak as i64).abs() <= 1,
            "peak bin {peak}, expected ~51"
        );
    }

    #[test]
    fn resample_rejects_zero_rate() {
        assert!(resample_linear(&[0.0], 0, 16_000).is_err());
        assert!(resample_linear(&[0.0], 16_000, 0).is_err());
    }

    // ---- Window / STFT ----

    #[test]
    fn hann_window_is_periodic_and_symmetric_about_centre() {
        let w = hann_window(8);
        assert!((w[0] - 0.0).abs() < 1e-6, "w[0] = {}", w[0]);
        assert!((w[4] - 1.0).abs() < 1e-6, "w[4] = {}", w[4]);
        for i in 1..4 {
            assert!((w[i] - w[8 - i]).abs() < 1e-6, "asymmetry at {i}");
        }
    }

    #[test]
    fn stft_matches_a_direct_dft() {
        // Guards against the FFT silently producing zeros: compare one frame of
        // the STFT against a naive O(n²) DFT of the same windowed block.
        let signal: Vec<f32> = (0..64).map(|i| ((i * 7) % 13) as f32 / 13.0 - 0.5).collect();
        let n_fft = 16;
        let frames = stft_power(&signal, n_fft, n_fft, false).expect("stft");
        assert!(!frames.is_empty());

        let window = hann_window(n_fft);
        for (t, frame) in frames.iter().enumerate() {
            for (k, &got) in frame.iter().enumerate() {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for n in 0..n_fft {
                    let x = f64::from(signal[t * n_fft + n] * window[n]);
                    let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                    re += x * angle.cos();
                    im += x * angle.sin();
                }
                let expected = (re * re + im * im) as f32;
                assert!(
                    (got - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "frame {t} bin {k}: got {got}, dft {expected}"
                );
            }
        }
    }

    #[test]
    fn stft_frame_count_matches_centering_convention() {
        let signal = vec![0.25f32; 1600];
        let frames = stft_power(&signal, 400, 160, true).expect("stft");
        // centred: padded length 1600 + 400 → (2000 - 400) / 160 + 1 = 11
        assert_eq!(frames.len(), 11);
        assert_eq!(frames[0].len(), 201);
    }

    #[test]
    fn stft_rejects_zero_parameters() {
        assert!(stft_power(&[0.0; 10], 0, 1, false).is_err());
        assert!(stft_power(&[0.0; 10], 4, 0, false).is_err());
    }

    // ---- Mel scale ----

    #[test]
    fn mel_scale_round_trips() {
        for hz in [0.0, 100.0, 500.0, 999.0, 1000.0, 2000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-6, "{hz} -> {back}");
        }
    }

    #[test]
    fn mel_scale_is_linear_below_the_break_point() {
        // Slaney: 200/3 Hz per mel below 1 kHz, so 1000 Hz == 15 mel exactly.
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        assert!((hz_to_mel(500.0) - 7.5).abs() < 1e-9);
        assert!((mel_to_hz(7.5) - 500.0).abs() < 1e-9);
    }

    #[test]
    fn mel_filterbank_is_nonnegative_and_triangular() {
        let bank = mel_filterbank(16_000, 400, 80, 0.0, 8000.0).expect("bank");
        assert_eq!(bank.len(), 80);
        assert_eq!(bank[0].len(), 201);
        for (m, filter) in bank.iter().enumerate() {
            assert!(
                filter.iter().all(|&w| w >= 0.0),
                "negative weight in filter {m}"
            );
            assert!(
                filter.iter().any(|&w| w > 0.0),
                "filter {m} is entirely zero"
            );
            // Support must be a single contiguous run (triangular).
            let support: Vec<usize> =
                filter.iter().enumerate().filter(|(_, &w)| w > 0.0).map(|(i, _)| i).collect();
            for pair in support.windows(2) {
                assert_eq!(pair[1], pair[0] + 1, "filter {m} support is not contiguous");
            }
            // Weights rise then fall (single local maximum).
            let peak = support
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    filter[a].partial_cmp(&filter[b]).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("non-empty support");
            for w in support.windows(2) {
                if w[1] <= peak {
                    assert!(filter[w[0]] <= filter[w[1]] + 1e-6, "filter {m} not rising");
                } else {
                    assert!(
                        filter[w[0]] >= filter[w[1]] - 1e-6,
                        "filter {m} not falling"
                    );
                }
            }
        }
    }

    #[test]
    fn mel_filterbank_centres_ascend_and_bracket_the_range() {
        let bank = mel_filterbank(16_000, 400, 20, 0.0, 8000.0).expect("bank");
        let mut previous_peak = 0usize;
        for filter in &bank {
            let peak = filter
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .expect("non-empty filter");
            assert!(peak >= previous_peak, "filter centres must ascend");
            previous_peak = peak;
        }
        assert!(
            previous_peak <= 200,
            "last centre must stay below Nyquist bin"
        );
    }

    #[test]
    fn mel_filterbank_rejects_bad_ranges() {
        assert!(mel_filterbank(16_000, 400, 0, 0.0, 8000.0).is_err());
        assert!(mel_filterbank(16_000, 400, 80, 4000.0, 1000.0).is_err());
    }

    // ---- Log-mel ----

    #[test]
    fn log_mel_locates_a_1khz_tone() {
        // A pure 1 kHz tone must put its energy in the mel band that brackets
        // 1 kHz. This fails against all-zero "features" and against an
        // HTK-scaled filterbank (whose centres sit elsewhere).
        let samples = sine(1000.0, 16_000, 16_000);
        let mel = log_mel_spectrogram(&samples, 16_000, MelConfig::default()).expect("mel");
        assert_eq!(mel.len(), 100, "16000 samples / hop 160");
        assert_eq!(mel[0].len(), 80);

        // Average over the middle frames to avoid the padded edges.
        let mut avg = [0.0f32; 80];
        for frame in &mel[20..80] {
            for (a, v) in avg.iter_mut().zip(frame.iter()) {
                *a += v / 60.0;
            }
        }
        let peak = avg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("non-empty mel frame");

        // Independently locate the band whose filter peaks nearest 1 kHz.
        let bank = mel_filterbank(16_000, 400, 80, 0.0, 8000.0).expect("bank");
        let expected = bank
            .iter()
            .enumerate()
            .map(|(m, f)| {
                let centre_bin = f
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (m, (centre_bin as f64 * 8000.0 / 200.0 - 1000.0).abs())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(m, _)| m)
            .expect("non-empty bank");

        assert!(
            (peak as i64 - expected as i64).abs() <= 2,
            "1 kHz energy landed in mel band {peak}, expected ~{expected}"
        );
    }

    #[test]
    fn log_mel_distinguishes_different_tones() {
        let low = log_mel_spectrogram(&sine(220.0, 16_000, 16_000), 16_000, MelConfig::default())
            .expect("mel low");
        let high = log_mel_spectrogram(&sine(3000.0, 16_000, 16_000), 16_000, MelConfig::default())
            .expect("mel high");
        let band = |m: &[Vec<f32>]| {
            let mut avg = [0.0f32; 80];
            for frame in &m[20..80] {
                for (a, v) in avg.iter_mut().zip(frame.iter()) {
                    *a += v / 60.0;
                }
            }
            avg.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        assert!(
            band(&low) < band(&high),
            "220 Hz must peak in a lower mel band than 3 kHz"
        );
    }

    #[test]
    fn log_mel_output_is_not_uniform() {
        let samples = sine(440.0, 16_000, 8_000);
        let mel = log_mel_spectrogram(&samples, 16_000, MelConfig::default()).expect("mel");
        let flat: Vec<f32> = mel.iter().flat_map(|f| f.iter().copied()).collect();
        let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
        let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 0.5,
            "log-mel dynamic range too small: {min}..{max}"
        );
        assert!(
            flat.iter().any(|&v| v != 0.0),
            "log-mel must not be all zeros"
        );
    }

    #[test]
    fn log_mel_rejects_empty_audio() {
        assert!(log_mel_spectrogram(&[], 16_000, MelConfig::default()).is_err());
    }

    #[test]
    fn apply_mel_filterbank_rejects_bin_mismatch() {
        let bank = vec![vec![1.0f32; 4]];
        let power = vec![vec![1.0f32; 5]];
        assert!(apply_mel_filterbank(&power, &bank).is_err());
    }

    #[test]
    fn rms_matches_hand_computation() {
        let samples = [3.0f32, 4.0];
        // sqrt((9 + 16) / 2) = sqrt(12.5)
        assert!((rms(&samples) - 12.5f32.sqrt()).abs() < 1e-6);
        assert_eq!(rms(&[]), 0.0);
    }
}
