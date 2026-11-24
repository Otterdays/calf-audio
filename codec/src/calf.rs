use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::fft::MdctEngine;
use crate::psycho::PsychoModel;
use crate::rans::{rans_decode, rans_encode};
use crate::wav::WavData;

const HOP_SIZE: usize = 1024;
const SUPERBLOCK_FRAMES: usize = 64;
const SHORT_M: usize = 256;

#[derive(Clone, Copy)]
struct FrameDesc {
    start: usize,
    m: usize,
    transient: bool,
}

pub(crate) fn transient_energy(padded: &[Vec<f32>], pos: usize, prev: &mut f32) -> bool {
    let end = (pos + 2 * HOP_SIZE).min(padded[0].len());
    let mut e = 0.0f32;
    for ch in 0..padded.len() {
        for i in pos..end {
            let x = padded[ch][i];
            e += x * x;
        }
    }
    let n = (end - pos).max(1) as f32;
    let cur = e / n;
    let is = *prev > 0.0 && cur > *prev * 20.0;
    *prev = cur;
    is
}

/// Quality profiles targeting <= 50% of MP3 bitrate with high perceptual fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// ~40-50 kbps stereo (~3x smaller than 128k MP3, ~4.5x smaller than 192k MP3)
    Ultra = 0,
    /// ~55-68 kbps stereo (~2x smaller than 128k MP3, ~3x smaller than 192k MP3)
    Balanced = 1,
    /// ~85-105 kbps stereo (~2x-3x smaller than 192k-320k MP3, near-transparent)
    Studio = 2,
}

impl Profile {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "ultra" | "ultra-light" | "ultralight" | "0" => Ok(Profile::Ultra),
            "balanced" | "default" | "1" => Ok(Profile::Balanced),
            "studio" | "high" | "2" => Ok(Profile::Studio),
            _ => Err(format!("Unknown profile '{s}' (use ultra, balanced, or studio)")),
        }
    }

    pub fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            0 => Ok(Profile::Ultra),
            1 => Ok(Profile::Balanced),
            2 => Ok(Profile::Studio),
            _ => Err(format!("Invalid CALF profile byte: {v}")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Profile::Ultra => "ultra",
            Profile::Balanced => "balanced",
            Profile::Studio => "studio",
        }
    }

    /// Quantizer step multiplier (higher = more compression, lower = higher fidelity)
    pub(crate) fn quant_scale(&self) -> f32 {
        match self {
            Profile::Ultra => 0.72,
            Profile::Balanced => 0.36,
            Profile::Studio => 0.16,
        }
    }

    /// Frequency (Hz) above which Intensity Stereo (Mid + 5-bit Pan) replaces full Side coding
    pub(crate) fn is_cutoff_hz(&self) -> f32 {
        match self {
            Profile::Ultra => 2800.0,
            Profile::Balanced => 3700.0,
            Profile::Studio => 6400.0,
        }
    }

    /// Frequency (Hz) above which Harmonic Bandwidth Extension (HBE) reconstructs highs
    pub(crate) fn hbe_cutoff_hz(&self) -> f32 {
        match self {
            Profile::Ultra => 10500.0,
            Profile::Balanced => 13000.0,
            Profile::Studio => 16500.0,
        }
    }

    /// Dead-zone rounding bias (smaller = wider dead zone = sparser zeros)
    pub(crate) fn deadzone_bias(&self) -> f32 {
        match self {
            Profile::Ultra => 0.34,
            Profile::Balanced => 0.38,
            Profile::Studio => 0.42,
        }
    }
}

#[derive(Debug)]
pub struct CalfInfo {
    pub sample_rate: u32,
    pub channels: u8,
    pub profile: Profile,
    pub total_samples: u64,
    pub num_frames: u32,
    pub title: String,
    pub file_bytes: usize,
    pub duration_secs: f64,
    pub bitrate_kbps: f64,
}

#[inline]
pub(crate) fn step_from_idx(idx: u8) -> f32 {
    if idx == 0 {
        0.0
    } else {
        // 1.5 dB step resolution across 63 levels (94.5 dB range)
        2.0f32.powf(((idx as f32) - 34.0) * 0.25)
    }
}

#[inline]
pub(crate) fn idx_from_step(step: f32) -> u8 {
    if step <= 0.0 {
        return 0;
    }
    let idx = (step.log2() * 4.0 + 34.0).round();
    idx.clamp(1.0, 63.0) as u8
}

#[inline]
pub(crate) fn energy_from_hbe_idx(idx: u8) -> f32 {
    if idx == 0 {
        0.0
    } else {
        // 2.0 dB RMS steps across 31 levels
        2.0f32.powf(((idx as f32) - 22.0) * 0.33333334)
    }
}

#[inline]
pub(crate) fn hbe_idx_from_rms(rms: f32) -> u8 {
    if rms < 1e-5 {
        return 0;
    }
    let idx = (rms.log2() * 3.0 + 22.0).round();
    idx.clamp(1.0, 31.0) as u8
}

/// Deterministic fast PRNG for Perceptual Noise Filling (PNF)
#[inline]
pub(crate) fn xorshift32(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    // Map to approximately [-1.0, 1.0]
    ((x as i32) as f32) * (1.0 / 2147483648.0)
}

/// Pack signed delta into unsigned byte via zig-zag encoding
#[inline]
pub(crate) fn zigzag_enc(delta: i32) -> u8 {
    let z = ((delta << 1) ^ (delta >> 31)) as u32;
    z.min(255) as u8
}

#[inline]
pub(crate) fn zigzag_dec(z: u8) -> i32 {
    let v = z as u32;
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

/// Bit packer for sign bits (1 bit per nonzero MDCT coefficient)
struct SignWriter {
    bytes: Vec<u8>,
    cur: u8,
    bits: u8,
}

impl SignWriter {
    fn new() -> Self {
        Self { bytes: Vec::new(), cur: 0, bits: 0 }
    }
    #[inline]
    fn push(&mut self, negative: bool) {
        if negative {
            self.cur |= 1 << self.bits;
        }
        self.bits += 1;
        if self.bits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.bits = 0;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

struct SignReader<'a> {
    bytes: &'a [u8],
    idx: usize,
    bits: u8,
}

impl<'a> SignReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, idx: 0, bits: 0 }
    }
    #[inline]
    fn next_neg(&mut self) -> bool {
        if self.idx >= self.bytes.len() {
            return false;
        }
        let neg = ((self.bytes[self.idx] >> self.bits) & 1) != 0;
        self.bits += 1;
        if self.bits == 8 {
            self.bits = 0;
            self.idx += 1;
        }
        neg
    }
}

/// Quantize a single band's MDCT coefficients using 0.75-power dead-zone companding.
/// Returns (scale_idx, pnf_level, quantized_coeffs).
pub(crate) fn quantize_band(
    spec: &[f32],
    mask_thresh: f32,
    tonality: f32,
    profile: Profile,
    allow_pnf: bool,
) -> (u8, u8, Vec<i32>) {
    let n = spec.len();
    let band_energy: f32 = spec.iter().map(|&x| x * x).sum();
    let max_abs = spec.iter().fold(0.0f32, |a, &x| a.max(x.abs()));

    // If completely masked or silent, skip coding bins!
    if max_abs < 1e-5 || band_energy < mask_thresh * 0.25 {
        return (0, 0, vec![0; n]);
    }

    // Target quantization step from perceptual masking threshold
    let noise_per_bin = (mask_thresh / (n as f32)).sqrt();
    let tonal_boost = 0.68 + 0.45 * (1.0 - tonality); // tighter for tonal bands
    let mut step = (noise_per_bin * profile.quant_scale() * tonal_boost).max(1e-4);

    // Prevent runaway quantized integers (> 400) on extreme peaks
    let min_step_for_peak = max_abs / 300.0f32.powf(4.0 / 3.0);
    if step < min_step_for_peak {
        step = min_step_for_peak;
    }

    let scale_idx = idx_from_step(step);
    let q_step = step_from_idx(scale_idx);
    let inv_step = 1.0 / q_step;
    let bias = profile.deadzone_bias();

    let mut q = Vec::with_capacity(n);
    let mut zero_energy = 0.0f32;
    let mut zero_count = 0usize;
    let mut nonzero_count = 0usize;

    for &x in spec {
        let u = (x.abs() * inv_step).powf(0.75);
        let mag = (u + bias).floor() as i32;
        if mag == 0 {
            q.push(0);
            zero_energy += x * x;
            zero_count += 1;
        } else {
            nonzero_count += 1;
            q.push(if x < 0.0 { -mag } else { mag });
        }
    }

    if nonzero_count == 0 && !allow_pnf {
        return (0, 0, vec![0; n]);
    }

    // Perceptual Noise Filling level for zeroed bins
    let pnf_level = if allow_pnf && zero_count > 0 && zero_energy > 0.0 {
        let zero_rms = (zero_energy / zero_count as f32).sqrt();
        let ratio = zero_rms / q_step;
        // Quantize ratio in [0..1.4] into 3 bits (0..7)
        ((ratio * 5.5).round() as i32).clamp(0, 7) as u8
    } else {
        0
    };

    if nonzero_count == 0 && pnf_level == 0 {
        return (0, 0, vec![0; n]);
    }

    (scale_idx, pnf_level, q)
}

/// Emit quantized coefficients into rANS magnitude stream (with zero-run tokens 0..=31),
/// packed sign bits, and escape stream.
pub(crate) fn emit_coeffs(q: &[i32], mag_stream: &mut Vec<u8>, signs: &mut SignWriter, escapes: &mut Vec<u8>) {
    let mut zrun = 0usize;
    for &v in q {
        if v == 0 {
            zrun += 1;
            if zrun == 32 {
                mag_stream.push(31); // run of 32 zeros
                zrun = 0;
            }
        } else {
            if zrun > 0 {
                mag_stream.push((zrun - 1) as u8);
                zrun = 0;
            }
            let mag = v.unsigned_abs();
            signs.push(v < 0);
            if mag <= 31 {
                mag_stream.push(31 + mag as u8); // 32..=62 -> mag 1..=31
            } else {
                mag_stream.push(63); // escape
                let capped = mag.min(65535) as u16;
                escapes.extend_from_slice(&capped.to_le_bytes());
            }
        }
    }
    if zrun > 0 {
        mag_stream.push((zrun - 1) as u8);
    }
}

pub(crate) fn read_coeffs(
    n: usize,
    mag_stream: &[u8],
    mag_pos: &mut usize,
    signs: &mut SignReader,
    escapes: &[u8],
    esc_pos: &mut usize,
    out: &mut [i32],
) -> Result<(), String> {
    let mut i = 0usize;
    while i < n {
        if *mag_pos >= mag_stream.len() {
            return Err("Unexpected EOF in spectral magnitude stream".into());
        }
        let tok = mag_stream[*mag_pos];
        *mag_pos += 1;
        if tok < 32 {
            let run = (tok as usize) + 1;
            for _ in 0..run {
                if i < n {
                    out[i] = 0;
                    i += 1;
                }
            }
        } else if tok < 63 {
            let mag = (tok - 31) as i32;
            let neg = signs.next_neg();
            out[i] = if neg { -mag } else { mag };
            i += 1;
        } else {
            if *esc_pos + 2 > escapes.len() {
                return Err("Unexpected EOF in escape stream".into());
            }
            let mag = u16::from_le_bytes([escapes[*esc_pos], escapes[*esc_pos + 1]]) as i32;
            *esc_pos += 2;
            let neg = signs.next_neg();
            out[i] = if neg { -mag } else { mag };
            i += 1;
        }
    }
    Ok(())
}

/// Encode PCM WAV data into CALF v2 binary bytes (variable long/short blocks).
pub fn encode_calf(wav: &WavData, profile: Profile, title: &str) -> Result<Vec<u8>, String> {
    let channels = wav.channels as usize;
    let total_samples = wav.samples[0].len();
    let sample_rate = wav.sample_rate;

    let num_long_slots = (total_samples + HOP_SIZE - 1) / HOP_SIZE + 1;
    let padded_len = (num_long_slots + 1) * HOP_SIZE;
    let mut padded = vec![vec![0.0f32; padded_len]; channels];
    for ch in 0..channels {
        padded[ch][HOP_SIZE..HOP_SIZE + total_samples].copy_from_slice(&wav.samples[ch]);
    }

    // First pass: transient detection + frame plan.
    let mut prev_energy = 0.0f32;
    let mut slot_transient = vec![false; num_long_slots];
    let mut frames: Vec<FrameDesc> = Vec::new();
    for slot in 0..num_long_slots {
        let pos = slot * HOP_SIZE;
        let tr = transient_energy(&padded, pos, &mut prev_energy);
        slot_transient[slot] = tr;
        if tr {
            for k in 0..(HOP_SIZE / SHORT_M) {
                frames.push(FrameDesc { start: pos + k * SHORT_M, m: SHORT_M, transient: true });
            }
        } else {
            frames.push(FrameDesc { start: pos, m: HOP_SIZE, transient: false });
        }
    }
    let total_frames = frames.len();

    // Header
    let mut out = Vec::new();
    out.extend_from_slice(b"CAL2");
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.push(wav.channels as u8);
    out.push(profile as u8);
    out.extend_from_slice(&(HOP_SIZE as u16).to_le_bytes());
    out.extend_from_slice(&(total_samples as u64).to_le_bytes());
    out.extend_from_slice(&(total_frames as u32).to_le_bytes());
    let title_bytes = title.as_bytes();
    let tlen = title_bytes.len().min(65535) as u16;
    out.extend_from_slice(&tlen.to_le_bytes());
    out.extend_from_slice(&title_bytes[..tlen as usize]);

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let sqrt2 = std::f32::consts::SQRT_2;

    // Process in superblocks of 64 frames.
    let mut frame_i = 0usize;
    while frame_i < frames.len() {
        let sb_frames = (frames.len() - frame_i).min(SUPERBLOCK_FRAMES);

        let mut meta_stream = Vec::new();
        let mut mag_stream = Vec::new();
        let mut sign_writer = SignWriter::new();
        let mut escape_stream = Vec::new();

        // Previous frame state for temporal delta coding (sized lazily per frame).
        // We keep separate vectors sized to current frame's band counts.
        let mut prev_m_scale: Vec<i32> = Vec::new();
        let mut prev_s_scale: Vec<i32> = Vec::new();
        let mut prev_hbe: Vec<i32> = Vec::new();
        let mut prev_pan: Vec<i32> = Vec::new();
        let mut have_prev = false;

        for _ in 0..sb_frames {
            let fd = frames[frame_i];
            frame_i += 1;
            let m = fd.m;
            let mdct = MdctEngine::new(m);
            let psycho = PsychoModel::new(sample_rate, m);

            let nb = psycho.bands.len();
            let is_start_band = psycho
                .bands
                .iter()
                .position(|b| b.center_hz >= profile.is_cutoff_hz())
                .unwrap_or(nb);
            let hbe_start_band = psycho
                .bands
                .iter()
                .position(|b| b.center_hz >= profile.hbe_cutoff_hz())
                .unwrap_or(nb);
            let pnf_start_band = psycho
                .bands
                .iter()
                .position(|b| b.center_hz >= 1200.0)
                .unwrap_or(nb);

            // Reset temporal state if band counts changed.
            let side_n = is_start_band.min(hbe_start_band);
            if !have_prev
                || prev_m_scale.len() != hbe_start_band
                || prev_s_scale.len() != side_n
                || prev_hbe.len() != nb - hbe_start_band
                || prev_pan.len() != nb - is_start_band
            {
                prev_m_scale = vec![0i32; hbe_start_band];
                prev_s_scale = vec![0i32; side_n];
                prev_hbe = vec![0i32; nb - hbe_start_band];
                prev_pan = vec![16i32; nb - is_start_band];
                have_prev = true;
            }

            let mut spec_l = vec![0.0f32; m];
            let mut spec_r = vec![0.0f32; m];
            let mut spec_m = vec![0.0f32; m];
            let mut spec_s = vec![0.0f32; m];

            let start = fd.start;
            mdct.forward(&padded[0][start..start + 2 * m], &mut spec_l);

            if channels == 2 {
                mdct.forward(&padded[1][start..start + 2 * m], &mut spec_r);
                for k in 0..m {
                    spec_m[k] = (spec_l[k] + spec_r[k]) * inv_sqrt2;
                    spec_s[k] = (spec_l[k] - spec_r[k]) * inv_sqrt2;
                }
            } else {
                spec_m.copy_from_slice(&spec_l);
            }

            meta_stream.push(if fd.transient { 1u8 } else { 0u8 });

            let (_m_e, m_mask, m_tonal) = psycho.compute_masking(&spec_m, fd.transient);

            // Mid channel
            for b in 0..hbe_start_band {
                let band = &psycho.bands[b];
                let allow_pnf = b >= pnf_start_band;
                let (scale_idx, pnf, q) = quantize_band(
                    &spec_m[band.start_bin..band.end_bin],
                    m_mask[b],
                    m_tonal[b],
                    profile,
                    allow_pnf,
                );
                meta_stream.push(zigzag_enc(scale_idx as i32 - prev_m_scale[b]));
                prev_m_scale[b] = scale_idx as i32;
                if scale_idx > 0 {
                    if allow_pnf {
                        meta_stream.push(pnf);
                    }
                    emit_coeffs(&q, &mut mag_stream, &mut sign_writer, &mut escape_stream);
                }
            }

            if channels == 2 {
                let (_s_e, s_mask, s_tonal) = psycho.compute_masking(&spec_s, fd.transient);
                let side_end = is_start_band.min(hbe_start_band);
                for b in 0..side_end {
                    let band = &psycho.bands[b];
                    let allow_pnf = b >= pnf_start_band;
                    let joint_mask = s_mask[b].max(m_mask[b] * 0.5);
                    let (scale_idx, pnf, q) = quantize_band(
                        &spec_s[band.start_bin..band.end_bin],
                        joint_mask,
                        s_tonal[b],
                        profile,
                        allow_pnf,
                    );
                    meta_stream.push(zigzag_enc(scale_idx as i32 - prev_s_scale[b]));
                    prev_s_scale[b] = scale_idx as i32;
                    if scale_idx > 0 {
                        if allow_pnf {
                            meta_stream.push(pnf);
                        }
                        emit_coeffs(&q, &mut mag_stream, &mut sign_writer, &mut escape_stream);
                    }
                }

                for b in is_start_band..nb {
                    let band = &psycho.bands[b];
                    let mut el = 0.0f32;
                    let mut er = 0.0f32;
                    for k in band.start_bin..band.end_bin {
                        el += spec_l[k] * spec_l[k];
                        er += spec_r[k] * spec_r[k];
                    }
                    let pan_idx = if el + er < 1e-10 {
                        16i32
                    } else {
                        let angle = er.sqrt().atan2(el.sqrt());
                        ((angle / std::f32::consts::FRAC_PI_2) * 31.0).round().clamp(0.0, 31.0) as i32
                    };
                    let pi = b - is_start_band;
                    meta_stream.push(zigzag_enc(pan_idx - prev_pan[pi]));
                    prev_pan[pi] = pan_idx;
                }
            }

            for b in hbe_start_band..nb {
                let band = &psycho.bands[b];
                let n = (band.end_bin - band.start_bin).max(1) as f32;
                let mut e = 0.0f32;
                for k in band.start_bin..band.end_bin {
                    e += spec_m[k] * spec_m[k];
                }
                let rms = (e / n).sqrt();
                let h_idx = hbe_idx_from_rms(rms) as i32;
                let hi = b - hbe_start_band;
                meta_stream.push(zigzag_enc(h_idx - prev_hbe[hi]));
                prev_hbe[hi] = h_idx;
            }
        }

        let meta_enc = rans_encode(&meta_stream);
        let mag_enc = rans_encode(&mag_stream);
        let sign_bytes = sign_writer.finish();

        out.extend_from_slice(&(meta_enc.len() as u32).to_le_bytes());
        out.extend_from_slice(&meta_enc);
        out.extend_from_slice(&(mag_enc.len() as u32).to_le_bytes());
        out.extend_from_slice(&mag_enc);
        out.extend_from_slice(&(sign_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&sign_bytes);
        out.extend_from_slice(&(escape_stream.len() as u32).to_le_bytes());
        out.extend_from_slice(&escape_stream);
    }

    let _ = sqrt2;
    Ok(out)
}

pub fn read_calf_info(data: &[u8]) -> Result<CalfInfo, String> {
    if data.len() < 24 || &data[0..4] != b"CAL2" {
        return Err("Not a valid CALF v2 file".into());
    }
    let sample_rate = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let channels = data[8];
    let profile = Profile::from_u8(data[9])?;
    let _hop = u16::from_le_bytes([data[10], data[11]]);
    let total_samples = u64::from_le_bytes([
        data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
    ]);
    let num_frames = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
    if data.len() < 26 {
        return Err("Truncated CALF header".into());
    }
    let tlen = u16::from_le_bytes([data[24], data[25]]) as usize;
    if data.len() < 26 + tlen {
        return Err("Truncated CALF title".into());
    }
    let title = String::from_utf8_lossy(&data[26..26 + tlen]).into_owned();
    let duration_secs = total_samples as f64 / sample_rate.max(1) as f64;
    let bitrate_kbps = if duration_secs > 0.0 {
        (data.len() as f64 * 8.0) / (duration_secs * 1000.0)
    } else {
        0.0
    };

    Ok(CalfInfo {
        sample_rate,
        channels,
        profile,
        total_samples,
        num_frames,
        title,
        file_bytes: data.len(),
        duration_secs,
        bitrate_kbps,
    })
}

/// Decode CALF v2 binary bytes back into PCM WavData.
pub fn decode_calf(data: &[u8]) -> Result<WavData, String> {
    let info = read_calf_info(data)?;
    let channels = info.channels as usize;
    let total_samples = info.total_samples as usize;
    let num_frames = info.num_frames as usize;
    let profile = info.profile;
    let sample_rate = info.sample_rate;

    let tlen = u16::from_le_bytes([data[24], data[25]]) as usize;
    let mut pos = 26 + tlen;

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let sqrt2 = std::f32::consts::SQRT_2;
    let mut rng_state: u32 = 0x51A7_C0DE;

    // Over-allocate; trim later.
    let padded_len = num_frames * 1024 + 2048;
    let mut out_padded = vec![vec![0.0f32; padded_len]; channels];

    let mut frame_start = 0usize;
    let mut frame_idx = 0usize;
    while frame_idx < num_frames {
        let sb_frames = (num_frames - frame_idx).min(SUPERBLOCK_FRAMES);

        fn read_slice<'a>(data: &'a [u8], pos: &mut usize) -> Result<&'a [u8], String> {
            if *pos + 4 > data.len() {
                return Err("Unexpected EOF reading superblock section length".into());
            }
            let len = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]) as usize;
            *pos += 4;
            if *pos + len > data.len() {
                return Err("Truncated superblock section".into());
            }
            let s = &data[*pos..*pos + len];
            *pos += len;
            Ok(s)
        }

        let meta_enc = read_slice(data, &mut pos)?;
        let mag_enc = read_slice(data, &mut pos)?;
        let sign_bytes = read_slice(data, &mut pos)?;
        let escape_stream = read_slice(data, &mut pos)?;

        let mut mp = 0;
        let meta_stream = rans_decode(meta_enc, &mut mp)?;
        let mut gp = 0;
        let mag_stream = rans_decode(mag_enc, &mut gp)?;

        let mut meta_pos = 0usize;
        let mut mag_pos = 0usize;
        let mut sign_reader = SignReader::new(sign_bytes);
        let mut esc_pos = 0usize;

        let mut prev_m_scale: Vec<i32> = Vec::new();
        let mut prev_s_scale: Vec<i32> = Vec::new();
        let mut prev_hbe: Vec<i32> = Vec::new();
        let mut prev_pan: Vec<i32> = Vec::new();

        for _ in 0..sb_frames {
            let bt = *meta_stream.get(meta_pos).ok_or("EOF meta block type")?;
            meta_pos += 1;
            let short = bt != 0;
            let m = if short { SHORT_M } else { HOP_SIZE };

            let mdct = MdctEngine::new(m);
            let psycho = PsychoModel::new(sample_rate, m);
            let nb = psycho.bands.len();
            let is_start_band = psycho
                .bands
                .iter()
                .position(|b| b.center_hz >= profile.is_cutoff_hz())
                .unwrap_or(nb);
            let hbe_start_band = psycho
                .bands
                .iter()
                .position(|b| b.center_hz >= profile.hbe_cutoff_hz())
                .unwrap_or(nb);
            let pnf_start_band = psycho
                .bands
                .iter()
                .position(|b| b.center_hz >= 1200.0)
                .unwrap_or(nb);

            let mut spec_m = vec![0.0f32; m];
            let mut spec_s = vec![0.0f32; m];
            let mut spec_l = vec![0.0f32; m];
            let mut spec_r = vec![0.0f32; m];
            let mut time_block = vec![0.0f32; 2 * m];
            let mut q_buf = vec![0i32; m];
            let mut pan_angles = vec![16i32; nb - is_start_band];

            let side_n = is_start_band.min(hbe_start_band);
            if prev_m_scale.len() != hbe_start_band
                || prev_s_scale.len() != side_n
                || prev_hbe.len() != nb - hbe_start_band
                || prev_pan.len() != nb - is_start_band
            {
                prev_m_scale = vec![0i32; hbe_start_band];
                prev_s_scale = vec![0i32; side_n];
                prev_hbe = vec![0i32; nb - hbe_start_band];
                prev_pan = vec![16i32; nb - is_start_band];
            }

            spec_m.fill(0.0);
            spec_s.fill(0.0);

            // Mid
            for b in 0..hbe_start_band {
                let band = &psycho.bands[b];
                let allow_pnf = b >= pnf_start_band;
                let delta = zigzag_dec(*meta_stream.get(meta_pos).ok_or("EOF meta m")?);
                meta_pos += 1;
                let scale_idx = (prev_m_scale[b] + delta).clamp(0, 63) as u8;
                prev_m_scale[b] = scale_idx as i32;

                if scale_idx > 0 {
                    let pnf = if allow_pnf {
                        let v = *meta_stream.get(meta_pos).ok_or("EOF meta pnf")?;
                        meta_pos += 1;
                        v
                    } else {
                        0
                    };
                    let n = band.end_bin - band.start_bin;
                    read_coeffs(n, &mag_stream, &mut mag_pos, &mut sign_reader, escape_stream, &mut esc_pos, &mut q_buf[..n])?;
                    let step = step_from_idx(scale_idx);
                    let noise_amp = (pnf as f32 / 5.5) * step * 1.15;
                    for j in 0..n {
                        let qv = q_buf[j];
                        if qv != 0 {
                            let mag = (qv.abs() as f32).powf(4.0 / 3.0) * step;
                            spec_m[band.start_bin + j] = if qv < 0 { -mag } else { mag };
                        } else if pnf > 0 {
                            spec_m[band.start_bin + j] = xorshift32(&mut rng_state) * noise_amp;
                        }
                    }
                }
            }

            if channels == 2 {
                let side_end = is_start_band.min(hbe_start_band);
                for b in 0..side_end {
                    let band = &psycho.bands[b];
                    let allow_pnf = b >= pnf_start_band;
                    let delta = zigzag_dec(*meta_stream.get(meta_pos).ok_or("EOF meta s")?);
                    meta_pos += 1;
                    let scale_idx = (prev_s_scale[b] + delta).clamp(0, 63) as u8;
                    prev_s_scale[b] = scale_idx as i32;

                    if scale_idx > 0 {
                        let pnf = if allow_pnf {
                            let v = *meta_stream.get(meta_pos).ok_or("EOF meta s pnf")?;
                            meta_pos += 1;
                            v
                        } else {
                            0
                        };
                        let n = band.end_bin - band.start_bin;
                        read_coeffs(n, &mag_stream, &mut mag_pos, &mut sign_reader, escape_stream, &mut esc_pos, &mut q_buf[..n])?;
                        let step = step_from_idx(scale_idx);
                        let noise_amp = (pnf as f32 / 5.5) * step * 1.15;
                        for j in 0..n {
                            let qv = q_buf[j];
                            if qv != 0 {
                                let mag = (qv.abs() as f32).powf(4.0 / 3.0) * step;
                                spec_s[band.start_bin + j] = if qv < 0 { -mag } else { mag };
                            } else if pnf > 0 {
                                spec_s[band.start_bin + j] = xorshift32(&mut rng_state) * noise_amp;
                            }
                        }
                    }
                }

                for b in is_start_band..nb {
                    let delta = zigzag_dec(*meta_stream.get(meta_pos).ok_or("EOF meta pan")?);
                    meta_pos += 1;
                    let pi = b - is_start_band;
                    let pan = (prev_pan[pi] + delta).clamp(0, 31);
                    prev_pan[pi] = pan;
                    pan_angles[pi] = pan;
                }
            }

            let src_start = if hbe_start_band > 0 {
                psycho.bands[hbe_start_band - 1].start_bin / 2
            } else {
                64
            };
            let src_end = if hbe_start_band < nb {
                psycho.bands[hbe_start_band].start_bin.max(src_start + 32)
            } else {
                m
            };
            let src_len = (src_end - src_start).max(1);

            for b in hbe_start_band..nb {
                let band = &psycho.bands[b];
                let delta = zigzag_dec(*meta_stream.get(meta_pos).ok_or("EOF meta hbe")?);
                meta_pos += 1;
                let hi = b - hbe_start_band;
                let h_idx = (prev_hbe[hi] + delta).clamp(0, 31) as u8;
                prev_hbe[hi] = h_idx as i32;

                let target_rms = energy_from_hbe_idx(h_idx);
                if target_rms > 0.0 {
                    let n = band.end_bin - band.start_bin;
                    let mut raw_e = 0.0f32;
                    for j in 0..n {
                        let src_bin = src_start + ((band.start_bin + j) % src_len);
                        let v = spec_m[src_bin] * 0.75 + xorshift32(&mut rng_state) * 0.25 * target_rms;
                        spec_m[band.start_bin + j] = v;
                        raw_e += v * v;
                    }
                    let cur_rms = (raw_e / n.max(1) as f32).sqrt();
                    if cur_rms > 1e-8 {
                        let gain = target_rms / cur_rms;
                        for k in band.start_bin..band.end_bin {
                            spec_m[k] *= gain;
                        }
                    }
                }
            }

            let start = frame_start;
            if channels == 2 {
                let side_end = is_start_band.min(hbe_start_band);
                for b in 0..side_end {
                    let band = &psycho.bands[b];
                    for k in band.start_bin..band.end_bin {
                        spec_l[k] = (spec_m[k] + spec_s[k]) * inv_sqrt2;
                        spec_r[k] = (spec_m[k] - spec_s[k]) * inv_sqrt2;
                    }
                }
                for b in is_start_band..nb {
                    let band = &psycho.bands[b];
                    let angle = (pan_angles[b - is_start_band] as f32 / 31.0) * std::f32::consts::FRAC_PI_2;
                    let gl = angle.cos() * sqrt2;
                    let gr = angle.sin() * sqrt2;
                    for k in band.start_bin..band.end_bin {
                        spec_l[k] = spec_m[k] * gl;
                        spec_r[k] = spec_m[k] * gr;
                    }
                }

                mdct.inverse(&spec_l, &mut time_block);
                for i in 0..2 * m {
                    if start + i < out_padded[0].len() {
                        out_padded[0][start + i] += time_block[i];
                    }
                }
                mdct.inverse(&spec_r, &mut time_block);
                for i in 0..2 * m {
                    if start + i < out_padded[1].len() {
                        out_padded[1][start + i] += time_block[i];
                    }
                }
            } else {
                mdct.inverse(&spec_m, &mut time_block);
                for i in 0..2 * m {
                    if start + i < out_padded[0].len() {
                        out_padded[0][start + i] += time_block[i];
                    }
                }
            }

            frame_start += m;
            frame_idx += 1;
        }
    }

    // Trim
    let mut samples = Vec::with_capacity(channels);
    for ch in 0..channels {
        let end = (HOP_SIZE + total_samples).min(out_padded[ch].len());
        samples.push(out_padded[ch][HOP_SIZE..end].to_vec());
    }

    Ok(WavData {
        sample_rate,
        channels: info.channels as u16,
        samples,
    })
}

pub fn write_bytes<P: AsRef<Path>>(path: P, bytes: &[u8]) -> Result<(), String> {
    let mut f = File::create(path).map_err(|e| format!("Failed creating output file: {e}"))?;
    f.write_all(bytes).map_err(|e| format!("Failed writing output file: {e}"))?;
    Ok(())
}

pub fn read_bytes<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(|e| format!("Failed opening file: {e}"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("Failed reading file: {e}"))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_v2() {
        let sr = 44100usize;
        let n = sr;
        let mut l = vec![0.0f32; n];
        let mut r = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f32 / sr as f32;
            l[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            r[i] = 0.5 * (2.0 * std::f32::consts::PI * 660.0 * t).sin();
        }
        let wav = WavData { sample_rate: 44100, channels: 2, samples: vec![l, r] };
        let enc = encode_calf(&wav, Profile::Balanced, "roundtrip").unwrap();
        let dec = decode_calf(&enc).unwrap();
        assert_eq!(dec.sample_rate, 44100);
        assert_eq!(dec.channels, 2);
        assert_eq!(dec.samples[0].len(), n);
    }
}
