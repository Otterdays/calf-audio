//! Legacy CALF v1 bitstream decoder (static per-superblock rANS, fixed long blocks).
//! Kept so files produced by the first release stay playable; new files are v2.

use std::f32::consts::FRAC_PI_2;

use crate::calf::{energy_from_hbe_idx, step_from_idx, xorshift32, CalfInfo, Profile};
use crate::fft::MdctEngine;
use crate::psycho::PsychoModel;
use crate::rans::rans_decode;
use crate::wav::WavData;

pub const CALF_V1_MAGIC: &[u8; 4] = b"CALF";
const HOP_SIZE: usize = 1024;
const SUPERBLOCK_FRAMES: usize = 64;

#[inline]
fn zigzag_dec(z: u8) -> i32 {
    let v = z as u32;
    ((v >> 1) as i32) ^ -((v & 1) as i32)
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

fn read_coeffs(
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

pub fn read_v1_info(data: &[u8]) -> Result<CalfInfo, String> {
    if data.len() < 24 || &data[0..4] != CALF_V1_MAGIC {
        return Err("Not a valid CALF file".into());
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

/// Decode CALF v1 binary bytes back into PCM WavData.
pub fn decode_v1(data: &[u8]) -> Result<WavData, String> {
    let info = read_v1_info(data)?;
    let channels = info.channels as usize;
    let total_samples = info.total_samples as usize;
    let num_frames = info.num_frames as usize;
    let profile = info.profile;

    let tlen = u16::from_le_bytes([data[24], data[25]]) as usize;
    let mut pos = 26 + tlen;

    let mdct = MdctEngine::new(HOP_SIZE);
    let psycho = PsychoModel::new(info.sample_rate, HOP_SIZE);
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

    let padded_len = (num_frames + 1) * HOP_SIZE;
    let mut out_padded = vec![vec![0.0f32; padded_len]; channels];

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let sqrt2 = std::f32::consts::SQRT_2;
    let mut rng_state: u32 = 0x51A7_C0DE;

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

        let mut prev_m_scale = vec![0i32; hbe_start_band];
        let mut prev_s_scale = vec![0i32; is_start_band.min(hbe_start_band)];
        let mut prev_hbe = vec![0i32; nb - hbe_start_band];
        let mut prev_pan = vec![16i32; nb - is_start_band];

        let mut spec_m = vec![0.0f32; HOP_SIZE];
        let mut spec_s = vec![0.0f32; HOP_SIZE];
        let mut spec_l = vec![0.0f32; HOP_SIZE];
        let mut spec_r = vec![0.0f32; HOP_SIZE];
        let mut time_block = vec![0.0f32; 2 * HOP_SIZE];
        let mut q_buf = vec![0i32; HOP_SIZE];
        let mut pan_angles = vec![16i32; nb - is_start_band];

        for _ in 0..sb_frames {
            spec_m.fill(0.0);
            spec_s.fill(0.0);

            // 1. Decode Mid bands
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

            // 2. Decode Stereo (Side + Intensity Stereo pan)
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

            // 3. Reconstruct Harmonic Bandwidth Extension (HBE) bands by spectral folding + envelope shaping
            let src_start = if hbe_start_band > 0 {
                psycho.bands[hbe_start_band - 1].start_bin / 2
            } else {
                64
            };
            let src_end = if hbe_start_band < nb {
                psycho.bands[hbe_start_band].start_bin.max(src_start + 32)
            } else {
                HOP_SIZE
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
                    // Fold lower-octave spectral structure + 35% decorrelated sparkle noise
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

            // 4. Reconstruct L/R spectra and run Inverse MDCT + Overlap-Add
            let start = frame_idx * HOP_SIZE;
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
                    let angle = (pan_angles[b - is_start_band] as f32 / 31.0) * FRAC_PI_2;
                    let gl = angle.cos() * sqrt2;
                    let gr = angle.sin() * sqrt2;
                    for k in band.start_bin..band.end_bin {
                        spec_l[k] = spec_m[k] * gl;
                        spec_r[k] = spec_m[k] * gr;
                    }
                }

                mdct.inverse(&spec_l, &mut time_block);
                for i in 0..2 * HOP_SIZE {
                    out_padded[0][start + i] += time_block[i];
                }
                mdct.inverse(&spec_r, &mut time_block);
                for i in 0..2 * HOP_SIZE {
                    out_padded[1][start + i] += time_block[i];
                }
            } else {
                mdct.inverse(&spec_m, &mut time_block);
                for i in 0..2 * HOP_SIZE {
                    out_padded[0][start + i] += time_block[i];
                }
            }

            frame_idx += 1;
        }
    }

    // Trim the leading HOP_SIZE padding and trailing pad to exact sample count
    let mut samples = Vec::with_capacity(channels);
    for ch in 0..channels {
        let end = (HOP_SIZE + total_samples).min(out_padded[ch].len());
        samples.push(out_padded[ch][HOP_SIZE..end].to_vec());
    }

    Ok(WavData {
        sample_rate: info.sample_rate,
        channels: info.channels as u16,
        samples,
    })
}
