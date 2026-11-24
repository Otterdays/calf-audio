//! Objective benchmark signals and quality metrics.
//!
//! Waveform SNR alone is misleading for a perceptual codec (noise filling,
//! bandwidth extension and intensity stereo are intentionally not waveform
//! matching), so we report three complementary measures:
//!   * SNR / segmental SNR  - waveform fidelity, segSNR punishes noise in quiet parts
//!   * band energy error    - mean |dB| deviation of per-Bark-band energy (tonal balance,
//!                            catches HBE / intensity-stereo level errors per channel)
//!   * pre-echo             - error energy in the 20 ms before each attack relative to
//!                            the attack energy (block-switching effectiveness)

use crate::fft::MdctEngine;
use crate::psycho::PsychoModel;
use crate::wav::WavData;

#[inline]
fn xorshift(rng: &mut u32) -> f32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    (*rng as i32) as f32 / 2147483648.0
}

/// 10 s chord progression with panned voices + off-beat hi-hat shimmer (the original benchmark).
pub fn music_signal(sr: u32, dur: usize) -> WavData {
    let n = sr as usize * dur;
    let mut l = vec![0.0f32; n];
    let mut r = vec![0.0f32; n];
    let mut rng: u32 = 0x12345678;
    let chords: &[[f32; 4]] = &[
        [130.81, 261.63, 329.63, 392.00],
        [110.00, 220.00, 261.63, 329.63],
        [87.31, 174.61, 220.00, 261.63],
        [98.00, 196.00, 246.94, 293.66],
    ];
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let chord = chords[((t / 2.5) as usize) % chords.len()];
        let env = (-((t * 2.0) % 1.0) * 3.5).exp();
        let mut sl = 0.0f32;
        let mut sr_s = 0.0f32;
        for (idx, &f) in chord.iter().enumerate() {
            let w = 2.0 * std::f32::consts::PI * f * t;
            let tone = 0.16 * w.sin() + 0.05 * (2.0 * w).sin() + 0.02 * (3.0 * w).sin();
            let pan = 0.3 + 0.15 * idx as f32;
            sl += tone * (1.0 - pan * 0.4) * (0.4 + 0.6 * env);
            sr_s += tone * (0.6 + pan * 0.4) * (0.4 + 0.6 * env);
        }
        let hat_env = (-((t * 4.0) % 1.0) * 28.0).exp();
        let noise = xorshift(&mut rng) * 0.06 * hat_env;
        l[i] = (sl + noise * 0.8).clamp(-0.98, 0.98);
        r[i] = (sr_s + noise * 1.1).clamp(-0.98, 0.98);
    }
    WavData { sample_rate: sr, channels: 2, samples: vec![l, r] }
}

/// Sharp percussive attacks (castanet / snare-like bursts) over a very quiet pad.
/// Returns the signal and the attack sample positions.
pub fn transient_signal(sr: u32, dur: usize) -> (WavData, Vec<usize>) {
    let n = sr as usize * dur;
    let mut l = vec![0.0f32; n];
    let mut r = vec![0.0f32; n];
    let mut rng: u32 = 0x9E3779B9;
    let mut onsets = Vec::new();
    // Irregular spacing so attacks land at every possible offset within a frame.
    let mut pos = (0.21 * sr as f32) as usize;
    let mut k = 0usize;
    while pos + sr as usize / 4 < n {
        onsets.push(pos);
        pos += (sr as f32 * (0.37 + 0.093 * (k % 5) as f32)) as usize + 211 * (k % 7);
        k += 1;
    }
    for i in 0..n {
        let t = i as f32 / sr as f32;
        // -40 dBFS sustained pad so pre-echo is clearly exposed
        let pad = 0.01 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
        l[i] = pad;
        r[i] = pad * 0.9;
    }
    for (idx, &o) in onsets.iter().enumerate() {
        let len = sr as usize / 5;
        let tone_f = 180.0 + 40.0 * (idx % 3) as f32;
        let pan = 0.25 + 0.5 * ((idx * 7) % 5) as f32 / 4.0;
        for j in 0..len.min(n - o) {
            let tt = j as f32 / sr as f32;
            let noise_env = (-tt * 60.0).exp();
            let body_env = (-tt * 18.0).exp();
            let s = 0.55 * xorshift(&mut rng) * noise_env
                + 0.35 * (2.0 * std::f32::consts::PI * tone_f * tt).sin() * body_env;
            l[o + j] += s * (1.0 - pan) * 1.4;
            r[o + j] += s * pan * 1.4;
        }
    }
    for i in 0..n {
        l[i] = l[i].clamp(-0.98, 0.98);
        r[i] = r[i].clamp(-0.98, 0.98);
    }
    (WavData { sample_rate: sr, channels: 2, samples: vec![l, r] }, onsets)
}

pub struct Metrics {
    pub snr_db: f64,
    pub seg_snr_db: f64,
    pub band_err_db: f64,
    pub pre_echo_db: Option<f64>,
}

pub fn measure(orig: &WavData, dec: &WavData, onsets: Option<&[usize]>) -> Metrics {
    let channels = orig.channels as usize;
    let n = orig.samples[0].len().min(dec.samples[0].len());

    // Global SNR
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for ch in 0..channels {
        for i in 0..n {
            let s = orig.samples[ch][i] as f64;
            let d = dec.samples[ch][i] as f64;
            sig += s * s;
            err += (s - d) * (s - d);
        }
    }
    let snr_db = 10.0 * (sig / err.max(1e-12)).log10();

    // Segmental SNR over 256-sample segments, clamped to [-10, 40] dB, skipping digital silence
    let seg = 256usize;
    let mut seg_sum = 0.0f64;
    let mut seg_cnt = 0usize;
    for ch in 0..channels {
        let mut i = 0;
        while i + seg <= n {
            let mut s_p = 0.0f64;
            let mut e_p = 0.0f64;
            for j in i..i + seg {
                let s = orig.samples[ch][j] as f64;
                let d = dec.samples[ch][j] as f64;
                s_p += s * s;
                e_p += (s - d) * (s - d);
            }
            if s_p > 1e-9 * seg as f64 {
                seg_sum += (10.0 * (s_p / e_p.max(1e-15)).log10()).clamp(-10.0, 40.0);
                seg_cnt += 1;
            }
            i += seg;
        }
    }
    let seg_snr_db = seg_sum / seg_cnt.max(1) as f64;

    // Band energy error per channel on 1024-bin MDCT frames
    let m = 1024usize;
    let mdct = MdctEngine::new(m);
    let psycho = PsychoModel::new(orig.sample_rate, m);
    let mut so = vec![0.0f32; m];
    let mut sd = vec![0.0f32; m];
    let mut be_sum = 0.0f64;
    let mut be_cnt = 0usize;
    // floor ~ -75 dBFS band energy so inaudible bands do not dominate
    let floor = (m as f32 * 0.5).powi(2) * 10f32.powf(-7.5);
    for ch in 0..channels {
        let mut start = 0;
        while start + 2 * m <= n {
            mdct.forward(&orig.samples[ch][start..start + 2 * m], &mut so);
            mdct.forward(&dec.samples[ch][start..start + 2 * m], &mut sd);
            for band in &psycho.bands {
                let eo: f32 = so[band.start_bin..band.end_bin].iter().map(|x| x * x).sum();
                let ed: f32 = sd[band.start_bin..band.end_bin].iter().map(|x| x * x).sum();
                if eo.max(ed) > floor {
                    let d = 10.0 * ((ed.max(floor) / eo.max(floor)) as f64).log10();
                    be_sum += d.abs();
                    be_cnt += 1;
                }
            }
            start += m;
        }
    }
    let band_err_db = be_sum / be_cnt.max(1) as f64;

    // Pre-echo: error energy in [onset-20ms, onset-1ms) relative to the attack energy (first 10 ms)
    let pre_echo_db = onsets.map(|ons| {
        let sr = orig.sample_rate as usize;
        let pre = sr / 50;
        let guard = sr / 1000;
        let att = sr / 100;
        let mut acc = 0.0f64;
        let mut cnt = 0usize;
        for &o in ons {
            if o < pre || o + att > n {
                continue;
            }
            let mut e_pre = 0.0f64;
            let mut e_att = 0.0f64;
            for ch in 0..channels {
                for j in o - pre..o - guard {
                    let d = (orig.samples[ch][j] - dec.samples[ch][j]) as f64;
                    e_pre += d * d;
                }
                for j in o..o + att {
                    let s = orig.samples[ch][j] as f64;
                    e_att += s * s;
                }
            }
            acc += 10.0 * ((e_pre / (pre - guard) as f64) / (e_att / att as f64).max(1e-12)).max(1e-12).log10();
            cnt += 1;
        }
        acc / cnt.max(1) as f64
    });

    Metrics { snr_db, seg_snr_db, band_err_db, pre_echo_db }
}
