/// 24-band Bark-scale psychoacoustic model with Absolute Threshold of Hearing (ATH),
/// asymmetric simultaneous masking (spreading function), and Spectral Flatness (SFM).

#[derive(Clone, Debug)]
pub struct BarkBand {
    pub start_bin: usize,
    pub end_bin: usize, // exclusive
    pub center_hz: f32,
    pub ath_power: f32, // minimum audible power per bin in MDCT scale
}

pub struct PsychoModel {
    pub bands: Vec<BarkBand>,
    pub spreading: Vec<Vec<f32>>, // [masker][maskee] linear attenuation
}

/// Terhardt Absolute Threshold of Hearing in dB SPL (relative to 0 dBFS = 96 dB SPL).
fn ath_db_spl(freq_hz: f32) -> f32 {
    let f_khz = (freq_hz.max(20.0)) / 1000.0;
    3.64 * f_khz.powf(-0.8)
        - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
        + 1e-3 * f_khz.powi(4)
}

impl PsychoModel {
    pub fn new(sample_rate: u32, num_bins: usize) -> Self {
        // Bark critical band edges in Hz
        const EDGES_HZ: &[f32] = &[
            0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0,
            1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0,
            3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 24000.0,
        ];

        let nyquist = sample_rate as f32 * 0.5;
        let hz_per_bin = nyquist / num_bins as f32;

        let mut bands = Vec::new();
        let mut prev_bin = 0usize;

        for w in EDGES_HZ.windows(2) {
            let mut end_bin = ((w[1] / hz_per_bin).round() as usize).min(num_bins);
            if end_bin <= prev_bin {
                end_bin = (prev_bin + 2).min(num_bins);
            }
            if prev_bin >= num_bins {
                break;
            }
            let center_hz = ((prev_bin + end_bin) as f32 * 0.5) * hz_per_bin;
            // Convert ATH dB SPL (with 96 dB SPL = full-scale sine) into MDCT bin power scale.
            // A full-scale sine wave has MDCT bin magnitude ~ M/2 (512), power ~ 2.6e5.
            // Reference 0 dBFS power in MDCT domain ~ (num_bins as f32 * 0.5)^2.
            let ath_spl = ath_db_spl(center_hz).clamp(-5.0, 75.0);
            let ath_rel_db = ath_spl - 96.0; // negative dB relative to full scale
            let ref_power = (num_bins as f32 * 0.5).powi(2);
            let ath_power = ref_power * 10.0f32.powf(ath_rel_db / 10.0);

            bands.push(BarkBand {
                start_bin: prev_bin,
                end_bin,
                center_hz,
                ath_power,
            });
            prev_bin = end_bin;
        }

        // Ensure last band reaches num_bins
        if let Some(last) = bands.last_mut() {
            last.end_bin = num_bins;
        }

        // Asymmetric Bark spreading function:
        // Downward masking: -25 dB/Bark, Upward masking: -10 dB/Bark
        let nb = bands.len();
        let mut spreading = vec![vec![0.0f32; nb]; nb];
        for i in 0..nb {
            for j in 0..nb {
                let dz = j as f32 - i as f32;
                let db = if dz >= 0.0 {
                    -10.5 * dz // upward masking (stronger)
                } else {
                    25.0 * dz // downward masking (steeper drop)
                };
                spreading[i][j] = 10.0f32.powf(db / 10.0);
            }
        }

        Self { bands, spreading }
    }

    /// Compute per-band masking threshold (total band noise power allowed) and tonality [0..1].
    /// `transient`: spectrum is an interleaved set of short blocks; its spectral flatness
    /// reflects temporal modulation rather than tonality, so treat every band as noise-like.
    pub fn compute_masking(&self, spec: &[f32], transient: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let nb = self.bands.len();
        let mut energies = vec![0.0f32; nb];
        let mut tonality = vec![0.0f32; nb];

        for (b_idx, band) in self.bands.iter().enumerate() {
            let n = (band.end_bin - band.start_bin).max(1) as f32;
            let mut sum_e = 0.0f32;
            let mut sum_log = 0.0f32;
            for k in band.start_bin..band.end_bin {
                let p = spec[k] * spec[k];
                sum_e += p;
                sum_log += (p + 1e-12).ln();
            }
            energies[b_idx] = sum_e;
            let arith = (sum_e / n).max(1e-12);
            let geom = (sum_log / n).exp();
            let sfm = (geom / arith).clamp(0.0, 1.0);
            // Tonality: 1.0 = pure tone, 0.0 = flat white noise
            let sfm_db = 10.0 * (sfm + 1e-9).log10();
            tonality[b_idx] = if transient { 0.0 } else { (sfm_db / -30.0).clamp(0.0, 1.0) };
        }

        let mut mask_thresh = vec![0.0f32; nb];
        for j in 0..nb {
            let mut spread_e = 0.0f32;
            for i in 0..nb {
                spread_e += energies[i] * self.spreading[i][j];
            }
            // Tone-masking-noise needs ~18 dB SNR; noise-masking-tone needs ~6 dB SNR
            let t = tonality[j];
            let offset_db = 18.0 * t + 7.5 * (1.0 - t);
            let masked = spread_e * 10.0f32.powf(-offset_db / 10.0);
            let n_bins = (self.bands[j].end_bin - self.bands[j].start_bin) as f32;
            let ath_band = self.bands[j].ath_power * n_bins;
            mask_thresh[j] = masked.max(ath_band);
        }

        (energies, mask_thresh, tonality)
    }
}
