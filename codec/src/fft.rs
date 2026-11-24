use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    pub fn add(self, o: Self) -> Self {
        Self { re: self.re + o.re, im: self.im + o.im }
    }
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        Self { re: self.re - o.re, im: self.im - o.im }
    }
    #[inline]
    pub fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

/// In-place radix-2 Cooley-Tukey FFT (forward: exp(-2*pi*i*k*n/N)).
pub fn fft_in_place(buf: &mut [Complex]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let angle = -2.0 * PI / (len as f64);
        let wlen = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let v = buf[start + k + half].mul(w);
                buf[start + k] = u.add(v);
                buf[start + k + half] = u.sub(v);
                w = w.mul(wlen);
            }
        }
        len <<= 1;
    }
}

/// Fast MDCT / IMDCT engine using a Vorbis window (exact Princen-Bradley TDAC)
/// and O(N log N) radix-2 FFT.
pub struct MdctEngine {
    pub m: usize,        // hop size (number of spectral bins), e.g. 1024
    pub n: usize,        // window size = 2 * m, e.g. 2048
    pub window: Vec<f64>,
    fwd_pre: Vec<Complex>,
    fwd_post: Vec<Complex>,
    inv_pre: Vec<Complex>,
    inv_post: Vec<Complex>,
}

impl MdctEngine {
    pub fn new(m: usize) -> Self {
        let n = 2 * m;
        assert!(n.is_power_of_two());

        // Vorbis window: w[i] = sin(pi/2 * sin^2(pi * (i + 0.5) / N))
        // Satisfies w[i]^2 + w[i + M]^2 = 1 analytically with -18 dB/oct sidelobes.
        let mut window = Vec::with_capacity(n);
        for i in 0..n {
            let s = (PI * (i as f64 + 0.5) / (n as f64)).sin();
            window.push((0.5 * PI * s * s).sin());
        }

        let n0 = 0.5 + (m as f64) * 0.5;
        let mut fwd_pre = Vec::with_capacity(n);
        for i in 0..n {
            let theta = -PI * (i as f64) / (n as f64);
            fwd_pre.push(Complex::new(theta.cos(), theta.sin()));
        }

        let mut fwd_post = Vec::with_capacity(m);
        for k in 0..m {
            let theta = -PI * n0 * (k as f64 + 0.5) / (m as f64);
            fwd_post.push(Complex::new(theta.cos(), theta.sin()));
        }

        let mut inv_pre = Vec::with_capacity(m);
        for k in 0..m {
            let theta = -PI * n0 * (k as f64) / (m as f64);
            inv_pre.push(Complex::new(theta.cos(), theta.sin()));
        }

        let mut inv_post = Vec::with_capacity(n);
        for i in 0..n {
            let theta = -PI * (i as f64 + n0) / (n as f64);
            inv_post.push(Complex::new(theta.cos(), theta.sin()));
        }

        Self { m, n, window, fwd_pre, fwd_post, inv_pre, inv_post }
    }

    /// Forward MDCT: 2M time samples -> M spectral coefficients (built-in Vorbis window).
    pub fn forward(&self, input: &[f32], out: &mut [f32]) {
        self.forward_windowed(input, &self.window, out);
    }

    /// Forward MDCT with a caller-supplied analysis window (block-switching transitions).
    pub fn forward_windowed(&self, input: &[f32], window: &[f64], out: &mut [f32]) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(window.len(), self.n);
        debug_assert_eq!(out.len(), self.m);

        let mut buf = vec![Complex::new(0.0, 0.0); self.n];
        for i in 0..self.n {
            let w_x = (input[i] as f64) * window[i];
            buf[i] = Complex::new(w_x * self.fwd_pre[i].re, w_x * self.fwd_pre[i].im);
        }

        fft_in_place(&mut buf);

        for k in 0..self.m {
            let c = buf[k].mul(self.fwd_post[k]);
            out[k] = c.re as f32;
        }
    }

    /// Inverse MDCT: M spectral coefficients -> 2M windowed time samples (built-in window).
    pub fn inverse(&self, spec: &[f32], out: &mut [f32]) {
        self.inverse_windowed(spec, &self.window, out);
    }

    /// Inverse MDCT with a caller-supplied synthesis window.
    pub fn inverse_windowed(&self, spec: &[f32], window: &[f64], out: &mut [f32]) {
        debug_assert_eq!(spec.len(), self.m);
        debug_assert_eq!(out.len(), self.n);

        let mut buf = vec![Complex::new(0.0, 0.0); self.n];
        for k in 0..self.m {
            let s = spec[k] as f64;
            buf[k] = Complex::new(s * self.inv_pre[k].re, s * self.inv_pre[k].im);
        }

        fft_in_place(&mut buf);

        let scale = 2.0 / (self.m as f64);
        for i in 0..self.n {
            let c = buf[i].mul(self.inv_post[i]);
            out[i] = (c.re * scale * window[i]) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdct_tdac_perfect_reconstruction() {
        let m = 1024;
        let engine = MdctEngine::new(m);
        let total = m * 6;
        let mut signal = vec![0.0f32; total];
        for i in 0..total {
            let t = i as f32 / 44100.0;
            signal[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1320.0 * t).cos();
        }

        let mut recon = vec![0.0f32; total];
        let mut spec = vec![0.0f32; m];
        let mut time_block = vec![0.0f32; 2 * m];

        for frame in 0..5 {
            let start = frame * m;
            engine.forward(&signal[start..start + 2 * m], &mut spec);
            engine.inverse(&spec, &mut time_block);
            for i in 0..2 * m {
                recon[start + i] += time_block[i];
            }
        }

        // Check interior frames (where both halves overlap)
        let mut max_err = 0.0f32;
        for i in m..(total - m) {
            let err = (signal[i] - recon[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-5, "TDAC reconstruction error too high: {}", max_err);
    }
}
