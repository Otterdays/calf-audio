//! Adaptive binary range coder (LZMA-style carry-propagating encoder) with
//! dual-rate probability estimation, plus symmetric "Coder" abstraction so the
//! bitstream syntax is written exactly once for both encoder and decoder.

const TOP: u32 = 1 << 24;
const PROB_BITS: u32 = 15;
const PROB_ONE: u32 = 1 << PROB_BITS;

/// Adaptive probability of a `0` bit, mixed from a fast (1/16) and a slow (1/128)
/// estimator: the fast one tracks non-stationary music, the slow one converges
/// to a precise long-run estimate. Both stay strictly inside (0, 1).
#[derive(Clone, Copy, Debug)]
pub struct Prob {
    fast: u16,
    slow: u16,
}

impl Default for Prob {
    fn default() -> Self {
        Self { fast: (PROB_ONE / 2) as u16, slow: (PROB_ONE / 2) as u16 }
    }
}

impl Prob {
    #[inline]
    fn p0(&self) -> u32 {
        (self.fast as u32 + self.slow as u32) >> 1
    }
    #[inline]
    fn update(&mut self, bit: bool) {
        if bit {
            self.fast -= self.fast >> 4;
            self.slow -= self.slow >> 7;
        } else {
            self.fast += ((PROB_ONE - self.fast as u32) >> 4) as u16;
            self.slow += ((PROB_ONE - self.slow as u32) >> 7) as u16;
        }
    }
}

pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1, out: Vec::new() }
    }

    #[inline]
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

pub struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self { data, pos: 0, range: 0xFFFF_FFFF, code: 0 };
        for _ in 0..5 {
            d.code = (d.code << 8) | d.next_byte() as u32;
        }
        d
    }
    #[inline]
    fn next_byte(&mut self) -> u8 {
        // Reading past the end yields zeros; corrupt input decodes garbage but never panics.
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
    #[inline]
    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next_byte() as u32;
        }
    }
    /// True when the decoder consumed more bytes than the payload holds (corruption check).
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len() + 4
    }
}

/// Symmetric coding interface: the encoder codes the given value and returns it,
/// the decoder ignores the given value and returns the decoded one.
pub trait Coder {
    fn bit(&mut self, p: &mut Prob, bit: bool) -> bool;
    fn bypass(&mut self, bit: bool) -> bool;

    /// `n` raw bits, MSB first.
    fn bypass_bits(&mut self, v: u32, n: u32) -> u32 {
        let mut out = 0u32;
        for i in (0..n).rev() {
            let b = self.bypass((v >> i) & 1 != 0);
            out = (out << 1) | b as u32;
        }
        out
    }

    /// Fixed-width symbol through an adaptive binary tree (`probs.len() >= 1 << n`).
    fn tree(&mut self, probs: &mut [Prob], v: u32, n: u32) -> u32 {
        let mut m = 1usize;
        for i in (0..n).rev() {
            let b = self.bit(&mut probs[m], (v >> i) & 1 != 0);
            m = (m << 1) | b as usize;
        }
        (m - (1 << n)) as u32
    }

    /// Unbounded non-negative integer: adaptive unary exponent (contexts `probs`,
    /// one per exponent position, last one shared) + bypass mantissa (Exp-Golomb shape).
    fn exp_golomb(&mut self, probs: &mut [Prob], v: u32) -> u32 {
        let vp1 = v as u64 + 1;
        let e_true = 63 - vp1.leading_zeros(); // floor(log2(v+1))
        let mut e = 0u32;
        loop {
            let ctx = (e as usize).min(probs.len() - 1);
            let more = self.bit(&mut probs[ctx], e < e_true);
            if !more || e >= 24 {
                break;
            }
            e += 1;
        }
        let mant = self.bypass_bits((vp1 as u32).wrapping_sub(1 << e), e);
        ((1u64 << e) + mant as u64 - 1) as u32
    }
}

impl Coder for RangeEncoder {
    #[inline]
    fn bit(&mut self, p: &mut Prob, bit: bool) -> bool {
        let bound = (self.range >> PROB_BITS) * p.p0();
        if bit {
            self.low += bound as u64;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        p.update(bit);
        self.normalize();
        bit
    }
    #[inline]
    fn bypass(&mut self, bit: bool) -> bool {
        self.range >>= 1;
        if bit {
            self.low += self.range as u64;
        }
        self.normalize();
        bit
    }
}

impl<'a> Coder for RangeDecoder<'a> {
    #[inline]
    fn bit(&mut self, p: &mut Prob, _bit: bool) -> bool {
        let bound = (self.range >> PROB_BITS) * p.p0();
        let bit = if self.code < bound {
            self.range = bound;
            false
        } else {
            self.code -= bound;
            self.range -= bound;
            true
        };
        p.update(bit);
        self.normalize();
        bit
    }
    #[inline]
    fn bypass(&mut self, _bit: bool) -> bool {
        self.range >>= 1;
        let bit = if self.code >= self.range {
            self.code -= self.range;
            true
        } else {
            false
        };
        self.normalize();
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syntax<C: Coder>(c: &mut C, vals: &mut [u32], probs: &mut [Prob], eg: &mut [Prob]) {
        for v in vals.iter_mut() {
            let small = c.bit(&mut probs[0], *v < 8);
            *v = if small { c.tree(&mut probs[1..], *v, 3) } else { c.exp_golomb(eg, v.wrapping_sub(8)) + 8 };
            let _ = c.bypass(*v & 1 == 1);
        }
    }

    #[test]
    fn roundtrip_mixed_symbols() {
        let mut rng = 0x1234_5678u32;
        let mut vals: Vec<u32> = (0..20000)
            .map(|i| {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                if i % 9 == 0 { rng % 100_000 } else { rng % 5 }
            })
            .collect();
        let orig = vals.clone();
        let mut enc = RangeEncoder::new();
        syntax(&mut enc, &mut vals, &mut [Prob::default(); 9], &mut [Prob::default(); 6]);
        let bytes = enc.finish();
        let mut out = vec![0u32; orig.len()];
        let mut dec = RangeDecoder::new(&bytes);
        syntax(&mut dec, &mut out, &mut [Prob::default(); 9], &mut [Prob::default(); 6]);
        assert_eq!(orig, out);
        assert!(!dec.overrun());
    }
}
