/// 32-bit Range Asymmetric Numeral Systems (rANS) entropy coder.
/// Reaches within ~0.01 bits/symbol of the Shannon entropy limit.
const PROB_BITS: u32 = 12;
const PROB_SCALE: u32 = 1 << PROB_BITS; // 4096
const RANS_L: u32 = 1 << 23;

#[derive(Clone, Debug)]
pub struct RansModel {
    pub freqs: [u16; 256],
    pub starts: [u16; 256],
    slot_to_sym: Vec<u8>,
}

impl RansModel {
    /// Build a normalized 12-bit rANS probability model from symbol counts.
    pub fn from_symbols(symbols: &[u8]) -> Self {
        let mut counts = [0u64; 256];
        for &s in symbols {
            counts[s as usize] += 1;
        }
        let total: u64 = counts.iter().sum();
        let mut freqs = [0u16; 256];

        if total == 0 {
            freqs[0] = PROB_SCALE as u16;
        } else {
            let mut active: Vec<usize> = (0..256).filter(|&i| counts[i] > 0).collect();
            let mut sum = 0u32;
            for &i in &active {
                let f = ((counts[i] * PROB_SCALE as u64 + total / 2) / total).max(1) as u16;
                freqs[i] = f;
                sum += f as u32;
            }
            // Adjust frequencies so sum == PROB_SCALE exactly
            active.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
            while sum > PROB_SCALE {
                for &i in &active {
                    if sum == PROB_SCALE {
                        break;
                    }
                    if freqs[i] > 1 {
                        freqs[i] -= 1;
                        sum -= 1;
                    }
                }
            }
            while sum < PROB_SCALE {
                for &i in &active {
                    if sum == PROB_SCALE {
                        break;
                    }
                    freqs[i] += 1;
                    sum += 1;
                }
            }
        }

        Self::from_freqs(freqs)
    }

    pub fn from_freqs(freqs: [u16; 256]) -> Self {
        let mut starts = [0u16; 256];
        let mut slot_to_sym = vec![0u8; PROB_SCALE as usize];
        let mut cum = 0u16;
        for s in 0..256 {
            starts[s] = cum;
            let f = freqs[s];
            for k in 0..f {
                slot_to_sym[(cum + k) as usize] = s as u8;
            }
            cum += f;
        }
        debug_assert_eq!(cum as u32, PROB_SCALE);
        Self { freqs, starts, slot_to_sym }
    }

    /// Serialize compact frequency table: [count_nonzero: u8] + [(sym: u8, freq: u16)]*
    pub fn serialize(&self, out: &mut Vec<u8>) {
        let active: Vec<(u8, u16)> = (0..256)
            .filter(|&i| self.freqs[i] > 0)
            .map(|i| (i as u8, self.freqs[i]))
            .collect();
        // Store count - 1 so 256 fits in u8
        out.push((active.len() - 1) as u8);
        for (sym, freq) in active {
            out.push(sym);
            out.extend_from_slice(&freq.to_le_bytes());
        }
    }

    pub fn deserialize(data: &[u8], pos: &mut usize) -> Result<Self, String> {
        if *pos >= data.len() {
            return Err("Unexpected EOF reading rANS table".into());
        }
        let n = (data[*pos] as usize) + 1;
        *pos += 1;
        if *pos + n * 3 > data.len() {
            return Err("Truncated rANS frequency table".into());
        }
        let mut freqs = [0u16; 256];
        let mut sum = 0u32;
        for _ in 0..n {
            let sym = data[*pos] as usize;
            let freq = u16::from_le_bytes([data[*pos + 1], data[*pos + 2]]);
            *pos += 3;
            freqs[sym] = freq;
            sum += freq as u32;
        }
        if sum != PROB_SCALE {
            return Err(format!("Corrupt rANS table: sum={sum} != {PROB_SCALE}"));
        }
        Ok(Self::from_freqs(freqs))
    }
}

/// Encode a byte stream using rANS + compact model header.
pub fn rans_encode(symbols: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
    if symbols.is_empty() {
        return out;
    }

    let model = RansModel::from_symbols(symbols);
    model.serialize(&mut out);

    // Encode symbols in reverse order
    let mut state: u32 = RANS_L;
    let mut bytes_rev = Vec::with_capacity(symbols.len());

    for &s in symbols.iter().rev() {
        let freq = model.freqs[s as usize] as u32;
        let start = model.starts[s as usize] as u32;
        let x_max = ((RANS_L as u64 >> PROB_BITS) << 8) * (freq as u64);
        while (state as u64) >= x_max {
            bytes_rev.push((state & 0xFF) as u8);
            state >>= 8;
        }
        state = ((state / freq) << PROB_BITS) + (state % freq) + start;
    }

    // Store payload length so decoder knows exact bounds
    let payload_len = (4 + bytes_rev.len()) as u32;
    out.extend_from_slice(&payload_len.to_le_bytes());
    // Emit final state big-endian, then reversed bytes (so forward read pops LIFO)
    out.extend_from_slice(&state.to_be_bytes());
    for &b in bytes_rev.iter().rev() {
        out.push(b);
    }
    out
}

/// Decode a rANS-encoded block starting at `data[*pos]`, advancing `*pos`.
pub fn rans_decode(data: &[u8], pos: &mut usize) -> Result<Vec<u8>, String> {
    if *pos + 4 > data.len() {
        return Err("Unexpected EOF reading rANS length".into());
    }
    let sym_len = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]) as usize;
    *pos += 4;
    if sym_len == 0 {
        return Ok(Vec::new());
    }

    let model = RansModel::deserialize(data, pos)?;
    if *pos + 4 > data.len() {
        return Err("Unexpected EOF reading rANS payload length".into());
    }
    let payload_len = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]) as usize;
    *pos += 4;
    if payload_len < 4 || *pos + payload_len > data.len() {
        return Err("Truncated rANS payload".into());
    }

    let end = *pos + payload_len;
    let mut state = u32::from_be_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;

    let mut out = Vec::with_capacity(sym_len);
    let mask = PROB_SCALE - 1;
    for _ in 0..sym_len {
        let slot = state & mask;
        let sym = model.slot_to_sym[slot as usize];
        out.push(sym);
        let freq = model.freqs[sym as usize] as u32;
        let start = model.starts[sym as usize] as u32;
        state = freq * (state >> PROB_BITS) + slot - start;
        while state < RANS_L && *pos < end {
            state = (state << 8) | (data[*pos] as u32);
            *pos += 1;
        }
    }
    *pos = end;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rans_roundtrip() {
        let mut data = Vec::new();
        for i in 0..5000 {
            let v = if i % 5 == 0 { (i % 17) as u8 } else { 0 };
            data.push(v);
        }
        let encoded = rans_encode(&data);
        assert!(encoded.len() < data.len() / 2, "rANS should compress skewed data well");
        let mut pos = 0;
        let decoded = rans_decode(&encoded, &mut pos).unwrap();
        assert_eq!(data, decoded);
        assert_eq!(pos, encoded.len());
    }
}
