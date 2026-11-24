use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    /// Planar f32 samples in [-1.0, 1.0], one Vec per channel.
    pub samples: Vec<Vec<f32>>,
}

pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<WavData, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open WAV: {e}"))?;
    let mut r = BufReader::new(file);
    let mut header = [0u8; 12];
    r.read_exact(&mut header).map_err(|e| format!("Invalid WAV header: {e}"))?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Not a valid RIFF/WAVE file".into());
    }

    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_bytes: Option<Vec<u8>> = None;

    loop {
        let mut chunk_hdr = [0u8; 8];
        if r.read_exact(&mut chunk_hdr).is_err() {
            break;
        }
        let id = &chunk_hdr[0..4];
        let size = u32::from_le_bytes([chunk_hdr[4], chunk_hdr[5], chunk_hdr[6], chunk_hdr[7]]) as usize;

        if id == b"fmt " {
            let mut fmt = vec![0u8; size];
            r.read_exact(&mut fmt).map_err(|e| format!("Failed reading fmt chunk: {e}"))?;
            if fmt.len() < 16 {
                return Err("Truncated fmt chunk".into());
            }
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if id == b"data" {
            let mut data = vec![0u8; size];
            r.read_exact(&mut data).map_err(|e| format!("Failed reading data chunk: {e}"))?;
            data_bytes = Some(data);
            break;
        } else {
            let skip = size + (size & 1);
            let mut junk = vec![0u8; skip];
            let _ = r.read_exact(&mut junk);
        }
    }

    let raw = data_bytes.ok_or("No data chunk found in WAV")?;
    if channels == 0 || channels > 2 {
        return Err(format!("Unsupported channel count: {channels} (must be 1 or 2)"));
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frame_bytes = bytes_per_sample * channels as usize;
    let num_frames = raw.len() / frame_bytes;
    let mut samples = vec![Vec::with_capacity(num_frames); channels as usize];

    match (audio_format, bits_per_sample) {
        (1, 16) => {
            for f in 0..num_frames {
                for ch in 0..channels as usize {
                    let off = f * frame_bytes + ch * 2;
                    let v = i16::from_le_bytes([raw[off], raw[off + 1]]);
                    samples[ch].push(v as f32 / 32768.0);
                }
            }
        }
        (1, 24) => {
            for f in 0..num_frames {
                for ch in 0..channels as usize {
                    let off = f * frame_bytes + ch * 3;
                    let sign = if raw[off + 2] & 0x80 != 0 { 0xFF } else { 0x00 };
                    let v = i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], sign]);
                    samples[ch].push(v as f32 / 8388608.0);
                }
            }
        }
        (3, 32) => {
            for f in 0..num_frames {
                for ch in 0..channels as usize {
                    let off = f * frame_bytes + ch * 4;
                    let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    samples[ch].push(v.clamp(-1.0, 1.0));
                }
            }
        }
        _ => {
            return Err(format!(
                "Unsupported WAV format={} bits={} (convert to 16-bit PCM first)",
                audio_format, bits_per_sample
            ));
        }
    }

    Ok(WavData { sample_rate, channels, samples })
}

pub fn write_wav<P: AsRef<Path>>(path: P, wav: &WavData) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create WAV: {e}"))?;
    let mut w = BufWriter::new(file);
    write_wav_to_writer(&mut w, wav)
}

pub fn write_wav_to_writer<W: Write>(w: &mut W, wav: &WavData) -> Result<(), String> {
    let ch = wav.channels as usize;
    let num_frames = wav.samples[0].len();
    let data_size = (num_frames * ch * 2) as u32;
    let riff_size = 36 + data_size;
    let byte_rate = wav.sample_rate * (ch as u32) * 2;
    let block_align = (ch as u16) * 2;

    let mut hdr = Vec::with_capacity(44);
    hdr.extend_from_slice(b"RIFF");
    hdr.extend_from_slice(&riff_size.to_le_bytes());
    hdr.extend_from_slice(b"WAVE");
    hdr.extend_from_slice(b"fmt ");
    hdr.extend_from_slice(&16u32.to_le_bytes());
    hdr.extend_from_slice(&1u16.to_le_bytes()); // PCM
    hdr.extend_from_slice(&wav.channels.to_le_bytes());
    hdr.extend_from_slice(&wav.sample_rate.to_le_bytes());
    hdr.extend_from_slice(&byte_rate.to_le_bytes());
    hdr.extend_from_slice(&block_align.to_le_bytes());
    hdr.extend_from_slice(&16u16.to_le_bytes());
    hdr.extend_from_slice(b"data");
    hdr.extend_from_slice(&data_size.to_le_bytes());

    w.write_all(&hdr).map_err(|e| format!("Write WAV header failed: {e}"))?;

    let mut buf = Vec::with_capacity(num_frames * ch * 2);
    for f in 0..num_frames {
        for c in 0..ch {
            let s = (wav.samples[c][f].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }
    w.write_all(&buf).map_err(|e| format!("Write WAV samples failed: {e}"))?;
    w.flush().map_err(|e| format!("Flush WAV failed: {e}"))?;
    Ok(())
}
