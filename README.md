# CALF — Compact Audio Listening Format (`.calf`)

A next-generation psychoacoustic audio compression format & zero-dependency Rust codec (`calf-codec`) paired with an ultra-lightweight native YouTube-to-`.calf` desktop app.

- **Live Landing Page:** https://otterdays.github.io/calf-audio/
- **Compression:** **35.5 – 64.2 kbps** stereo (`-50%` to `-72%` smaller than 128k MP3, `22× – 39.7×` smaller than 16-bit PCM WAV)
- **Speed:** ~35 ms to encode 10 seconds of 44.1 kHz stereo audio (~285× realtime)

## Architecture
1. **Vorbis-Windowed Fast MDCT** (`codec/src/fft.rs`) — 2048-point `O(N log N)` MDCT with exact Princen-Bradley TDAC (`< 1e-5` error).
2. **24-Band Bark Psychoacoustic Model** (`codec/src/psycho.rs`) — Terhardt ATH, asymmetric simultaneous masking, and Spectral Flatness tonality.
3. **Joint M/S + Intensity Stereo + HBE + PNF** (`codec/src/calf.rs`) — High-frequency pan-angle coding, Harmonic Bandwidth Extension, and Perceptual Noise Filling.
4. **32-bit rANS Entropy Coder** (`codec/src/rans.rs`) — Custom Range Asymmetric Numeral Systems coder over 64-frame superblocks.

## Quickstart

```bash
# Launch the native desktop YouTube → .calf app
./run.sh

# Or build & use the Rust CLI directly
cd codec && cargo build --release
./target/release/calf-codec encode input.wav output.calf --profile balanced
./target/release/calf-codec decode output.calf restored.wav
./target/release/calf-codec synth-bench
```
