<div align="center">
  <img src="docs/icon.svg" alt="CALF Audio Logo" width="84" />
  <h1>CALF Audio (<code>.calf</code>)</h1>
  <p>
    <strong>Compact Audio Listening Format</strong> — Turn YouTube videos into tiny, high-quality <code>.calf</code> audio files on your computer.
  </p>
  <p>
    <a href="https://otterdays.github.io/calf-audio/">Live Landing Page</a> •
    <a href="#how-to-install-and-run">Quick Start</a> •
    <a href="#how-to-use-the-app">User Guide</a> •
    <a href="#quality-modes-explained">Quality Modes</a> •
    <a href="#for-developers-and-power-users">CLI &amp; Technical Docs</a>
  </p>
</div>

---

## What is CALF?

**CALF** is two things in one simple package:

1. **A super-compact audio file format (`.calf`)** that shrinks music and speech down to **half (or even less) the file size of a standard MP3**, while keeping clear stereo sound.
2. **A lightweight desktop app** where you paste a YouTube link, click **Convert**, and play back your saved `.calf` files right away—no browser tabs, ads, or complicated menus.

> **How much space does it save?**  
> On a typical 10-second stereo music clip:
> - **Uncompressed WAV:** ~1,764 KB
> - **Standard 128k MP3:** ~160 KB
> - **CALF (`balanced`):** ~60 KB (**62.6% smaller than MP3**, **~29× smaller than WAV**)

---

## Quality Modes Explained

When converting audio, you can choose between three quality presets depending on whether you want the smallest possible file or maximum audio detail:

| Preset | Typical Bitrate | Space Saved vs. 128k MP3 | Size vs. Uncompressed WAV | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`ultra`** | ~35.5 kbps | **~72% smaller** | ~39.7× smaller | Podcasts, voice, or squeezing maximum songs onto limited storage |
| **`balanced`** *(default)* | ~47.9 kbps | **~63% smaller** | ~29.4× smaller | Everyday music listening — best mix of tiny size and crisp sound |
| **`studio`** | ~64.2 kbps | **~50% smaller** | ~22.0× smaller | High-detail music where you want extra clarity in cymbals and vocals |

*Encoding 10 seconds of stereo audio takes roughly **35 milliseconds** (~285× faster than real-time).*

---

## What You Need Before Starting

Even if you have never written code before, you only need **two free tools** installed on your computer. Everything else (including Python 3.15, `yt-dlp` for YouTube, and `ffmpeg` for audio decoding) is downloaded automatically when you start the app.

1. **Rust (Cargo)** — Needed once to build the fast audio encoder.
   - Install from [rustup.rs](https://rustup.rs/) or run in your terminal:
     ```bash
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
     ```
2. **`uv` (Python manager)** — Automatically sets up Python and the app's libraries in one step.
   - Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) or run:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

> **Note for Linux users:** The desktop window uses Python's built-in `tkinter` library and plays audio with `paplay`, `aplay`, `pw-play`, or your system's default audio player. If you prefer a browser interface instead of a desktop window, see [Web UI Mode](#prefer-using-your-web-browser).

---

## How to Install and Run

Open your **Terminal** app and copy-paste these commands one at a time:

### 1. Download this project
```bash
git clone https://github.com/Otterdays/calf-audio.git
cd calf-audio
```
*(If you downloaded the ZIP from GitHub instead, unzip it, open a Terminal inside that folder, and skip straight to Step 2.)*

### 2. Launch the app
```bash
./run.sh
```
**What happens the first time you run this?**
- It builds the `calf-codec` audio engine automatically (this takes a few seconds the first time, and is instant afterward).
- `uv` sets up Python and downloads `yt-dlp` and `ffmpeg` into a self-contained folder inside the project (it won't clutter the rest of your computer).
- The **CALF Audio** desktop window opens!

---

## How to Use the App

The app window has three simple sections:

1. **Output Location**  
   - Shows where your `.calf` files will be saved (defaults to the `output/` folder inside the project).
   - Click **Browse…** to pick any folder (like your `Music` or `Downloads` folder), then click **Save**.
2. **Convert YouTube URL to `.calf`**  
   - Paste any YouTube link into the text box.
   - Pick a quality mode (`ultra`, `balanced`, or `studio`).
   - *(Optional)* Check **"Also save 128k MP3 side-by-side for comparison"** if you want both files so you can hear and compare the file sizes yourself.
   - Click **Convert**.
   - **Want to test without a YouTube link?** Click **Run 8s Synth Test** to instantly generate an 8-second musical test track and see how small `.calf` makes it.
3. **Saved `.calf` Files**  
   - Lists every `.calf` file in your output folder along with its size, bitrate, length, and quality mode.
   - Click a file in the list and press **▶ Play Selected `.calf`** to listen, or **■ Stop** to stop playback.
   - Click **Open Output Folder** to view your saved files in your file manager.

### Prefer Using Your Web Browser?

Instead of the desktop window, you can run the included local web app:

```bash
uv run python app/server.py
```

Then open **http://127.0.0.1:8765** in any web browser to convert, browse, and play `.calf` files directly in your browser.

---

## For Developers and Power Users

### Command-Line Usage (`calf-codec`)

You can build and run the standalone Rust CLI directly with **zero external Rust crate dependencies**:

```bash
# 1. Build the release binary
cd codec
cargo build --release

# 2. Encode a WAV file to .calf (profiles: ultra, balanced, studio)
./target/release/calf-codec encode input.wav output.calf --profile balanced --title "My Track"

# 3. Decode a .calf file back to standard WAV (use "-" to stream WAV to stdout)
./target/release/calf-codec decode output.calf restored.wav

# 4. Inspect metadata (size, bitrate, duration, sample rate, channels, profile, title)
./target/release/calf-codec info output.calf

# 5. Run the built-in 10-second synthetic music & transient benchmark
./target/release/calf-codec synth-bench
```

<details>
<summary><strong>🧠 How <code>.calf</code> Compression Works Under the Hood (Click to expand)</strong></summary>

1. **Vorbis-Windowed Fast MDCT (`codec/src/fft.rs`)**  
   Splits stereo audio into overlapping 2048-sample frequency blocks using an `O(N log N)` Modified Discrete Cosine Transform with Princen-Bradley time-domain aliasing cancellation (`< 1e-5` reconstruction error).
2. **24-Band Bark Psychoacoustic Model (`codec/src/psycho.rs`)**  
   Models how human ears actually hear—calculating the Terhardt absolute threshold of hearing, asymmetric masking (loud sounds hiding quiet nearby frequencies), and Spectral Flatness tonality to discard inaudible detail.
3. **Joint M/S + Intensity Stereo + HBE + PNF (`codec/src/calf.rs`)**  
   - **Mid/Side coding** removes redundant information shared between left and right speakers.
   - **Intensity Stereo** replaces high-frequency stereo channels with a single spectrum plus compact 5-bit pan angles.
   - **Harmonic Bandwidth Extension (HBE)** & **Perceptual Noise Filling (PNF)** reconstruct airy upper frequencies cleanly without wasting bits.
4. **32-bit rANS Entropy Coder (`codec/src/rans.rs`)**  
   Packs the quantized frequency coefficients tightly across 64-frame superblocks using Range Asymmetric Numeral Systems.

</details>

---

## What's Inside This Project

| File / Folder | What It Does |
| :--- | :--- |
| `run.sh` | One-click launcher script that builds the codec (if needed) and opens the desktop app |
| `app/desktop.py` | Native desktop app interface (built with Python `tkinter`) |
| `app/server.py` | Shared YouTube-to-`.calf` converter logic + optional local web server (`http://127.0.0.1:8765`) |
| `app/index.html` | Interface for the optional local web server |
| `codec/` | Zero-dependency Rust source code for the `.calf` encoder, decoder, and benchmark tool |
| `docs/` & `web/` | Static landing page and logo (`docs/icon.svg`) hosted on GitHub Pages |
| `output/` | Default folder where converted `.calf` audio files are saved |
| `config.json` | Automatically created file that remembers your chosen output folder and default profile |

---

## Troubleshooting

- **`Permission denied` when running `./run.sh`**  
  Make the script executable first, then run it again:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```
- **`cargo: command not found` or `uv: command not found`**  
  After installing Rust or `uv`, close and reopen your Terminal window so your system detects the new commands.
- **No sound when clicking Play in the desktop app**  
  The desktop player looks for `paplay`, `aplay`, `pw-play`, or `xdg-open` on Linux. Alternatively, run `uv run python app/server.py` and play files right inside your web browser at `http://127.0.0.1:8765`.
- **Running the Rust unit tests**  
  ```bash
  cd codec && cargo test
  ```
