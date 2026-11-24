#!/usr/bin/env python3
"""
CALF (Compact Audio Listening Format) — Extremely lightweight YT2Audio MVP server.
"""

import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import urllib.parse
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
CONFIG_PATH = ROOT_DIR / "config.json"
CALF_BIN = ROOT_DIR / "codec" / "target" / "release" / "calf-codec"
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_OUTPUT_DIR = str((ROOT_DIR / "output").resolve())

_config_lock = threading.Lock()


def load_config() -> dict:
    with _config_lock:
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if "output_dir" in cfg:
                    return cfg
            except Exception:
                pass
        cfg = {"output_dir": DEFAULT_OUTPUT_DIR, "default_profile": "balanced"}
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg


def save_config(cfg: dict) -> dict:
    with _config_lock:
        out_dir = os.path.expanduser(cfg.get("output_dir", DEFAULT_OUTPUT_DIR).strip() or DEFAULT_OUTPUT_DIR)
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        new_cfg = {
            "output_dir": str(out_path),
            "default_profile": cfg.get("default_profile", "balanced"),
        }
        CONFIG_PATH.write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")
        return new_cfg


def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean[:120] or "audio"


def pick_directory_native(initial_dir: str) -> str | None:
    """Try native Linux folder pickers (zenity, kdialog, tkinter)."""
    for cmd in (
        ["zenity", "--file-selection", "--directory", f"--filename={initial_dir}/", "--title=Select Output Folder"],
        ["kdialog", "--getexistingdirectory", initial_dir, "--title", "Select Output Folder"],
    ):
        if shutil.which(cmd[0]):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
                return None
            except Exception:
                pass
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial_dir, title="Select Output Folder")
        root.destroy()
        return chosen or None
    except Exception:
        return None


def encode_wav_to_calf(wav_path: Path, calf_path: Path, profile: str, title: str) -> dict:
    if not CALF_BIN.exists():
        raise RuntimeError(f"calf-codec binary not found at {CALF_BIN}. Run `cargo build --release` in codec/.")
    proc = subprocess.run(
        [str(CALF_BIN), "encode", str(wav_path), str(calf_path), "--profile", profile, "--title", title],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "calf-codec encode failed")
    return json.loads(proc.stdout.strip())


def get_calf_info(calf_path: Path) -> dict | None:
    if not CALF_BIN.exists():
        return None
    proc = subprocess.run(
        [str(CALF_BIN), "info", str(calf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        return None


def convert_youtube_to_calf(url: str, profile: str, output_dir: str, also_mp3: bool = False) -> dict:
    out_path = Path(os.path.expanduser(output_dir)).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="calf_yt_") as tmpdir:
        tmp = Path(tmpdir)
        dl_template = str(tmp / "source.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": dl_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title") or "youtube_audio"

        downloaded = next((p for p in tmp.iterdir() if p.name.startswith("source.")), None)
        if not downloaded:
            raise RuntimeError("yt-dlp did not produce an audio file")

        wav_path = tmp / "pcm.wav"
        ff = subprocess.run(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(downloaded),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if ff.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {ff.stderr[-300:]}")

        wav_bytes = wav_path.stat().st_size
        safe_name = sanitize_filename(title)
        calf_path = out_path / f"{safe_name}.calf"
        stats = encode_wav_to_calf(wav_path, calf_path, profile, title)

        mp3_path = None
        mp3_actual_bytes = None
        if also_mp3:
            mp3_path = out_path / f"{safe_name}.mp3"
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", str(wav_path), "-b:a", "128k", str(mp3_path)],
                capture_output=True,
                check=False,
            )
            if mp3_path.exists():
                mp3_actual_bytes = mp3_path.stat().st_size

        dur = max(stats["duration_secs"], 0.001)
        mp3_128_est = mp3_actual_bytes or int((128_000 / 8) * dur)
        mp3_192_est = int((192_000 / 8) * dur)
        calf_bytes = stats["bytes"]

        return {
            "title": title,
            "filename": calf_path.name,
            "saved_path": str(calf_path),
            "mp3_path": str(mp3_path) if mp3_path and mp3_path.exists() else None,
            "profile": profile,
            "duration_secs": stats["duration_secs"],
            "calf_bytes": calf_bytes,
            "calf_kbps": stats["bitrate_kbps"],
            "encode_ms": stats["encode_ms"],
            "wav_bytes": wav_bytes,
            "mp3_128_bytes": mp3_128_est,
            "mp3_192_bytes": mp3_192_est,
            "savings_vs_mp3_128_pct": round((1.0 - calf_bytes / max(mp3_128_est, 1)) * 100.0, 1),
            "savings_vs_mp3_192_pct": round((1.0 - calf_bytes / max(mp3_192_est, 1)) * 100.0, 1),
            "ratio_vs_wav": round(wav_bytes / max(calf_bytes, 1), 1),
        }


def generate_demo_calf(profile: str, output_dir: str) -> dict:
    """Generate an 8-second stereo musical test WAV and encode it to .calf for instant testing."""
    out_path = Path(os.path.expanduser(output_dir)).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    sr = 44100
    dur = 8
    n = sr * dur
    chords = [
        (130.81, 261.63, 329.63, 392.00),
        (110.00, 220.00, 261.63, 329.63),
        (87.31, 174.61, 220.00, 261.63),
        (98.00, 196.00, 246.94, 293.66),
    ]

    with tempfile.TemporaryDirectory(prefix="calf_demo_") as tmpdir:
        wav_path = Path(tmpdir) / "demo.wav"
        frames = bytearray()
        two_pi = 2.0 * math.pi
        for i in range(n):
            t = i / sr
            chord = chords[int(t / 2.0) % len(chords)]
            env = math.exp(-((t * 2.0) % 1.0) * 3.2)
            sl = 0.0
            sr_s = 0.0
            for idx, f in enumerate(chord):
                w = two_pi * f * t
                tone = 0.16 * math.sin(w) + 0.045 * math.sin(2.0 * w) + 0.015 * math.sin(3.0 * w)
                pan = 0.3 + 0.15 * idx
                sl += tone * (1.0 - pan * 0.4) * (0.4 + 0.6 * env)
                sr_s += tone * (0.6 + pan * 0.4) * (0.4 + 0.6 * env)
            il = max(-32767, min(32767, int(sl * 32767)))
            ir = max(-32767, min(32767, int(sr_s * 32767)))
            frames += struct.pack("<hh", il, ir)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(frames)

        wav_bytes = wav_path.stat().st_size
        calf_path = out_path / f"CALF_Synth_Demo_{profile}.calf"
        stats = encode_wav_to_calf(wav_path, calf_path, profile, f"CALF Synth Demo ({profile})")
        mp3_128_est = int((128_000 / 8) * dur)
        mp3_192_est = int((192_000 / 8) * dur)
        calf_bytes = stats["bytes"]
        return {
            "title": f"CALF Synth Demo ({profile})",
            "filename": calf_path.name,
            "saved_path": str(calf_path),
            "mp3_path": None,
            "profile": profile,
            "duration_secs": stats["duration_secs"],
            "calf_bytes": calf_bytes,
            "calf_kbps": stats["bitrate_kbps"],
            "encode_ms": stats["encode_ms"],
            "wav_bytes": wav_bytes,
            "mp3_128_bytes": mp3_128_est,
            "mp3_192_bytes": mp3_192_est,
            "savings_vs_mp3_128_pct": round((1.0 - calf_bytes / max(mp3_128_est, 1)) * 100.0, 1),
            "savings_vs_mp3_192_pct": round((1.0 - calf_bytes / max(mp3_192_est, 1)) * 100.0, 1),
            "ratio_vs_wav": round(wav_bytes / max(calf_bytes, 1), 1),
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html = (APP_DIR / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if path in ("/about", "/format", "/calf"):
            html = (WEB_DIR / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if path == "/api/config":
            cfg = load_config()
            presets = [
                {"label": "./output", "path": DEFAULT_OUTPUT_DIR},
                {"label": "~/Music", "path": str((Path.home() / "Music").resolve())},
                {"label": "~/Downloads", "path": str((Path.home() / "Downloads").resolve())},
            ]
            self._send_json({"config": cfg, "presets": presets})
            return

        if path == "/api/files":
            cfg = load_config()
            out_dir = Path(cfg["output_dir"])
            files = []
            if out_dir.exists():
                for p in sorted(out_dir.glob("*.calf"), key=lambda x: x.stat().st_mtime, reverse=True):
                    info = get_calf_info(p)
                    if info:
                        files.append(
                            {
                                "filename": p.name,
                                "path": str(p),
                                "title": info.get("title") or p.stem,
                                "bytes": info["bytes"],
                                "bitrate_kbps": info["bitrate_kbps"],
                                "duration_secs": info["duration_secs"],
                                "profile": info["profile"],
                            }
                        )
            self._send_json({"output_dir": str(out_dir), "files": files})
            return

        if path == "/api/play":
            qs = urllib.parse.parse_qs(parsed.query)
            fname = qs.get("file", [""])[0]
            cfg = load_config()
            out_dir = Path(cfg["output_dir"]).resolve()
            target = (out_dir / os.path.basename(fname)).resolve()
            if not target.exists() or target.parent != out_dir or target.suffix != ".calf":
                self._send_json({"error": "File not found"}, status=404)
                return

            proc = subprocess.run(
                [str(CALF_BIN), "decode", str(target), "-"],
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                self._send_json({"error": proc.stderr.decode("utf-8", "ignore")}, status=500)
                return

            wav_bytes = proc.stdout
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(wav_bytes)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json()
        except Exception as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, status=400)
            return

        if path == "/api/config":
            try:
                cfg = save_config(body)
                self._send_json({"ok": True, "config": cfg})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/browse":
            cfg = load_config()
            chosen = pick_directory_native(cfg["output_dir"])
            if chosen:
                cfg = save_config({**cfg, "output_dir": chosen})
                self._send_json({"ok": True, "selected": chosen, "config": cfg})
            else:
                self._send_json({"ok": False, "selected": None, "config": cfg})
            return

        if path == "/api/convert":
            url = (body.get("url") or "").strip()
            profile = (body.get("profile") or "balanced").strip().lower()
            also_mp3 = bool(body.get("also_mp3", False))
            if not url:
                self._send_json({"error": "Please enter a YouTube URL."}, status=400)
                return
            cfg = load_config()
            out_dir = (body.get("output_dir") or cfg["output_dir"]).strip()
            if out_dir != cfg["output_dir"]:
                cfg = save_config({**cfg, "output_dir": out_dir})
            try:
                result = convert_youtube_to_calf(url, profile, cfg["output_dir"], also_mp3=also_mp3)
                self._send_json({"ok": True, "result": result})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/demo":
            profile = (body.get("profile") or "balanced").strip().lower()
            cfg = load_config()
            try:
                result = generate_demo_calf(profile, cfg["output_dir"])
                self._send_json({"ok": True, "result": result})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "Not found"}, status=404)


def main():
    port = int(os.environ.get("PORT", "8765"))
    cfg = load_config()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CALF — Compact Audio Listening Format (.calf) MVP running at http://127.0.0.1:{port}")
    print(f"Output folder: {cfg['output_dir']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
