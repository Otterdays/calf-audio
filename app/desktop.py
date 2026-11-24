#!/usr/bin/env python3
"""
CALF Audio (.calf) — Ultra-Lightweight Native Desktop App (Tkinter + yt-dlp + calf-codec).
No browser, no localhost server — launches directly as a native desktop window.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from server import (
    CALF_BIN,
    convert_youtube_to_calf,
    generate_demo_calf,
    get_calf_info,
    load_config,
    save_config,
)

BG = "#0e1118"
PANEL = "#161b26"
PANEL2 = "#1e2536"
FG = "#edf2fc"
MUTED = "#8c99b5"
ACCENT = "#5c7cff"
MINT = "#38e8b0"
DANGER = "#ff6b7a"


def fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"


class CalfDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CALF Audio (.calf) — Compact Audio Listening Format")
        self.geometry("760x660")
        self.minsize(680, 580)
        self.configure(bg=BG)

        self.cfg = load_config()
        self.play_proc: subprocess.Popen | None = None
        self.tmp_wav: str | None = None

        self._setup_style()
        self._build_ui()
        self.refresh_files()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Sans", 10))
        style.configure("Card.TLabel", background=PANEL, foreground=FG, font=("Sans", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Sans", 9))
        style.configure("Title.TLabel", background=BG, foreground=FG, font=("Sans", 16, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Sans", 9))
        style.configure("Section.TLabel", background=PANEL, foreground=MUTED, font=("Sans", 8, "bold"))
        style.configure("TButton", background=PANEL2, foreground=FG, borderwidth=0, padding=(12, 7), font=("Sans", 9, "bold"))
        style.map("TButton", background=[("active", "#2a334a")])
        style.configure("Primary.TButton", background=ACCENT, foreground="#071018", borderwidth=0, padding=(14, 7), font=("Sans", 9, "bold"))
        style.map("Primary.TButton", background=[("active", MINT)])
        style.configure("TCheckbutton", background=PANEL, foreground=MUTED, font=("Sans", 9))
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=FG, arrowcolor=FG)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)

        # Header
        hdr = ttk.Frame(outer)
        hdr.pack(fill="x", pady=(0, 14))
        ttk.Label(hdr, text="CALF Audio  (.calf)", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hdr,
            text="Compact Audio Listening Format — Native Desktop YouTube → .calf Converter",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # 1. Output Location Card
        dir_card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        dir_card.pack(fill="x", pady=(0, 12))
        ttk.Label(dir_card, text="OUTPUT LOCATION", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        dir_row = ttk.Frame(dir_card, style="Card.TFrame")
        dir_row.pack(fill="x")
        self.dir_var = tk.StringVar(value=self.cfg["output_dir"])
        dir_entry = tk.Entry(
            dir_row,
            textvariable=self.dir_var,
            bg=PANEL2,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            font=("Monospace", 9),
            highlightthickness=1,
            highlightbackground="#2b354f",
            highlightcolor=ACCENT,
        )
        dir_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        ttk.Button(dir_row, text="Browse…", command=self.browse_dir).pack(side="left", padx=(0, 6))
        ttk.Button(dir_row, text="Save", command=self.save_dir).pack(side="left")

        # 2. YouTube to .CALF Converter Card
        conv_card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        conv_card.pack(fill="x", pady=(0, 12))
        ttk.Label(conv_card, text="CONVERT YOUTUBE URL TO .CALF", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        url_row = ttk.Frame(conv_card, style="Card.TFrame")
        url_row.pack(fill="x")
        self.url_var = tk.StringVar()
        url_entry = tk.Entry(
            url_row,
            textvariable=self.url_var,
            bg=PANEL2,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            font=("Sans", 10),
            highlightthickness=1,
            highlightbackground="#2b354f",
            highlightcolor=ACCENT,
        )
        url_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        self.profile_var = tk.StringVar(value="balanced")
        prof_box = ttk.Combobox(
            url_row,
            textvariable=self.profile_var,
            values=["ultra", "balanced", "studio"],
            state="readonly",
            width=10,
        )
        prof_box.pack(side="left", padx=(0, 8))

        self.convert_btn = ttk.Button(url_row, text="Convert", style="Primary.TButton", command=self.start_convert)
        self.convert_btn.pack(side="left")

        opt_row = ttk.Frame(conv_card, style="Card.TFrame")
        opt_row.pack(fill="x", pady=(10, 0))
        self.also_mp3_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_row,
            text="Also save 128k MP3 side-by-side for comparison",
            variable=self.also_mp3_var,
        ).pack(side="left")
        self.demo_btn = ttk.Button(opt_row, text="Run 8s Synth Test", command=self.start_demo)
        self.demo_btn.pack(side="right")

        self.status_var = tk.StringVar(value="Ready. Paste a YouTube link or run the 8s synth test.")
        self.status_lbl = tk.Label(
            conv_card,
            textvariable=self.status_var,
            bg=PANEL,
            fg=MINT,
            font=("Monospace", 9),
            anchor="w",
            justify="left",
            wraplength=690,
        )
        self.status_lbl.pack(fill="x", pady=(10, 0))

        # 3. Saved .CALF Files + Native Player
        lib_card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        lib_card.pack(fill="both", expand=True)

        lib_hdr = ttk.Frame(lib_card, style="Card.TFrame")
        lib_hdr.pack(fill="x", pady=(0, 8))
        ttk.Label(lib_hdr, text="SAVED .CALF FILES", style="Section.TLabel").pack(side="left")
        ttk.Button(lib_hdr, text="Refresh", command=self.refresh_files).pack(side="right")

        self.listbox = tk.Listbox(
            lib_card,
            bg=PANEL2,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#071018",
            relief="flat",
            borderwidth=0,
            font=("Monospace", 9),
            activestyle="none",
        )
        self.listbox.pack(fill="both", expand=True, pady=(0, 10))
        self.file_paths: list[Path] = []

        ctrl_row = ttk.Frame(lib_card, style="Card.TFrame")
        ctrl_row.pack(fill="x")
        ttk.Button(ctrl_row, text="▶ Play Selected .calf", style="Primary.TButton", command=self.play_selected).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl_row, text="■ Stop", command=self.stop_playback).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl_row, text="Open Output Folder", command=self.open_output_folder).pack(side="right")

    def set_status(self, msg: str, color: str = MINT):
        self.after(0, lambda: (self.status_var.set(msg), self.status_lbl.config(fg=color)))

    def browse_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.dir_var.get(), title="Select Output Folder")
        if chosen:
            self.dir_var.set(chosen)
            self.save_dir()

    def save_dir(self):
        try:
            self.cfg = save_config({**self.cfg, "output_dir": self.dir_var.get()})
            self.dir_var.set(self.cfg["output_dir"])
            self.set_status(f"Output folder saved: {self.cfg['output_dir']}", MINT)
            self.refresh_files()
        except Exception as e:
            messagebox.showerror("Invalid Folder", str(e))

    def open_output_folder(self):
        path = self.cfg["output_dir"]
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", path])

    def refresh_files(self):
        self.listbox.delete(0, tk.END)
        self.file_paths.clear()
        out_dir = Path(self.cfg["output_dir"])
        if not out_dir.exists():
            return
        for p in sorted(out_dir.glob("*.calf"), key=lambda x: x.stat().st_mtime, reverse=True):
            info = get_calf_info(p)
            if info:
                self.file_paths.append(p)
                dur = int(info["duration_secs"])
                line = f" {p.name:<38} | {fmt_bytes(info['bytes']):>8} | {info['bitrate_kbps']:>5.1f} kbps | {dur // 60}:{dur % 60:02d} | {info['profile']}"
                self.listbox.insert(tk.END, line)

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.after(0, lambda: (self.convert_btn.config(state=state), self.demo_btn.config(state=state)))

    def start_convert(self):
        url = self.url_var.get().strip()
        if not url:
            self.set_status("Please paste a YouTube URL first.", DANGER)
            return
        self.save_dir()
        profile = self.profile_var.get()
        also_mp3 = self.also_mp3_var.get()
        self._set_busy(True)
        self.set_status("Downloading audio via yt-dlp & compressing with CALF v2…", "#9ec5ff")

        def worker():
            try:
                r = convert_youtube_to_calf(url, profile, self.cfg["output_dir"], also_mp3=also_mp3)
                msg = (
                    f"Saved {r['filename']} — {fmt_bytes(r['calf_bytes'])} ({r['calf_kbps']:.1f} kbps) | "
                    f"-{r['savings_vs_mp3_128_pct']}% vs 128k MP3 | {r['ratio_vs_wav']}x smaller than WAV ({r['encode_ms']:.0f}ms)"
                )
                self.set_status(msg, MINT)
                self.after(0, self.refresh_files)
            except Exception as e:
                self.set_status(f"Error: {e}", DANGER)
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def start_demo(self):
        self.save_dir()
        profile = self.profile_var.get()
        self._set_busy(True)
        self.set_status("Generating 8s stereo musical test & encoding to .calf…", "#9ec5ff")

        def worker():
            try:
                r = generate_demo_calf(profile, self.cfg["output_dir"])
                msg = (
                    f"Demo saved: {r['filename']} — {fmt_bytes(r['calf_bytes'])} ({r['calf_kbps']:.1f} kbps) | "
                    f"-{r['savings_vs_mp3_128_pct']}% vs 128k MP3 | {r['ratio_vs_wav']}x smaller than WAV"
                )
                self.set_status(msg, MINT)
                self.after(0, self.refresh_files)
            except Exception as e:
                self.set_status(f"Error: {e}", DANGER)
            finally:
                self._set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def stop_playback(self):
        if self.play_proc and self.play_proc.poll() is None:
            self.play_proc.terminate()
        self.play_proc = None
        if self.tmp_wav and os.path.exists(self.tmp_wav):
            try:
                os.remove(self.tmp_wav)
            except OSError:
                pass
            self.tmp_wav = None

    def play_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            self.set_status("Select a .calf file from the list first.", DANGER)
            return
        target = self.file_paths[sel[0]]
        self.stop_playback()

        fd, wav_path = tempfile.mkstemp(prefix="calf_play_", suffix=".wav")
        os.close(fd)
        self.tmp_wav = wav_path

        proc = subprocess.run([str(CALF_BIN), "decode", str(target), wav_path], capture_output=True, text=True)
        if proc.returncode != 0:
            self.set_status(f"Decode failed: {proc.stderr.strip()}", DANGER)
            return

        for player in ("paplay", "aplay", "pw-play"):
            if shutil.which(player):
                self.play_proc = subprocess.Popen([player, wav_path])
                self.set_status(f"Playing {target.name} via {player}…", MINT)
                return

        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", wav_path])
            self.set_status(f"Opened decoded {target.name} in system audio player.", MINT)
        else:
            self.set_status("No audio player found (paplay/aplay/pw-play).", DANGER)

    def _on_close(self):
        self.stop_playback()
        self.destroy()


if __name__ == "__main__":
    CalfDesktopApp().mainloop()
