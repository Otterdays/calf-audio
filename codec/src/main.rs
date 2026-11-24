mod calf;
mod fft;
mod psycho;
mod rans;
mod wav;

use std::env;
use std::io;
use std::process;
use std::time::Instant;

use calf::{decode_calf, encode_calf, read_bytes, read_calf_info, write_bytes, Profile};
use wav::{read_wav, write_wav, write_wav_to_writer, WavData};

fn print_usage() {
    eprintln!(
        "Compact Audio Listening Format (CALF v1) Codec\n\
         Usage:\n\
           calf-codec encode <input.wav> <output.calf> [--profile ultra|balanced|studio] [--title \"...\"]\n\
           calf-codec decode <input.calf> <output.wav | ->\n\
           calf-codec info   <input.calf>\n\
           calf-codec synth-bench\n"
    );
}

fn run_synth_bench() -> Result<(), String> {
    // Generate a 10-second rich stereo musical test signal (chord progression + bass + hi-hat shimmer)
    let sr = 44100u32;
    let dur = 10;
    let n = (sr as usize) * dur;
    let mut l = vec![0.0f32; n];
    let mut r = vec![0.0f32; n];
    let mut rng: u32 = 0x12345678;

    let chords: &[[f32; 4]] = &[
        [130.81, 261.63, 329.63, 392.00], // Cmaj
        [110.00, 220.00, 261.63, 329.63], // Am
        [87.31, 174.61, 220.00, 261.63],  // Fmaj
        [98.00, 196.00, 246.94, 293.66],  // Gmaj
    ];

    for i in 0..n {
        let t = i as f32 / sr as f32;
        let chord = chords[((t / 2.5) as usize) % chords.len()];
        let beat_phase = (t * 2.0) % 1.0;
        let env = (-beat_phase * 3.5).exp();

        let mut sl = 0.0f32;
        let mut sr_s = 0.0f32;
        for (idx, &f) in chord.iter().enumerate() {
            let w = 2.0 * std::f32::consts::PI * f * t;
            let tone = 0.16 * w.sin() + 0.05 * (2.0 * w).sin() + 0.02 * (3.0 * w).sin();
            let pan = 0.3 + 0.15 * idx as f32;
            sl += tone * (1.0 - pan * 0.4) * (0.4 + 0.6 * env);
            sr_s += tone * (0.6 + pan * 0.4) * (0.4 + 0.6 * env);
        }
        // Subtle hi-hat shimmer on off-beats
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let hat_phase = (t * 4.0) % 1.0;
        let hat_env = (-hat_phase * 28.0).exp();
        let noise = ((rng as i32) as f32 / 2147483648.0) * 0.06 * hat_env;
        l[i] = (sl + noise * 0.8).clamp(-0.98, 0.98);
        r[i] = (sr_s + noise * 1.1).clamp(-0.98, 0.98);
    }

    let wav = WavData { sample_rate: sr, channels: 2, samples: vec![l, r] };
    let wav_bytes = n * 4 + 44;
    let mp3_128_bytes = (128_000 / 8) * dur;
    let mp3_192_bytes = (192_000 / 8) * dur;

    println!("=== CALF v1 Benchmark (10s 44.1kHz Stereo Musical Signal) ===");
    println!("Raw WAV (16-bit PCM): {:>9} bytes (1411.2 kbps)", wav_bytes);
    println!("MP3 @ 128 kbps ref:   {:>9} bytes ( 128.0 kbps)", mp3_128_bytes);
    println!("MP3 @ 192 kbps ref:   {:>9} bytes ( 192.0 kbps)", mp3_192_bytes);
    println!("-------------------------------------------------------------");

    for profile in [Profile::Ultra, Profile::Balanced, Profile::Studio] {
        let t0 = Instant::now();
        let encoded = encode_calf(&wav, profile, "Synth Benchmark")?;
        let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let decoded = decode_calf(&encoded)?;
        let dec_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Compute Signal-to-Noise Ratio (SNR dB)
        let mut sig_p = 0.0f64;
        let mut err_p = 0.0f64;
        for ch in 0..2 {
            for i in 0..n {
                let s = wav.samples[ch][i] as f64;
                let d = decoded.samples[ch][i] as f64;
                sig_p += s * s;
                err_p += (s - d) * (s - d);
            }
        }
        let snr_db = 10.0 * (sig_p / err_p.max(1e-12)).log10();
        let kbps = (encoded.len() as f64 * 8.0) / (dur as f64 * 1000.0);
        let vs_mp3_128 = 100.0 * (1.0 - encoded.len() as f64 / mp3_128_bytes as f64);
        let vs_wav = wav_bytes as f64 / encoded.len() as f64;

        println!(
            "CALF [{:<8}]: {:>7} bytes | {:>5.1} kbps | {:>+5.1}% vs 128k MP3 | {:>4.1}x vs WAV | SNR {:>4.1} dB | enc {:>5.1}ms dec {:>5.1}ms",
            profile.name(),
            encoded.len(),
            kbps,
            -vs_mp3_128,
            vs_wav,
            snr_db,
            enc_ms,
            dec_ms
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let res = match args[1].as_str() {
        "encode" => {
            if args.len() < 4 {
                print_usage();
                process::exit(1);
            }
            let input = &args[2];
            let output = &args[3];
            let mut profile = Profile::Balanced;
            let mut title = String::new();
            let mut i = 4;
            while i < args.len() {
                match args[i].as_str() {
                    "--profile" | "-p" if i + 1 < args.len() => {
                        match Profile::from_str(&args[i + 1]) {
                            Ok(p) => profile = p,
                            Err(e) => {
                                eprintln!("Error: {e}");
                                process::exit(1);
                            }
                        }
                        i += 2;
                    }
                    "--title" | "-t" if i + 1 < args.len() => {
                        title = args[i + 1].clone();
                        i += 2;
                    }
                    _ => i += 1,
                }
            }
            let t0 = Instant::now();
            read_wav(input).and_then(|wav| {
                let encoded = encode_calf(&wav, profile, &title)?;
                write_bytes(output, &encoded)?;
                let info = read_calf_info(&encoded)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "{{\"bytes\":{},\"bitrate_kbps\":{:.2},\"duration_secs\":{:.3},\"sample_rate\":{},\"channels\":{},\"profile\":\"{}\",\"encode_ms\":{:.1}}}",
                    info.file_bytes, info.bitrate_kbps, info.duration_secs, info.sample_rate, info.channels, info.profile.name(), ms
                );
                Ok(())
            })
        }
        "decode" => {
            if args.len() < 4 {
                print_usage();
                process::exit(1);
            }
            let input = &args[2];
            let output = &args[3];
            read_bytes(input).and_then(|bytes| {
                let wav = decode_calf(&bytes)?;
                if output == "-" {
                    let stdout = io::stdout();
                    let mut handle = stdout.lock();
                    write_wav_to_writer(&mut handle, &wav)
                } else {
                    write_wav(output, &wav)
                }
            })
        }
        "info" => {
            if args.len() < 3 {
                print_usage();
                process::exit(1);
            }
            read_bytes(&args[2]).and_then(|bytes| {
                let info = read_calf_info(&bytes)?;
                println!(
                    "{{\"bytes\":{},\"bitrate_kbps\":{:.2},\"duration_secs\":{:.3},\"sample_rate\":{},\"channels\":{},\"profile\":\"{}\",\"title\":{:?}}}",
                    info.file_bytes, info.bitrate_kbps, info.duration_secs, info.sample_rate, info.channels, info.profile.name(), info.title
                );
                Ok(())
            })
        }
        "synth-bench" => run_synth_bench(),
        _ => {
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = res {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
