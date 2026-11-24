mod bench;
mod calf;
mod calf_v1;
mod fft;
mod psycho;
mod rans;
mod wav;

use std::env;
use std::io;
use std::process;
use std::time::Instant;

use calf::{decode_calf, encode_calf, read_bytes, read_calf_info, write_bytes, Profile};
use wav::{read_wav, write_wav, write_wav_to_writer};

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
    let sr = 44100u32;
    let dur = 10usize;
    let music = bench::music_signal(sr, dur);
    let (trans, onsets) = bench::transient_signal(sr, dur);
    let mp3_128_bytes = (128_000 / 8) * dur;
    for (name, wav, ons) in [("music", &music, None), ("transients", &trans, Some(onsets.as_slice()))] {
        println!("=== {name}: {dur}s 44.1kHz stereo | 128k MP3 ref {mp3_128_bytes} bytes ===");
        for profile in [Profile::Ultra, Profile::Balanced, Profile::Studio] {
            let t0 = Instant::now();
            let encoded = encode_calf(wav, profile, "bench")?;
            let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            let decoded = decode_calf(&encoded)?;
            let dec_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let m = bench::measure(wav, &decoded, ons);
            let kbps = (encoded.len() as f64 * 8.0) / (dur as f64 * 1000.0);
            let pe = m.pre_echo_db.map(|v| format!(" | pre-echo {v:>6.1} dB")).unwrap_or_default();
            println!(
                "  [{:<8}] {:>7} B {:>5.1} kbps ({:>+5.1}% vs MP3-128) | SNR {:>5.2} | segSNR {:>5.2} | band err {:>4.2} dB{} | enc {:>5.1}ms dec {:>5.1}ms",
                profile.name(), encoded.len(), kbps, -100.0 * (1.0 - encoded.len() as f64 / mp3_128_bytes as f64),
                m.snr_db, m.seg_snr_db, m.band_err_db, pe, enc_ms, dec_ms
            );
        }
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
                let wav = if bytes.starts_with(b"CALF") {
                    calf_v1::decode_v1(&bytes)?
                } else {
                    decode_calf(&bytes)?
                };
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
                let info = if bytes.starts_with(b"CALF") {
                    calf_v1::read_v1_info(&bytes)?
                } else {
                    read_calf_info(&bytes)?
                };
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
