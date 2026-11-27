import argparse
import random
import subprocess
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from audiomentations import AddBackgroundNoise, Gain, Shift

SAMPLE_RATE = 16000
N_MELS = 64
CLIP_SECONDS = 2.0  # every clip is padded/cropped to this, so all spectrograms get the same shape
MIN_DB = -80.0      # power_to_db floor, used to scale spectrograms to 0..1


def load_audio(path, sr=SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-vn",               # skip video
        "-ac", "1",          # mono
        "-ar", str(sr),      # resample
        "-f", "f32le", "-",  # raw float32 to stdout
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def fix_length(y, sr=SAMPLE_RATE, seconds=CLIP_SECONDS):
    return librosa.util.fix_length(y, size=int(sr * seconds))


def normalize(S_db, min_db=MIN_DB):
    # dB values (min_db..0) -> 0..1, the range the model gets as input
    return np.clip((S_db - min_db) / -min_db, 0.0, 1.0).astype(np.float32)


def mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def pitch_shift(y, sr, steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def add_background_noise(y, sr, noise_dir, min_snr_db=3.0, max_snr_db=30.0):
    transform = AddBackgroundNoise(sounds_path=noise_dir, min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=1.0)
    return transform(samples=y, sample_rate=sr)

def change_gain(y, sr, min_gain_db=-6.0, max_gain_db=6.0):
    transform = Gain(min_gain_db=min_gain_db, max_gain_db=max_gain_db, p=1.0)
    return transform(samples=y, sample_rate=sr)

def time_shift(y, sr, max_shift=0.1):
    # Only moves the sound earlier, so the zeros end up at the end like the real padding
    # (zeros at the start never happen in the real data)
    transform = Shift(min_shift=-max_shift, max_shift=0.0, shift_unit="fraction", rollover=False, p=1.0)
    return transform(samples=y, sample_rate=sr)

def mask_end(y, max_fraction=0.35):
    # Zeroes the end of the clip, like the zero padding in the *_padded files, so that
    # padding shows up in every class and the model can't use it to tell classes apart
    cut = int(len(y) * (1 - random.uniform(0, max_fraction)))
    y = y.copy()
    y[cut:] = 0
    return y

def save_image(S_db, out_path, sr=SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel spectrogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save a mel spectrogram image of an audio/video file.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--noise", type=Path, help="folder of background noise files (wav/flac/mp3/ogg)")
    parser.add_argument("--min-snr", type=float, default=3.0, help="min SNR in dB (default 3)")
    parser.add_argument("--max-snr", type=float, default=30.0, help="max SNR in dB (default 30)")
    parser.add_argument("--gain", type=float, default=0.0, help="max gain change in dB, +/- (default 0 = off)")
    parser.add_argument("--shift", type=float, default=0.1, help="max time shift earlier, as fraction of length (default 0.1, 0 = off)")
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output or in_path.with_suffix(".png")

    y = load_audio(in_path)
    print(f"Loaded {len(y) / SAMPLE_RATE:.2f} s of audio from {in_path}")
    y = pitch_shift(y, SAMPLE_RATE, 3)
    if args.shift:
        y = time_shift(y, SAMPLE_RATE, args.shift)
    if args.gain:
        y = change_gain(y, SAMPLE_RATE, -args.gain, args.gain)
    if args.noise:
        y = add_background_noise(y, SAMPLE_RATE, args.noise, args.min_snr, args.max_snr)
        print(f"Added background noise from {args.noise}")
    save_image(mel_spectrogram(y), out_path)
    print(f"Saved spectrogram to {out_path}")


if __name__ == "__main__":
    main()
