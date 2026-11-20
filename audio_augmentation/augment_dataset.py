import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from mel_spectrogram import (
    SAMPLE_RATE, add_background_noise, change_gain, load_audio, mel_spectrogram, pitch_shift, save_image, time_shift,
)

SCRIPT_DIR = Path(__file__).parent
AUDIO_EXTENSIONS ={".mp4", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mov", ".mkv"}


def augment(y, sr, args):
    # Random settings for every copy, so each one is different
    if args.pitch:
        y = pitch_shift(y, sr, random.uniform(-args.pitch, args.pitch))
    if args.shift:
        y = time_shift(y, sr, args.shift)
    if args.gain:
        y = change_gain(y, sr, -args.gain, args.gain)
    if args.noise:
        y = add_background_noise(y, sr, args.noise, args.min_snr, args.max_snr)

    # Gain can push the signal above 1.0, which would clip when saved
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak
    return y


def save_sample(y, base_path):
    # Saves base_path.wav and its mel spectrogram as base_path.png
    sf.write(base_path.with_suffix(".wav"), y, SAMPLE_RATE)
    save_image(mel_spectrogram(y), base_path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(
        description="Create augmented copies of every file in audio_samples/<class>/, "
                    "saved to augmented_samples/<class>/synt/ (originals in <class>/clean/)."
    )
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "audio_samples")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "augmented_samples")
    parser.add_argument("--copies", type=int, default=5, help="augmented copies per file (default 5)")
    parser.add_argument("--pitch", type=float, default=2.0, help="max pitch shift in semitones, +/- (default 2, 0 = off)")
    parser.add_argument("--shift", type=float, default=0.2, help="max time shift as fraction of length (default 0.2, 0 = off)")
    parser.add_argument("--gain", type=float, default=6.0, help="max gain change in dB, +/- (default 6, 0 = off)")
    parser.add_argument("--noise", type=Path, help="folder of background noise files (wav/flac/mp3/ogg)")
    parser.add_argument("--min-snr", type=float, default=3.0, help="min SNR in dB (default 3)")
    parser.add_argument("--max-snr", type=float, default=30.0, help="max SNR in dB (default 30)")
    args = parser.parse_args()

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        print(f"No audio files found in {args.input}")
        return

    for path in files:
        class_dir = args.output / path.parent.relative_to(args.input)  # keeps the class folder, e.g. fault1/
        clean_dir = class_dir / "clean"
        synt_dir = class_dir / "synt"
        clean_dir.mkdir(parents=True, exist_ok=True)
        synt_dir.mkdir(parents=True, exist_ok=True)

        y = load_audio(path)
        # Unaugmented original, converted to the same format (16 kHz mono wav) as the synthesized ones
        save_sample(y, clean_dir / path.stem)
        for i in range(args.copies):
            save_sample(augment(y, SAMPLE_RATE, args), synt_dir / f"{path.stem}_aug{i}")
        print(f"{path} -> {clean_dir.name}/ + {args.copies} files in {synt_dir.name}/ ({class_dir})")

    print(f"Done: {len(files) * args.copies} augmented files from {len(files)} originals")


if __name__ == "__main__":
    main()
