import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from mel_spectrogram import (
    SAMPLE_RATE, add_background_noise, change_gain, fix_length, load_audio, mel_spectrogram, normalize,
    pitch_shift, save_image, time_shift,
)

SCRIPT_DIR = Path(__file__).parent
AUDIO_EXTENSIONS = {".mp4", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mov", ".mkv"}
SPLITS = ("train", "val", "test")
# e.g. val_medm8_561be482_000002_padded -> recording "561be482", segment 2
NAME_PATTERN = re.compile(r"_([0-9a-f]{8})_(\d+)(?:_padded)?$")


def recording_and_segment(path):
    m = NAME_PATTERN.search(path.stem)
    return (m.group(1), int(m.group(2))) if m else (path.stem, 0)


def split_class(files, fractions, rng):
    """Split one class into train/val/test.

    Segments cut from the same recording sound almost identical, so they are kept in the
    same split; otherwise the model is tested on audio it has already heard. Classes with
    fewer than 3 recordings can't be split that way, so their segments are split in time
    order instead (start of the recording -> train, end -> test).
    """
    recordings = defaultdict(list)
    for f in files:
        recordings[recording_and_segment(f)[0]].append(f)

    targets = [frac * len(files) for frac in fractions]
    result = {s: [] for s in SPLITS}

    if len(recordings) < len(SPLITS):
        ordered = sorted(files, key=lambda f: recording_and_segment(f))
        n_train = round(targets[0])
        n_val = max(1, round(targets[1]))
        result["train"] = ordered[:n_train]
        result["val"] = ordered[n_train:n_train + n_val]
        result["test"] = ordered[n_train + n_val:]
        return result

    groups = list(recordings.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # place big recordings first, small ones fill the gaps
    for group in groups:
        # Give the recording to the split that is furthest below its target size
        split = max(SPLITS, key=lambda s: targets[SPLITS.index(s)] - len(result[s]))
        result[split].extend(group)
    return result


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
    # Saves base_path.wav and base_path.png, returns the normalized spectrogram for the .npz
    S_db = mel_spectrogram(y)
    sf.write(base_path.with_suffix(".wav"), y, SAMPLE_RATE)
    save_image(S_db, base_path.with_suffix(".png"))
    return normalize(S_db)


def main():
    parser = argparse.ArgumentParser(
        description="Split audio_samples/<class>/ into train/val/test, augment the training files and save "
                    "wav + png to augmented_audio/<class>/<split>/ and all spectrograms to augmented_audio/<split>.npz."
    )
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "audio_samples")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "augmented_audio")
    parser.add_argument("--overwrite", action="store_true", help="delete the output folder first if it exists")
    parser.add_argument("--val", type=float, default=0.15, help="fraction of each class used for validation (default 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="fraction of each class used for testing (default 0.15)")
    parser.add_argument("--seed", type=int, default=0, help="random seed, same seed = same split and augmentations")
    parser.add_argument("--copies", type=int, default=5, help="augmented copies per training file (default 5)")
    parser.add_argument("--pitch", type=float, default=2.0, help="max pitch shift in semitones, +/- (default 2, 0 = off)")
    parser.add_argument("--shift", type=float, default=0.2, help="max time shift as fraction of length (default 0.2, 0 = off)")
    parser.add_argument("--gain", type=float, default=6.0, help="max gain change in dB, +/- (default 6, 0 = off)")
    parser.add_argument("--noise", type=Path, help="folder of background noise files (wav/flac/mp3/ogg)")
    parser.add_argument("--min-snr", type=float, default=3.0, help="min SNR in dB (default 3)")
    parser.add_argument("--max-snr", type=float, default=30.0, help="max SNR in dB (default 30)")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            parser.error(f"{args.output} already exists, use --overwrite to replace it")
        shutil.rmtree(args.output)

    random.seed(args.seed)
    np.random.seed(args.seed)  # audiomentations draws its random values from numpy
    rng = random.Random(args.seed)
    fractions = (1 - args.val - args.test, args.val, args.test)

    classes = sorted(d.name for d in args.input.iterdir() if d.is_dir())
    if not classes:
        print(f"No class folders found in {args.input}")
        return

    data = {s: {"X": [], "y": [], "files": []} for s in SPLITS}
    for label, cls in enumerate(classes):
        files = sorted(p for p in (args.input / cls).iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        splits = split_class(files, fractions, rng)
        counts = ", ".join(f"{s} {len(splits[s])}" for s in SPLITS)
        print(f"{cls}: {len(files)} files -> {counts}")

        for split, split_files in splits.items():
            out_dir = args.output / cls / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for path in split_files:
                y = fix_length(load_audio(path))
                samples = [(path.stem, y)]
                # Only the training set is augmented, val/test stay real recordings
                if split == "train":
                    samples += [(f"{path.stem}_aug{i}", augment(y, SAMPLE_RATE, args)) for i in range(args.copies)]
                for name, audio in samples:
                    data[split]["X"].append(save_sample(audio, out_dir / name))
                    data[split]["y"].append(label)
                    data[split]["files"].append(f"{cls}/{split}/{name}.wav")

    for split in SPLITS:
        out_path = args.output / f"{split}.npz"
        np.savez_compressed(
            out_path,
            X=np.stack(data[split]["X"]),  # (N, n_mels, frames), float32 in 0..1
            y=np.array(data[split]["y"], dtype=np.int64),
            classes=np.array(classes),
            files=np.array(data[split]["files"]),
        )
        print(f"Saved {out_path} with X shape {np.stack(data[split]['X']).shape}")


if __name__ == "__main__":
    main()
