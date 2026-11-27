```bash
pip install -r requirements.txt
```

Run:

```bash
python augment_dataset.py
```

Output:
```
augmented_audio/
├── fault1/
│   ├── train/   sample.wav, sample.png, sample_aug0.wav, sample_aug0.png, ...
│   ├── val/     sample.wav, sample.png   (real recordings only, no augmentation)
│   └── test/    sample.wav, sample.png
├── train.npz    all training spectrograms, for the model
├── val.npz
└── test.npz
```

Each class is split 70/15/15. Segments from the same recording (same id in the file name,
e.g. `medm8_561be482_000001`) are kept in the same split. A class with fewer than 3 recordings
is split in time order instead.

Load a split for training:

```python
d = np.load("augmented_audio/train.npz")
X, y = d["X"], d["y"]   # X: (N, 64, 126) float32 in 0..1, y: class index
classes = d["classes"]  # class names, in label order
files = d["files"]      # source wav of each row
```

| Option | Default | Meaning |
|---|---|---|
| `--copies N` | 5 | modified copies per training file |
| `--val` / `--test` | 0.15 / 0.15 | fraction of each class for validation / testing |
| `--seed` | 0 | random seed (same seed = same split) |
| `--overwrite` | off | replace an existing `augmented_audio/` |
| `--noise DIR` | off | folder with background noise files to mix in |
| `--min-snr` / `--max-snr` | 3 / 30 | noise level in dB (lower = more noise) |
| `--pitch` | 0.5 | max pitch shift in semitones (0 = off) |
| `--shift` | 0.1 | max time shift earlier, as fraction of the clip (0 = off) |
| `--gain` | 0 | max volume change in dB (0 = off, has no effect on the spectrograms since they are scaled to their own max) |
| `--mask` | 0.35 | max fraction of the clip end set to zero, so zero padding appears in every class (0 = off) |
| `--mask-prob` | 0.5 | share of augmented copies that get masked |
