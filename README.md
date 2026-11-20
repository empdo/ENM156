```bash
pip install -r requirements.txt
```

Put your recordings (mp4, wav, mp3, ...) in one folder per class:

```
audio_samples/
├── fault1/
├── fault2/
└── fault3/
```

Then run:

```bash
python augment_dataset.py
```

Output:
```
augmented_samples/fault1/
├── clean/   sample.wav, sample.png
└── synt/    sample_aug0.wav, sample_aug0.png, ...
```

| Option | Default | Meaning |
|---|---|---|
| `--copies N` | 5 | modified copies per file |
| `--noise DIR` | off | folder with background noise files to mix in |
| `--min-snr` / `--max-snr` | 3 / 30 | noise level in dB (lower = more noise) |
| `--pitch` | 2 | max pitch shift in semitones (0 = off) |
| `--shift` | 0.2 | max time shift as fraction of the clip (0 = off) |
| `--gain` | 6 | max volume change in dB (0 = off) |
