
from pathlib import Path
import sys
import sounddevice as sd

# Hitta gruppens kod för mel-spektrogram
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "audio_augmentation"))

from mel_spectrogram import (
    SAMPLE_RATE,
    CLIP_SECONDS,
    fix_length,
    mel_spectrogram,
    normalize,
    save_image,
)

# Spela in 2 sekunder ljud
CHUNK_SIZE = int(SAMPLE_RATE * CLIP_SECONDS)

print("Spelar in 2 sekunder ljud...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
) as stream:
    audio, overflowed = stream.read(CHUNK_SIZE)

if overflowed:
    print("Varning: ljuddata kan ha tappats!")

# Omvandla ljudet till en endimensionell signal
y = fix_length(audio[:, 0])

# Skapa mel-spektrogram med gruppens funktion
S_db = mel_spectrogram(y)

# Normalisera till värden mellan 0 och 1
S_normalized = normalize(S_db)

print("Mel-spektrogram skapat!")
print("Shape:", S_normalized.shape)
print("Min:", S_normalized.min())
print("Max:", S_normalized.max())

# Spara en bild så att vi kan se spektrogrammet
output = Path(__file__).parent / "test_mel.png"
save_image(S_db, output)

print("Bild sparad:", output)
