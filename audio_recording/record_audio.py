
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 32000  # Två sekund ljud

print("Mikrofonen lyssnar...")
print("Tryck Ctrl+C för att avsluta.")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    ) as stream:

        while True:
            audio, overflowed = stream.read(CHUNK_SIZE)

            # Beräkna ljudets styrka
            volume = np.sqrt(np.mean(audio ** 2))

            print(f"Ljudstyrka: {volume:.4f}")

            if overflowed:
                print("Varning: ljuddata har tappats!")

except KeyboardInterrupt:
    print("\nInspelningen avslutades.")
