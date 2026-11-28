# Till Pi gruppen

Vi har tränat en första AI modell och sparat den i `ai/artifacts/model.joblib`. Den kan läsa en WAV fil och ge en ljudklass. Vi har inte testat den med mikrofonen på Pi än.

Ljudet görs om till mono med 16 kHz. Modellen använder 2 sekunder ljud och ett mel spektrogram med 64 band, FFT 1024 och hop 256. Spektrogrammet blir 64 × 126.

Använd `predict_file` i `ai/model_utils.py` för att läsa in ljud och få en förutsägelse. `interpret_prediction` i `ai/status_mapping.py` gör om ljudklassen till en status.

| Ljudklass | Status |
|---|---|
| `felm4` | `fault1` |
| `korrm4` | `normal` |
| `medm8` | `normal` |
| `utanm8` | `fault2` |

`bakgrund` är förberedd som `idle`, men den klassen är inte tränad ännu. Okända klasser ger `uncertain`. Om de ska visas måste även servern och gränssnittet hantera de statusarna. I paketet finns en separat testkopia i `ui`. Byt inte ut gruppens egna UI filer utan att jämföra dem först.

## Testa

1. Installera biblioteken i `ai/requirements.txt`.
2. Kör `python3 ai/test_status.py`.
3. Testa en WAV fil med `python3 ai/predict.py FIL.wav`.
4. Använd samma ljudinställningar i Pi koden som vid träningen.
5. Mät hur snabbt modellen svarar och hur mycket minne den använder.
6. Testa med mikrofonen och skriv ner hur ofta modellen missar fel eller ger falska larm.

Vi har inte gjort quantization eftersom vi inte vet om det behövs. Modellen ska inte användas ensam för att stoppa maskinen automatiskt.
