# AI för ENM156

Vi har gjort en första modell som känner igen ljud från skruvmonteringen. Vi testade ExtraTrees och Logistic Regression. Båda fick samma resultat på testklippen, så vi valde att fortsätta med ExtraTrees.

## Ljuden

| Klass | Vad den betyder | Status |
|---|---|---|
| `felm4` | M4 är fel | `fault1` |
| `korrm4` | M4 är korrekt | `normal` |
| `medm8` | M8 finns med | `normal` |
| `utanm8` | M8 saknas | `fault2` |

Vi har förberett `bakgrund` som `idle`. Den är inte med i den tränade modellen eftersom vi inte hade sådana ljud när vi tränade.

## Så gjorde vi

Vi använde 324 ljudklipp från projektet. Koden läser in ljuden, gör mel spektrogram och räknar ut värden som modellen kan använda. Vi delade sedan upp klippen i träning, validering och test.

| Resultat | ExtraTrees | Logistic Regression |
|---|---:|---:|
| Träningsklipp | 227 | 227 |
| Valideringsklipp | 50 | 50 |
| Testklipp | 47 | 47 |
| Rätt på testet | 41 | 41 |
| Träffsäkerhet | 87,2 % | 87,2 % |
| Missade fel | 4 | 4 |
| Falska larm | 2 | 2 |

Resultat för varje klass finns i `ai/artifacts/comparison.json`.

## Köra koden

Lägg mappen `ai` bredvid gruppens mapp `audio_augmentation`. Installera biblioteken:

```bash
python3 -m pip install -r ai/requirements.txt
```

Kontrollera statusarna:

```bash
python3 ai/test_status.py
```

Kör modelljämförelsen:

```bash
python3 ai/compare_models.py --data audio_augmentation/audio_samples
```

Testa med en ljudfil:

```bash
python3 ai/predict.py SOKVAG_TILL_LJUDFIL.wav
```

Byt ut filnamnet mot en riktig WAV fil. Programmet visar vilken klass modellen tror att ljudet tillhör och vilken status den får.

`ai/send_to_ui.py` kan skicka resultat från sparade WAV filer till testgränssnittet i `ui`. Det är inte en färdig lösning för mikrofoninspelning på Pi.

## Det som återstår

Modellen är inte testad på Pi än. Vi behöver också träna om när bakgrundsljuden är färdiga. Testresultatet behöver tolkas försiktigt eftersom flera klipp kan komma från samma ursprungliga inspelning. Modellen missade fyra fel i vår testuppdelning.

När modellen kopplas till Pi behöver ljudet behandlas på samma sätt som under träningen. Pi gruppen behöver också kontrollera hastighet och minnesanvändning innan vi kan avgöra om modellen behöver göras mindre.
