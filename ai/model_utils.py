from pathlib import Path
import joblib
import librosa
import numpy as np

SR = 16000
SECONDS = 2
N_MELS = 64
N_FFT = 1024
HOP = 256


def spectrogram_from_audio(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = librosa.util.fix_length(y, size=SR * SECONDS)
    spec = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                          hop_length=HOP, n_mels=N_MELS)
    db = librosa.power_to_db(spec, ref=np.max)
    return np.clip((db + 80.0) / 80.0, 0, 1).astype(np.float32)


def features_from_spectrogram(spec):
    if spec.shape != (64, 126):
        raise ValueError(f"Spektrogrammet ska vara 64x126, fick {spec.shape}")

    pooling = spec.reshape(8, 8, 9, 14).mean(axis=(1, 3)).ravel()
    freq_mean = spec.mean(axis=1).reshape(8, 8).mean(axis=1)
    freq_std = spec.std(axis=1).reshape(8, 8).mean(axis=1)
    time_mean = spec.mean(axis=0).reshape(9, 14).mean(axis=1)
    time_std = spec.std(axis=0).reshape(9, 14).mean(axis=1)
    return np.concatenate([pooling, freq_mean, freq_std, time_mean, time_std]).astype(np.float32)


def predict_file(model_file, audio_file):
    model = joblib.load(model_file)
    x = features_from_spectrogram(spectrogram_from_audio(audio_file))
    p = model.predict_proba(x[None, :])[0]
    classes = list(model.classes_)
    return {"prediction": classes[int(np.argmax(p))],
            "scores": {cls: float(s) for cls, s in zip(classes, p)}}
