import librosa
import numpy as np


def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def fix_size(spec):
    if spec.shape[1] < 128:
        pad = 128 - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)))
    else:
        spec = spec[:, :128]
    return spec
