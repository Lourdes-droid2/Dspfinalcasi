import soundfile as sf
import numpy as np
from scipy.signal import resample

def load_signal_from_wav(filename, target_fs=48000, normalize=False):
    try:
        signal, original_fs = sf.read(filename)
        print(f"✔️ Archivo cargado: '{filename}' ({original_fs} Hz)")
        if signal.ndim > 1:
            signal = signal[:, 0]
        if original_fs != target_fs:
            num_samples = int(len(signal) * target_fs / original_fs)
            signal = resample(signal, num_samples)
        if normalize:
            signal = signal / np.max(np.abs(signal))
        return signal, target_fs
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {filename}")
        return None, None
    except Exception as e:
        print(f"❌ Error al cargar archivo: {e}")
        return None, None