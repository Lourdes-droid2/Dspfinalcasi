import soundfile as sf
from scipy.signal import resample
import numpy as np

def load_signal_from_wav(filename, target_fs=48000, normalize=False):
    """
    Carga un archivo WAV y lo remuestrea a target_fs si es necesario.

    Parámetros:
        filename (str): Ruta al archivo WAV.
        target_fs (int): Frecuencia de muestreo deseada.
        normalize (bool): Si True, normaliza la señal entre -1 y 1.

    Retorna:
        tuple: (señal como np.array, frecuencia de muestreo final) o (None, None) si hay error.
    """
    try:
        # Carga el archivo
        signal, original_fs = sf.read(filename)
        print(f"✔️ Archivo cargado: '{filename}' ({original_fs} Hz)")

        # Si tiene múltiples canales, usar solo el primero
        if signal.ndim > 1:
            signal = signal[:, 0]
            print(f"ℹ️ Se seleccionó solo el canal izquierdo (mono)")

        # Remuestreo si es necesario
        if original_fs != target_fs:
            print(f"🔁 Remuestreando de {original_fs} Hz a {target_fs} Hz...")
            num_samples = int(len(signal) * target_fs / original_fs)
            signal = resample(signal, num_samples)

        # Normalización opcional
        if normalize:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val
                print("📏 Señal normalizada entre -1 y 1.")

        return signal, target_fs

    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {filename}")
        return None, None
    except Exception as e:
        print(f"❌ Error al cargar o procesar '{filename}': {e}")
        return None, None
