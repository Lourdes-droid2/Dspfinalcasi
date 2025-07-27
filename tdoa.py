import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from typing import Tuple, Dict, Optional

class EstimadorTDOA:
    def __init__(self, fs: int = 48000, c: float = 343.0):
        self.fs = fs
        self.c = c
        self.dt = 1.0 / fs

    def correlacion_cruzada(self, x1: np.ndarray, x2: np.ndarray, metodo: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        correlation = np.correlate(x1, x2, mode=metodo)
        if metodo == 'full':
            lags = np.arange(-len(x2) + 1, len(x1))
        elif metodo == 'same':
            lags = np.arange(-len(x2)//2, len(x1)//2)
        else:
            lags = np.arange(len(x1) - len(x2) + 1)
        return correlation, lags

    def gcc_basico(self, x1: np.ndarray, x2: np.ndarray, ventana: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        if ventana:
            window = signal.get_window(ventana, min_len)
            x1 = x1 * window
            x2 = x2 * window
        X1 = fft(x1, n=2*min_len-1)
        X2 = fft(x2, n=2*min_len-1)
        cross_spectrum = X1 * np.conj(X2)
        gcc = np.real(ifft(cross_spectrum))
        gcc = np.fft.fftshift(gcc)
        lags = np.arange(-min_len + 1, min_len)
        return gcc, lags

    def gcc_phat(self, x1: np.ndarray, x2: np.ndarray, epsilon: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        n_fft = 2 * min_len - 1
        X1 = fft(x1, n=n_fft)
        X2 = fft(x2, n=n_fft)
        cross_spectrum = X1 * np.conj(X2)
        magnitude = np.abs(cross_spectrum) + epsilon
        phat_spectrum = cross_spectrum / magnitude
        gcc_phat = np.real(ifft(phat_spectrum))
        gcc_phat = np.fft.fftshift(gcc_phat)
        lags = np.arange(-min_len + 1, min_len)
        return gcc_phat, lags

    def gcc_scot(self, x1: np.ndarray, x2: np.ndarray, epsilon: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        min_len = min(len(x1), len(x2))
        x1 = x1[:min_len]
        x2 = x2[:min_len]
        n_fft = 2 * min_len - 1
        X1 = fft(x1, n=n_fft)
        X2 = fft(x2, n=n_fft)
        S11 = X1 * np.conj(X1)
        S22 = X2 * np.conj(X2)
        S12 = X1 * np.conj(X2)
        scot_weight = 1.0 / (np.sqrt(S11 * S22) + epsilon)
        scot_spectrum = S12 * scot_weight
        gcc_scot = np.real(ifft(scot_spectrum))
        gcc_scot = np.fft.fftshift(gcc_scot)
        lags = np.arange(-min_len + 1, min_len)
        return gcc_scot, lags

    def estimar_tdoa_par(self, mic1: np.ndarray, mic2: np.ndarray, metodo: str = 'gcc_phat',
                         busqueda_pico: str = 'max', ventana_busqueda: Optional[Tuple[float, float]] = None) -> Dict:
        if metodo == 'correlacion':
            correlation, lags = self.correlacion_cruzada(mic1, mic2)
        elif metodo == 'gcc':
            correlation, lags = self.gcc_basico(mic1, mic2)
        elif metodo == 'gcc_phat':
            correlation, lags = self.gcc_phat(mic1, mic2)
        elif metodo == 'gcc_scot':
            correlation, lags = self.gcc_scot(mic1, mic2)
        else:
            raise ValueError(f"Método desconocido: {metodo}")
        if ventana_busqueda:
            min_samples = int(ventana_busqueda[0] * self.fs)
            max_samples = int(ventana_busqueda[1] * self.fs)
            valid_indices = (lags >= min_samples) & (lags <= max_samples)
            if np.any(valid_indices):
                correlation = correlation[valid_indices]
                lags = lags[valid_indices]
        if busqueda_pico == 'max':
            max_idx = np.argmax(np.abs(correlation))
            tdoa_samples = lags[max_idx]
            confidence = np.abs(correlation[max_idx])
        elif busqueda_pico == 'interpolacion':
            max_idx = np.argmax(np.abs(correlation))
            if 0 < max_idx < len(correlation) - 1:
                y1, y2, y3 = np.abs(correlation[max_idx-1:max_idx+2])
                a = (y1 - 2*y2 + y3) / 2
                b = (y3 - y1) / 2
                if a != 0:
                    offset = -b / (2*a)
                    tdoa_samples = lags[max_idx] + offset
                else:
                    tdoa_samples = lags[max_idx]
            else:
                tdoa_samples = lags[max_idx]
            confidence = np.abs(correlation[max_idx])
        tdoa_seconds = tdoa_samples * self.dt
        correlation_normalized = correlation / np.max(np.abs(correlation))
        return {
            'tdoa_samples': float(tdoa_samples),
            'tdoa_seconds': float(tdoa_seconds),
            'confidence': float(confidence),
            'correlation': correlation,
            'lags': lags,
            'correlation_normalized': correlation_normalized,
            'metodo': metodo,
            'max_idx': int(max_idx) if 'max_idx' in locals() else None
        }