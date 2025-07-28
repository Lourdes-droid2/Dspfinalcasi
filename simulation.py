import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from typing import Tuple, List, Optional, Dict

class SimuladorDOA:
    def __init__(self, fs: int = 16000):
        self.fs = fs
        self.c = 343
        self.array_geometry = None
        self.room = None
        self.signals = {}
        self.source_positions = []

    def crear_array_microfonos(self, num_mics: int = 4, spacing: float = 0.1, center_pos: List[float] = [2.0, 1.5, 1.0]) -> np.ndarray:
        positions = np.zeros((3, num_mics))
        for i in range(num_mics):
            positions[0, i] = center_pos[0] + (i - (num_mics - 1) / 2) * spacing
            positions[1, i] = center_pos[1]
            positions[2, i] = center_pos[2]
        self.array_geometry = {
            'num_mics': num_mics,
            'positions': positions,
            'spacing': spacing,
            'center': center_pos
        }
        return positions

    def simular_ambiente_anecoico(self, room_size: List[float] = [10, 8, 3], max_order=0, absorption=1, air_absorption=False):
        import pyroomacoustics as pra
        self.room = pra.ShoeBox(room_size, fs=self.fs, max_order=max_order, absorption=absorption, air_absorption=air_absorption)
        if self.array_geometry is None:
            self.crear_array_microfonos()
        self.room.add_microphone_array(self.array_geometry['positions'])
        return self.room

    def simular_ambiente_reverberante(self, room_size: List[float] = [6, 4, 3], rt60: float = 0.3):
        import pyroomacoustics as pra
        volume = np.prod(room_size)
        surface_area = 2 * (room_size[0]*room_size[1] + room_size[0]*room_size[2] + room_size[1]*room_size[2])
        absorption = 0.161 * volume / (rt60 * surface_area)
        absorption = min(absorption, 0.99)
        self.room = pra.ShoeBox(room_size, fs=self.fs, max_order=10, absorption=absorption, air_absorption=True)
        if self.array_geometry is None:
            self.crear_array_microfonos()
        self.room.add_microphone_array(self.array_geometry['positions'])
        return self.room

    def agregar_fuente(self, signal: np.ndarray, azimuth: float, distance: float = 2.0, elevation: float = 0.0) -> List[float]:
        if self.room is None:
            raise ValueError("Debe crear un ambiente primero")
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        center = self.array_geometry['center']
        source_pos = [
            float(center[0] + distance * np.cos(elevation_rad) * np.cos(azimuth_rad)),
            float(center[1] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad)),
            float(center[2] + distance * np.sin(elevation_rad))
        ]
        self.room.add_source(source_pos, signal=signal)
        self.source_positions.append({
            'position': source_pos,
            'azimuth': float(azimuth),
            'elevation': float(elevation),
            'distance': float(distance)
        })
        return source_pos

    def simular_propagacion(self, agregar_ruido: bool = False, snr_db: float = 20):
        if self.room is None:
            raise ValueError("Debe crear un ambiente y agregar fuentes primero")
        self.room.simulate()
        mic_signals = self.room.mic_array.signals
        if agregar_ruido:
            for i in range(mic_signals.shape[0]):
                signal_power = np.mean(mic_signals[i, :] ** 2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.sqrt(noise_power) * np.random.randn(mic_signals.shape[1])
                mic_signals[i, :] += noise
        self.signals = {
            'mic_signals': mic_signals,
            'fs': self.fs,
            'num_mics': mic_signals.shape[0],
            'length': mic_signals.shape[1],
            'snr_db': snr_db if agregar_ruido else None
        }

def crear_senal_prueba(tipo: str = "tono", duracion: float = 2.0, fs: int = 16000) -> np.ndarray:
    t = np.linspace(0, duracion, int(duracion * fs))
    if tipo == "tono":
        return np.sin(2 * np.pi * 1000 * t)
    elif tipo == "chirp":
        from scipy import signal
        return signal.chirp(t, f0=500, f1=2000, t1=duracion, method='linear')
    elif tipo == "ruido":
        return np.random.randn(len(t))
    elif tipo == "impulso":
        impulso = np.zeros(len(t))
        impulso[len(t)//4] = 1.0
        return impulso
    else:
        return np.sin(2 * np.pi * 1000 * t)