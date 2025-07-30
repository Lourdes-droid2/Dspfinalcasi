import os
import pandas as pd
import numpy as11 np
import matplotlib.pyplot as plt

from simulation import SimuladorDOA, crear_senal_prueba
from tdoa import EstimadorTDOA
from doa import EstimadorDOA

print("=== SIMULADOR DOA CON RUIDO AMBIENTE - PRUEBA ===")

# 1. PARÁMETROS ACÚSTICOS
ambiente_tipo = input("Tipo de ambiente (1=Anecoico, 2=Reverberante) [default: 2]: ") or "2"
room_size = [
    float(input("Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0),
    float(input("Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
    float(input("Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)
]
if ambiente_tipo == "2":
    rt60 = float(input("RT60 en segundos [default: 0.3]: ") or 0.3)
else:
    rt60 = None
absorption = float(input("Coeficiente de absorción de la sala [default: 1]: ") or 1)

# 2. PARÁMETROS DEL ARRAY
num_mics = int(input("Cantidad de micrófonos [default: 4]: ") or 4)
spacing = float(input("Separación entre micrófonos (m) [default: 0.1]: ") or 0.1)

# 3. PARÁMETROS DE LA FUENTE
azimuth_real = float(input("Azimuth en grados [default: 60]: ") or 60.0)
distancia_real = float(input("Distancia en metros [default: 2.0]: ") or 2.0)
while distancia_real < 2.0:
    print("Se debe mantener la condición de campo lejano con una distancia mayor a 2 metros.")
    distancia_real = float(input("Ingrese una distancia en metros mayor a 2.0: ") or 2.0)
elevacion_real = float(input("Elevación en grados [default: 0.0]: ") or 0.0)

# 4. PARÁMETROS DE SEÑAL Y SIMULACIÓN
fs = int(input("Frecuencia de muestreo [default: 16000]: ") or 16000)
metodo_tdoa = input("Método TDOA (correlacion/gcc/gcc_phat/gcc_scot) [default: gcc_phat]: ") or "gcc_phat"

# 5. CARGA DE SEÑAL
wav_path = input("Ingrese la ruta completa del archivo WAV para cargar (ejemplo: simulaciones/p227_004.wav): ")
while not wav_path or not os.path.isfile(wav_path):
    print("Ruta inválida o archivo no encontrado. Por favor, ingrese una ruta válida.")
    wav_path = input("Ingrese la ruta completa del archivo WAV para cargar (ejemplo: simulaciones/p227_004.wav): ")

from scipy.io import wavfile
fs_loaded, signal = wavfile.read(wav_path)
signal = signal.astype(np.float32) / 32767.0  # Normalizar

if fs_loaded != fs:
    print(f"Advertencia: La frecuencia de muestreo cargada ({fs_loaded}) es diferente de la configurada ({fs}). Se actualizará.")
    fs = fs_loaded

# 6. SIMULACIÓN
sim = SimuladorDOA(fs=fs)
sim.crear_array_microfonos(num_mics=num_mics, spacing=spacing)
if ambiente_tipo == "1":
    sim.simular_ambiente_anecoico(room_size=room_size, absorption=absorption, max_order=0, air_absorption=False)
else:
    sim.simular_ambiente_reverberante(room_size=room_size, rt60=rt60)
sim.agregar_fuente(signal, azimuth=azimuth_real, distance=distancia_real, elevation=elevacion_real)
sim.simular_propagacion(agregar_ruido=True, snr_db=20)
mic_signals = sim.signals['mic_signals']

# 7. ESTIMACIÓN TDOA
pares_microfonos = [(i, j) for i in range(num_mics) for j in range(i+1, num_mics)]
estimador_tdoa = EstimadorTDOA(fs=fs)
resultados_tdoa = {}
for i, j in pares_microfonos:
    resultado = estimador_tdoa.estimar_tdoa_par(mic_signals[i], mic_signals[j], metodo=metodo_tdoa)
    key = f"mic_{i+1}_mic_{j+1}"
    resultados_tdoa[key] = resultado

# 8. ESTIMACIÓN DOA
estimador_doa = EstimadorDOA(c=343.0)
resultados_doa = estimador_doa.calcular_angulo_arribo(resultados_tdoa, spacing, geometria='linear')
resultado_promedio = estimador_doa.promediar_angulos(resultados_doa, metodo='circular')

# 9. GUARDADO DE RESULTADOS
output_dir = "resultados_csv"
os.makedirs(output_dir, exist_ok=True)
nombre_experimento = input("Ingrese nombre del experimento (ej: num_mics, spacing, distance, elevation, fs, rt60, metodo_tdoa): ")
csv_path = os.path.join(output_dir, f"resultados_{nombre_experimento}.csv")

df_resultado = pd.DataFrame([{
    'ambiente': ambiente_tipo,
    'num_mics': num_mics,
    'spacing': spacing,
    'distance': distancia_real,
    'elevation': elevacion_real,
    'fs': fs,
    'rt60': rt60 if ambiente_tipo == "2" else None,
    'metodo_tdoa': metodo_tdoa,
    'error_promedio': abs(resultado_promedio['angulo_promedio_deg'] - azimuth_real) if resultado_promedio and resultado_promedio.get('valido', False) else None
}])

if os.path.isfile(csv_path):
    df_existente = pd.read_csv(csv_path)
    df_final = pd.concat([df_existente, df_resultado], ignore_index=True)
else:
    df_final = df_resultado
df_final.to_csv(csv_path, index=False)
print(f"Resultado guardado en {csv_path}")

print("\n=== Análisis DOA/TDOA completado ===")