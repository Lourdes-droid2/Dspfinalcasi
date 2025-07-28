import os
import numpy as np
from scipy.io import wavfile
from simulation import SimuladorDOA, crear_senal_prueba
from tdoa import EstimadorTDOA
from doa import EstimadorDOA
from evaluacion import EvaluadorDOA

print("=== SIMULADOR DOA CON RUIDO AMBIENTE - PRUEBA ===")

# Parámetros de ambiente
ambiente_tipo = input("   Tipo de ambiente (1=Anecoico, 2=Reverberante) [default: 2]: ") or "2"
sim = SimuladorDOA(fs=16000)

if ambiente_tipo == "1":
    sim.simular_ambiente_anecoico(
        room_size=[float(input("   Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0), 
                    float(input("   Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
                    float(input("   Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)],
                    max_order = 0,
                    absorption = float(input("Ingrese el coeficiente de absorción de la sala (default 1): ") or 1),
                    air_absorption=False
    )
else:
    sim.simular_ambiente_reverberante(
        room_size=[float(input("   Ancho del recinto (X) en metros [default: 6.0]: ") or 6.0), 
                    float(input("   Largo del recinto (Y) en metros [default: 4.0]: ") or 4.0),
                    float(input("   Altura del recinto (Z) en metros [default: 3.0]: ") or 3.0)], 
        rt60= float(input("   RT60 en segundos [default: 0.3]: ") or 0.3),
    )

# Cargar señal WAV
wav_path = input("Ingrese la ruta completa del archivo WAV para cargar (ejemplo: simulaciones/p227_004.wav): ")
while not wav_path or not os.path.isfile(wav_path):
    print("Ruta inválida o archivo no encontrado. Por favor, ingrese una ruta válida.")
    wav_path = input("Ingrese la ruta completa del archivo WAV para cargar (ejemplo: simulaciones/p227_004.wav): ")

fs_loaded, signal = wavfile.read(wav_path)
signal = signal.astype(np.float32) / 32767.0

if fs_loaded != sim.fs:
    print(f"Advertencia: La frecuencia de muestreo cargada ({fs_loaded}) es diferente de la configurada ({sim.fs}). Se actualizará.")
    sim.fs = fs_loaded

# Configuración de fuente
azimuth_real = float(input(f"     Azimuth en grados [default: 60]: ") or 60.0)
distancia_real = float(input("     Distancia en metros [default: 2.0]: ") or 2.0)
while distancia_real < 2.0:
    print("Se debe mantener la condición de campo lejano con una distancia mayor a 2 metros.")
    distancia_real = float(input("Ingrese una distancia en metros mayor a 2.0: ") or 2.0)
elevacion_real = float(input("     Elevación en grados [default: 0.0]: ") or 0.0)

sim.agregar_fuente(signal, azimuth=azimuth_real, distance=distancia_real, elevation=elevacion_real)
sim.simular_propagacion(agregar_ruido=True, snr_db=20)

# TDOA
estimador_tdoa = EstimadorTDOA(fs=sim.fs)
mic_signals = sim.signals['mic_signals']
num_mics = mic_signals.shape[0]

pares_microfonos = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
resultados_tdoa = {}
for i, j in pares_microfonos:
    resultado = estimador_tdoa.estimar_tdoa_par(mic_signals[i], mic_signals[j], metodo="gcc_phat")
    key = f"mic_{i+1}_mic_{j+1}"
    resultados_tdoa[key] = resultado

# DOA
estimador_doa = EstimadorDOA(c=343.0)
spacing = sim.array_geometry['spacing']
array_positions = sim.array_geometry['positions']
resultados_doa = estimador_doa.calcular_angulo_arribo(resultados_tdoa, spacing, geometria='linear')

# Promediado de ángulos
resultado_promedio = estimador_doa.promediar_angulos(resultados_doa, metodo='circular')

# Exportar resultados a CSV
import pandas as pd
df_resultados = pd.DataFrame([
    {
        'mic_pair': k,
        'tdoa_seconds': v['tdoa_seconds'],
        'doa_deg': resultados_doa[k]['angulo_deg'] if resultados_doa[k]['valido'] else None,
        'confidence': v['confidence']
    }
    for k, v in resultados_tdoa.items()
])
df_resultados.to_csv("resultados_doa_tdoa.csv", index=False)
print("Resultados exportados a resultados_doa_tdoa.csv")

# Visualización
estimador_doa.visualizar_estimaciones(resultados_doa, angulo_real=azimuth_real)

# Reporte
reporte = estimador_doa.generar_reporte(resultados_doa, resultado_promedio, angulo_real=azimuth_real)
print(reporte)