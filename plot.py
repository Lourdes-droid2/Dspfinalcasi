import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulacion import SimuladorDOA, crear_senal_prueba
from tdoa import EstimadorTDOA
from doa import EstimadorDOA

sns.set(style="whitegrid")
plt.style.use('default')

# Parámetros a variar
num_mics_list = [2, 4, 8]
spacing_list = [0.05, 0.1, 0.2]
distance_list = [2.0, 3.0, 5.0]
elevation_list = [0, 15, 30]
fs_list = [8000, 16000, 32000]
rt60_list = [0.1, 0.3, 0.6]
ambientes = ['anechoico', 'reverberante']
metodos_tdoa = ['correlacion', 'gcc_phat']

# Experimentos
resultados = []

for ambiente in ambientes:
    for fs in fs_list:
        for num_mics in num_mics_list:
            for spacing in spacing_list:
                for distance in distance_list:
                    for elev in elevation_list:
                        for rt60 in rt60_list if ambiente == 'reverberante' else [None]:
                            for metodo_tdoa in metodos_tdoa:
                                # Simulación
                                sim = SimuladorDOA(fs=fs)
                                sim.crear_array_microfonos(num_mics=num_mics, spacing=spacing)
                                if ambiente == 'anechoico':
                                    sim.simular_ambiente_anecoico()
                                else:
                                    sim.simular_ambiente_reverberante(rt60=rt60)
                                signal = crear_senal_prueba("chirp", duracion=1.0, fs=fs)
                                azimuth_real = 60.0
                                sim.agregar_fuente(signal, azimuth=azimuth_real, distance=distance, elevation=elev)
                                sim.simular_propagacion(agregar_ruido=True, snr_db=20)
                                mic_signals = sim.signals['mic_signals']
                                pares_microfonos = [(i, j) for i in range(num_mics) for j in range(i+1, num_mics)]
                                estimador_tdoa = EstimadorTDOA(fs=fs)
                                resultados_tdoa = {}
                                for i, j in pares_microfonos:
                                    resultado = estimador_tdoa.estimar_tdoa_par(mic_signals[i], mic_signals[j], metodo=metodo_tdoa)
                                    key = f"mic_{i+1}_mic_{j+1}"
                                    resultados_tdoa[key] = resultado
                                estimador_doa = EstimadorDOA(c=343.0)
                                spacing_actual = spacing
                                resultados_doa = estimador_doa.calcular_angulo_arribo(resultados_tdoa, spacing_actual, geometria='linear')
                                resultado_promedio = estimador_doa.promediar_angulos(resultados_doa, metodo='circular')
                                for k, v in resultados_tdoa.items():
                                    resultados.append({
                                        'ambiente': ambiente,
                                        'fs': fs,
                                        'num_mics': num_mics,
                                        'spacing': spacing,
                                        'distance': distance,
                                        'elevation': elev,
                                        'rt60': rt60 if ambiente == 'reverberante' else None,
                                        'metodo_tdoa': metodo_tdoa,
                                        'mic_pair': k,
                                        'tdoa_seconds': v['tdoa_seconds'],
                                        'doa_deg': resultados_doa[k]['angulo_deg'] if resultados_doa[k]['valido'] else None,
                                        'confidence': v['confidence'],
                                        'error_doa': abs(resultados_doa[k]['angulo_deg'] - azimuth_real) if resultados_doa[k]['valido'] else None,
                                        'error_promedio': abs(resultado_promedio['angulo_promedio_deg'] - azimuth_real) if resultado_promedio['valido'] else None
                                    })

# Guardar resultados
df = pd.DataFrame(resultados)
df.to_csv("resultados_experimentos.csv", index=False)
print("Resultados exportados a resultados_experimentos.csv")

# Graficar error promedio vs parámetros
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='num_mics', y='error_promedio', hue='ambiente')
plt.title("Error promedio DOA vs cantidad de micrófonos")
plt.xlabel("Cantidad de micrófonos")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='spacing', y='error_promedio', hue='ambiente')
plt.title("Error promedio DOA vs separación entre micrófonos")
plt.xlabel("Separación (m)")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='distance', y='error_promedio', hue='ambiente')
plt.title("Error promedio DOA vs distancia fuente-array")
plt.xlabel("Distancia fuente-array (m)")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='elevation', y='error_promedio', hue='ambiente')
plt.title("Error promedio DOA vs elevación fuente")
plt.xlabel("Elevación fuente (°)")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='fs', y='error_promedio', hue='ambiente')
plt.title("Error promedio DOA vs frecuencia de muestreo")
plt.xlabel("Frecuencia de muestreo (Hz)")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[df['ambiente']=='reverberante'], x='rt60', y='error_promedio')
plt.title("Error promedio DOA vs RT60 (solo reverberante)")
plt.xlabel("RT60 (s)")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='metodo_tdoa', y='error_promedio', hue='ambiente')
plt.title("Error promedio DOA vs método de TDOA")
plt.xlabel("Método TDOA")
plt.ylabel("Error promedio DOA (°)")
plt.grid(True)
plt.tight_layout()
plt.show()