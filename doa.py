import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, least_squares
from scipy.stats import circmean, circstd

class EstimadorDOA:
    def __init__(self, c: float = 343.0):
        self.c = c

    def calcular_angulo_arribo(self, tdoas: Dict, distancia_micros: float,
                              geometria: str = 'linear',
                              mic_positions: Optional[np.ndarray] = None) -> Dict:
        if geometria == 'linear':
            return self._calcular_angulo_lineal(tdoas, distancia_micros)
        elif geometria == 'arbitrary':
            if mic_positions is None:
                raise ValueError("Se requieren posiciones de micrófonos para geometría arbitraria")
            return self._calcular_angulo_arbitrario(tdoas, mic_positions)
        else:
            raise ValueError(f"Geometría no soportada: {geometria}")

    def _calcular_angulo_lineal(self, tdoas: Dict, spacing: float) -> Dict:
        angulos = {}
        for par_key, tdoa_data in tdoas.items():
            tdoa_sec = tdoa_data['tdoa_seconds']
            sin_theta = (self.c * tdoa_sec) / spacing
            if abs(sin_theta) <= 1.0:
                theta_rad = np.pi/2 - np.arcsin(sin_theta)
                theta_deg = np.degrees(theta_rad)
                confidence = tdoa_data.get('confidence', 1.0)
                uncertainty_rad = np.arcsin(self.c / (spacing * confidence)) if confidence > 0 else np.pi/2
                uncertainty_deg = np.degrees(uncertainty_rad)
                angulos[par_key] = {
                    'angulo_rad': float(theta_rad),
                    'angulo_deg': float(theta_deg),
                    'sin_theta': float(sin_theta),
                    'uncertainty_deg': float(uncertainty_deg),
                    'confidence': float(confidence),
                    'tdoa_usado': float(tdoa_sec),
                    'valido': True,
                    'metodo': 'trigonometria_lineal'
                }
            else:
                angulos[par_key] = {
                    'angulo_rad': None,
                    'angulo_deg': None,
                    'sin_theta': float(sin_theta),
                    'uncertainty_deg': None,
                    'confidence': tdoa_data.get('confidence', 0.0),
                    'tdoa_usado': float(tdoa_sec),
                    'valido': False,
                    'error': f'TDOA fuera de rango físico (sin_theta = {sin_theta:.3f})',
                    'metodo': 'trigonometria_lineal'
                }
        return angulos

    def _calcular_angulo_arbitrario(self, tdoas: Dict, mic_positions: np.ndarray) -> Dict:
        tdoa_values = []
        mic_pairs = []
        for par_key, tdoa_data in tdoas.items():
            if tdoa_data.get('valido', True):
                parts = par_key.split('_')
                mic1_idx = int(parts[1]) - 1
                mic2_idx = int(parts[3]) - 1
                tdoa_values.append(tdoa_data['tdoa_seconds'])
                mic_pairs.append((mic1_idx, mic2_idx))
        if len(tdoa_values) < 2:
            return {'error': 'Insuficientes TDOAs válidos para triangulación'}
        def objetivo(theta_phi):
            theta, phi = theta_phi
            error_total = 0
            source_dir = np.array([
                np.cos(phi) * np.cos(theta),
                np.cos(phi) * np.sin(theta),
                np.sin(phi)
            ])
            for i, (mic1_idx, mic2_idx) in enumerate(mic_pairs):
                pos1 = mic_positions[:, mic1_idx]
                pos2 = mic_positions[:, mic2_idx]
                dist_diff = np.dot(source_dir, pos2 - pos1)
                tdoa_teorico = dist_diff / self.c
                error_total += (tdoa_teorico - tdoa_values[i])**2
            return error_total
        resultado = minimize(objetivo, x0=[0, 0], method='BFGS')
        if resultado.success:
            theta_opt, phi_opt = resultado.x
            return {
                'angulo_azimuth_rad': float(theta_opt),
                'angulo_azimuth_deg': float(np.degrees(theta_opt)),
                'angulo_elevation_rad': float(phi_opt),
                'angulo_elevation_deg': float(np.degrees(phi_opt)),
                'error_optimizacion': float(resultado.fun),
                'valido': True,
                'metodo': 'optimizacion_arbitraria'
            }
        else:
            return {
                'error': 'Optimización falló',
                'valido': False,
                'metodo': 'optimizacion_arbitraria'
            }

    def promediar_angulos(self, angulos: Dict, metodo: str = 'circular',
                         pesos: Optional[Dict] = None) -> Dict:
        angulos_validos = []
        confidencias = []
        pares_validos = []
        for par_key, angulo_data in angulos.items():
            if angulo_data.get('valido', False):
                angulos_validos.append(angulo_data['angulo_deg'])
                confidencias.append(angulo_data.get('confidence', 1.0))
                pares_validos.append(par_key)
        if not angulos_validos:
            return {
                'angulo_promedio_deg': None,
                'std_deg': None,
                'num_estimaciones': 0,
                'valido': False,
                'error': 'No hay ángulos válidos para promediar'
            }
        angulos_rad = np.deg2rad(angulos_validos)
        if metodo == 'circular':
            if pesos:
                pesos_array = np.array([pesos.get(par, 1.0) for par in pares_validos])
                pesos_array = pesos_array / np.sum(pesos_array)
                x_sum = np.sum(pesos_array * np.cos(angulos_rad))
                y_sum = np.sum(pesos_array * np.sin(angulos_rad))
                angulo_promedio_rad = np.arctan2(y_sum, x_sum)
            else:
                angulo_promedio_rad = circmean(angulos_rad)
            std_rad = circstd(angulos_rad)
        elif metodo == 'aritmetico':
            angulo_promedio_rad = np.mean(angulos_rad)
            std_rad = np.std(angulos_rad)
        elif metodo == 'ponderado':
            if pesos:
                pesos_array = np.array([pesos.get(par, 1.0) for par in pares_validos])
            else:
                pesos_array = np.array(confidencias)
            pesos_array = pesos_array / np.sum(pesos_array)
            angulo_promedio_rad = np.sum(pesos_array * angulos_rad)
            varianza_ponderada = np.sum(pesos_array * (angulos_rad - angulo_promedio_rad)**2)
            std_rad = np.sqrt(varianza_ponderada)
        else:
            raise ValueError(f"Método de promediado desconocido: {metodo}")
        angulo_promedio_deg = np.degrees(angulo_promedio_rad)
        std_deg = np.degrees(std_rad)
        angulo_promedio_deg = ((angulo_promedio_deg + 180) % 360) - 180
        return {
            'angulo_promedio_deg': float(angulo_promedio_deg),
            'angulo_promedio_rad': float(angulo_promedio_rad),
            'std_deg': float(std_deg),
            'std_rad': float(std_rad),
            'num_estimaciones': len(angulos_validos),
            'angulos_individuales': angulos_validos,
            'confidencias': confidencias,
            'pares_usados': pares_validos,
            'metodo_promedio': metodo,
            'valido': True
        }

    def evaluar_ambiguedad(self, angulos: Dict) -> Dict:
        angulos_validos = [a['angulo_deg'] for a in angulos.values() if a.get('valido', False)]
        if len(angulos_validos) < 2:
            return {
                'tiene_ambiguedad': False,
                'dispersion': 0.0,
                'rango': 0.0
            }
        std_angulos = np.std(angulos_validos)
        rango_angulos = np.max(angulos_validos) - np.min(angulos_validos)
        umbral_ambiguedad = 10.0
        tiene_ambiguedad = std_angulos > umbral_ambiguedad or rango_angulos > 2 * umbral_ambiguedad
        return {
            'tiene_ambiguedad': tiene_ambiguedad,
            'dispersion': float(std_angulos),
            'rango': float(rango_angulos),
            'num_estimaciones': len(angulos_validos),
            'angulos': angulos_validos
        }

    def visualizar_estimaciones(self, angulos: Dict, angulo_real: Optional[float] = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        pares = []
        angulos_est = []
        confidencias = []
        for par_key, angulo_data in angulos.items():
            if angulo_data.get('valido', False):
                pares.append(par_key.replace('_', '-'))
                angulos_est.append(angulo_data['angulo_deg'])
                confidencias.append(angulo_data.get('confidence', 1.0))
        if angulos_est:
            colors = plt.cm.viridis(np.array(confidencias) / max(confidencias))
            bars = ax1.bar(range(len(pares)), angulos_est, color=colors)
            ax1.set_xlabel('Par de Micrófonos')
            ax1.set_ylabel('Ángulo Estimado (°)')
            ax1.set_title('Estimaciones por Par de Micrófonos')
            ax1.set_xticks(range(len(pares)))
            ax1.set_xticklabels(pares, rotation=45)
            ax1.grid(True, alpha=0.3)
            if angulo_real is not None:
                ax1.axhline(y=angulo_real, color='red', linestyle='--',
                            label=f'Ángulo Real: {angulo_real:.1f}°')
                ax1.legend()
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                        norm=plt.Normalize(vmin=min(confidencias),
                                        vmax=max(confidencias)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1)
            cbar.set_label('Confianza')
        ax2 = plt.subplot(122, projection='polar')
        if angulos_est:
            angulos_rad = np.deg2rad(angulos_est)
            radios = confidencias
            scatter = ax2.scatter(angulos_rad, radios, c=confidencias,
                                cmap='viridis', s=100, alpha=0.7)
            if angulo_real is not None:
                ax2.axvline(x=np.deg2rad(angulo_real), color='red',
                            linestyle='--', label=f'Real: {angulo_real:.1f}°')
            ax2.set_title('Vista Polar - Ángulos vs Confianza')
            ax2.set_ylim(0, max(radios) * 1.1)
            ax2.grid(True)
        plt.tight_layout()
        plt.show()

    def generar_reporte(self, angulos: Dict, angulo_promedio: Dict,
                        angulo_real: Optional[float] = None) -> str:
        reporte = "=== REPORTE DE ESTIMACIÓN DOA ===\n\n"
        num_total = len(angulos)
        num_validos = sum(1 for a in angulos.values() if a.get('valido', False))
        reporte += f"Estimaciones totales: {num_total}\n"
        reporte += f"Estimaciones válidas: {num_validos}\n"
        reporte += f"Tasa de éxito: {num_validos/num_total*100:.1f}%\n\n"
        if angulo_promedio.get('valido', False):
            reporte += f"ÁNGULO PROMEDIO: {angulo_promedio['angulo_promedio_deg']:.2f}° ± {angulo_promedio['std_deg']:.2f}°\n"
            reporte += f"Método de promedio: {angulo_promedio['metodo_promedio']}\n"
            if angulo_real is not None:
                error = abs(angulo_promedio['angulo_promedio_deg'] - angulo_real)
                reporte += f"Error vs ángulo real: {error:.2f}°\n"
        reporte += "\n"
        reporte += "ESTIMACIONES INDIVIDUALES:\n"
        for par_key, angulo_data in angulos.items():
            if angulo_data.get('valido', False):
                reporte += f"  {par_key}: {angulo_data['angulo_deg']:.2f}° "
                reporte += f"(confianza: {angulo_data.get('confidence', 'N/A'):.3f})\n"
            else:
                reporte += f"  {par_key}: INVÁLIDO - {angulo_data.get('error', 'Error desconocido')}\n"
        return reporte

def crear_geometria_array(tipo: str, num_mics: int, spacing: float = 0.1) -> np.ndarray:
    positions = np.zeros((3, num_mics))
    if tipo == 'linear':
        for i in range(num_mics):
            positions[0, i] = i * spacing - (num_mics - 1) * spacing / 2
    elif tipo == 'circular':
        radio = spacing * num_mics / (2 * np.pi)
        for i in range(num_mics):
            angle = 2 * np.pi * i / num_mics
            positions[0, i] = radio * np.cos(angle)
            positions[1, i] = radio * np.sin(angle)
    elif tipo == 'L_shape':
        half_mics = num_mics // 2
        for i in range(half_mics):
            positions[0, i] = i * spacing
        for i in range(half_mics, num_mics):
            positions[1, i] = (i - half_mics + 1) * spacing
    return positions

# Ejemplo de uso y testing
if __name__ == "__main__":
    print("=== TESTING MÓDULO DOA ===")
    estimador = EstimadorDOA()
    spacing = 0.1
    angulo_real = 30.0
    tdoa_real = spacing * np.sin(np.deg2rad(angulo_real)) / estimador.c
    tdoas_simulados = {
        'mic_1_mic_2': {
            'tdoa_seconds': tdoa_real,
            'confidence': 0.9,
            'valido': True
        },
        'mic_1_mic_3': {
            'tdoa_seconds': 2 * tdoa_real,
            'confidence': 0.8,
            'valido': True
        },
        'mic_1_mic_4': {
            'tdoa_seconds': 3 * tdoa_real,
            'confidence': 0.7,
            'valido': True
        }
    }
    print(f"Ángulo real: {angulo_real}°")
    print(f"TDOA teórico: {tdoa_real*1000:.3f} ms")
    angulos = estimador.calcular_angulo_arribo(tdoas_simulados, spacing)
    print("\n=== ÁNGULOS ESTIMADOS ===")
    for par, angulo_data in angulos.items():
        if angulo_data['valido']:
            print(f"{par}: {angulo_data['angulo_deg']:.2f}° ± {angulo_data['uncertainty_deg']:.2f}°")
        else:
            print(f"{par}: INVÁLIDO - {angulo_data['error']}")
    angulo_promedio = estimador.promediar_angulos(angulos, metodo='ponderado')
    print(f"\n=== ÁNGULO PROMEDIO ===")
    if angulo_promedio['valido']:
        print(f"Ángulo promedio: {angulo_promedio['angulo_promedio_deg']:.2f}° ± {angulo_promedio['std_deg']:.2f}°")
        error = abs(angulo_promedio['angulo_promedio_deg'] - angulo_real)
        print(f"Error: {error:.2f}°")
    ambiguedad = estimador.evaluar_ambiguedad(angulos)
    print(f"\nAmbigüedad detectada: {ambiguedad['tiene_ambiguedad']}")
    print(f"Dispersión: {ambiguedad['dispersion']:.2f}°")
    estimador.visualizar_estimaciones(angulos, angulo_real)
    reporte = estimador.generar_reporte(angulos, angulo_promedio, angulo_real)
    print(f"\n{reporte}")
    print("\n¡Testing DOA completado!")