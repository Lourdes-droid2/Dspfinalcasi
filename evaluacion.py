from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import numpy as np  

class EvaluadorDOA:
    def __init__(self):
        pass

    def calcular_error(self, estimado: Union[float, List[float]], real: Union[float, List[float]],
                      tipo_error: str = 'absoluto') -> Dict:
        if tipo_error == 'absoluto':
            error = np.abs(np.array(estimado) - np.array(real))
        elif tipo_error == 'relativo':
            error = np.abs(np.array(estimado) - np.array(real)) / (np.abs(np.array(real)) + 1e-12)
        else:
            raise ValueError("Tipo de error no soportado")
        return {
            'error': error,
            'tipo': tipo_error
        }

    def evaluar_metodo_tdoa(self, estimador_tdoa, signals: np.ndarray, tdoas_reales: Dict,
                            metodos: List[str] = ['correlacion', 'gcc_phat', 'gcc_scot']) -> Dict:
        resultados = {}
        for metodo in metodos:
            try:
                resultado = estimador_tdoa.estimar_tdoa_array(signals, metodo=metodo)
                resultados[metodo] = resultado
            except Exception as e:
                resultados[metodo] = {'error': str(e)}
        return resultados