"""
Módulo de Análisis de Probabilidad y Convergencia de Variables Aleatorias.
Implementa soluciones analíticas (vía SymPy) y numéricas (vía NumPy/SciPy).
"""

from typing import Tuple, Dict, List
import numpy as np
import sympy as sp
from fractions import Fraction


class ExperimentoDados:
    """Modela el experimento de lanzamiento de dos dados y evalúa independencia/condicionales."""
    
    def __init__(self, caras: int = 6):
        self.caras = caras
        self.omega = {(i, j) for i in range(1, caras + 1) for j in range(1, caras + 1)}
        self.cardinalidad = len(self.omega)

    def evento_suma(self, n: int) -> set:
        return {o for o in self.omega if sum(o) == n}

    def evento_diferencia(self, m: int) -> set:
        return {o for o in self.omega if abs(o[0] - o[1]) >= m}

    def calcular_probabilidad_condicional(self, evento_a: set, evento_b: set) -> Fraction:
        """Calcula P(A | B) = P(A ∩ B) / P(B)"""
        interseccion = evento_a.intersection(evento_b)
        if not evento_b:
            raise ValueError("El evento condicionante B no puede ser vacío.")
        return Fraction(len(interseccion), len(evento_b))

    def analizar_independencia(self) -> List[Tuple[int, int]]:
        """Determina para qué parejas (n, m) los eventos S_n y D_m son independientes."""
        resultados_indep = []
        rango_s = range(2, 2 * self.caras + 1)
        rango_d = range(0, self.caras)

        for n in rango_s:
            sn = self.evento_suma(n)
            prob_sn = Fraction(len(sn), self.cardinalidad)
            for m in rango_d:
                dm = self.evento_diferencia(m)
                prob_dm = Fraction(len(dm), self.cardinalidad)
                prob_intersec = Fraction(len(sn.intersection(dm)), self.cardinalidad)

                if prob_intersec == prob_sn * prob_dm:
                    resultados_indep.append((n, m))
        return resultados_indep


class SimulaciónConvergencia:
    """Demuestra empíricamente teoremas de convergencia c.s., en probabilidad y en ley."""

    @staticmethod
    def exponencial_borel_cantelli(theta_n_func, n_sim: int = 5000, max_n: int = 1000) -> np.ndarray:
        """
        Simula X_n ~ Exp(theta_n) para evaluar convergencia casi segura.
        """
        trayectorias = np.zeros((n_sim, max_n))
        for n in range(1, max_n + 1):
            scale = 1.0 / theta_n_func(n)
            trayectorias[:, n - 1] = np.random.exponential(scale=scale, size=n_sim)
        return trayectorias

    @staticmethod
    def limite_maximo_uniforme(n_samples: int = 1000, size: int = 100) -> np.ndarray:
        """
        Evalúa la convergencia en ley de Y_n = n * (1 - max(U_1, ..., U_n)) -> Exp(1).
        """
        u = np.random.uniform(low=0.0, high=1.0, size=(n_samples, size))
        m_n = np.max(u, axis=1)
        y_n = size * (1.0 - m_n)
        return y_n


if __name__ == "__main__":
    print("=== 1. Análisis de Dados y Probabilidad Condicional ===")
    exp = ExperimentoDados()
    s7 = exp.evento_suma(7)
    d3 = exp.evento_diferencia(3)

    p_s7_dado_d3 = exp.calcular_probabilidad_condicional(s7, d3)
    p_d3_dado_s7 = exp.calcular_probabilidad_condicional(d3, s7)

    print(f"P(S_7 | D_3) = {p_s7_dado_d3}")
    print(f"P(D_3 | S_7) = {p_d3_dado_s7}")
    print(f"Parejas de independencia (n, m): {exp.analizar_independencia()}")

    print("\n=== 2. Simulación de Convergencia en Ley ===")
    y_n = SimulaciónConvergencia.limite_maximo_uniforme(n_samples=5000, size=500)
    print(f"Media empírica de n*(1-M_n): {np.mean(y_n):.4f} (Teórico Exp(1) = 1.0)")