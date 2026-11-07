"""
Módulo: local_search.py
Propósito: Optimizar el orden de visita de los puntos de entrega mediante Búsqueda Local (Hill Climbing).

Decisiones de diseño:
- Representación: Permutación de los índices de las entregas [0, 1, ..., N-1].
  La ruta completa siempre inicia en 'Inicio', visita todas las entregas en el orden dado,
  y finaliza en 'Destino'.
- Función objetivo / Costo: Suma de costos reales de camino (calculados por A*) más penalización
  si el consumo excede la capacidad del recurso disponible:
    Costo_Total = Costo_Rutas + max(0, Consumo - Capacidad) * Penalizacion_Por_Unidad
- Vecindad explícita: Intercambio de dos posiciones (2-swap). Para 6 entregas, hay C(6, 2) = 15 vecinos.
- Algoritmo: Hill Climbing de máxima pendiente descendente (Steepest Descent para minimización de costo) con detección de mesetas (plateaus).
- Se ejecutan al menos 10 soluciones iniciales aleatorias para analizar óptimos locales y variabilidad.
- Trazabilidad: el CSV guarda la solución inicial de cada corrida y cuántos movimientos laterales
  (sobre mesetas) se hicieron, para poder sustentar el análisis de mesetas con datos.
"""

import csv
import os
import random
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scenario import Escenario, cargar_escenario
from src.heuristic_search import construir_matriz_costos


def calcular_costo_solucion(
    orden: list[int],
    matriz_costos: list[list[float]],
    capacidad_recurso: float,
    penalizacion_exceso: float = 10.0,
) -> tuple[float, float, float]:
    """
    Calcula el costo total de una solución (permutación de entregas).

    Índices en la matriz de costos:
      - 0: Inicio
      - 1 a N: Entregas (donde entrega k tiene índice k+1)
      - N+1: Destino

    Retorna:
      - (costo_total, costo_base, penalizacion)
    """
    n_entregas = len(orden)
    idx_inicio = 0
    idx_destino = n_entregas + 1

    # 1. Tramo desde Inicio hasta la primera entrega
    primer_pto = orden[0] + 1
    costo_base = matriz_costos[idx_inicio][primer_pto]

    # 2. Tramos intermedios entre entregas consecutivas
    for k in range(len(orden) - 1):
        origen_k = orden[k] + 1
        destino_k = orden[k + 1] + 1
        costo_base += matriz_costos[origen_k][destino_k]

    # 3. Tramo desde la última entrega hasta el Destino
    ultimo_pto = orden[-1] + 1
    costo_base += matriz_costos[ultimo_pto][idx_destino]

    # 4. Evaluación de la restricción de recurso (combustible / batería)
    consumo_recurso = costo_base  # El consumo es proporcional al desplazamiento
    exceso = max(0.0, consumo_recurso - capacidad_recurso)
    penalizacion = exceso * penalizacion_exceso

    costo_total = costo_base + penalizacion
    return costo_total, costo_base, penalizacion


def generar_vecinos_intercambio(solucion: list[int]) -> list[tuple[list[int], tuple[int, int]]]:
    """
    Genera todos los vecinos intercambiando dos posiciones (operador 2-swap).
    Para N=6 elementos, genera exactamente N*(N-1)/2 = 15 vecinos.
    """
    vecinos = []
    n = len(solucion)
    for i in range(n):
        for j in range(i + 1, n):
            vecino = list(solucion)
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append((vecino, (i, j)))
    return vecinos


def hill_climbing(
    orden_inicial: list[int],
    matriz_costos: list[list[float]],
    capacidad_recurso: float,
    penalizacion_exceso: float = 10.0,
    max_pasos_meseta: int = 5,
) -> dict[str, Any]:
    """
    QUÉ HACE:
        Optimiza el orden de visita de las entregas usando Hill Climbing de máxima pendiente descendente
        (Steepest Descent para minimización de costo), partiendo de una solución dada.
    POR QUÉ ASÍ:
        - Como buscamos minimizar costo, el algoritmo busca el vecino con la mayor reducción de costo.
        - Evalúa exhaustivamente todos los C(N, 2) = 15 vecinos por 2-swap en cada paso.
        - Admite hasta 'max_pasos_meseta' (5) movimientos laterales consecutivos sobre mesetas
          (mismo costo) para permitir explorar hombros del espacio de búsqueda sin quedarse
          estancado inmediatamente. Un movimiento lateral puede regresar a la solución anterior
          (oscilar); el tope de 5 evita ciclos infinitos.
        - 'movimientos_laterales' cuenta el total de movimientos laterales hechos en la corrida
          (a diferencia de 'pasos_meseta', que se reinicia con cada mejora).
    COMPLEJIDAD EN TIEMPO:
        O(I * (N^2 / 2) * N), donde I es el número de iteraciones hasta converger (típicamente 3 a 7),
        N es el número de entregas (N=6), generando 15 vecinos evaluados en O(N) cada uno.
    """
    actual = list(orden_inicial)
    costo_actual, base_actual, pen_actual = calcular_costo_solucion(
        actual, matriz_costos, capacidad_recurso, penalizacion_exceso
    )

    historial = [(actual, costo_actual)]
    pasos_meseta = 0
    movimientos_laterales = 0
    iteraciones = 0
    tipo_terminacion = "optimo_local"

    while True:
        iteraciones += 1
        vecinos = generar_vecinos_intercambio(actual)
        mejor_vecino = None
        mejor_costo = float("inf")
        mejor_base = 0.0
        mejor_pen = 0.0

        for vec, _ in vecinos:
            c_tot, c_base, c_pen = calcular_costo_solucion(
                vec, matriz_costos, capacidad_recurso, penalizacion_exceso
            )
            if c_tot < mejor_costo:
                mejor_costo = c_tot
                mejor_base = c_base
                mejor_pen = c_pen
                mejor_vecino = vec

        # Caso 1: Estricta mejora
        if mejor_costo < costo_actual - 1e-9:
            actual = mejor_vecino
            costo_actual = mejor_costo
            base_actual = mejor_base
            pen_actual = mejor_pen
            pasos_meseta = 0
            historial.append((actual, costo_actual))

        # Caso 2: Meseta (mismo costo)
        elif abs(mejor_costo - costo_actual) <= 1e-9 and pasos_meseta < max_pasos_meseta:
            pasos_meseta += 1
            movimientos_laterales += 1
            actual = mejor_vecino
            costo_actual = mejor_costo
            historial.append((actual, costo_actual))
            if pasos_meseta == max_pasos_meseta:
                tipo_terminacion = "meseta_agotada"
                break

        # Caso 3: Ningún vecino mejora ni empata -> Óptimo Local alcanzado
        else:
            tipo_terminacion = "optimo_local"
            break

    return {
        "solucion_inicial": orden_inicial,
        "costo_inicial": historial[0][1],
        "solucion_final": actual,
        "costo_final": costo_actual,
        "costo_base": base_actual,
        "penalizacion": pen_actual,
        "iteraciones": iteraciones,
        "mejora": round(historial[0][1] - costo_actual, 4),
        "terminacion": tipo_terminacion,
        "pasos": len(historial),
        "movimientos_laterales": movimientos_laterales,
    }


def ejecutar_multi_start_hill_climbing(
    escenario: Escenario,
    num_soluciones: int = 10,
    semilla: int = 42,
    ruta_salida_csv: str | None = None,
) -> list[dict[str, Any]]:
    """
    Ejecuta Hill Climbing desde al menos 10 soluciones iniciales aleatorias,
    registrando los óptimos locales, mesetas y variabilidad.

    Reproducibilidad: una sola semilla genera, en secuencia, las permutaciones iniciales de las
    10 corridas; por eso todas las filas del CSV muestran la misma semilla.
    """
    # 1. Obtener matriz de costos con A*
    _, _, matriz_costos, _ = construir_matriz_costos(escenario, metodo="astar")

    rng = random.Random(semilla)
    n_entregas = len(escenario.entregas)
    base_entregas = list(range(n_entregas))

    resultados = []

    for i in range(num_soluciones):
        # Generar permutación aleatoria
        sol_inicial = list(base_entregas)
        rng.shuffle(sol_inicial)

        t_inicio = time.perf_counter()
        res = hill_climbing(
            orden_inicial=sol_inicial,
            matriz_costos=matriz_costos,
            capacidad_recurso=escenario.capacidad_recurso,
            penalizacion_exceso=escenario.penalizacion_exceso_recurso,
        )
        t_total = time.perf_counter() - t_inicio
        res["tiempo_seg"] = round(t_total, 6)
        res["ejecucion"] = i + 1
        res["semilla"] = semilla
        resultados.append(res)

    if ruta_salida_csv:
        os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
        # utf-8-sig agrega la marca BOM para que Excel muestre bien los acentos
        with open(ruta_salida_csv, "w", newline="", encoding="utf-8-sig") as f:
            campos = [
                "ejecucion", "semilla", "vecindad", "max_mesetas", "costo_inicial", "costo_final",
                "costo_base", "penalizacion", "mejora", "iteraciones",
                "pasos", "terminacion", "tiempo_seg", "solucion_final",
                "solucion_inicial", "movimientos_laterales",
            ]
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            for r in resultados:
                fila = dict(r)
                fila["vecindad"] = "2-swap"
                fila["max_mesetas"] = 5
                fila["solucion_final"] = str(fila["solucion_final"])
                fila["solucion_inicial"] = str(fila["solucion_inicial"])
                writer.writerow({c: fila[c] for c in campos})

    return resultados


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    ruta_esc = os.path.join(base_dir, "..", "config", "escenarios", "escenario_1_facil.json")
    esc = cargar_escenario(ruta_esc)

    csv_salida = os.path.join(base_dir, "..", "results", "local_search_results.csv")
    res = ejecutar_multi_start_hill_climbing(esc, num_soluciones=10, semilla=42, ruta_salida_csv=csv_salida)

    print("=== Resultados de Búsqueda Local (Hill Climbing - 10 Soluciones Iniciales) ===")
    for r in res:
        print(f"Run {r['ejecucion']:2d}: Inicial={r['costo_inicial']:.1f} -> Final={r['costo_final']:.1f} (Mejora: {r['mejora']:.1f}) [{r['terminacion']}] (movimientos laterales: {r['movimientos_laterales']})")

    costos_finales = [r["costo_final"] for r in res]
    print(f"\nMejor costo hallado: {min(costos_finales):.1f}")
    print(f"Peor óptimo local: {max(costos_finales):.1f}")
    print(f"Variabilidad (Diferencia): {max(costos_finales) - min(costos_finales):.1f}")
