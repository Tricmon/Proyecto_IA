"""
Script Maestro: run_all.py
==========================
Ejecuta todo el proyecto de punta a punta con un solo comando:

1. Búsqueda heurística (A* y Greedy)
2. Búsqueda local (Hill Climbing)
3. Metaheurísticas (Recocido Simulado + Genético vs Fuerza Bruta)
4. Teoría de juegos (matriz 3x3 y equilibrios de Nash)
5. Minimax y Poda Alfa-Beta
6. Pipeline de integración + generación de gráficas y CSVs
"""

import json
import os
import sys

# Asegurar que se puedan importar los módulos de src/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.scenario import cargar_escenario
from src.heuristic_search import ejecutar_experimentos_heuristicos, construir_matriz_costos
from src.local_search import ejecutar_multi_start_hill_climbing
from src.metaheuristics import ejecutar_experimento_metaheuristicas
from src.game_theory import ejecutar_analisis_teoria_juegos
from src.adversarial import comparar_minimax_y_alfa_beta
from src.pipeline import ejecutar_pipeline
from src.visualization import (
    graficar_busqueda_heuristica,
    graficar_busqueda_local,
    graficar_metaheuristicas,
    graficar_minimax_vs_alfabeta,
    graficar_escenario_y_ruta,
)


def principal():
    # ------------------------------------------------------------------
    # Configuración de rutas
    # ------------------------------------------------------------------
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_dir = os.path.join(base_dir, "config")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Cargar parámetros (incluyendo la semilla)
    # ------------------------------------------------------------------
    params_path = os.path.join(config_dir, "params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Semilla global (si no existe en params.json, usa 55)
    semilla_global = params.get("semilla_global", 55)

    print("=" * 80)
    print("  SISTEMA INTELIGENTE DE PLANIFICACIÓN, OPTIMIZACIÓN Y DECISIONES ADVERSARIAS")
    print(f"  Semilla global de reproducibilidad: {semilla_global}")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 1. Búsqueda Heurística (A* y Greedy)
    # ------------------------------------------------------------------
    print(">> [1/6] Ejecutando Módulo A: Búsqueda Heurística...")

    escenarios_rutas = [
        os.path.join(config_dir, "escenarios", "escenario_1_facil.json"),
        os.path.join(config_dir, "escenarios", "escenario_2_obstaculos.json"),
        os.path.join(config_dir, "escenarios", "escenario_3_trampa_greedy.json"),
        os.path.join(config_dir, "escenarios", "escenario_4_sin_solucion.json"),
    ]

    csv_heuristico = os.path.join(results_dir, "heuristic_search_results.csv")
    datos_heuristicos = ejecutar_experimentos_heuristicos(escenarios_rutas, csv_heuristico)

    grafica_1 = os.path.join(results_dir, "grafica_1_busqueda_heuristica.png")
    graficar_busqueda_heuristica(datos_heuristicos, grafica_1)

    print(f"   -> CSV: {csv_heuristico}")
    print(f"   -> Gráfica 1: {grafica_1}\n")

    # ------------------------------------------------------------------
    # 2. Búsqueda Local (Hill Climbing)
    # ------------------------------------------------------------------
    print(">> [2/6] Ejecutando Módulo B: Búsqueda Local (Hill Climbing)...")

    esc_base = cargar_escenario(escenarios_rutas[0])
    csv_local = os.path.join(results_dir, "local_search_results.csv")

    datos_local = ejecutar_multi_start_hill_climbing(
        esc_base,
        num_soluciones=10,
        semilla=semilla_global,
        ruta_salida_csv=csv_local
    )

    grafica_2 = os.path.join(results_dir, "grafica_2_busqueda_local_convergencia.png")
    graficar_busqueda_local(datos_local, grafica_2)

    print(f"   -> CSV: {csv_local}")
    print(f"   -> Gráfica 2: {grafica_2}\n")

    # ------------------------------------------------------------------
    # 3. Metaheurísticas vs Fuerza Bruta
    # ------------------------------------------------------------------
    print(">> [3/6] Ejecutando Módulo C: Metaheurísticas vs Fuerza Bruta...")

    csv_meta_runs = os.path.join(results_dir, "metaheuristics_runs.csv")
    csv_meta_resumen = os.path.join(results_dir, "metaheuristics_comparison.csv")

    fb, runs_meta, resumen_meta = ejecutar_experimento_metaheuristicas(
        esc_base,
        num_ejecuciones=20,
        presupuesto_evaluaciones=1000,
        semilla_base=semilla_global,
        ruta_runs_csv=csv_meta_runs,
        ruta_resumen_csv=csv_meta_resumen,
    )

    grafica_3 = os.path.join(results_dir, "grafica_3_metaheuristicas_comparacion.png")
    graficar_metaheuristicas(runs_meta, fb["mejor_costo"], grafica_3)

    print(f"   -> Óptimo Global (Fuerza Bruta): {fb['mejor_costo']}")
    print(f"   -> CSVs: {csv_meta_runs} | {csv_meta_resumen}")
    print(f"   -> Gráfica 3: {grafica_3}\n")

    # ------------------------------------------------------------------
    # 4. Teoría de Juegos
    # ------------------------------------------------------------------
    print(">> [4/6] Ejecutando Módulo D: Teoría de Juegos...")

    csv_juegos = os.path.join(results_dir, "game_theory_matrix.csv")
    analisis_juegos = ejecutar_analisis_teoria_juegos(
        ruta_salida_csv=csv_juegos,
        costo_optimo_base=fb["mejor_costo"]
    )

    print(f"   -> CSV: {csv_juegos}")
    print(f"   -> Equilibrios de Nash encontrados: {len(analisis_juegos['equilibrios_nash'])}\n")

    # ------------------------------------------------------------------
    # 5. Minimax y Alfa-Beta
    # ------------------------------------------------------------------
    print(">> [5/6] Ejecutando Módulo E: Minimax y Poda Alfa-Beta...")

    csv_adversario = os.path.join(results_dir, "minimax_alphabeta_comparison.csv")
    datos_adversarios = comparar_minimax_y_alfa_beta(
        recurso_inicial=7,
        ruta_salida_csv=csv_adversario
    )

    grafica_4 = os.path.join(results_dir, "grafica_4_minimax_vs_alfabeta.png")
    graficar_minimax_vs_alfabeta(datos_adversarios, grafica_4)

    print(f"   -> CSV: {csv_adversario}")
    print(f"   -> Gráfica 4: {grafica_4}\n")

    # ------------------------------------------------------------------
    # 6. Pipeline de Integración + Mapa final
    # ------------------------------------------------------------------
    print(">> [6/6] Ejecutando Integración Completa (Pipeline)...")

    plan = ejecutar_pipeline(escenarios_rutas[0], semilla=semilla_global)

    # Reconstruir la ruta celda a celda para la visualización
    _, _, _, matriz_rutas = construir_matriz_costos(esc_base, metodo="astar")
    secuencia_puntos_celdas = []

    orden_indices = [0] + [i + 1 for i in plan["orden_inteligente"]] + [len(esc_base.entregas) + 1]

    for k in range(len(orden_indices) - 1):
        origen = orden_indices[k]
        destino = orden_indices[k + 1]
        tramo = matriz_rutas[origen][destino]

        if k > 0 and tramo:
            secuencia_puntos_celdas.extend(tramo[1:])
        else:
            secuencia_puntos_celdas.extend(tramo)

    grafica_5 = os.path.join(results_dir, "grafica_5_tablero_ruta_final.png")
    graficar_escenario_y_ruta(esc_base, secuencia_puntos_celdas, grafica_5)

    print(f"   -> Gráfica 5 (Mapa y Ruta Final): {grafica_5}\n")

    print("=" * 80)
    print("  ¡EJECUCIÓN COMPLETADA EXITOSAMENTE!")
    print("  Todos los artefactos se generaron en la carpeta results/")
    print("=" * 80)


if __name__ == "__main__":
    principal()