"""
Módulo: metaheuristics.py
Propósito: Optimización de permutaciones mediante Recocido Simulado y Algoritmo Genético,
comparados rigurosamente bajo el MISMO presupuesto de evaluaciones contra Fuerza Bruta (720 combinaciones).

Decisiones de diseño:
1. Presupuesto común de evaluaciones: Ambos métodos comparten exactamente el mismo límite
   de llamadas a la función de aptitud/costo (ej. 1000 evaluaciones).
2. Recocido Simulado (SA):
   - Regla de aceptación de Metropolis: P = exp(-delta / T) para movimientos desfavorables (delta > 0).
   - Enfriamiento geométrico: T_{k+1} = T_k * alfa.
3. Algoritmo Genético (GA):
   - Representación: Permutación de 6 enteros sin repetición.
   - Selección: Torneo de tamaño k=3 (balance entre presión de selección y diversidad).
   - Cruza: Order Crossover (OX) preserva el orden relativo sin generar duplicados.
   - Mutación: Swap de dos genes con probabilidad p_mutacion.
   - Elitismo: Se conserva al mejor individuo para evitar regresión.
4. Fuerza Bruta:
   - Se evalúan las 6! = 720 permutaciones posibles para obtener el óptimo global exacto de referencia.
"""

import csv
import itertools
import math
import os
import random
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scenario import Escenario, cargar_escenario
from src.heuristic_search import construir_matriz_costos
from src.local_search import calcular_costo_solucion


def evaluar_fuerza_bruta(
    matriz_costos: list[list[float]],
    capacidad_recurso: float,
    penalizacion_exceso: float = 10.0,
    n_entregas: int = 6,
) -> dict[str, Any]:
    """
    Evalúa TODAS las N! permutaciones posibles (6! = 720) para conocer
    el óptimo global exacto del problema.
    """
    t_inicio = time.perf_counter()
    mejor_orden = None
    mejor_costo = float("inf")
    mejor_base = 0.0
    mejor_pen = 0.0
    total_evaluaciones = 0

    for perm in itertools.permutations(range(n_entregas)):
        orden = list(perm)
        c_tot, c_base, c_pen = calcular_costo_solucion(
            orden, matriz_costos, capacidad_recurso, penalizacion_exceso
        )
        total_evaluaciones += 1
        if c_tot < mejor_costo:
            mejor_costo = c_tot
            mejor_base = c_base
            mejor_pen = c_pen
            mejor_orden = orden

    t_total = time.perf_counter() - t_inicio
    return {
        "metodo": "Fuerza Bruta",
        "mejor_orden": mejor_orden,
        "mejor_costo": mejor_costo,
        "costo_base": mejor_base,
        "penalizacion": mejor_pen,
        "evaluaciones": total_evaluaciones,
        "tiempo_seg": round(t_total, 6),
    }


def recocido_simulado(
    matriz_costos: list[list[float]],
    capacidad_recurso: float,
    penalizacion_exceso: float = 10.0,
    n_entregas: int = 6,
    presupuesto_evaluaciones: int = 1000,
    temperatura_inicial: float = 100.0,
    temperatura_final: float = 0.01,
    alfa: float = 0.95,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    QUÉ HACE:
        Optimiza la secuencia de visitas simulando el proceso físico de recocido térmico,
        escapando de óptimos locales al permitir transiciones que empeoran el costo.
    POR QUÉ ASÍ:
        - Aplica el criterio de Metropolis: mejoras (delta < 0) se aceptan siempre;
          empeoramientos (delta > 0) se aceptan con probabilidad P = exp(-delta / T).
        - Enfriamiento geométrico T <- T * alfa: a altas temperaturas favorece la exploración
          global del espacio de permutaciones; a bajas temperaturas intensifica la explotación local.
    COMPLEJIDAD EN TIEMPO:
        O(K * N), donde K es el presupuesto de evaluaciones (K = 1000) y N es el número
        de entregas (N = 6). Cada evaluación de costo de ruta toma O(N).
    """
    if rng is None:
        rng = random.Random()

    t_inicio = time.perf_counter()

    # Solución inicial aleatoria
    actual = list(range(n_entregas))
    rng.shuffle(actual)
    costo_actual, _, _ = calcular_costo_solucion(
        actual, matriz_costos, capacidad_recurso, penalizacion_exceso
    )

    mejor_sol = list(actual)
    mejor_costo = costo_actual
    evaluaciones = 1

    temp = temperatura_inicial

    # Ajuste del ritmo de enfriamiento según el presupuesto
    pasos_por_temperatura = max(1, presupuesto_evaluaciones // 150)
    paso_en_temp = 0

    while evaluaciones < presupuesto_evaluaciones:
        # Generar vecino por 2-swap
        i, j = rng.sample(range(n_entregas), 2)
        vecino = list(actual)
        vecino[i], vecino[j] = vecino[j], vecino[i]

        costo_vecino, _, _ = calcular_costo_solucion(
            vecino, matriz_costos, capacidad_recurso, penalizacion_exceso
        )
        evaluaciones += 1

        delta = costo_vecino - costo_actual

        # Regla de Metropolis: si mejora (delta < 0), aceptar siempre;
        # si empeora, aceptar con probabilidad P = exp(-delta / T)

        #def algoritmo_genetico(
        if delta < 0:
            actual = vecino
            costo_actual = costo_vecino
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual
                mejor_sol = list(actual)
        else:
            prob = math.exp(-delta / max(temp, 1e-8))
            if rng.random() < prob:
                actual = vecino
                costo_actual = costo_vecino

        # Enfriamiento geométrico
        paso_en_temp += 1
        if paso_en_temp >= pasos_por_temperatura:
            temp = max(temperatura_final, temp * alfa)
            paso_en_temp = 0

    t_total = time.perf_counter() - t_inicio
    c_tot, c_base, c_pen = calcular_costo_solucion(
        mejor_sol, matriz_costos, capacidad_recurso, penalizacion_exceso
    )

    return {
        "metodo": "Recocido Simulado",
        "mejor_orden": mejor_sol,
        "mejor_costo": c_tot,
        "costo_base": c_base,
        "penalizacion": c_pen,
        "evaluaciones": evaluaciones,
        "tiempo_seg": round(t_total, 6),
    }


def cruza_ox(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """
    QUÉ HACE:
        Operador de Cruza de Orden (Order Crossover - OX) para cromosomas de permutación.
        1. Copia un subsegmento continuo aleatorio del progenitor 1 intacto en el descendiente.
        2. Rellena las posiciones restantes con los alelos del progenitor 2 en su orden relativo,
           comenzando desde el corte final de forma circular.
    POR QUÉ ASÍ:
        Garantiza que el descendiente sea una permutación válida sin elementos duplicados ni faltantes,
        preservando el orden y la adyacencia relativa de tramos de ruta de ambos progenitores.
    COMPLEJIDAD EN TIEMPO:
        O(N) en tiempo y O(N) en memoria, donde N = len(p1) = 6.
    """
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    
    hijo = [None] * n
    # Copiar segmento de p1
    hijo[a:b + 1] = p1[a:b + 1]
    genes_presentes = set(hijo[a:b + 1])

    # Llenar el resto en orden circular desde b+1 usando p2
    idx_p2 = (b + 1) % n
    idx_hijo = (b + 1) % n

    for _ in range(n):
        gen = p2[idx_p2]
        if gen not in genes_presentes:
            hijo[idx_hijo] = gen
            genes_presentes.add(gen)
            idx_hijo = (idx_hijo + 1) % n
        idx_p2 = (idx_p2 + 1) % n

    return hijo


def mutacion_swap(individuo: list[int], rng: random.Random) -> list[int]:
    """
    QUÉ HACE:
        Operador de mutación por intercambio de dos posiciones aleatorias en el individuo.
    POR QUÉ ASÍ:
        Introduce diversidad genética al intercambiar el orden de dos entregas sin alterar
        la validez del cromosoma como permutación (mantiene exactamente los 6 índices).
    COMPLEJIDAD EN TIEMPO:
        O(1) tras el copiado del arreglo.
    """
    mutado = list(individuo)
    i, j = rng.sample(range(len(individuo)), 2)
    mutado[i], mutado[j] = mutado[j], mutado[i]
    return mutado


def algoritmo_genetico(
    matriz_costos: list[list[float]],
    capacidad_recurso: float,
    penalizacion_exceso: float = 10.0,
    n_entregas: int = 6,
    presupuesto_evaluaciones: int = 1000,
    tamano_poblacion: int = 20,
    tamano_torneo: int = 3,
    prob_cruza: float = 0.9,
    prob_mutacion: float = 0.2,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    QUÉ HACE:
        Evoluciona una población de secuencias de entrega usando principios de selección natural:
        selección por torneo, cruza de orden OX, mutación swap y elitismo.
    POR QUÉ ASÍ:
        - Selección por torneo (k=3): Mantiene una presión selectiva equilibrada sin riesgo
          de que un solo individuo domine la población de inmediato.
        - Cruza OX + Mutación Swap: Diseñados para problemas de ruteo/permutación (TSP-like).
        - Elitismo: Conserva las 2 mejores soluciones para asegurar que la mejor cota nunca empeore.
    COMPLEJIDAD EN TIEMPO:
        O(K * N), donde K es el presupuesto estricto de evaluaciones (K = 1000) y N = 6.
    """
    if rng is None:
        rng = random.Random()

    t_inicio = time.perf_counter()

    # 1. Inicializar población
    poblacion = []
    base = list(range(n_entregas))
    evaluaciones = 0

    for _ in range(tamano_poblacion):
        ind = list(base)
        rng.shuffle(ind)
        c_tot, _, _ = calcular_costo_solucion(
            ind, matriz_costos, capacidad_recurso, penalizacion_exceso
        )
        evaluaciones += 1
        poblacion.append((c_tot, ind))

    # Ordenar por costo (menor es mejor)
    poblacion.sort(key=lambda x: x[0])
    mejor_costo_global = poblacion[0][0]
    mejor_ind_global = list(poblacion[0][1])

    # 2. Bucle generacional hasta agotar presupuesto
    while evaluaciones < presupuesto_evaluaciones:
        nueva_poblacion = []

        # Elitismo: conservar los mejores 2 individuos directamente
        nueva_poblacion.append(poblacion[0])
        if len(poblacion) > 1:
            nueva_poblacion.append(poblacion[1])

        while len(nueva_poblacion) < tamano_poblacion and evaluaciones < presupuesto_evaluaciones:
            # Selección por torneo
            aspirantes1 = rng.sample(poblacion, min(tamano_torneo, len(poblacion)))
            padre1 = min(aspirantes1, key=lambda x: x[0])[1]

            aspirantes2 = rng.sample(poblacion, min(tamano_torneo, len(poblacion)))
            padre2 = min(aspirantes2, key=lambda x: x[0])[1]

            # Cruza OX
            if rng.random() < prob_cruza:
                hijo = cruza_ox(padre1, padre2, rng)
            else:
                hijo = list(padre1)

            # Mutación por intercambio
            if rng.random() < prob_mutacion:
                hijo = mutacion_swap(hijo, rng)

            # Evaluación
            c_hijo, _, _ = calcular_costo_solucion(
                hijo, matriz_costos, capacidad_recurso, penalizacion_exceso
            )
            evaluaciones += 1
            nueva_poblacion.append((c_hijo, hijo))

            if c_hijo < mejor_costo_global:
                mejor_costo_global = c_hijo
                mejor_ind_global = list(hijo)

        poblacion = sorted(nueva_poblacion, key=lambda x: x[0])

    t_total = time.perf_counter() - t_inicio
    c_tot, c_base, c_pen = calcular_costo_solucion(
        mejor_ind_global, matriz_costos, capacidad_recurso, penalizacion_exceso
    )

    return {
        "metodo": "Algoritmo Genético",
        "mejor_orden": mejor_ind_global,
        "mejor_costo": c_tot,
        "costo_base": c_base,
        "penalizacion": c_pen,
        "evaluaciones": evaluaciones,
        "tiempo_seg": round(t_total, 6),
    }


def ejecutar_experimento_metaheuristicas(
    escenario: Escenario,
    num_ejecuciones: int = 20,
    presupuesto_evaluaciones: int = 1000,
    semilla_base: int = 100,
    ruta_runs_csv: str | None = None,
    ruta_resumen_csv: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Ejecuta 20 repeticiones de Recocido Simulado y 20 de Algoritmo Genético
    con el mismo presupuesto de evaluaciones, comparándolos contra Fuerza Bruta.
    """
    _, _, matriz_costos, _ = construir_matriz_costos(escenario, metodo="astar")
    n_entregas = len(escenario.entregas)

    # 1. Fuerza Bruta (Benchmark exacto)
    resultado_fb = evaluar_fuerza_bruta(
        matriz_costos, escenario.capacidad_recurso, escenario.penalizacion_exceso_recurso, n_entregas
    )
    optimo_exacto = resultado_fb["mejor_costo"]

    todos_los_runs = []

    # 2. Recocido Simulado (20 ejecuciones)
    for run_idx in range(num_ejecuciones):
        semilla = semilla_base + run_idx
        rng = random.Random(semilla)
        res_sa = recocido_simulado(
            matriz_costos,
            escenario.capacidad_recurso,
            escenario.penalizacion_exceso_recurso,
            n_entregas=n_entregas,
            presupuesto_evaluaciones=presupuesto_evaluaciones,
            rng=rng,
        )
        res_sa["ejecucion"] = run_idx + 1
        res_sa["semilla"] = semilla
        res_sa["alcanzo_optimo"] = (abs(res_sa["mejor_costo"] - optimo_exacto) < 1e-6)
        todos_los_runs.append(res_sa)

    # 3. Algoritmo Genético (20 ejecuciones)
    for run_idx in range(num_ejecuciones):
        semilla = semilla_base + 500 + run_idx
        rng = random.Random(semilla)
        res_ga = algoritmo_genetico(
            matriz_costos,
            escenario.capacidad_recurso,
            escenario.penalizacion_exceso_recurso,
            n_entregas=n_entregas,
            presupuesto_evaluaciones=presupuesto_evaluaciones,
            rng=rng,
        )
        res_ga["ejecucion"] = run_idx + 1
        res_ga["semilla"] = semilla
        res_ga["alcanzo_optimo"] = (abs(res_ga["mejor_costo"] - optimo_exacto) < 1e-6)
        todos_los_runs.append(res_ga)

    # 4. Estadísticas agregadas
    resumen_filas = []
    for metodo in ["Recocido Simulado", "Algoritmo Genético"]:
        runs_m = [r for r in todos_los_runs if r["metodo"] == metodo]
        costos = [r["mejor_costo"] for r in runs_m]
        tiempos = [r["tiempo_seg"] for r in runs_m]
        evals = [r["evaluaciones"] for r in runs_m]
        aciertos_optimo = sum(1 for r in runs_m if r["alcanzo_optimo"])
        config_str = "T0=100.0, alfa=0.95, Tfin=0.01" if metodo == "Recocido Simulado" else "Pob=20, Torneo=3, Cruza=0.9, Mut=0.2"

        resumen_filas.append({
            "metodo": metodo,
            "presupuesto_evaluaciones": presupuesto_evaluaciones,
            "configuracion_parametros": config_str,
            "optimo_global_referencia": optimo_exacto,
            "mejor_costo": min(costos),
            "peor_costo": max(costos),
            "promedio_costo": round(statistics.mean(costos), 3),
            "desv_est_costo": round(statistics.stdev(costos), 3) if len(costos) > 1 else 0.0,
            "promedio_tiempo_seg": round(statistics.mean(tiempos), 5),
            "evaluaciones_promedio": round(statistics.mean(evals), 1),
            "tasa_exito_optimo": f"{aciertos_optimo}/{num_ejecuciones} ({aciertos_optimo/num_ejecuciones*100:.1f}%)",
        })

    # Guardar en CSV si se indican rutas
    if ruta_runs_csv:
        os.makedirs(os.path.dirname(ruta_runs_csv), exist_ok=True)
        with open(ruta_runs_csv, "w", newline="", encoding="utf-8") as f:
            campos = [
                "metodo", "ejecucion", "semilla", "presupuesto_evaluaciones", "parametros", "mejor_costo",
                "costo_base", "penalizacion", "evaluaciones", "tiempo_seg", "alcanzo_optimo"
            ]
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            for r in todos_los_runs:
                fila = dict(r)
                fila["presupuesto_evaluaciones"] = presupuesto_evaluaciones
                fila["parametros"] = "T0=100, alfa=0.95" if r["metodo"] == "Recocido Simulado" else "Pob=20, OX=0.9, Mut=0.2"
                writer.writerow({c: fila[c] for c in campos})

    if ruta_resumen_csv:
        os.makedirs(os.path.dirname(ruta_resumen_csv), exist_ok=True)
        with open(ruta_resumen_csv, "w", newline="", encoding="utf-8") as f:
            campos = [
                "metodo", "presupuesto_evaluaciones", "configuracion_parametros", "optimo_global_referencia",
                "mejor_costo", "peor_costo", "promedio_costo", "desv_est_costo", "promedio_tiempo_seg",
                "evaluaciones_promedio", "tasa_exito_optimo"
            ]
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resumen_filas)

    return resultado_fb, todos_los_runs, resumen_filas


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    ruta_esc = os.path.join(base_dir, "..", "config", "escenarios", "escenario_1_facil.json")
    esc = cargar_escenario(ruta_esc)

    csv_runs = os.path.join(base_dir, "..", "results", "metaheuristics_runs.csv")
    csv_resumen = os.path.join(base_dir, "..", "results", "metaheuristics_comparison.csv")

    fb, runs, resumen = ejecutar_experimento_metaheuristicas(
        esc,
        num_ejecuciones=20,
        presupuesto_evaluaciones=1000,
        semilla_base=100,
        ruta_runs_csv=csv_runs,
        ruta_resumen_csv=csv_resumen,
    )

    print(f"=== FUERZA BRUTA (720 permutaciones): Óptimo={fb['mejor_costo']} en {fb['tiempo_seg']}s ===")
    print(f"Mejor orden exacto: {fb['mejor_orden']}\n")

    print("=== RESUMEN COMPARATIVO (20 ejecuciones con 1000 evaluaciones cada una) ===")
    for row in resumen:
        print(f"[{row['metodo']}]")
        print(f"  Mejor={row['mejor_costo']}, Prom={row['promedio_costo']}, DesvEst={row['desv_est_costo']}")
        print(f"  Tiempo Prom={row['promedio_tiempo_seg']}s, Éxito Óptimo={row['tasa_exito_optimo']}")
