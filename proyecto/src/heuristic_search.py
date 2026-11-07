"""
Módulo: heuristic_search.py
Propósito: Implementar búsqueda heurística en cuadrícula utilizando una única función general
donde se parametriza la función de evaluación:
  - A*: f(n) = g(n) + h(n)
  - Greedy Best-First Search: f(n) = h(n)

Decisiones de diseño:
- Se implementa el algoritmo a mano usando 'heapq' como cola de prioridad (requisito: sin networkx ni scipy).
- Heurística admisible: h(n) = Distancia Manhattan * costo_mínimo_de_celda.
  Justificación de admisibilidad: La distancia Manhattan representa el menor número de pasos
  posibles entre dos celdas sin diagonales ni obstáculos. Al multiplicarla por el costo mínimo
  global de cualquier celda transitable (c_min >= 1), se garantiza matemáticamente que h(n) <= h*(n)
  para todo nodo n (nunca sobreestima el costo real). Además, es consistente (monótona).
- Se construye la matriz de distancias/costos reales para alimentar a los módulos posteriores.
"""

import csv
import heapq
import os
import sys
import time
from typing import Any

# Permitir importaciones relativas al directorio raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scenario import Escenario, cargar_escenario


def heuristica_manhattan(pos_actual: tuple[int, int], pos_meta: tuple[int, int], costo_minimo: int) -> int:
    """
    Calcula la heurística admisible y consistente basada en Manhattan escalada por c_min.
    h(n) = (|r1 - r2| + |c1 - c2|) * c_min
    """
    dr = abs(pos_actual[0] - pos_meta[0])
    dc = abs(pos_actual[1] - pos_meta[1])
    return (dr + dc) * costo_minimo


def busqueda_heuristica(
    escenario: Escenario,
    origen: tuple[int, int],
    destino: tuple[int, int],
    metodo: str = "astar",
) -> dict[str, Any]:
    """
     A* usa f(n) = g(n) + h(n) (garantía de cota óptima); Greedy usa f(n) = h(n) (búsqueda miope).
        - La heurística Manhattan multiplicada por c_min (costo mínimo de celda transitable)
          garantiza matemáticamente que h(n) <= h*(n) (admisible) y h(n) <= c + h(n') (consistente).
    
        O((V + E) log V) en grafo con V <= 144 celdas y E <= 4V aristas.
        En el peor caso de árbol de búsqueda O(b^d) donde b <= 4 y d es la longitud del camino.
    """
    metodo_normalizado = metodo.strip().lower()
    if metodo_normalizado not in ("astar", "greedy"):
        raise ValueError(f"Método de búsqueda desconocido: '{metodo}'. Use 'astar' o 'greedy'.")

    # Si origen o destino son obstáculos o inválidos
    if escenario.es_obstaculo(origen[0], origen[1]) or escenario.es_obstaculo(destino[0], destino[1]):
        return {
            "encontrado": False,
            "ruta": [],
            "costo": float("inf"),
            "nodos_expandidos": 0,
            "tiempo_seg": 0.0,
            "metodo": metodo_normalizado,
        }

    # Caso trivial: ya estamos en el destino
    if origen == destino:
        return {
            "encontrado": True,
            "ruta": [origen],
            "costo": 0,
            "nodos_expandidos": 0,
            "tiempo_seg": 0.0,
            "metodo": metodo_normalizado,
        }

    tiempo_inicio = time.perf_counter()
    c_min = escenario.costo_minimo_global()

    # Estructuras de búsqueda
    # La cola de prioridad almacena tuplas: (prioridad, contador, nodo_actual)
    # El contador actúa como desempate determinista para evitar comparar tuplas (r, c).
    contador = 0
    frontera = []
    
    g_cost = {origen: 0}
    padre = {origen: None}
    cerrados = set()
    nodos_expandidos = 0

    h_inicial = heuristica_manhattan(origen, destino, c_min)
    prioridad_inicial = (0 + h_inicial) if metodo_normalizado == "astar" else h_inicial
    heapq.heappush(frontera, (prioridad_inicial, contador, origen))

    encontrado = False

    while frontera:
        prioridad_actual, _, actual = heapq.heappop(frontera)

        if actual in cerrados:
            continue

        cerrados.add(actual)
        nodos_expandidos += 1

        # Comprobación de meta
        if actual == destino:
            encontrado = True
            break

        # Explorar vecinos en 4 direcciones
        for vecino in escenario.obtener_vecinos(actual[0], actual[1]):
            if vecino in cerrados:
                continue

            costo_paso = escenario.costo_celda(vecino[0], vecino[1])
            nuevo_g = g_cost[actual] + costo_paso

            # Si encontramos un mejor camino hacia el vecino
            if vecino not in g_cost or nuevo_g < g_cost[vecino]:
                g_cost[vecino] = nuevo_g
                padre[vecino] = actual
                h_val = heuristica_manhattan(vecino, destino, c_min)

                if metodo_normalizado == "astar":
                    prioridad = nuevo_g + h_val
                else:  # 'greedy'
                    prioridad = h_val

                contador += 1
                heapq.heappush(frontera, (prioridad, contador, vecino))

    tiempo_total = time.perf_counter() - tiempo_inicio

    if not encontrado:
        return {
            "encontrado": False,
            "ruta": [],
            "costo": float("inf"),
            "nodos_expandidos": nodos_expandidos,
            "tiempo_seg": round(tiempo_total, 6),
            "metodo": metodo_normalizado,
        }

    # Reconstrucción de la ruta desde destino hasta origen
    ruta = []
    curr = destino
    while curr is not None:
        ruta.append(curr)
        curr = padre[curr]
    ruta.reverse()

    costo_total = g_cost[destino]

    return {
        "encontrado": True,
        "ruta": ruta,
        "costo": costo_total,
        "nodos_expandidos": nodos_expandidos,
        "tiempo_seg": round(tiempo_total, 6),
        "metodo": metodo_normalizado,
    }


def a_estrella(escenario: Escenario, origen: tuple[int, int], destino: tuple[int, int]) -> dict[str, Any]:
    """Envoltorio específico para A*."""
    return busqueda_heuristica(escenario, origen, destino, metodo="astar")


def greedy_best_first(escenario: Escenario, origen: tuple[int, int], destino: tuple[int, int]) -> dict[str, Any]:
    """Envoltorio específico para Greedy Best-First Search."""
    return busqueda_heuristica(escenario, origen, destino, metodo="greedy")


def construir_matriz_costos(
    escenario: Escenario, metodo: str = "astar"
) -> tuple[list[tuple[int, int]], list[str], list[list[float]], list[list[list[tuple[int, int]]]]]:
    """
    Construye la matriz completa de costos y rutas entre todos los puntos clave:
    [Inicio, E1, E2, E3, E4, E5, E6, Destino].
    
    Retorna:
        - puntos: Lista de coordenadas.
        - etiquetas: Nombres legibles ['Inicio', 'E1', ..., 'Destino'].
        - matriz_costos: Matriz 2D de NxN con costos mínimos.
        - matriz_rutas: Matriz 2D de NxN con las secuencias de celdas.
    """
    puntos = [escenario.inicio] + list(escenario.entregas) + [escenario.destino]
    etiquetas = ["Inicio"] + [f"E{i+1}" for i in range(len(escenario.entregas))] + ["Destino"]
    n = len(puntos)

    matriz_costos = [[0.0] * n for _ in range(n)]
    matriz_rutas = [[[] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matriz_costos[i][j] = 0.0
                matriz_rutas[i][j] = [puntos[i]]
            else:
                res = busqueda_heuristica(escenario, puntos[i], puntos[j], metodo=metodo)
                if not res["encontrado"]:
                    matriz_costos[i][j] = float("inf")
                    matriz_rutas[i][j] = []
                else:
                    matriz_costos[i][j] = float(res["costo"])
                    matriz_rutas[i][j] = res["ruta"]

    return puntos, etiquetas, matriz_costos, matriz_rutas


def ejecutar_experimentos_heuristicos(rutas_escenarios: list[str], ruta_salida_csv: str) -> list[dict[str, Any]]:
    """
    Ejecuta comparaciones entre A* y Greedy en todos los escenarios provistos
    y guarda los resultados cuantitativos en un archivo CSV.
    """
    resultados = []

    for ruta_esc in rutas_escenarios:
        esc = cargar_escenario(ruta_esc)

        # Probar desde Inicio hasta Destino
        for metodo in ["astar", "greedy"]:
            res = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo=metodo)
            fila = {
                "escenario": esc.nombre,
                "metodo": metodo.upper(),
                "origen": str(esc.inicio),
                "destino": str(esc.destino),
                "encontrado": res["encontrado"],
                "costo": res["costo"] if res["encontrado"] else "INF",
                "longitud_pasos": len(res["ruta"]) if res["encontrado"] else 0,
                "nodos_expandidos": res["nodos_expandidos"],
                "tiempo_seg": res["tiempo_seg"],
            }
            resultados.append(fila)

    os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
    with open(ruta_salida_csv, "w", newline="", encoding="utf-8") as f:
        campos = [
            "escenario", "metodo", "origen", "destino",
            "encontrado", "costo", "longitud_pasos", "nodos_expandidos", "tiempo_seg"
        ]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    return resultados


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    archivos = [
        os.path.join(base_dir, "..", "config", "escenarios", "escenario_1_facil.json"),
        os.path.join(base_dir, "..", "config", "escenarios", "escenario_2_obstaculos.json"),
        os.path.join(base_dir, "..", "config", "escenarios", "escenario_3_trampa_greedy.json"),
        os.path.join(base_dir, "..", "config", "escenarios", "escenario_4_sin_solucion.json"),
    ]
    csv_salida = os.path.join(base_dir, "..", "results", "heuristic_search_results.csv")
    res = ejecutar_experimentos_heuristicos(archivos, csv_salida)
    print(f"=== Experimentos Heurísticos Finalizados. Resultados guardados en: {csv_salida} ===")
    for r in res:
        print(f"[{r['escenario']}] {r['metodo']}: costo={r['costo']}, expandidos={r['nodos_expandidos']}, tiempo={r['tiempo_seg']}s, exito={r['encontrado']}")
