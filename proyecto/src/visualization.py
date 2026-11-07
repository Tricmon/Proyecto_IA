"""
Módulo: visualization.py
Propósito: Generar automáticamente las 5 figuras y gráficas requeridas
a partir de los datos experimentales reales generados por código.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Generación sin ventana gráfica para servidores y terminales
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def graficar_busqueda_heuristica(datos_heuristicos: list[dict], ruta_guardado: str):
    """
    Gráfica 1: Comparativa A* vs Greedy en costo y nodos expandidos.
    """
    # Filtrar solo escenarios con solución
    validos = [d for d in datos_heuristicos if d["encontrado"]]
    escenarios = sorted(list(set(d["escenario"] for d in validos)))

    costos_astar = [next(d["costo"] for d in validos if d["escenario"] == esc and d["metodo"] == "ASTAR") for esc in escenarios]
    costos_greedy = [next(d["costo"] for d in validos if d["escenario"] == esc and d["metodo"] == "GREEDY") for esc in escenarios]

    nodos_astar = [next(d["nodos_expandidos"] for d in validos if d["escenario"] == esc and d["metodo"] == "ASTAR") for esc in escenarios]
    nodos_greedy = [next(d["nodos_expandidos"] for d in validos if d["escenario"] == esc and d["metodo"] == "GREEDY") for esc in escenarios]

    x = range(len(escenarios))
    ancho = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subgráfico 1: Costo
    ax1.bar([i - ancho/2 for i in x], costos_astar, width=ancho, label="A*", color="#2b5c8f")
    ax1.bar([i + ancho/2 for i in x], costos_greedy, width=ancho, label="Greedy", color="#d95f02")
    ax1.set_title("Costo de Ruta Encontrada (Menor es mejor)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.split(" - ")[1] for e in escenarios], rotation=15)
    ax1.set_ylabel("Costo Total")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Subgráfico 2: Nodos Expandidos
    ax2.bar([i - ancho/2 for i in x], nodos_astar, width=ancho, label="A*", color="#2b5c8f")
    ax2.bar([i + ancho/2 for i in x], nodos_greedy, width=ancho, label="Greedy", color="#d95f02")
    ax2.set_title("Nodos Expandidos (Esfuerzo de Búsqueda)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.split(" - ")[1] for e in escenarios], rotation=15)
    ax2.set_ylabel("Nodos Expandidos")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado, dpi=150)
    plt.close()


def graficar_busqueda_local(datos_local: list[dict], ruta_guardado: str):
    """
    Gráfica 2: Dispersión y mejora de Hill Climbing en 10 ejecuciones.
    """
    runs = [d["ejecucion"] for d in datos_local]
    costo_ini = [d["costo_inicial"] for d in datos_local]
    costo_fin = [d["costo_final"] for d in datos_local]

    plt.figure(figsize=(9, 5))
    plt.plot(runs, costo_ini, "o--", color="#d95f02", label="Costo Solución Inicial (Aleatoria)")
    plt.plot(runs, costo_fin, "s-", color="#2b5c8f", linewidth=2, label="Costo Óptimo Local Alcanzado")
    plt.axhline(min(costo_fin), color="green", linestyle=":", label=f"Mejor Óptimo Hallado ({min(costo_fin):.1f})")

    plt.title("Hill Climbing: Convergencia desde 10 Soluciones Iniciales")
    plt.xlabel("Número de Ejecución")
    plt.ylabel("Costo de la Ruta")
    plt.xticks(runs)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado, dpi=150)
    plt.close()


def graficar_metaheuristicas(runs_meta: list[dict], optimo_fb: float, ruta_guardado: str):
    """
    Gráfica 3: Comparación de costos entre Recocido Simulado y Algoritmo Genético en 20 runs.
    """
    runs_sa = [r["mejor_costo"] for r in runs_meta if r["metodo"] == "Recocido Simulado"]
    runs_ga = [r["mejor_costo"] for r in runs_meta if r["metodo"] == "Algoritmo Genético"]

    plt.figure(figsize=(9, 5))
    plt.boxplot([runs_sa, runs_ga], tick_labels=["Recocido Simulado", "Algoritmo Genético"], patch_artist=True,
                boxprops=dict(facecolor="#c6dbef", color="#2b5c8f"),
                medianprops=dict(color="#d95f02", linewidth=2))
    plt.axhline(optimo_fb, color="green", linestyle="--", linewidth=1.5, label=f"Óptimo Fuerza Bruta ({optimo_fb:.1f})")

    plt.title("Metaheurísticas: Distribución de Costos en 20 Ejecuciones (1000 Evals)")
    plt.ylabel("Costo de Solución")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado, dpi=150)
    plt.close()


def graficar_minimax_vs_alfabeta(datos_adversarios: list[dict], ruta_guardado: str):
    """
    Gráfica 4: Nodos explorados en Minimax vs Poda Alfa-Beta según el orden de jugadas.
    """
    # Filtrar solo el árbol completo para ver el impacto del orden de jugadas
    filas_arbol = [d for d in datos_adversarios if "Árbol Completo" in d["configuracion"]]
    if not filas_arbol:
        filas_arbol = datos_adversarios[:3]

    etiquetas = [f"Orden {d['orden_jugadas']}" for d in filas_arbol]
    nodos_mm = [d["nodos_minimax"] for d in filas_arbol]
    nodos_ab = [d["nodos_alfa_beta"] for d in filas_arbol]

    x = range(len(etiquetas))
    ancho = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - ancho/2 for i in x], nodos_mm, width=ancho, label="Minimax Puro", color="#756bb1")
    plt.bar([i + ancho/2 for i in x], nodos_ab, width=ancho, label="Minimax con Poda Alfa-Beta", color="#31a354")

    for i in x:
        pct = filas_arbol[i]["reduccion_nodos_pct"]
        plt.text(i + ancho/2, nodos_ab[i] + 2, f"-{pct:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#1c6b34")

    plt.title("Impacto del Orden de Jugadas en la Poda Alfa-Beta (Árbol Completo)")
    plt.xlabel("Estrategia de Ordenamiento de Jugadas")
    plt.ylabel("Nodos Visitados")
    plt.xticks(x, etiquetas)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado, dpi=150)
    plt.close()


def graficar_escenario_y_ruta(escenario, secuencia_puntos, ruta_guardado: str):
    """
    Gráfica 5: Mapa 12x12 del escenario con obstáculos, puntos clave y la ruta trazada.
    """
    import numpy as np
    matriz = np.array(escenario.cuadricula, dtype=float)

    plt.figure(figsize=(8, 8))
    # Mapa de calor de costos
    plt.imshow(matriz, cmap="YlOrBr", origin="upper")
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label("Costo de Paso (0 = Obstáculo)", rotation=270, labelpad=15)

    # Dibujar obstáculos
    for r in range(escenario.filas):
        for c in range(escenario.columnas):
            if escenario.es_obstaculo(r, c):
                plt.scatter(c, r, color="black", marker="s", s=180)

    # Dibujar ruta
    if secuencia_puntos:
        coords_y = [p[0] for p in secuencia_puntos]
        coords_x = [p[1] for p in secuencia_puntos]
        plt.plot(coords_x, coords_y, color="#2b5c8f", linewidth=2.5, linestyle="-", label="Ruta Planificada", zorder=4)

    # Dibujar inicio y fin
    plt.scatter(escenario.inicio[1], escenario.inicio[0], color="green", s=250, marker="o", label="Inicio (I)", zorder=5)
    plt.scatter(escenario.destino[1], escenario.destino[0], color="red", s=250, marker="X", label="Destino (D)", zorder=5)

    # Dibujar entregas
    for idx, (er, ec) in enumerate(escenario.entregas):
        plt.scatter(ec, er, color="blue", s=180, marker="^", zorder=5)
        plt.text(ec + 0.2, er - 0.2, f"E{idx+1}", color="darkblue", fontweight="bold")

    plt.title(f"Plan Logístico Integral: {escenario.nombre}")
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.xticks(range(escenario.columnas))
    plt.yticks(range(escenario.filas))
    plt.grid(True, color="gray", linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    plt.savefig(ruta_guardado, dpi=150)
    plt.close()
