"""
Módulo: pipeline.py
Propósito: Integrar los 5 módulos en un flujo unificado y reproducible:
Cargar escenario -> A* matriz de costos -> Minimax/Juego determinan recurso disponible
-> Metaheurística optimiza la ruta -> Compara plan coordinado vs ignorar al competidor.

Cómo influyen los módulos estratégicos en el plan final:
- Minimax con alfa-beta decide cuántas unidades de recarga se aseguran (recurso_asegurado), lo que
  reduce la capacidad de recurso disponible para optimizar el orden de entregas.
- Teoría de juegos fija la estrategia nuestra (equilibrio seleccionado) y el sobrecosto de
  interferencia que se presupuesta: el peor caso entre las mejores respuestas del rival a nuestra estrategia.
"""

import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scenario import cargar_escenario
from src.heuristic_search import construir_matriz_costos
from src.game_theory import (
    calcular_utilidades_estrategicas,
    calcular_mejores_respuestas,
    encontrar_equilibrios_nash,
    seleccionar_equilibrio_pipeline,
    interferencia_j1,
    peor_interferencia_j1,
)
from src.adversarial import EstadoJuegoRecursos, minimax_alfa_beta
from src.metaheuristics import recocido_simulado, calcular_costo_solucion

# --- Supuestos del modelo (documentar en el informe) ---
COSTO_BASE_JUEGO = 47.0      # Costo óptimo exacto (fuerza bruta) del escenario 1; base de la matriz de pagos
RECARGA_REQUERIDA = 35.0     # Unidades de recarga que el plan necesita
UNIDADES_POR_ACCION = 5.0    # Unidades de recarga que equivale cada unidad tomada en el juego minimax
ESTRATEGIA_INGENUA = 0       # A0 (Directa): lo que haría un plan que no considera al competidor


def ejecutar_pipeline(ruta_escenario: str, semilla: int = 42) -> dict:
    # 1. Cargar escenario logístico
    esc = cargar_escenario(ruta_escenario)
    rng = random.Random(semilla)

    # 2. A* calcula la matriz de costos y rutas reales entre puntos clave
    puntos, etiquetas, matriz_costos, matriz_rutas = construir_matriz_costos(esc, metodo="astar")

    # 3. Minimax y Teoría de Juegos determinan la interacción por el recurso
    _, accion_mm = minimax_alfa_beta(EstadoJuegoRecursos(recurso_restante=7), 0, -float("inf"), float("inf"), True)
    recurso_asegurado = accion_mm * UNIDADES_POR_ACCION
    _, _, matriz_juego = calcular_utilidades_estrategicas(costo_optimo_base=COSTO_BASE_JUEGO)
    equilibrios = encontrar_equilibrios_nash(matriz_juego)
    eq_sel = seleccionar_equilibrio_pipeline(equilibrios, criterio="max_pago_j1")

    # Interferencia presupuestada: jugamos la fila del equilibrio y el rival puede responder con
    # cualquiera de sus mejores respuestas (hay empate), así que se toma la peor para nosotros.
    _, br_j2 = calcular_mejores_respuestas(matriz_juego)
    respuestas_rival = br_j2[eq_sel["fila"]]
    sobrecosto_adversario = max(interferencia_j1(eq_sel["fila"], j) for j in respuestas_rival)

    # 4. Optimización metaheurística con la restricción real adversaria
    capacidad_ajustada = esc.capacidad_recurso - (RECARGA_REQUERIDA - recurso_asegurado)
    res_inteligente = recocido_simulado(matriz_costos, capacidad_ajustada, esc.penalizacion_exceso_recurso, len(esc.entregas), rng=rng)

    # 5. Plan Ingenuo: ignora al rival (capacidad máxima) y queda expuesto al peor caso de interferencia
    res_ingenuo = recocido_simulado(matriz_costos, esc.capacidad_recurso, esc.penalizacion_exceso_recurso, len(esc.entregas), rng=random.Random(semilla))
    sobrecosto_ingenuo = peor_interferencia_j1(ESTRATEGIA_INGENUA)
    costo_total_inteligente = res_inteligente["mejor_costo"] + sobrecosto_adversario
    costo_real_ingenuo = res_ingenuo["mejor_costo"] + sobrecosto_ingenuo

    # Imprimir resumen ejecutivo del plan
    print(f"\n================ RESUMEN PIPELINE: {esc.nombre.upper()} ================")
    print(f"Equilibrio de Nash adoptado: Perfil (A{eq_sel['fila']}, B{eq_sel['columna']}) con pagos ({eq_sel['pago_j1']}, {eq_sel['pago_j2']}) [Criterio: Mayor pago J1]")
    print(f"Ruta óptima inteligente: {[etiquetas[i+1] for i in res_inteligente['mejor_orden']]}")
    print(f"Plan Inteligente (con IA Adversaria) -> Costo base {res_inteligente['mejor_costo']:.1f} + interferencia {sobrecosto_adversario:.1f} = {costo_total_inteligente:.1f}")
    print(f"Plan Ingenuo (Ignora competidor)     -> Costo base {res_ingenuo['mejor_costo']:.1f} + peor caso {sobrecosto_ingenuo:.1f} = {costo_real_ingenuo:.1f}")
    print(f"Ahorro estratégico obtenido:         -> {costo_real_ingenuo - costo_total_inteligente:.1f} unidades")
    print("=" * 60)

    return {
        "orden_inteligente": res_inteligente["mejor_orden"],
        "costo_inteligente": costo_total_inteligente,
        "costo_ingenuo": costo_real_ingenuo,
        "costo_base_inteligente": res_inteligente["mejor_costo"],
        "costo_base_ingenuo": res_ingenuo["mejor_costo"],
        "sobrecosto_inteligente": sobrecosto_adversario,
        "sobrecosto_ingenuo": sobrecosto_ingenuo,
    }


if __name__ == "__main__":
    ruta = os.path.join(os.path.dirname(__file__), "..", "config", "escenarios", "escenario_1_facil.json")
    ejecutar_pipeline(ruta)
