"""
Suite de pruebas integrales para todos los módulos del proyecto (pytest).
Cubre los casos límite y propiedades teóricas requeridas por la rúbrica.
"""

import os
import random
import pytest

from src.scenario import cargar_escenario, EscenarioInvalidoError
from src.heuristic_search import busqueda_heuristica, construir_matriz_costos
from src.local_search import hill_climbing, calcular_costo_solucion
from src.metaheuristics import recocido_simulado, evaluar_fuerza_bruta
from src.game_theory import calcular_utilidades_estrategicas, encontrar_equilibrios_nash, calcular_mejores_respuestas
from src.adversarial import EstadoJuegoRecursos, minimax_puro, minimax_alfa_beta, ContadorRendimiento


@pytest.fixture
def rutas_escenarios():
    base = os.path.join(os.path.dirname(__file__), "..", "config", "escenarios")
    return {
        "facil": os.path.join(base, "escenario_1_facil.json"),
        "obstaculos": os.path.join(base, "escenario_2_obstaculos.json"),
        "trampa": os.path.join(base, "escenario_3_trampa_greedy.json"),
        "sin_solucion": os.path.join(base, "escenario_4_sin_solucion.json"),
    }


def test_1_escenario_sin_solucion(rutas_escenarios):
    """Caso límite: El escenario 4 no tiene camino; A* y Greedy deben reportar sin solución de forma limpia."""
    esc = cargar_escenario(rutas_escenarios["sin_solucion"])
    res_a = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo="astar")
    res_g = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo="greedy")

    assert not res_a["encontrado"]
    assert res_a["costo"] == float("inf")
    assert not res_g["encontrado"]
    assert res_g["costo"] == float("inf")


def test_2_entrada_invalida_obstaculo(tmp_path):
    """Caso límite: Detección de errores cuando una coordenada cae sobre un obstáculo."""
    ruta_temp = tmp_path / "invalido.json"
    ruta_temp.write_text('{"nombre": "x", "filas": 2, "columnas": 2, "inicio": [0,0], "destino": [1,1], "entregas": [[0,1],[1,0],[0,0]], "capacidad_recurso": 10, "cuadricula": [[0, 1], [1, 1]]}', encoding="utf-8")
    with pytest.raises(EscenarioInvalidoError):
        cargar_escenario(str(ruta_temp))


def test_3_astar_menor_o_igual_greedy(rutas_escenarios):
    """Propiedad matemática: Por admisibilidad y consistencia, el costo de A* es <= al de Greedy."""
    for nombre in ["facil", "obstaculos", "trampa"]:
        esc = cargar_escenario(rutas_escenarios[nombre])
        res_a = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo="astar")
        res_g = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo="greedy")
        assert res_a["costo"] <= res_g["costo"]


def test_4_trampa_greedy_estricta(rutas_escenarios):
    """Verificación de la trampa: Greedy es engañado por Manhattan y obtiene un costo estrictamente peor."""
    esc = cargar_escenario(rutas_escenarios["trampa"])
    res_a = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo="astar")
    res_g = busqueda_heuristica(esc, esc.inicio, esc.destino, metodo="greedy")
    assert res_a["costo"] < res_g["costo"]
    assert res_a["costo"] == 21
    assert res_g["costo"] == 51


def test_5_minimax_igual_a_alfabeta():
    """Teorema: Minimax y Minimax con Poda Alfa-Beta deben retornar el mismo valor para todo estado."""
    for recurso in [5, 6, 7]:
        for lim in [2, 3, None]:
            e = EstadoJuegoRecursos(recurso_restante=recurso)
            val_mm, _ = minimax_puro(e, profundidad=0, es_max=True, limite_profundidad=lim)
            val_ab, _ = minimax_alfa_beta(e, profundidad=0, alfa=float("-inf"), beta=float("inf"), es_max=True, limite_profundidad=lim)
            assert abs(val_mm - val_ab) < 1e-9


def test_6_alfabeta_visita_menos_o_igual_nodos():
    """Eficiencia: La poda Alfa-Beta nunca explora más nodos que Minimax estándar."""
    e = EstadoJuegoRecursos(recurso_restante=7)
    c_mm = ContadorRendimiento()
    c_ab = ContadorRendimiento()

    minimax_puro(e, 0, True, None, c_mm)
    minimax_alfa_beta(e, 0, float("-inf"), float("inf"), True, None, c_ab)

    assert c_ab.nodos_visitados <= c_mm.nodos_visitados
    assert c_ab.podas_realizadas > 0


def test_7_misma_semilla_mismo_resultado(rutas_escenarios):
    """Reproducibilidad: Dos ejecuciones estocásticas con la misma semilla dan resultados idénticos."""
    esc = cargar_escenario(rutas_escenarios["facil"])
    _, _, matriz_costos, _ = construir_matriz_costos(esc, metodo="astar")

    res1 = recocido_simulado(matriz_costos, esc.capacidad_recurso, n_entregas=6, rng=random.Random(123))
    res2 = recocido_simulado(matriz_costos, esc.capacidad_recurso, n_entregas=6, rng=random.Random(123))

    assert res1["mejor_orden"] == res2["mejor_orden"]
    assert res1["mejor_costo"] == res2["mejor_costo"]


def test_8_hill_climbing_nunca_empeora(rutas_escenarios):
    """Invariante: Hill Climbing garantiza que costo_final <= costo_inicial."""
    esc = cargar_escenario(rutas_escenarios["facil"])
    _, _, matriz_costos, _ = construir_matriz_costos(esc, metodo="astar")

    sol_inicial = [5, 4, 3, 2, 1, 0]
    res = hill_climbing(sol_inicial, matriz_costos, esc.capacidad_recurso)
    assert res["costo_final"] <= res["costo_inicial"]


def test_9_fuerza_bruta_es_cota_inferior_absoluta(rutas_escenarios):
    """Optimalidad global: Ninguna solución evaluada puede superar en costo al óptimo de Fuerza Bruta."""
    esc = cargar_escenario(rutas_escenarios["facil"])
    _, _, matriz_costos, _ = construir_matriz_costos(esc, metodo="astar")

    fb = evaluar_fuerza_bruta(matriz_costos, esc.capacidad_recurso, n_entregas=6)
    c_opt = fb["mejor_costo"]

    # Probar 10 permutaciones aleatorias
    rng = random.Random(999)
    for _ in range(10):
        perm = list(range(6))
        rng.shuffle(perm)
        c, _, _ = calcular_costo_solucion(perm, matriz_costos, esc.capacidad_recurso)
        assert c >= c_opt


def test_10_estabilidad_equilibrio_nash():
    """Teoría de Juegos: En los equilibrios de Nash identificados, ningún jugador puede mejorar unilateralmente."""
    _, _, matriz = calcular_utilidades_estrategicas()
    equilibrios = encontrar_equilibrios_nash(matriz)
    br_j1, br_j2 = calcular_mejores_respuestas(matriz)

    assert len(equilibrios) > 0
    for eq in equilibrios:
        i, j = eq["fila"], eq["columna"]
        assert i in br_j1[j]
        assert j in br_j2[i]
