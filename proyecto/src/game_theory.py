"""
Módulo: game_theory.py
Propósito: Modelar y resolver la interacción estratégica entre nuestra empresa logística
y un competidor mediante Teoría de Juegos (matriz 3x3 de información completa).

Decisiones de diseño y justificación para defensa:
1. Jugadores:
   - Jugador 1 (Nosotros): Operador logístico principal.
   - Jugador 2 (Competidor): Operador rival en la misma zona de reparto.
2. Estrategias:
   - A0 / B0: 'Directa / Ocupación': Demanda máxima del recurso en la estación de recarga central.
   - A1 / B1: 'Equilibrada': Demanda moderada, compartiendo la infraestructura.
   - A2 / B2: 'Conservadora / Periférica': Desvío perimetral que evita por completo la estación.
3. Matriz de pagos calculada con una fórmula (NO puesta celda por celda a mano):
   Utilidad = Ingreso_Venta (120) - [Costo_Ruta + Sobrecosto_Interferencia].
   Los números que alimentan la fórmula (sobrecosto de ruta de cada estrategia, costo base del
   rival y matriz de interferencia) son SUPUESTOS DEL MODELO, definidos como constantes al inicio
   del módulo. No salen de A*; hay que presentarlos como parámetros del modelo en el informe.
4. Resolución:
   - Detección exhaustiva de mejores respuestas celda por celda.
   - Detección de estrategias estrictamente dominadas.
   - Distinción entre equilibrios de Nash estrictos y débiles (empates).
   - Selección explícita de equilibrio para el pipeline bajo criterio de dominancia de pagos de J1.
"""

import csv
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Parámetros del modelo (supuestos documentados)
# ---------------------------------------------------------------------------
INGRESO_ENTREGA = 120.0  # Beneficio fijo por completar las entregas

# Costo extra de ruta de J1 según su estrategia (A0, A1, A2) sobre el costo óptimo base
SOBRECOSTO_RUTA_J1 = [0.0, 6.0, 15.0]

# Costo base del competidor J2 según su estrategia (B0, B1, B2)
COSTO_BASE_J2 = [48.0, 52.0, 60.0]

# Interferencia por coincidencia en la estación central: INTERFERENCIA[i][j] = (penal_J1, penal_J2)
INTERFERENCIA = [
    # B0 (Ocupación)        B1 (Compartido)       B2 (Periférico)
    [(25.0, 20.0),          (10.0, 8.0),          (0.0, 0.0)],    # A0
    [(12.0, 15.0),          (4.0, 4.0),           (0.0, 0.0)],    # A1
    [(0.0, 0.0),            (0.0, 0.0),           (0.0, 0.0)],    # A2
]


def interferencia_j1(fila: int, columna: int) -> float:
    """Sobrecosto de interferencia que sufre J1 (nosotros) cuando J1 juega 'fila' y J2 juega 'columna'."""
    return INTERFERENCIA[fila][columna][0]


def peor_interferencia_j1(fila: int) -> float:
    """
    Peor sobrecosto posible para J1 al jugar la estrategia 'fila', sin importar qué haga J2.
    Representa el caso de un plan que NO anticipó al competidor y queda expuesto a lo peor.
    """
    return max(interferencia_j1(fila, j) for j in range(len(INTERFERENCIA[fila])))


def calcular_utilidades_estrategicas(
    costo_optimo_base: float = 47.0,
) -> tuple[list[str], list[str], list[list[tuple[float, float]]]]:
    """
    Calcula la matriz de pagos 3x3 basándose en los costos de operación y congestión.

    Retorna:
      - estrategias_j1: Nombres de estrategias del Jugador 1.
      - estrategias_j2: Nombres de estrategias del Jugador 2.
      - matriz_pagos: Matriz 3x3 donde cada celda es una tupla (pago_j1, pago_j2).
    """
    estrategias_j1 = [
        "A0: Directa (Alta recarga)",
        "A1: Equilibrada (Media recarga)",
        "A2: Conservadora (Cero recarga / Desvío)",
    ]
    estrategias_j2 = [
        "B0: Ocupación Estación",
        "B1: Uso Compartido",
        "B2: Operación Periférica",
    ]

    # Costos base según la ruta elegida por J1
    costo_base_j1 = [costo_optimo_base + extra for extra in SOBRECOSTO_RUTA_J1]

    matriz_pagos = [[(0.0, 0.0) for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            pen_j1, pen_j2 = INTERFERENCIA[i][j]
            costo_total_j1 = costo_base_j1[i] + pen_j1
            costo_total_j2 = COSTO_BASE_J2[j] + pen_j2

            utilidad_j1 = round(INGRESO_ENTREGA - costo_total_j1, 2)
            utilidad_j2 = round(INGRESO_ENTREGA - costo_total_j2, 2)

            matriz_pagos[i][j] = (utilidad_j1, utilidad_j2)

    return estrategias_j1, estrategias_j2, matriz_pagos


def calcular_mejores_respuestas(
    matriz_pagos: list[list[tuple[float, float]]]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    QUÉ HACE:
        Determina para cada acción del adversario cuáles son las acciones óptimas que
        maximizan el beneficio de cada jugador (Best Responses).
    POR QUÉ ASÍ:
        Permite verificar la condición formal de equilibrio de Nash: un perfil (i*, j*)
        es equilibrio si y solo si i* in BR_1(j*) y j* in BR_2(i*).
    COMPLEJIDAD EN TIEMPO:
        O(M * N) donde M=3 filas y N=3 columnas. Exactamente 9 comparaciones por jugador.
    """
    n_filas = len(matriz_pagos)
    n_cols = len(matriz_pagos[0])

    br_j1 = {}  # columna j -> lista de mejores filas i
    for j in range(n_cols):
        max_pago = max(matriz_pagos[i][j][0] for i in range(n_filas))
        mejores = [i for i in range(n_filas) if abs(matriz_pagos[i][j][0] - max_pago) < 1e-9]
        br_j1[j] = mejores

    br_j2 = {}  # fila i -> lista de mejores columnas j
    for i in range(n_filas):
        max_pago = max(matriz_pagos[i][j][1] for j in range(n_cols))
        mejores = [j for j in range(n_cols) if abs(matriz_pagos[i][j][1] - max_pago) < 1e-9]
        br_j2[i] = mejores

    return br_j1, br_j2


def detectar_estrategias_dominadas(
    matriz_pagos: list[list[tuple[float, float]]]
) -> tuple[list[str], list[str]]:
    """
    QUÉ HACE:
        Detecta estrategias estrictamente dominadas para ambos jugadores.
    POR QUÉ ASÍ:
        Una estrategia X es estrictamente dominada por Y si Y ofrece un pago mayor que X
        ante CADA decisión del adversario. Un jugador racional nunca juega una estrategia así.
    COMPLEJIDAD EN TIEMPO:
        O(M^2 * N) para J1 y O(N^2 * M) para J2; con matriz 3x3 son pocas comparaciones.
    """
    n_filas = len(matriz_pagos)
    n_cols = len(matriz_pagos[0])
    dominadas_j1 = []
    dominadas_j2 = []

    # Dominancia en J1 (filas)
    for i1 in range(n_filas):
        for i2 in range(n_filas):
            if i1 == i2:
                continue
            if all(matriz_pagos[i2][j][0] > matriz_pagos[i1][j][0] for j in range(n_cols)):
                dominadas_j1.append(f"A{i1} estrictamente dominada por A{i2}")

    # Dominancia en J2 (columnas)
    for j1 in range(n_cols):
        for j2 in range(n_cols):
            if j1 == j2:
                continue
            if all(matriz_pagos[i][j2][1] > matriz_pagos[i][j1][1] for i in range(n_filas)):
                dominadas_j2.append(f"B{j1} estrictamente dominada por B{j2}")

    return dominadas_j1, dominadas_j2


def encontrar_equilibrios_nash(
    matriz_pagos: list[list[tuple[float, float]]]
) -> list[dict[str, Any]]:
    """
    QUÉ HACE:
        Encuentra todos los Equilibrios de Nash en estrategias puras celda por celda.
        Marca cada equilibrio como estricto o débil.
    POR QUÉ ASÍ:
        Un perfil es Nash si cada jugador juega una mejor respuesta a la jugada del otro.
        Es débil cuando algún jugador tiene más de una mejor respuesta (hay empate en su pago),
        es decir, podría cambiar de estrategia sin perder nada.
    COMPLEJIDAD EN TIEMPO:
        O(M * N) después de calcular las mejores respuestas.
    """
    br_j1, br_j2 = calcular_mejores_respuestas(matriz_pagos)
    equilibrios = []

    for i in range(len(matriz_pagos)):
        for j in range(len(matriz_pagos[0])):
            if i in br_j1[j] and j in br_j2[i]:
                # Es débil si para J1 o para J2 hay más de una mejor respuesta en esa columna/fila
                es_debil = (len(br_j1[j]) > 1) or (len(br_j2[i]) > 1)
                tipo = "Débil (Empate)" if es_debil else "Estricto"
                equilibrios.append({
                    "fila": i,
                    "columna": j,
                    "pago_j1": matriz_pagos[i][j][0],
                    "pago_j2": matriz_pagos[i][j][1],
                    "tipo": tipo,
                    "es_debil": es_debil,
                })

    return equilibrios


def seleccionar_equilibrio_pipeline(
    equilibrios: list[dict[str, Any]],
    criterio: str = "max_pago_j1",
) -> dict[str, Any]:
    """
    Selecciona un equilibrio de Nash específico para alimentar el pipeline de decisión.
    Único criterio implementado ('max_pago_j1'): maximizar la utilidad de nuestra empresa (J1).
    Si hay empate en el pago de J1, desempata por mayor pago de J2 (con J1 igual, equivale a
    mayor pago conjunto). Cualquier otro criterio lanza ValueError para no aparentar uno que no existe.
    """
    if criterio != "max_pago_j1":
        raise ValueError(f"Criterio no implementado: {criterio!r}. Solo existe 'max_pago_j1'.")
    if not equilibrios:
        raise ValueError("No se encontraron equilibrios de Nash puros para seleccionar.")

    # Ordenar por pago J1 descendente, luego pago J2 descendente
    seleccionado = max(equilibrios, key=lambda eq: (eq["pago_j1"], eq["pago_j2"]))
    return seleccionado


def ejecutar_analisis_teoria_juegos(
    ruta_salida_csv: str | None = None,
    costo_optimo_base: float = 47.0,
) -> dict[str, Any]:
    """
    Ejecuta el análisis completo de teoría de juegos, selecciona el equilibrio para el pipeline
    y guarda la matriz completa en CSV (incluye columnas de mejor respuesta de cada jugador).
    """
    est_j1, est_j2, matriz = calcular_utilidades_estrategicas(costo_optimo_base=costo_optimo_base)
    br_j1, br_j2 = calcular_mejores_respuestas(matriz)
    dom_j1, dom_j2 = detectar_estrategias_dominadas(matriz)
    equilibrios = encontrar_equilibrios_nash(matriz)
    eq_pipeline = seleccionar_equilibrio_pipeline(equilibrios)

    if ruta_salida_csv:
        os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
        # utf-8-sig agrega la marca BOM para que Excel muestre bien los acentos
        with open(ruta_salida_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Estrategia_J1", "Estrategia_J2", "Pago_J1", "Pago_J2",
                "Es_Nash", "Tipo_Equilibrio", "Seleccionado_Pipeline",
                "Es_Mejor_Respuesta_J1", "Es_Mejor_Respuesta_J2",
            ])
            for i in range(3):
                for j in range(3):
                    pago1, pago2 = matriz[i][j]
                    match_eq = next((e for e in equilibrios if e["fila"] == i and e["columna"] == j), None)
                    es_nash = match_eq is not None
                    tipo = match_eq["tipo"] if match_eq else "No es Nash"
                    es_sel = (es_nash and eq_pipeline["fila"] == i and eq_pipeline["columna"] == j)
                    es_mr_j1 = i in br_j1[j]
                    es_mr_j2 = j in br_j2[i]
                    writer.writerow([est_j1[i], est_j2[j], pago1, pago2, es_nash, tipo, es_sel, es_mr_j1, es_mr_j2])

    return {
        "estrategias_j1": est_j1,
        "estrategias_j2": est_j2,
        "matriz_pagos": matriz,
        "mejores_respuestas_j1": br_j1,
        "mejores_respuestas_j2": br_j2,
        "dominancia_j1": dom_j1,
        "dominancia_j2": dom_j2,
        "equilibrios_nash": equilibrios,
        "equilibrio_pipeline": eq_pipeline,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    csv_salida = os.path.join(base_dir, "..", "results", "game_theory_matrix.csv")
    analisis = ejecutar_analisis_teoria_juegos(ruta_salida_csv=csv_salida)

    print("=== MATRIZ DE PAGOS 3x3 (Jugador 1, Jugador 2) ===")
    print(f"{'':<35} | {analisis['estrategias_j2'][0]:<22} | {analisis['estrategias_j2'][1]:<22} | {analisis['estrategias_j2'][2]:<22}")
    print("-" * 110)
    for i, est1 in enumerate(analisis["estrategias_j1"]):
        fila_str = f"{est1:<35} | "
        for j in range(3):
            p1, p2 = analisis["matriz_pagos"][i][j]
            fila_str += f"({p1:5.1f}, {p2:5.1f}){'':<10} | "
        print(fila_str)

    print("\n=== MEJORES RESPUESTAS ===")
    for j, filas in analisis["mejores_respuestas_j1"].items():
        print(f"Si J2 juega B{j}, la(s) mejor(es) respuesta(s) de J1: {', '.join('A' + str(i) for i in filas)}")
    for i, cols in analisis["mejores_respuestas_j2"].items():
        print(f"Si J1 juega A{i}, la(s) mejor(es) respuesta(s) de J2: {', '.join('B' + str(j) for j in cols)}")

    print("\n=== ESTRATEGIAS ESTRICTAMENTE DOMINADAS ===")
    print(f"J1: {analisis['dominancia_j1'] if analisis['dominancia_j1'] else 'Ninguna'}")
    print(f"J2: {analisis['dominancia_j2'] if analisis['dominancia_j2'] else 'Ninguna'}")

    print("\n=== EQUILIBRIOS DE NASH PUROS ===")
    for eq in analisis["equilibrios_nash"]:
        i, j = eq["fila"], eq["columna"]
        p1, p2 = eq["pago_j1"], eq["pago_j2"]
        print(f"-> Equilibrio [{eq['tipo']}]: J1='{analisis['estrategias_j1'][i]}' vs J2='{analisis['estrategias_j2'][j]}' con pagos ({p1}, {p2})")

    sel = analisis["equilibrio_pipeline"]
    print("\n=== EQUILIBRIO SELECCIONADO PARA EL PIPELINE ===")
    print("Criterio: Mayor pago para nuestra empresa (J1)")
    print(f"Seleccionado: J1='{analisis['estrategias_j1'][sel['fila']]}' vs J2='{analisis['estrategias_j2'][sel['columna']]}' -> Pagos: ({sel['pago_j1']}, {sel['pago_j2']}) [{sel['tipo']}]")
