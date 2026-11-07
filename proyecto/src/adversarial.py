"""
Módulo: adversarial.py
Propósito: Implementar Minimax y Minimax con Poda Alfa-Beta para un subproblema por turnos
de asignación adversaria de recursos compartidos (reserva de 7 unidades).

Decisiones de diseño y justificación para defensa:
- Se modela un juego finito secuencial de información perfecta de 2 jugadores (MAX vs MIN).
- Estado: Tupla (recurso_restante, acumulado_max, acumulado_min).
- Acciones legales: Tomar 1, 2 o 3 unidades (sin exceder el remanente).
- Función de utilidad:
    U = (acumulado_max - acumulado_min) * 10 + bonificación_última_unidad (+5 si MAX, -5 si MIN).
- Evaluación aproximada heurística para límite de profundidad:
    Eval(s) = (acumulado_max - acumulado_min) * 10 + ventaja_de_turno (+2 si MAX, -2 si MIN).
- Poda Alfa-Beta: Descarta ramas que no pueden alterar el resultado final manteniendo cotas [alfa, beta].
- Orden de jugadas: Se evalúa cómo el orden heurístico [3,2,1], inverso [1,2,3] y aleatorio afecta la cantidad de podas.
- Definición de "poda" en los contadores: se cuenta una poda solo cuando el corte (beta <= alfa)
  deja al menos una jugada hermana SIN explorar. Si el corte ocurre en la última jugada del nodo,
  no se ahorra ningún nodo y por eso no se cuenta. Así, "podas = 0" siempre implica 0% de reducción.
"""

import csv
import os
import random
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class EstadoJuegoRecursos:
    """
    Representa el estado en el juego de asignación de recursos.
    """
    def __init__(
        self,
        recurso_restante: int = 7,
        acumulado_max: int = 0,
        acumulado_min: int = 0,
    ):
        self.recurso_restante = recurso_restante
        self.acumulado_max = acumulado_max
        self.acumulado_min = acumulado_min

    def es_terminal(self) -> bool:
        """El juego termina cuando se agotan las unidades del recurso (recurso_restante <= 0)."""
        return self.recurso_restante <= 0

    def acciones_legales(self, orden: str = "descendente", semilla: int = 42) -> list[int]:
        """
        Retorna las acciones posibles (tomar 1 a min(3, restante) unidades).

        Permite evaluar el impacto del ordenamiento de jugadas en la poda Alfa-Beta:
          - 'descendente': [3, 2, 1] (orden heurístico voraz, busca podas tempranas)
          - 'ascendente':  [1, 2, 3] (orden inverso, peor caso para poda)
          - 'aleatorio':   permutación pseudoaleatoria reproducible con semilla
        """
        if self.es_terminal():
            return []
        max_toma = min(3, self.recurso_restante)
        acciones = list(range(1, max_toma + 1))

        if orden == "descendente":
            return list(reversed(acciones))
        elif orden == "ascendente":
            return acciones
        elif orden == "aleatorio":
            rng = random.Random(semilla + self.recurso_restante)
            copia = list(acciones)
            rng.shuffle(copia)
            return copia
        return list(reversed(acciones))

    def aplicar_accion(self, accion: int, es_max: bool) -> "EstadoJuegoRecursos":
        """Genera el nuevo estado tras tomar 'accion' unidades."""
        nuevo_restante = self.recurso_restante - accion
        if es_max:
            return EstadoJuegoRecursos(
                recurso_restante=nuevo_restante,
                acumulado_max=self.acumulado_max + accion,
                acumulado_min=self.acumulado_min,
            )
        else:
            return EstadoJuegoRecursos(
                recurso_restante=nuevo_restante,
                acumulado_max=self.acumulado_max,
                acumulado_min=self.acumulado_min + accion,
            )

    def utilidad_terminal(self, ultimo_en_mover_es_max: bool) -> float:
        """
        Calcula la recompensa terminal en estados hojas.
        Premia la diferencia neta de recursos y otorga un bono (+5 / -5) al jugador que tomó la última unidad.
        """
        diferencia = (self.acumulado_max - self.acumulado_min) * 10.0
        bono = 5.0 if ultimo_en_mover_es_max else -5.0
        return diferencia + bono

    def evaluar_aproximada(self, es_turno_max: bool) -> float:
        """
        Función de evaluación estática cuando la búsqueda se corta por límite de profundidad.
        Estima la ventaja esperada sumando la diferencia actual y un bono táctico por posesión de turno.
        """
        diferencia = (self.acumulado_max - self.acumulado_min) * 10.0
        bono_turno = 2.0 if es_turno_max else -2.0
        return diferencia + bono_turno


class ContadorRendimiento:
    """
    Contabiliza los nodos visitados y las podas realizadas durante la búsqueda.

    - nodos_visitados: cada llamada a la función de búsqueda cuenta como un nodo.
    - podas_realizadas: cortes (beta <= alfa) que dejaron al menos una jugada hermana
      sin explorar (poda efectiva). Un corte en la última jugada no ahorra nada y no se cuenta.
    """
    def __init__(self):
        self.nodos_visitados = 0
        self.podas_realizadas = 0

    def reiniciar(self):
        self.nodos_visitados = 0
        self.podas_realizadas = 0


def minimax_puro(
    estado: EstadoJuegoRecursos,
    profundidad: int,
    es_max: bool,
    limite_profundidad: int | None = None,
    contador: ContadorRendimiento | None = None,
    orden_jugadas: str = "descendente",
    semilla: int = 42,
) -> tuple[float, int | None]:
    """
    QUÉ HACE:
        Calcula el valor óptimo del juego explorando exhaustivamente el árbol de decisiones
        asumiendo que MAX busca maximizar la utilidad y MIN busca minimizarla.
    POR QUÉ ASÍ:
        Sirve como referencia teórica fundamental para comprobar que la poda Alfa-Beta no altera
        el valor minimax del juego bajo ninguna circunstancia.
    COMPLEJIDAD EN TIEMPO:
        O(b^d), donde b es el factor de ramificación (b <= 3) y d es la profundidad del árbol (d <= 7).
        En el árbol completo con 7 unidades visita exactamente 96 nodos (sin importar el orden de jugadas).
    """
    if contador:
        contador.nodos_visitados += 1

    # Casos base: estado terminal o límite de profundidad
    if estado.es_terminal():
        return estado.utilidad_terminal(ultimo_en_mover_es_max=not es_max), None

    if limite_profundidad is not None and profundidad >= limite_profundidad:
        return estado.evaluar_aproximada(es_turno_max=es_max), None

    acciones = estado.acciones_legales(orden=orden_jugadas, semilla=semilla)

    if es_max:
        mejor_valor = float("-inf")
        mejor_accion = None
        for accion in acciones:
            sucesor = estado.aplicar_accion(accion, es_max=True)
            val, _ = minimax_puro(sucesor, profundidad + 1, False, limite_profundidad, contador, orden_jugadas, semilla)
            if val > mejor_valor:
                mejor_valor = val
                mejor_accion = accion
        return mejor_valor, mejor_accion
    else:
        mejor_valor = float("inf")
        mejor_accion = None
        for accion in acciones:
            sucesor = estado.aplicar_accion(accion, es_max=False)
            val, _ = minimax_puro(sucesor, profundidad + 1, True, limite_profundidad, contador, orden_jugadas, semilla)
            if val < mejor_valor:
                mejor_valor = val
                mejor_accion = accion
        return mejor_valor, mejor_accion


def minimax_alfa_beta(
    estado: EstadoJuegoRecursos,
    profundidad: int,
    alfa: float,
    beta: float,
    es_max: bool,
    limite_profundidad: int | None = None,
    contador: ContadorRendimiento | None = None,
    orden_jugadas: str = "descendente",
    semilla: int = 42,
) -> tuple[float, int | None]:
    """
    QUÉ HACE:
        Calcula el valor óptimo del juego idéntico a Minimax, pero descartando (podando)
        ramas enteras del árbol cuando se comprueba que beta <= alfa.
    POR QUÉ ASÍ:
        Alfa representa la mejor opción asegurada hasta el momento para MAX a lo largo del camino,
        y Beta la mejor para MIN. Si en un nodo MIN se encuentra un valor <= alfa, MAX nunca permitirá
        que el juego tome esa rama, haciendo superfluo explorar los hijos restantes.
    CONTEO DE PODAS:
        Solo se cuenta como poda el corte que deja al menos una jugada hermana sin explorar.
    COMPLEJIDAD EN TIEMPO:
        Mejor caso: O(b^(d/2)) con ordenamiento óptimo de jugadas.
        Peor caso: O(b^d) si las mejores jugadas se exploran al final.
        Espacio: O(d) proporcional a la profundidad máxima de recursión.
    """
    if contador:
        contador.nodos_visitados += 1

    # Casos base
    if estado.es_terminal():
        return estado.utilidad_terminal(ultimo_en_mover_es_max=not es_max), None

    if limite_profundidad is not None and profundidad >= limite_profundidad:
        return estado.evaluar_aproximada(es_turno_max=es_max), None

    acciones = estado.acciones_legales(orden=orden_jugadas, semilla=semilla)

    if es_max:
        mejor_valor = float("-inf")
        mejor_accion = None
        for indice, accion in enumerate(acciones):
            sucesor = estado.aplicar_accion(accion, es_max=True)
            val, _ = minimax_alfa_beta(sucesor, profundidad + 1, alfa, beta, False, limite_profundidad, contador, orden_jugadas, semilla)
            if val > mejor_valor:
                mejor_valor = val
                mejor_accion = accion
            alfa = max(alfa, mejor_valor)
            if beta <= alfa:
                # Poda beta: solo cuenta si quedaban jugadas hermanas sin explorar
                if contador and indice < len(acciones) - 1:
                    contador.podas_realizadas += 1
                break
        return mejor_valor, mejor_accion
    else:
        mejor_valor = float("inf")
        mejor_accion = None
        for indice, accion in enumerate(acciones):
            sucesor = estado.aplicar_accion(accion, es_max=False)
            val, _ = minimax_alfa_beta(sucesor, profundidad + 1, alfa, beta, True, limite_profundidad, contador, orden_jugadas, semilla)
            if val < mejor_valor:
                mejor_valor = val
                mejor_accion = accion
            beta = min(beta, mejor_valor)
            if beta <= alfa:
                # Poda alfa: solo cuenta si quedaban jugadas hermanas sin explorar
                if contador and indice < len(acciones) - 1:
                    contador.podas_realizadas += 1
                break
        return mejor_valor, mejor_accion


def comparar_minimax_y_alfa_beta(
    recurso_inicial: int = 7,
    limites_profundidad: list[int | None] | None = None,
    ruta_salida_csv: str | None = None,
    semilla: int = 42,
) -> list[dict[str, Any]]:
    """
    Ejecuta experimentos comparativos para validar:
    1. Que el valor minimax devuelto sea estrictamente idéntico entre Minimax y Alfa-Beta.
    2. El impacto de tres órdenes de jugadas ([3,2,1], [1,2,3] y aleatorio con semilla) en la poda.
    """
    if limites_profundidad is None:
        limites_profundidad = [2, 3, 4, None]

    ordenes_a_probar = ["descendente", "ascendente", "aleatorio"]
    resultados = []

    for lim in limites_profundidad:
        for orden in ordenes_a_probar:
            estado_inicial = EstadoJuegoRecursos(recurso_restante=recurso_inicial)

            # Minimax Puro
            cont_mm = ContadorRendimiento()
            t0 = time.perf_counter()
            val_mm, act_mm = minimax_puro(
                estado_inicial, profundidad=0, es_max=True, limite_profundidad=lim,
                contador=cont_mm, orden_jugadas=orden, semilla=semilla
            )
            t_mm = time.perf_counter() - t0

            # Minimax Alfa-Beta
            cont_ab = ContadorRendimiento()
            t1 = time.perf_counter()
            val_ab, act_ab = minimax_alfa_beta(
                estado_inicial, profundidad=0, alfa=float("-inf"), beta=float("inf"),
                es_max=True, limite_profundidad=lim, contador=cont_ab,
                orden_jugadas=orden, semilla=semilla
            )
            t_ab = time.perf_counter() - t1

            # Verificación de igualdad matemática
            valores_iguales = (abs(val_mm - val_ab) < 1e-9)
            reduccion_nodos = (
                (cont_mm.nodos_visitados - cont_ab.nodos_visitados) / cont_mm.nodos_visitados * 100.0
                if cont_mm.nodos_visitados > 0 else 0.0
            )

            etiqueta_prof = f"Profundidad {lim}" if lim is not None else "Árbol Completo"
            etiqueta_orden = "[3,2,1]" if orden == "descendente" else ("[1,2,3]" if orden == "ascendente" else "Aleatorio")

            fila = {
                "configuracion": etiqueta_prof,
                "orden_jugadas": etiqueta_orden,
                "semilla": semilla,
                "recurso_inicial": recurso_inicial,
                "limite_profundidad": str(lim) if lim is not None else "Ninguno",
                "valor_minimax": val_mm,
                "valor_alfa_beta": val_ab,
                "son_iguales": valores_iguales,
                "accion_optima_max": act_ab,
                "nodos_minimax": cont_mm.nodos_visitados,
                "nodos_alfa_beta": cont_ab.nodos_visitados,
                "podas_alfa_beta": cont_ab.podas_realizadas,
                "reduccion_nodos_pct": round(reduccion_nodos, 2),
                "tiempo_minimax_ms": round(t_mm * 1000, 4),
                "tiempo_alfa_beta_ms": round(t_ab * 1000, 4),
            }
            resultados.append(fila)

    if ruta_salida_csv:
        os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
        # utf-8-sig agrega la marca BOM para que Excel muestre bien los acentos
        with open(ruta_salida_csv, "w", newline="", encoding="utf-8-sig") as f:
            campos = [
                "configuracion", "orden_jugadas", "semilla", "recurso_inicial", "limite_profundidad",
                "valor_minimax", "valor_alfa_beta", "son_iguales", "accion_optima_max",
                "nodos_minimax", "nodos_alfa_beta", "podas_alfa_beta",
                "reduccion_nodos_pct", "tiempo_minimax_ms", "tiempo_alfa_beta_ms"
            ]
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados)

    return resultados


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    csv_salida = os.path.join(base_dir, "..", "results", "minimax_alphabeta_comparison.csv")
    comparaciones = comparar_minimax_y_alfa_beta(recurso_inicial=7, ruta_salida_csv=csv_salida, semilla=42)

    print("=== COMPARACIÓN MINIMAX vs MINIMAX CON PODA ALFA-BETA ===")
    for c in comparaciones:
        print(f"[{c['configuracion']} | Orden {c['orden_jugadas']}] Val={c['valor_alfa_beta']} (Iguales: {c['son_iguales']}) | Nodos: MM={c['nodos_minimax']} vs AB={c['nodos_alfa_beta']} (Podas: {c['podas_alfa_beta']}, -{c['reduccion_nodos_pct']}%)")
