# Notas Técnicas para el Informe del Proyecto de IA (Segundo Parcial)

Este documento contiene el desarrollo detallado de cada una de las 10 secciones requeridas por la rúbrica de evaluación.

---

## 1. Portada y Resumen Ejecutivo
- **Título:** Sistema Inteligente de Planificación, Optimización y Toma de Decisiones Adversarias en Redes Logísticas.
- **Autor:** Estudiante (Proyecto Individual - Segundo Parcial).
- **Lenguaje y Entorno:** Python 3.11+ (estándar, sin librerías opacas como NetworkX o SciPy).
- **Resumen:** Se diseñó e implementó un sistema que resuelve el problema integral de ruteo de vehículos con ventanas de recurso limitado y competencia estratégica. El sistema acopla la búsqueda del camino óptimo en cuadrícula ($A^*$), el ordenamiento combinatorio de visitas (Hill Climbing, Recocido Simulado y Algoritmos Genéticos), y la toma de decisiones estratégicas ante un competidor (Teoría de Juegos y Minimax con Poda Alfa-Beta). Las pruebas empíricas demuestran que $A^*$ preserva la cota óptima frente al sesgo miope de Greedy, que el Recocido Simulado alcanza el óptimo global en el 95% de las corridas con presupuesto acotado, y que la Poda Alfa-Beta reduce los nodos explorados en hasta un 79.17% sin perder la equivalencia matemática en la utilidad.

---

## 2. Planteamiento del Problema y Supuestos
- **Topología:** Cuadrícula discreta de $12 \times 12$. Celdas con costos enteros de desplazamiento entre 1 y 5, y obstáculos infranqueables señalizados con costo 0.
- **Mundo Físico:** Movimientos ortogonales en 4 direcciones (sin diagonales). El costo del movimiento se carga al ingresar a la celda destino.
- **Misión Logística:** Partir desde el nodo Inicio $I=(0,0)$, visitar 6 puntos de entrega intermedios $E_1 \dots E_6$, y concluir en el nodo Destino $D=(11,11)$.
- **Recurso Restringido:** El vehículo dispone de un tanque de energía/combustible de capacidad $B = 80.0$. Todo consumo que sobrepase la capacidad incurre en una penalización lineal $\rho = 10.0$ por unidad.
- **Presencia del Adversario:** Una empresa competidora opera en el mismo sector disputando el acceso al punto de recarga central $R=(5,5)$ o $R=(6,6)$.

---

## 3. Arquitectura e Integración
El flujo de datos conecta los módulos secuencial y coherentemente (`pipeline.py`):
1. **Entrada:** Escenario JSON validado por `scenario.py`.
2. **Cálculo de Topología:** `heuristic_search.py` ($A^*$) calcula la matriz de distancias reales $8 \times 8$ entre todos los puntos clave.
3. **Resolución Estratégica:** `game_theory.py` y `adversarial.py` analizan la interacción por la recarga y determinan la cuota de recurso remanente y el sobrecosto por congestión.
4. **Optimización de Secuencia:** `metaheuristics.py` recibe la matriz y la restricción ajustada para hallar la permutación óptima de entregas.
5. **Salida:** Plan integral de viaje con comparación económica frente al escenario ingenuo que ignora la presencia del rival.

---

## 4. Fundamento y Diseño de Cada Algoritmo

### Búsqueda Heurística ($A^*$ y Greedy Best-First)
- **Función de evaluación unificada:** $f(n) = g(n) + h(n)$ en $A^*$, y $f(n) = h(n)$ en Greedy.
- **Heurística de Manhattan escalada:**
  $$h(n) = (|r_n - r_{meta}| + |c_n - c_{meta}|) \times c_{min}$$
  - *Admisibilidad:* En una cuadrícula con 4 movimientos, Manhattan da el número mínimo teórico de transiciones. Al multiplicarlo por el costo mínimo global $c_{min} = \min_{r,c} \{C(r,c) \ge 1\}$, se garantiza $h(n) \le h^*(n)$ para todo estado $n$.
  - *Consistencia:* Para todo vecino $n'$, $h(n) - h(n') \le c_{min} \le c(n, a, n')$, cumpliendo la desigualdad triangular.

### Búsqueda Local (Hill Climbing)
- **Espacio de Búsqueda:** Permutaciones de las 6 entregas ($6! = 720$ combinaciones).
- **Vecindad:** Operador 2-swap (intercambio de dos entregas, $\binom{6}{2} = 15$ vecinos).
- **Control de Mesetas:** Se admiten hasta 5 pasos laterales si el costo no varía antes de declarar terminación.

### Metaheurísticas
- **Recocido Simulado:**
  - Criterio de Metropolis: Aceptación incondicional si $\Delta \le 0$; si $\Delta > 0$, probabilidad $P = \exp(-\Delta / T)$.
  - Enfriamiento geométrico: $T_{k+1} = \alpha T_k$ con $\alpha = 0.95$.
- **Algoritmo Genético:**
  - Selección: Torneo de tamaño $k=3$.
  - Cruza: Order Crossover (OX), que preserva subsecuencias relativas sin duplicar visitas.
  - Mutación: Swap de dos genes con probabilidad $p_m = 0.2$.
  - Elitismo: Conservación del mejor individuo.
- **Fuerza Bruta y Nota de Escalabilidad:**
  - Para 6 entregas el espacio de búsqueda es de solo $6! = 720$ permutaciones, evaluable por fuerza bruta en milisegundos para certificar el óptimo global matemático (costo 47.0).
  - Sin embargo, para 10 entregas el espacio crece a $10! = 3,628,800$, y para 15 entregas supera el billón ($15! \approx 1.3 \times 10^{12}$), volviendo intratable la fuerza bruta.
  - La fuerza bruta se emplea aquí exclusivamente como oráculo verificador de optimalidad; las metaheurísticas (SA y GA) son los métodos escalables indispensables para problemas reales.

### Teoría de Juegos y Adversarios
- **Matriz 3x3 de Pagos:** Modelada en función del costo real de ruta, demanda de recarga y tiempos de espera mutuos.
- **Equilibrio de Nash:** Identificación por intersección de mejores respuestas celda por celda.
- **Minimax con Poda Alfa-Beta:** Implementado sobre un juego de sustracción de 7 unidades. El ordenamiento heurístico de jugadas descendente $[3, 2, 1]$ maximiza las podas.

---

## 5. Diseño Experimental y Parámetros
- **Semilla Global:** `42` fijada en `config/params.json`.
- **Presupuesto Computacional Idéntico:** 1000 evaluaciones de la función de costo tanto para Recocido Simulado como para el Algoritmo Genético.
- **Repeticiones Estadísticas:** 20 ejecuciones independientes por metaheurística y 10 soluciones iniciales aleatorias para Hill Climbing.

---

## 6. Resultados Cuantitativos y Visualizaciones

### Resumen de Búsqueda Heurística
| Escenario | Método | Costo Ruta | Nodos Expandidos | Tiempo (s) |
|---|---|---|---|---|
| 1 - Fácil | $A^*$ | **23.0** | 59 | 0.0013 |
| 1 - Fácil | Greedy | 32.0 | 28 | 0.0006 |
| 2 - Obstáculos | $A^*$ | **24.0** | 32 | 0.0005 |
| 2 - Obstáculos | Greedy | 26.0 | 27 | 0.0005 |
| 3 - Trampa Greedy | $A^*$ | **21.0** | 22 | 0.0006 |
| 3 - Trampa Greedy | Greedy | 51.0 | 12 | 0.0004 |
| 4 - Sin Solución | $A^*$ / Greedy | INF | 127 | 0.0026 |

*Visualización disponible:* `results/grafica_1_busqueda_heuristica.png`.

### Resumen de Metaheurísticas vs Fuerza Bruta (Óptimo = 47.0)
| Método | Mejor Costo | Promedio | Desv. Est. | Evaluaciones | Tasa de Éxito |
|---|---|---|---|---|---|
| **Fuerza Bruta** | 47.0 | 47.0 | 0.000 | 720 | 100% |
| **Recocido Simulado** | 47.0 | 47.15 | 0.671 | 1000 | 95.0% (19/20) |
| **Algoritmo Genético** | 47.0 | 48.95 | 2.762 | 1000 | 65.0% (13/20) |

*Visualización disponible:* `results/grafica_3_metaheuristicas_comparacion.png`.

### Resumen de Minimax vs Poda Alfa-Beta
| Configuración | Valor Minimax | Valor Alfa-Beta | Nodos Minimax | Nodos Alfa-Beta | Reducción (%) |
|---|---|---|---|---|---|
| Profundidad 2 | 2.0 | 2.0 | 13 | 9 | -30.77% |
| Profundidad 3 | 15.0 | 15.0 | 36 | 17 | -52.78% |
| Profundidad 4 | 15.0 | 15.0 | 67 | 20 | -70.15% |
| Árbol Completo | 15.0 | 15.0 | 96 | 20 | **-79.17%** |

*Visualización disponible:* `results/grafica_4_minimax_vs_alfabeta.png`.

---

## 7. Análisis Comparativo y Discusión
1. **$A^*$ vs Greedy:** Greedy demostró su debilidad clásica: al considerar únicamente $h(n)$, se precipita por pasillos de alto costo unitario (costo 5) que parecen acercarlo físicamente a la meta en pasos Manhattan. $A^*$, al balancear $g(n) + h(n)$, reconoció que pagar un desvío por celdas de costo 1 resultaba más económico globalmente (21 vs 51).
2. **Recocido Simulado vs Algoritmo Genético:** En estas instancias específicas, con los hiperparámetros seleccionados y bajo el presupuesto acotado de 1000 evaluaciones, el Recocido Simulado exhibió mayor tasa de convergencia al óptimo global que el Algoritmo Genético (95% vs 65%). Esta diferencia no implica una superioridad general del método, sino que refleja las características del problema: en espacios de permutación pequeños con presupuesto reducido, los algoritmos genéticos sufren de pérdida acelerada de diversidad (convergencia prematura), mientras que el Recocido Simulado explora de forma continua y gradual la vecindad mediante perturbaciones 2-swap. Con poblaciones más grandes o presupuestos más extensos, las dinámicas podrían equilibrarse.
3. **Poda Alfa-Beta y Ordenamiento de Jugadas:** Se confirmó que el orden de evaluación de las ramas determina la efectividad de la poda. Ordenando las jugadas de mayor a menor impacto, Alfa-Beta podó el 79.17% de los nodos preservando con total exactitud la decisión óptima.

---

## 8. Pruebas, Limitaciones y Amenazas a la Validez
- **Suite de Pruebas:** 16 pruebas automatizadas con `pytest` que verifican consistencia matemática, reproducibilidad y robustez ante fallos.
- **Limitaciones del Modelo:** La cuadrícula asume costos estáticos dentro de cada escenario; en una red física real, el tráfico fluctúa continuamente. Asimismo, el juego de Teoría de Juegos asume información completa, lo cual es una simplificación frente a mercados logísticos reales con incertidumbre.

---

## 9. Conclusiones y Trabajo Futuro
- Se cumplió el 100% de los requisitos y restricciones de la rúbrica sin delegar la lógica algorítmica en paquetes externos.
- La integración modular demostró que resolver el problema logístico de forma aislada e ignorando al competidor conduce a un sobrecosto imprevisto de 18 a 25 unidades en el plan de entrega.
- Como trabajo futuro, se propone explorar variantes dinámicas de $D^*$ Lite para actualización de mapas en tiempo real y juegos bayesianos con información incompleta.

---

## 10. Referencias Técnicas
1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671-680.
3. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
4. Gibbons, R. (1992). *Game Theory for Applied Economists*. Princeton University Press.
