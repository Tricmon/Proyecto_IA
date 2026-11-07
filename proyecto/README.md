# Sistema Inteligente de Planificación, Optimización y Toma de Decisiones Adversarias

Proyecto individual de Inteligencia Artificial (Segundo Parcial) enfocado en la resolución de problemas logísticos complejos mediante la integración de **Búsqueda Heurística**, **Búsqueda Local**, **Metaheurísticas**, **Teoría de Juegos** y **Búsqueda Adversaria (Minimax / Alfa-Beta)**.

**Autor:** Emiliano Hernandez Garcia — Fundamentos de Inteligencia Artificial, Universidad Tecnológica de Aguascalientes.

---

## 1. Requisitos e Instalación

### Requisitos del Sistema
- **Python:** 3.11+ (desarrollado y probado en Python 3.13)
- **Sistema Operativo:** Windows / Linux / macOS
- **Librerías externas:** Únicamente `matplotlib` (para generar las visualizaciones) y `pytest` (para ejecutar las pruebas automatizadas). El resto del sistema utiliza estrictamente la biblioteca estándar de Python (`heapq`, `random`, `json`, `csv`, `time`, `statistics`). Los algoritmos están implementados a mano, sin bibliotecas que oculten su lógica (por ejemplo, sin `networkx` ni `scipy`).

### Instalación desde Cero

1. Clonar o descargar el repositorio y situarse en la carpeta raíz del proyecto:
   ```bash
   cd proyecto
   ```

2. (Opcional pero recomendado) Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En Linux / macOS:
   source venv/bin/activate
   ```

3. Instalar las dependencias permitidas:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Ejecución Rápida

Para reproducir todos los experimentos, generar los archivos CSV, construir las gráficas y ejecutar el pipeline integrado con una sola orden:

```bash
python run_all.py
```

Para correr la suite de pruebas unitarias y de límites matemáticos (18 pruebas):
```bash
python -m pytest -v
```

### Reproducibilidad y semillas

- La semilla global del experimento es **55** y se define en `config/params.json` (`semilla_global`). Se imprime al inicio de `python run_all.py`.
- El módulo de búsqueda local registra la semilla en `results/local_search_results.csv`.
- Cada ejecución de Recocido Simulado y de Algoritmo Genético registra su propia semilla en `results/metaheuristics_runs.csv`.
- El módulo de Minimax y Alfa-Beta usa por defecto la semilla **42** para el orden aleatorio de jugadas, y la registra en `results/minimax_alphabeta_comparison.csv`.
- Todos los parámetros experimentales (número de ejecuciones, presupuesto de evaluaciones, parámetros de SA y GA, límites de Minimax) están en `config/params.json`.

---

## 3. Modelo del Escenario Logístico

- **Cuadrícula y Movimientos:** Mapa matricial de $12 \times 12$. Los movimientos permitidos son estrictamente en 4 direcciones ortogonales: Arriba $(-1, 0)$, Abajo $(+1, 0)$, Izquierda $(0, -1)$ y Derecha $(0, +1)$. No se admiten diagonales.
- **Costos de Paso y Obstáculos:** Cada celda posee un costo de desplazamiento entero entre 1 y 5 según el modelo (la cuadrícula del escenario 1 usa valores de 1 a 3). Las celdas con valor `0` representan obstáculos físicos intransitables. El costo de una transición es el valor de la celda a la que se ingresa.
- **Puntos Clave:** Todo escenario cuenta con un punto de `Inicio (I)`, un punto de `Destino Final (D)` y 6 puntos de entrega intermedios (`E1` a `E6`). Opcionalmente se ubica una estación de `Recarga (R)`.
- **Restricción de Recurso:** El vehículo dispone de una capacidad limitada de combustible o batería (ej. 80.0 unidades en el escenario 1). Si el consumo de la ruta excede dicha capacidad, se aplica una penalización por unidad de exceso:
  $$\text{Costo Total} = \text{Costo Base de Rutas} + \max(0, \text{Consumo} - \text{Capacidad}) \times \text{Penalización}$$
- **Interacción Adversaria:** Un competidor logístico opera en la misma zona. Sus decisiones determinan si la estación de recarga está libre, compartida o saturada, impactando el costo de operación o la cantidad de recurso disponible.
- **Escenarios incluidos** (en `config/escenarios/`): 1 (fácil), 2 (muchos obstáculos), 3 (trampa para Greedy) y 4 (sin solución, caso límite).
- **Validación de entradas:** el sistema rechaza claves faltantes, valores fuera de rango, puntos sobre obstáculos, puntos duplicados, costos inválidos, capacidad de recurso inválida y entregas duplicadas, y detecta escenarios sin solución.

---

## 4. Estructura del Repositorio

```text
proyecto/
├── README.md                                  # Guía completa de uso y defensa
├── requirements.txt                           # matplotlib y pytest
├── run_all.py                                 # Script maestro de ejecución global
├── config/
│   ├── params.json                            # Parámetros experimentales y semillas
│   └── escenarios/
│       ├── escenario_1_facil.json             # 12x12 moderado con rutas despejadas
│       ├── escenario_2_obstaculos.json        # 12x12 denso con laberintos
│       ├── escenario_3_trampa_greedy.json     # Trampa donde Greedy falla vs A*
│       └── escenario_4_sin_solucion.json      # Destino bloqueado (caso límite)
├── src/
│   ├── scenario.py                            # Carga, validaciones y modelo de datos
│   ├── heuristic_search.py                    # A* y Greedy con una sola función
│   ├── local_search.py                        # Hill Climbing (2-swap) multi-start
│   ├── metaheuristics.py                      # Recocido Simulado, Algoritmo Genético y Fuerza Bruta
│   ├── game_theory.py                         # Matriz 3x3, mejores respuestas y Nash
│   ├── adversarial.py                         # Minimax y Poda Alfa-Beta para recursos
│   ├── pipeline.py                            # Integración de extremo a extremo
│   └── visualization.py                       # Generación de figuras PNG
├── tests/
│   ├── test_scenario.py                       # 8 pruebas de validación de entradas y casos límite
│   └── test_all_modules.py                    # 10 pruebas de propiedades teóricas
├── results/                                   # Archivos CSV generados y 5 gráficas PNG
└── report/
    └── report_notes.md                        # Notas estructuradas para el informe técnico
```

### Archivos generados en `results/`

| Archivo | Contenido |
|---|---|
| `heuristic_search_results.csv` | A* y Greedy en los 4 escenarios (costo, pasos, nodos expandidos, tiempo) |
| `local_search_results.csv` | 10 ejecuciones de Hill Climbing (costos, iteraciones, terminación) |
| `metaheuristics_runs.csv` | Cada ejecución de SA y GA con su semilla |
| `metaheuristics_comparison.csv` | Resumen de SA y GA: mejor, peor, promedio, desviación estándar, tiempo y tasa de éxito |
| `game_theory_matrix.csv` | Matriz de pagos 3x3, mejores respuestas y equilibrios de Nash |
| `minimax_alphabeta_comparison.csv` | Nodos visitados y podas de Minimax y Alfa-Beta por profundidad y orden de jugadas |
| `grafica_1` a `grafica_5` (`.png`) | Búsqueda heurística, convergencia de Hill Climbing, metaheurísticas, Alfa-Beta y mapa con la ruta final |

---

## 5. Resumen de Módulos Implementados

### Módulo A: Búsqueda Heurística (`src/heuristic_search.py`)
- Implementa una función unificada `busqueda_heuristica` que parametriza la función de evaluación:
  - **A\*:** $f(n) = g(n) + h(n)$
  - **Greedy Best-First:** $f(n) = h(n)$
- **Heurística Admisible y Consistente:**
  $$h(n) = (|r_n - r_{meta}| + |c_n - c_{meta}|) \times c_{min}$$
  Donde $c_{min}$ es el costo mínimo de cualquier celda transitable del mapa ($c_{min} \ge 1$). Como Manhattan representa la cota inferior absoluta de pasos y $c_{min}$ el menor costo por paso, $h(n) \le h^*(n)$ siempre.
- Genera la matriz de costos y rutas reales entre `[Inicio, E1..E6, Destino]`.
- Se prueba en 4 escenarios (3 con solución y 1 sin solución).

### Módulo B: Búsqueda Local (`src/local_search.py`)
- Espacio de búsqueda: Permutaciones de las 6 entregas ($6! = 720$ estados posibles).
- Operador de vecindad: Intercambio de dos posiciones (2-swap, $15$ vecinos por estado).
- Algoritmo: Hill Climbing de máxima pendiente descendente (*Steepest Descent para minimización de costo*) con control de mesetas (*plateaus*, hasta 5 movimientos laterales).
- Ejecuta 10 soluciones iniciales aleatorias para contrastar óptimos locales vs óptimo global.

### Módulo C: Metaheurísticas (`src/metaheuristics.py`)
- **Presupuesto Idéntico:** 1000 evaluaciones de la función objetivo por ejecución.
- **Recocido Simulado:** Regla de Metropolis $P = \exp(-\Delta / T)$ y enfriamiento geométrico $T \leftarrow \alpha T$ ($T_0 = 100.0$, $\alpha = 0.95$, $T_{fin} = 0.01$).
- **Algoritmo Genético:** Tamaño de población $P=20$, selección por torneo ($k=3$), Cruza de Orden (*Order Crossover - OX*, probabilidad 0.9), mutación por intercambio (probabilidad 0.2) y elitismo de 2 individuos.
- **Fuerza Bruta y Nota de Escalabilidad:** Evalúa las 720 permutaciones posibles para determinar el óptimo global matemático exacto (costo 47.0). Para 6 entregas son 720 permutaciones (manejable en milisegundos), pero para 10 entregas serían 3,628,800 y para 15 más de un billón. Por ello, la fuerza bruta actúa aquí como verificador de optimalidad, mientras que SA y GA son las herramientas escalables para instancias reales.
- 20 ejecuciones estadísticas por método para calcular mejor costo, promedio, desviación estándar y tasa de éxito.

### Módulo D: Teoría de Juegos (`src/game_theory.py`)
- Juego simultáneo $3 \times 3$ entre nuestra empresa y el competidor.
- Pagos calculados programáticamente a partir del modelo logístico de costos y fricción en la estación de recarga (no puestos a mano).
- Algoritmo celda por celda para calcular Mejores Respuestas, dominancia y Equilibrios de Nash puros, distinguiendo equilibrios estrictos y débiles.

### Módulo E: Minimax y Poda Alfa-Beta (`src/adversarial.py`)
- Juego secuencial de asignación de una reserva de 7 unidades de recarga compartida (acciones: tomar 1, 2 o 3 unidades).
- Se verifica, con experimentos y pruebas automatizadas, que Minimax puro y Alfa-Beta devuelven **exactamente el mismo valor**, y que Alfa-Beta reduce los nodos visitados mediante podas (hasta 79.17% en el árbol completo con el orden de jugadas [3,2,1]).
- Con límite de profundidad se usa una evaluación aproximada: $(\text{acumulado}_{MAX} - \text{acumulado}_{MIN}) \times 10 \pm 2$ según el turno.

### Integración (`src/pipeline.py`)
- Enlaza el mapa, la matriz de costos $A^*$, el juego estratégico y la metaheurística para emitir un plan de despacho final, demostrando el beneficio económico de modelar formalmente al competidor.

---

## 6. Resultados de Referencia (semilla global 55)

Al ejecutar `python run_all.py` se espera ver, entre otros, estos resultados (escenario 1 - fácil):

- Óptimo global por fuerza bruta: **47.0**.
- Equilibrios de Nash encontrados: **4**; el pipeline adopta el perfil (A0, B2) con pagos (73.0, 60.0).
- Plan Inteligente: costo base 47.0 + interferencia 10.0 = **57.0**.
- Plan Ingenuo: costo base 47.0 + peor caso 25.0 = **72.0**.
- Ahorro estratégico: **15.0** unidades.

Al ejecutar `python -m pytest -v` se esperan **18 pruebas aprobadas** (`18 passed`).

Los resultados detallados y su interpretación se encuentran en el informe técnico.
