"""
Módulo: scenario.py
Propósito: Cargar, validar y representar el escenario logístico en cuadrícula.

Decisiones de diseño:
- Se usa una clase simple 'Escenario' con atributos directos para facilitar la comprensión.
- Se define una excepción propia 'EscenarioInvalidoError' para reportar errores claros de configuración.
- El costo de movimiento hacia una celda corresponde al valor de dicha celda (1 a 5).
- Un valor de 0 en la cuadrícula representa un obstáculo intransitable.
- Los movimientos permitidos son en 4 direcciones (arriba, abajo, izquierda, derecha); sin diagonales.
"""

import json
import os


class EscenarioInvalidoError(Exception):
    """Excepción lanzada cuando el archivo de escenario no cumple con la especificación requerida."""
    pass


class Escenario:
    """
    Representa el mapa logístico y todas sus reglas.
    
    Atributos:
        nombre (str): Nombre descriptivo del escenario.
        descripcion (str): Explicación breve del propósito del mapa.
        filas (int): Número de filas en la cuadrícula (ej. 12).
        columnas (int): Número de columnas en la cuadrícula (ej. 12).
        inicio (tuple[int, int]): Coordenada (fila, columna) del punto de partida.
        destino (tuple[int, int]): Coordenada (fila, columna) del destino final.
        entregas (list[tuple[int, int]]): Lista de coordenadas a visitar antes del destino.
        cuadricula (list[list[int]]): Matriz donde 0 es obstáculo y 1-5 es el costo de paso.
        capacidad_recurso (float): Límite disponible de combustible/batería.
        penalizacion_exceso_recurso (float): Costo añadido por cada unidad que exceda el límite.
        punto_recarga (tuple[int, int] | None): Coordenada opcional de la estación de recarga.
        costo_base_recarga (float): Costo asociado al uso de la estación de recarga.
    """

    def __init__(
        self,
        nombre: str,
        descripcion: str,
        filas: int,
        columnas: int,
        inicio: tuple[int, int],
        destino: tuple[int, int],
        entregas: list[tuple[int, int]],
        cuadricula: list[list[int]],
        capacidad_recurso: float,
        penalizacion_exceso_recurso: float = 10.0,
        punto_recarga: tuple[int, int] | None = None,
        costo_base_recarga: float = 15.0,
    ):
        self.nombre = nombre
        self.descripcion = descripcion
        self.filas = filas
        self.columnas = columnas
        self.inicio = inicio
        self.destino = destino
        self.entregas = entregas
        self.cuadricula = cuadricula
        self.capacidad_recurso = capacidad_recurso
        self.penalizacion_exceso_recurso = penalizacion_exceso_recurso
        self.punto_recarga = punto_recarga
        self.costo_base_recarga = costo_base_recarga

    def es_valida(self, r: int, c: int) -> bool:
        """Verifica si la celda (r, c) está dentro de los límites de la cuadrícula."""
        return 0 <= r < self.filas and 0 <= c < self.columnas

    def es_obstaculo(self, r: int, c: int) -> bool:
        """Determina si la celda es un obstáculo intransitable (valor 0)."""
        if not self.es_valida(r, c):
            return True
        return self.cuadricula[r][c] == 0

    def costo_celda(self, r: int, c: int) -> int:
        """
        Retorna el costo de ingresar a la celda (r, c).
        Lanza error si es una celda inválida o un obstáculo.
        """
        if not self.es_valida(r, c):
            raise IndexError(f"La coordenada ({r}, {c}) está fuera del mapa ({self.filas}x{self.columnas}).")
        if self.es_obstaculo(r, c):
            raise ValueError(f"La celda ({r}, {c}) es un obstáculo intransitable.")
        return self.cuadricula[r][c]

    def obtener_vecinos(self, r: int, c: int) -> list[tuple[int, int]]:
        """
        Retorna los vecinos transitables en 4 direcciones ortogonales:
        Arriba (-1, 0), Abajo (+1, 0), Izquierda (0, -1), Derecha (0, +1).
        No incluye diagonales ni celdas con obstáculos.
        """
        movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        vecinos_validos = []
        for dr, dc in movimientos:
            nr, nc = r + dr, c + dc
            if self.es_valida(nr, nc) and not self.es_obstaculo(nr, nc):
                vecinos_validos.append((nr, nc))
        return vecinos_validos

    def costo_minimo_global(self) -> int:
        """
        Encuentra el costo mínimo entre todas las celdas transitables del mapa.
        Fundamental para construir una heurística admisible en A*:
        h(n) = Manhattan(n, meta) * costo_minimo_global.
        """
        costos = [
            self.cuadricula[r][c]
            for r in range(self.filas)
            for c in range(self.columnas)
            if self.cuadricula[r][c] > 0
        ]
        if not costos:
            raise EscenarioInvalidoError("La cuadrícula no contiene ninguna celda transitable.")
        return min(costos)

    def mostrar(self, ruta: list[tuple[int, int]] | None = None) -> str:
        """
        Genera una vista en texto ASCII del tablero para depurar y presentar.
        Muestra:
          'I ' para inicio
          'D ' para destino
          'E1'..'E6' para puntos de entrega
          'R ' para punto de recarga
          '##' para obstáculos
          '* ' para celdas que forman parte de una ruta recibida
          '.c' donde c es el costo (1 a 5) en celdas normales
        """
        mapa_str = []
        set_ruta = set(ruta) if ruta else set()

        # Diccionario de entregas para saber su número E1, E2, ...
        entregas_dict = {coord: f"E{i+1}" for i, coord in enumerate(self.entregas)}

        # Encabezado de columnas
        encabezado = "    " + " ".join(f"{c:2d}" for c in range(self.columnas))
        mapa_str.append(encabezado)
        mapa_str.append("    " + "---" * self.columnas)

        for r in range(self.filas):
            fila_chars = []
            for c in range(self.columnas):
                pos = (r, c)
                if pos == self.inicio:
                    fila_chars.append("I ")
                elif pos == self.destino:
                    fila_chars.append("D ")
                elif pos in entregas_dict:
                    fila_chars.append(f"{entregas_dict[pos]:2s}")
                elif self.punto_recarga and pos == self.punto_recarga:
                    fila_chars.append("R ")
                elif self.es_obstaculo(r, c):
                    fila_chars.append("##")
                elif pos in set_ruta:
                    fila_chars.append("* ")
                else:
                    fila_chars.append(f".{self.cuadricula[r][c]}")
            mapa_str.append(f"{r:2d} | " + " ".join(fila_chars))

        return "\n".join(mapa_str)


def _validar_coordenada(coord, nombre_campo: str, filas: int, columnas: int) -> tuple[int, int]:
    """Valida que una coordenada sea una lista o tupla de dos enteros dentro del mapa."""
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        raise EscenarioInvalidoError(
            f"El campo '{nombre_campo}' debe ser una lista o tupla de 2 enteros [fila, columna]. Recibido: {coord}"
        )
    r, c = coord
    if not (isinstance(r, int) and isinstance(c, int)) or isinstance(r, bool) or isinstance(c, bool):
        raise EscenarioInvalidoError(
            f"Las coordenadas de '{nombre_campo}' deben ser números enteros. Recibido: [{type(r).__name__}, {type(c).__name__}]"
        )
    if not (0 <= r < filas and 0 <= c < columnas):
        raise EscenarioInvalidoError(
            f"La coordenada '{nombre_campo}' ({r}, {c}) está fuera de los límites del mapa (0 a {filas-1}, 0 a {columnas-1})."
        )
    return (r, c)


def cargar_escenario(ruta_archivo: str) -> Escenario:
    """
    Carga un escenario desde un archivo JSON y realiza validaciones exhaustivas.
    
    Validaciones implementadas:
    1. Existencia del archivo y sintaxis JSON válida.
    2. Presencia de todas las claves requeridas.
    3. Dimensiones coherentes de la cuadrícula (filas y columnas positivas y concordantes).
    4. Valores de celda válidos (enteros entre 0 y 5).
    5. Coordenadas de inicio, destino y entregas dentro de rango.
    6. Inicio, destino o entregas NO pueden estar sobre obstáculos (valor 0).
    7. No pueden existir coordenadas duplicadas entre puntos clave.
    8. Debe haber al menos 3 entregas (la rúbrica pide 6 en el problema completo).
    9. Capacidad de recurso positiva.
    
    Retorna:
        Instancia validada de 'Escenario'.
    """
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo de escenario: '{ruta_archivo}'")

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except json.JSONDecodeError as e:
        raise EscenarioInvalidoError(f"Error de sintaxis JSON en '{ruta_archivo}': {str(e)}")

    if not isinstance(datos, dict):
        raise EscenarioInvalidoError("El archivo de configuración debe contener un objeto JSON raíz.")

    # 1. Validar claves requeridas
    claves_requeridas = [
        "nombre", "filas", "columnas", "inicio", "destino",
        "entregas", "cuadricula", "capacidad_recurso"
    ]
    for clave in claves_requeridas:
        if clave not in datos:
            raise EscenarioInvalidoError(f"Falta la clave requerida en el JSON: '{clave}'")

    nombre = str(datos["nombre"])
    descripcion = str(datos.get("descripcion", ""))

    # 2. Validar dimensiones
    filas = datos["filas"]
    columnas = datos["columnas"]
    if not isinstance(filas, int) or not isinstance(columnas, int) or filas <= 0 or columnas <= 0:
        raise EscenarioInvalidoError(f"Las dimensiones deben ser enteros positivos. Recibido: {filas}x{columnas}")

    cuadricula = datos["cuadricula"]
    if not isinstance(cuadricula, list) or len(cuadricula) != filas:
        raise EscenarioInvalidoError(
            f"La cuadrícula debe tener exactamente {filas} filas. Se encontraron {len(cuadricula) if isinstance(cuadricula, list) else 'tipo invalido'}."
        )

    for i, fila in enumerate(cuadricula):
        if not isinstance(fila, list) or len(fila) != columnas:
            raise EscenarioInvalidoError(
                f"La fila {i} de la cuadrícula debe tener {columnas} columnas. Longitud encontrada: {len(fila) if isinstance(fila, list) else 'tipo invalido'}."
            )
        for j, val in enumerate(fila):
            if not isinstance(val, int) or isinstance(val, bool):
                raise EscenarioInvalidoError(
                    f"El valor en la celda ({i}, {j}) debe ser un número entero. Recibido: {val} (tipo {type(val).__name__})."
                )
            if val < 0 or val > 5:
                raise EscenarioInvalidoError(
                    f"El valor de la celda ({i}, {j}) es {val}. Debe ser 0 (obstáculo) o entre 1 y 5 (costo de paso)."
                )

    # 3. Validar coordenadas de puntos especiales
    inicio = _validar_coordenada(datos["inicio"], "inicio", filas, columnas)
    destino = _validar_coordenada(datos["destino"], "destino", filas, columnas)

    # Validar entregas
    entregas_raw = datos["entregas"]
    if not isinstance(entregas_raw, list) or len(entregas_raw) < 3:
        raise EscenarioInvalidoError(
            f"El campo 'entregas' debe ser una lista con al menos 3 coordenadas. Se recibieron {len(entregas_raw) if isinstance(entregas_raw, list) else 0}."
        )

    entregas = []
    for idx, e in enumerate(entregas_raw):
        coord_entrega = _validar_coordenada(e, f"entrega[{idx}]", filas, columnas)
        entregas.append(coord_entrega)

    # Punto de recarga opcional
    punto_recarga = None
    if "punto_recarga" in datos and datos["punto_recarga"] is not None:
        punto_recarga = _validar_coordenada(datos["punto_recarga"], "punto_recarga", filas, columnas)

    # 4. Validar que puntos especiales no caigan sobre obstáculos
    if cuadricula[inicio[0]][inicio[1]] == 0:
        raise EscenarioInvalidoError(f"El punto de inicio {inicio} está ubicado sobre un obstáculo.")

    if cuadricula[destino[0]][destino[1]] == 0:
        raise EscenarioInvalidoError(f"El punto de destino {destino} está ubicado sobre un obstáculo.")

    for idx, (er, ec) in enumerate(entregas):
        if cuadricula[er][ec] == 0:
            raise EscenarioInvalidoError(f"La entrega {idx+1} en ({er}, {ec}) está ubicada sobre un obstáculo.")

    if punto_recarga and cuadricula[punto_recarga[0]][punto_recarga[1]] == 0:
        raise EscenarioInvalidoError(f"El punto de recarga {punto_recarga} está ubicado sobre un obstáculo.")

    # 5. Validar que no haya puntos duplicados
    puntos_clave = [("inicio", inicio), ("destino", destino)] + [
        (f"entrega_{i+1}", coord) for i, coord in enumerate(entregas)
    ]
    coords_vistas = {}
    for nombre_pto, coord in puntos_clave:
        if coord in coords_vistas:
            raise EscenarioInvalidoError(
                f"Conflicto de posición: '{nombre_pto}' y '{coords_vistas[coord]}' comparten la misma celda {coord}."
            )
        coords_vistas[coord] = nombre_pto

    # 6. Validar recurso
    capacidad = datos["capacidad_recurso"]
    if not isinstance(capacidad, (int, float)) or capacidad <= 0:
        raise EscenarioInvalidoError(
            f"La 'capacidad_recurso' debe ser un número positivo. Recibido: {capacidad}"
        )

    penalizacion = float(datos.get("penalizacion_exceso_recurso", 10.0))
    costo_recarga = float(datos.get("costo_base_recarga", 15.0))

    return Escenario(
        nombre=nombre,
        descripcion=descripcion,
        filas=filas,
        columnas=columnas,
        inicio=inicio,
        destino=destino,
        entregas=entregas,
        cuadricula=cuadricula,
        capacidad_recurso=float(capacidad),
        penalizacion_exceso_recurso=penalizacion,
        punto_recarga=punto_recarga,
        costo_base_recarga=costo_recarga,
    )


if __name__ == "__main__":
    # Prueba rápida de ejecución directa
    ruta_ejemplo = os.path.join(
        os.path.dirname(__file__), "..", "config", "escenarios", "escenario_1_facil.json"
    )
    esc = cargar_escenario(ruta_ejemplo)
    print(f"=== Escenario cargado: {esc.nombre} ===")
    print(f"Dimensiones: {esc.filas}x{esc.columnas}")
    print(f"Inicio: {esc.inicio}, Destino: {esc.destino}")
    print(f"Entregas ({len(esc.entregas)}): {esc.entregas}")
    print(f"Capacidad recurso: {esc.capacidad_recurso}")
    print(f"Costo mínimo global: {esc.costo_minimo_global()}")
    print("\nVisualización del tablero:")
    print(esc.mostrar())
