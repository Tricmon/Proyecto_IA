"""
Pruebas unitarias para el módulo de escenario y sus validaciones (scenario.py).
"""

import json
import os
import pytest
from src.scenario import cargar_escenario, EscenarioInvalidoError


def _crear_json_temporal(tmp_path, datos_dict):
    """Crea un archivo json temporal para pruebas de validación."""
    p = tmp_path / "temp_escenario.json"
    p.write_text(json.dumps(datos_dict), encoding="utf-8")
    return str(p)


def _base_valida():
    """Retorna un diccionario de escenario válido para usar como plantilla de pruebas."""
    return {
        "nombre": "Prueba",
        "descripcion": "Escenario base para test",
        "filas": 3,
        "columnas": 3,
        "inicio": [0, 0],
        "destino": [2, 2],
        "entregas": [[0, 1], [1, 1], [2, 1]],
        "capacidad_recurso": 20.0,
        "cuadricula": [
            [1, 2, 1],
            [1, 1, 3],
            [2, 1, 1],
        ],
    }


def test_carga_escenario_valido():
    """Prueba que el escenario base del proyecto cargue correctamente y conserve sus atributos."""
    ruta = os.path.join(
        os.path.dirname(__file__), "..", "config", "escenarios", "escenario_1_facil.json"
    )
    esc = cargar_escenario(ruta)
    assert esc.filas == 12
    assert esc.columnas == 12
    assert esc.inicio == (0, 0)
    assert esc.destino == (11, 11)
    assert len(esc.entregas) == 6
    assert esc.costo_minimo_global() == 1
    assert not esc.es_obstaculo(0, 0)
    assert esc.es_obstaculo(1, 1)


def test_validacion_clave_faltante(tmp_path):
    """Detecta error cuando falta una clave obligatoria (ej. 'cuadricula')."""
    datos = _base_valida()
    del datos["cuadricula"]
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="Falta la clave requerida"):
        cargar_escenario(ruta)


def test_validacion_fuera_de_rango(tmp_path):
    """Detecta error si una coordenada excede las dimensiones."""
    datos = _base_valida()
    datos["inicio"] = [5, 0]  # Fuera de 3x3
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="fuera de los límites"):
        cargar_escenario(ruta)


def test_validacion_punto_sobre_obstaculo(tmp_path):
    """Detecta error si el inicio o una entrega cae en obstáculo (0)."""
    datos = _base_valida()
    datos["cuadricula"][0][0] = 0  # Inicio en obstáculo
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="obstáculo"):
        cargar_escenario(ruta)


def test_validacion_puntos_duplicados(tmp_path):
    """Detecta error si el inicio y una entrega coinciden."""
    datos = _base_valida()
    datos["entregas"][0] = [0, 0]  # Igual que inicio
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="Conflicto de posición"):
        cargar_escenario(ruta)


def test_validacion_costo_invalido(tmp_path):
    """Detecta error si una celda tiene costo fuera de 0..5."""
    datos = _base_valida()
    datos["cuadricula"][1][0] = 9  # Inválido
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="Debe ser 0 .* o entre 1 y 5"):
        cargar_escenario(ruta)


def test_validacion_capacidad_recurso_invalida(tmp_path):
    """Detecta error si la capacidad del recurso es 0 o negativa."""
    datos = _base_valida()
    datos["capacidad_recurso"] = 0.0  # Inválido
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="capacidad_recurso.*positivo"):
        cargar_escenario(ruta)


def test_validacion_dos_entregas_duplicadas(tmp_path):
    """Detecta error si dos puntos de entrega comparten la misma coordenada."""
    datos = _base_valida()
    datos["entregas"][1] = datos["entregas"][0]  # Entrega 1 igual a Entrega 2
    ruta = _crear_json_temporal(tmp_path, datos)
    with pytest.raises(EscenarioInvalidoError, match="Conflicto de posición"):
        cargar_escenario(ruta)
