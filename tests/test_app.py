"""Prueba de humo de la interfaz: Prueba A, Prueba B y comparación, sin navegador."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _boton(at, texto):
    return next(b for b in at.button if texto in b.label)


def test_flujo_prueba_a_y_b():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception
    _boton(at, "Prueba A").click().run()
    _boton(at, "Ejecutar").click().run()
    assert not at.exception
    _boton(at, "Prueba B").click().run()
    _boton(at, "Ejecutar").click().run()
    assert not at.exception
    ej = at.session_state["ejecuciones"]
    assert len(ej) == 2
    assert ej[0]["pesos"]["nl"] == 0.5 and ej[1]["pesos"]["dp"] == 0.35
    assert ej[0]["etiqueta"].startswith("Prueba A") and ej[1]["etiqueta"].startswith("Prueba B")


def test_prueba_c_cambia_recomendacion_y_usa_rival_alcanzable():
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120).run()
    _boton(at, "Prueba C").click().run()
    _boton(at, "Ejecutar").click().run()
    assert not at.exception
    u = at.session_state["ultima"]
    assert u["sel"].ganador == "ABC" and u["sel"].segundo == "ANO"
    assert u["rival"] == "PSO" and u["cm"].factible
