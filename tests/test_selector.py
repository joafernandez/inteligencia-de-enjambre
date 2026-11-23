import numpy as np
import pytest

from src import selector as S
from src.pso import ConfigPSO
from src.sboxes import cargar_metricas_reportadas, cargar_sboxes

NORM = S.normalizar(S.matriz_decision(S.tabla_propiedades(cargar_sboxes())))


def test_normalizacion_en_rango_y_extremos():
    assert NORM.min().min() >= 0 and NORM.max().max() <= 1
    for c in S.CLAVES:
        assert NORM[c].max() == 1 and NORM[c].min() == 0


def test_pesos_se_normalizan():
    w = S.vector_pesos({k: 2 for k in S.CLAVES})
    assert w.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        S.vector_pesos({k: 0 for k in S.CLAVES})


def test_exhaustiva_equivale_a_argmax():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.dirichlet(np.ones(6))
        r = S.seleccion_exhaustiva(NORM, w)
        assert r.ganador == NORM.index[np.argmax(NORM.to_numpy() @ w)]
        assert r.ranking["puntaje"].is_monotonic_decreasing


def test_aportes_suman_puntaje():
    r = S.seleccion_exhaustiva(NORM, S.PESOS_PRUEBA_A)
    aportes = r.ranking[[f"aporte_{k}" for k in S.CLAVES]].sum(axis=1)
    np.testing.assert_allclose(aportes, r.ranking["puntaje"])


def test_alternativas_dominadas_nunca_ganan():
    dom = S.dominadas(NORM)
    frec = S.frecuencia_victoria(NORM, n=20_000)
    for alt, por in dom.items():
        if por:
            assert frec[alt] == 0


def test_pso_cambio_minimo_hace_ganar_al_objetivo():
    r = S.seleccion_exhaustiva(NORM, S.PESOS_PRUEBA_B)
    cm = S.pso_cambio_minimo(NORM, S.PESOS_PRUEBA_B, r.segundo, ConfigPSO(semilla=1))
    assert cm.factible
    nuevo = S.seleccion_exhaustiva(NORM, cm.pesos)
    assert nuevo.ganador == r.segundo


def test_pso_no_peor_que_montecarlo_con_menos_evaluaciones():
    r = S.seleccion_exhaustiva(NORM, S.PESOS_PRUEBA_B)
    cm = S.pso_cambio_minimo(NORM, S.PESOS_PRUEBA_B, r.segundo, ConfigPSO(semilla=1))
    mc, n_mc = S.montecarlo_cambio_minimo(NORM, S.PESOS_PRUEBA_B, r.segundo, n=50_000)
    assert cm.pso.evaluaciones < n_mc
    assert cm.distancia <= mc + 1e-3


def test_pso_reporta_infactible_para_dominadas():
    cm = S.pso_cambio_minimo(NORM, S.PESOS_IGUALES, "GA", ConfigPSO(semilla=0, iteraciones=30))
    assert not cm.factible


def test_pso_directo_reproducible():
    a = S.pso_seleccion_directa(NORM, S.PESOS_PRUEBA_A, ConfigPSO(n_particulas=10, iteraciones=20, semilla=5))
    b = S.pso_seleccion_directa(NORM, S.PESOS_PRUEBA_A, ConfigPSO(n_particulas=10, iteraciones=20, semilla=5))
    assert a.ganador == b.ganador and a.evaluaciones == 210


# ---------------- reproducción del método EDAS del artículo (Tablas 10-14)

REP = cargar_metricas_reportadas()
EDAS = S.edas(S.matriz_tabla10(REP), S.ART_TABLA11_PESOS, S.ART_TABLA10_BENEFICIO)


def test_edas_promedio_reproduce_tabla12():
    np.testing.assert_allclose(EDAS["AV"], [108.3571, 0.4311, 104.7143, 0.4921, 0.03683, 0.1295],
                               rtol=5e-4)


def test_edas_pda_reproduce_tabla13_fila_pso():
    np.testing.assert_allclose(EDAS["PDA"][1], [0.0290, 0.1750, 0.0505, 0.0205, 0.3636, 0.1550], atol=6e-4)


def test_edas_nda_reproduce_tabla14_fila_fa():
    np.testing.assert_allclose(EDAS["NDA"][6], [0.0125, 0.9704, 0.0068, 0.1284, 0.0606, 0.0254], atol=6e-4)


def test_edas_extremos_coinciden_con_tabla16():
    assert np.argmax(EDAS["AS"]) == 1 and np.argmin(EDAS["AS"]) == 6   # PSO primero, FA último
