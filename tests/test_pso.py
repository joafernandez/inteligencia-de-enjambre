import numpy as np

from src.pso import ConfigPSO, pso_minimizar


def esfera(x):
    return float(((x - 0.3) ** 2).sum())


def test_minimiza_esfera():
    r = pso_minimizar(esfera, dim=4, config=ConfigPSO(semilla=1))
    assert r.mejor_valor < 1e-6
    np.testing.assert_allclose(r.mejor_x, 0.3, atol=1e-3)


def test_reproducible_con_semilla():
    a = pso_minimizar(esfera, dim=3, config=ConfigPSO(semilla=7, iteraciones=20))
    b = pso_minimizar(esfera, dim=3, config=ConfigPSO(semilla=7, iteraciones=20))
    assert a.historial == b.historial and np.array_equal(a.mejor_x, b.mejor_x)


def test_historial_monotono_y_conteo_evaluaciones():
    cfg = ConfigPSO(n_particulas=8, iteraciones=15, semilla=3)
    r = pso_minimizar(esfera, dim=2, config=cfg)
    assert all(b <= a for a, b in zip(r.historial, r.historial[1:]))
    assert r.evaluaciones == 8 * 16


def test_respeta_limites():
    r = pso_minimizar(lambda x: -float(x.sum()), dim=3, limite_inf=-1, limite_sup=2,
                      config=ConfigPSO(semilla=0))
    assert np.all(r.mejor_x <= 2) and np.all(r.mejor_x >= -1)
    np.testing.assert_allclose(r.mejor_x, 2, atol=1e-6)
