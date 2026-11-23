"""Las métricas se validan contra valores conocidos de la S-box de AES, generada
algorítmicamente (inverso en GF(2^8) + transformación afín) en lugar de copiarse."""
import numpy as np
import pytest

from src import metricas as m


def _gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ 0x11B) if a & 0x80 else a << 1
        b >>= 1
    return r


def _aes_sbox():
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if _gf_mul(a, b) == 1:
                inv[a] = b
                break
    out = []
    for x in range(256):
        b = inv[x]
        s = b
        for k in range(1, 5):
            s ^= ((b << k) | (b >> (8 - k))) & 0xFF
        out.append(s ^ 0x63)
    return out


AES = _aes_sbox()
IDENTIDAD = list(range(256))


def test_aes_generada_correctamente():
    assert AES[0x00] == 0x63 and AES[0x01] == 0x7C and AES[0x53] == 0xED


def test_walsh_funcion_constante():
    w = m.walsh_hadamard(np.zeros(256, dtype=int))
    assert w[0] == 256 and np.all(w[1:] == 0)


def test_nonlinealidad_aes_es_112():
    assert list(m.nonlinealidades(AES)) == [112] * 8


def test_identidad_es_lineal():
    assert list(m.nonlinealidades(IDENTIDAD)) == [0] * 8
    assert m.uniformidad_diferencial(IDENTIDAD) == 256
    assert m.probabilidad_lineal(IDENTIDAD) == 0.5


def test_diferencial_y_lineal_aes():
    assert m.uniformidad_diferencial(AES) == 4
    assert m.probabilidad_lineal(AES) == 16 / 256


def test_ddt_filas_suman_256_y_cero_trivial():
    ddt = m.tabla_ddt(AES)
    assert np.all(ddt.sum(axis=1) == 256)
    assert ddt[0, 0] == 256


def test_sac_y_bic_aes():
    p = m.calcular_propiedades(AES)
    assert p.sac_prom == pytest.approx(0.504883, abs=1e-5)   # valor de literatura
    assert p.bic_nl_min == 112
    assert p.biyectiva


def test_rechaza_entradas_invalidas():
    with pytest.raises(ValueError):
        m.calcular_propiedades([0] * 10)
    with pytest.raises(ValueError):
        m.calcular_propiedades([300] * 256)
    assert not m.es_biyectiva([0] * 256)
