"""Propiedades criptográficas de S-boxes biyectivas de 8x8 bits.

Todas las métricas se calculan a partir de los 256 valores de la S-box.
Convenciones (las habituales en la literatura de diseño de S-boxes):

* Funciones coordenadas f_j(x) = bit j de S(x), j = 0..7.
* Nonlinealidad (NL) de una función booleana f de n bits:
      NL(f) = 2^(n-1) - max_a |W_f(a)| / 2
  con W_f(a) = sum_x (-1)^(f(x) XOR a·x)  (espectro de Walsh-Hadamard).
  Se reporta mín/máx/promedio sobre las 8 funciones coordenadas.
* SAC (Strict Avalanche Criterion): matriz 8x8 con
      M[i, j] = P_x[ f_j(x) != f_j(x XOR e_i) ]
  Ideal: 0.5 en cada celda.
* BIC-NL: NL de f_j XOR f_k para los 28 pares j < k.
* BIC-SAC: SAC de f_j XOR f_k para cada par j < k y cada bit de entrada i
  (matriz 28x8). Como en el artículo de referencia, mín/máx se toman sobre el
  promedio de cada par (28 valores); el desvío se mide sobre las 224 celdas.
* Uniformidad diferencial (DU): máximo de la tabla DDT para a != 0.
  DP = DU / 256.
* LP (probabilidad de aproximación lineal):
      LP = max_{a, b != 0} |#{x : a·x = b·S(x)} - 128| / 256
  (definición usada en el artículo de referencia: p. ej. 36/256 = 0.1406).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np

N_BITS = 8
N = 1 << N_BITS  # 256


def _validar(sbox) -> np.ndarray:
    s = np.asarray(sbox, dtype=np.int64)
    if s.shape != (N,):
        raise ValueError(f"La S-box debe tener {N} valores; tiene {s.size}.")
    if s.min() < 0 or s.max() > N - 1:
        raise ValueError("Los valores deben estar en el rango 0..255.")
    return s


def es_biyectiva(sbox) -> bool:
    s = _validar(sbox)
    return np.unique(s).size == N


def _paridad(v: np.ndarray) -> np.ndarray:
    """Paridad (XOR de bits) de cada entero de 8 bits."""
    v = v.copy()
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return v & 1


def walsh_hadamard(f: np.ndarray) -> np.ndarray:
    """Espectro de Walsh de una función booleana dada como vector 0/1 de largo 2^n."""
    w = (1 - 2 * np.asarray(f, dtype=np.int64)).copy()
    h = 1
    while h < w.size:
        w = w.reshape(-1, 2, h)
        a, b = w[:, 0, :].copy(), w[:, 1, :].copy()
        w[:, 0, :], w[:, 1, :] = a + b, a - b
        w = w.reshape(-1)
        h *= 2
    return w


def nonlinealidad_booleana(f: np.ndarray) -> int:
    w = walsh_hadamard(f)
    return int(N // 2 - np.abs(w).max() // 2)


def funciones_coordenadas(sbox) -> np.ndarray:
    """Matriz 8x256: fila j = bit j de S(x)."""
    s = _validar(sbox)
    return np.array([(s >> j) & 1 for j in range(N_BITS)])


def nonlinealidades(sbox) -> np.ndarray:
    return np.array([nonlinealidad_booleana(f) for f in funciones_coordenadas(sbox)])


def matriz_sac(sbox) -> np.ndarray:
    s = _validar(sbox)
    x = np.arange(N)
    m = np.zeros((N_BITS, N_BITS))
    for i in range(N_BITS):
        diff = s ^ s[x ^ (1 << i)]
        for j in range(N_BITS):
            m[i, j] = ((diff >> j) & 1).mean()
    return m


def bic_nl(sbox) -> np.ndarray:
    f = funciones_coordenadas(sbox)
    return np.array([nonlinealidad_booleana(f[j] ^ f[k])
                     for j, k in combinations(range(N_BITS), 2)])


def bic_sac(sbox) -> np.ndarray:
    """Matriz 28x8: SAC de f_j XOR f_k frente a la inversión del bit de entrada i."""
    f = funciones_coordenadas(sbox)
    x = np.arange(N)
    filas = []
    for j, k in combinations(range(N_BITS), 2):
        g = f[j] ^ f[k]
        filas.append([(g != g[x ^ (1 << i)]).mean() for i in range(N_BITS)])
    return np.array(filas)


def tabla_ddt(sbox) -> np.ndarray:
    s = _validar(sbox)
    x = np.arange(N)
    ddt = np.zeros((N, N), dtype=np.int64)
    for a in range(N):
        np.add.at(ddt[a], s ^ s[x ^ a], 1)
    return ddt


def uniformidad_diferencial(sbox) -> int:
    return int(tabla_ddt(sbox)[1:].max())


def tabla_lat(sbox) -> np.ndarray:
    """LAT[a, b] = #{x : a·x = b·S(x)} - 128."""
    s = _validar(sbox)
    x = np.arange(N)
    pa = _paridad((np.arange(N)[:, None] & x[None, :]))          # a·x   (256x256)
    pb = _paridad((np.arange(N)[:, None] & s[None, :]))          # b·S(x)
    # coincidencias = sum_x [pa[a,x] == pb[b,x]]
    sa, sb = 1 - 2 * pa, 1 - 2 * pb
    corr = sa @ sb.T                                             # 2*coinc - 256
    return corr // 2


def probabilidad_lineal(sbox) -> float:
    lat = tabla_lat(sbox)
    return float(np.abs(lat[:, 1:]).max() / N)


@dataclass
class Propiedades:
    nl_min: float
    nl_max: float
    nl_prom: float
    sac_min: float
    sac_max: float
    sac_prom: float
    sac_desvio: float      # promedio de |M[i,j] - 0.5|
    bic_nl_min: float
    bic_nl_max: float
    bic_nl_prom: float
    bic_sac_min: float
    bic_sac_max: float
    bic_sac_prom: float
    bic_sac_desvio: float  # promedio de |BIC-SAC - 0.5|
    du: int
    dp: float
    lp: float
    biyectiva: bool

    def a_dict(self) -> dict:
        return asdict(self)


def calcular_propiedades(sbox) -> Propiedades:
    nl = nonlinealidades(sbox)
    sac = matriz_sac(sbox)
    bnl = bic_nl(sbox)
    bsac = bic_sac(sbox)
    du = uniformidad_diferencial(sbox)
    return Propiedades(
        nl_min=float(nl.min()), nl_max=float(nl.max()), nl_prom=float(nl.mean()),
        sac_min=float(sac.min()), sac_max=float(sac.max()), sac_prom=float(sac.mean()),
        sac_desvio=float(np.abs(sac - 0.5).mean()),
        bic_nl_min=float(bnl.min()), bic_nl_max=float(bnl.max()), bic_nl_prom=float(bnl.mean()),
        bic_sac_min=float(bsac.mean(axis=1).min()), bic_sac_max=float(bsac.mean(axis=1).max()),
        bic_sac_prom=float(bsac.mean()),
        bic_sac_desvio=float(np.abs(bsac - 0.5).mean()),
        du=du, dp=du / N, lp=probabilidad_lineal(sbox),
        biyectiva=es_biyectiva(sbox),
    )
