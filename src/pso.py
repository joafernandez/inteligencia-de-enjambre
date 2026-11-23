"""Particle Swarm Optimization (PSO) genérico, reproducible y sin dependencias extra.

Implementa la variante clásica con factor de inercia (Shi & Eberhart, 1998) y los
coeficientes de constricción de Clerc & Kennedy (2002): w = 0.7298, c1 = c2 = 1.49618.

    v <- w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
    x <- clip(x + v, límites)

Minimiza una función f: R^d -> R. Toda la aleatoriedad proviene de un único
``numpy.random.Generator`` inicializado con ``semilla``, por lo que una misma
configuración produce exactamente el mismo resultado.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np


@dataclass
class ConfigPSO:
    n_particulas: int = 30
    iteraciones: int = 100
    inercia: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    vmax_frac: float = 0.25   # velocidad máxima como fracción del rango
    semilla: int = 42

    def a_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoPSO:
    mejor_x: np.ndarray
    mejor_valor: float
    historial: list[float] = field(default_factory=list)  # mejor valor global por iteración
    evaluaciones: int = 0
    iteracion_mejor: int = 0
    config: ConfigPSO = field(default_factory=ConfigPSO)


def pso_minimizar(f: Callable[[np.ndarray], float], dim: int,
                  limite_inf: float | np.ndarray = 0.0,
                  limite_sup: float | np.ndarray = 1.0,
                  config: ConfigPSO | None = None,
                  x_inicial: np.ndarray | None = None) -> ResultadoPSO:
    """Minimiza ``f`` en la caja [limite_inf, limite_sup]^dim.

    ``x_inicial`` (opcional) se inserta como una de las partículas iniciales;
    útil para arrancar la búsqueda desde una configuración conocida.
    """
    cfg = config or ConfigPSO()
    rng = np.random.default_rng(cfg.semilla)
    lo = np.broadcast_to(np.asarray(limite_inf, float), (dim,))
    hi = np.broadcast_to(np.asarray(limite_sup, float), (dim,))
    vmax = cfg.vmax_frac * (hi - lo)

    x = rng.uniform(lo, hi, size=(cfg.n_particulas, dim))
    if x_inicial is not None:
        x[0] = np.clip(x_inicial, lo, hi)
    v = rng.uniform(-vmax, vmax, size=(cfg.n_particulas, dim))

    valores = np.array([f(p) for p in x])
    evals = cfg.n_particulas
    pbest, pbest_val = x.copy(), valores.copy()
    g = int(np.argmin(pbest_val))
    gbest, gbest_val = pbest[g].copy(), float(pbest_val[g])
    historial, it_mejor = [gbest_val], 0

    for it in range(1, cfg.iteraciones + 1):
        r1 = rng.random((cfg.n_particulas, dim))
        r2 = rng.random((cfg.n_particulas, dim))
        v = cfg.inercia * v + cfg.c1 * r1 * (pbest - x) + cfg.c2 * r2 * (gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        valores = np.array([f(p) for p in x])
        evals += cfg.n_particulas
        mejora = valores < pbest_val
        pbest[mejora], pbest_val[mejora] = x[mejora], valores[mejora]
        g = int(np.argmin(pbest_val))
        if pbest_val[g] < gbest_val:
            gbest, gbest_val, it_mejor = pbest[g].copy(), float(pbest_val[g]), it
        historial.append(gbest_val)

    return ResultadoPSO(mejor_x=gbest, mejor_valor=gbest_val, historial=historial,
                        evaluaciones=evals, iteracion_mejor=it_mejor, config=cfg)
