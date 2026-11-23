"""Selección multicriterio de S-boxes.

Flujo:
1. ``matriz_decision``: criterios crudos calculados por ``metricas``.
2. ``normalizar``: min-max por criterio, orientado para que 1 = mejor y 0 = peor.
3. ``seleccion_exhaustiva``: suma ponderada (SAW) sobre las 7 alternativas.
   Es el MÉTODO DE REFERENCIA: con 7 alternativas evaluarlas todas es exacto y barato.
4. PSO se usa donde sí aporta, en el espacio CONTINUO de pesos:
   * ``analisis_robustez``: para cada S-box, PSO busca el cambio mínimo de
     prioridades (pesos) que la convertiría en la ganadora. Mide cuán estable es
     la recomendación frente a la elección subjetiva de pesos.
   * Se contrasta contra una búsqueda aleatoria densa (Monte Carlo) como referencia.
   * ``pso_seleccion_directa`` usa PSO para elegir la S-box directamente; se
     incluye sólo para mostrar que llega al mismo resultado que la búsqueda
     exhaustiva y que, para 7 alternativas, no tiene ventaja.
5. ``edas``: implementación del método del artículo (EDAS) para validar la reproducción.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metricas import calcular_propiedades
from .pso import ConfigPSO, ResultadoPSO, pso_minimizar

# --------------------------------------------------------------------------- criterios

CRITERIOS: list[dict] = [
    {"clave": "nl", "columna": "nl_prom", "sentido": "max",
     "nombre": "No linealidad",
     "ayuda": "Qué tan lejos está la S-box de una función lineal (mayor = mejor). "
              "Dificulta el criptoanálisis lineal y algebraico."},
    {"clave": "sac", "columna": "sac_desvio", "sentido": "min",
     "nombre": "Efecto avalancha (SAC)",
     "ayuda": "Cambiar 1 bit de entrada debería cambiar cada bit de salida con prob. 0.5. "
              "Se mide el desvío promedio respecto de 0.5 (menor = mejor)."},
    {"clave": "bic_nl", "columna": "bic_nl_prom", "sentido": "max",
     "nombre": "Independencia de bits (BIC-NL)",
     "ayuda": "No linealidad de las combinaciones XOR de pares de bits de salida (mayor = mejor)."},
    {"clave": "bic_sac", "columna": "bic_sac_desvio", "sentido": "min",
     "nombre": "Independencia de bits (BIC-SAC)",
     "ayuda": "Avalancha de las combinaciones XOR de pares de bits de salida; desvío respecto de 0.5 (menor = mejor)."},
    {"clave": "dp", "columna": "du", "sentido": "min",
     "nombre": "Resistencia diferencial",
     "ayuda": "Uniformidad diferencial (máximo de la tabla DDT). Menor = más difícil el criptoanálisis diferencial."},
    {"clave": "lp", "columna": "lp", "sentido": "min",
     "nombre": "Resistencia lineal (LP)",
     "ayuda": "Sesgo máximo de una aproximación lineal entrada/salida. Menor = más difícil el criptoanálisis lineal."},
]
CLAVES = [c["clave"] for c in CRITERIOS]
NOMBRES = {c["clave"]: c["nombre"] for c in CRITERIOS}

# Configuraciones de la demostración (ver README). No se ajustaron para forzar resultados.
PESOS_PRUEBA_A = {"nl": 0.50, "sac": 0.10, "bic_nl": 0.10, "bic_sac": 0.10, "dp": 0.10, "lp": 0.10}
PESOS_PRUEBA_B = {"nl": 0.075, "sac": 0.35, "bic_nl": 0.075, "bic_sac": 0.075, "dp": 0.35, "lp": 0.075}
PESOS_PRUEBA_C = {"nl": 0.0, "sac": 1.0, "bic_nl": 0.0, "bic_sac": 0.0, "dp": 0.0, "lp": 0.0}
PESOS_IGUALES = {k: 1 / len(CLAVES) for k in CLAVES}


def tabla_propiedades(sboxes) -> pd.DataFrame:
    """Propiedades completas calculadas por el prototipo, una fila por S-box."""
    return pd.DataFrame({s.id: calcular_propiedades(s.valores).a_dict() for s in sboxes}).T


def matriz_decision(propiedades: pd.DataFrame) -> pd.DataFrame:
    return propiedades[[c["columna"] for c in CRITERIOS]].astype(float).set_axis(CLAVES, axis=1)


def normalizar(matriz: pd.DataFrame) -> pd.DataFrame:
    """Min-max por columna orientado: 1 = mejor alternativa del conjunto, 0 = peor.

    Si todas las alternativas empatan en un criterio, ese criterio vale 1 para
    todas (no discrimina). La normalización es RELATIVA al conjunto evaluado.
    """
    out = pd.DataFrame(index=matriz.index)
    for c in CRITERIOS:
        col = matriz[c["clave"]]
        rango = col.max() - col.min()
        if rango == 0:
            out[c["clave"]] = 1.0
        elif c["sentido"] == "max":
            out[c["clave"]] = (col - col.min()) / rango
        else:
            out[c["clave"]] = (col.max() - col) / rango
    return out


def vector_pesos(pesos: dict | np.ndarray) -> np.ndarray:
    w = np.array([pesos[k] for k in CLAVES], float) if isinstance(pesos, dict) else np.asarray(pesos, float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Los pesos deben ser no negativos y no todos cero.")
    return w / w.sum()


# ------------------------------------------------------------------ método de referencia

@dataclass
class ResultadoSeleccion:
    pesos: dict
    ranking: pd.DataFrame          # puntaje, posición y contribución por criterio
    ganador: str
    segundo: str
    margen: float                  # puntaje(ganador) - puntaje(segundo)
    evaluaciones: int


def seleccion_exhaustiva(norm: pd.DataFrame, pesos) -> ResultadoSeleccion:
    w = vector_pesos(pesos)
    contrib = norm[CLAVES] * w
    puntaje = contrib.sum(axis=1)
    ranking = contrib.add_prefix("aporte_")
    ranking.insert(0, "puntaje", puntaje)
    ranking = ranking.sort_values("puntaje", ascending=False, kind="stable")
    ranking.insert(1, "posicion", np.arange(1, len(ranking) + 1))
    ganador, segundo = ranking.index[0], ranking.index[1]
    return ResultadoSeleccion(
        pesos={k: float(v) for k, v in zip(CLAVES, w.round(6))}, ranking=ranking, ganador=ganador, segundo=segundo,
        margen=float(ranking["puntaje"].iloc[0] - ranking["puntaje"].iloc[1]),
        evaluaciones=len(norm))


def dominadas(norm: pd.DataFrame) -> dict[str, list[str]]:
    """Para cada alternativa, lista de alternativas que la dominan en sentido de Pareto
    (iguales o mejores en todos los criterios y estrictamente mejores en alguno).
    Una alternativa dominada nunca puede ser la única ganadora con pesos no negativos."""
    X = norm[CLAVES].to_numpy()
    res = {}
    for i, a in enumerate(norm.index):
        res[a] = [b for j, b in enumerate(norm.index)
                  if j != i and (X[j] >= X[i]).all() and (X[j] > X[i]).any()]
    return res


def rival_alcanzable(norm: pd.DataFrame, sel: ResultadoSeleccion) -> str | None:
    """Alternativa no ganadora mejor ubicada que NO está dominada (la única clase
    de alternativa que podría ganar cambiando los pesos)."""
    dom = dominadas(norm)
    for alt in sel.ranking.index[1:]:
        if not dom[alt]:
            return alt
    return None


# ---------------------------------------------------------------------- uso de PSO

def _a_simplex(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return np.full_like(x, 1 / x.size) if s <= 0 else x / s


def distancia_pesos(w1: np.ndarray, w2: np.ndarray) -> float:
    """Distancia de variación total: fracción del peso total que hay que reasignar (0 a 1)."""
    return float(0.5 * np.abs(np.asarray(w1) - np.asarray(w2)).sum())


@dataclass
class CambioMinimo:
    objetivo: str
    factible: bool
    distancia: float               # fracción de importancia a reasignar (0..1)
    pesos: dict                    # pesos encontrados
    pso: ResultadoPSO | None = None
    distancia_montecarlo: float | None = None
    evaluaciones_montecarlo: int = 0


def pso_cambio_minimo(norm: pd.DataFrame, pesos, objetivo: str,
                      config: ConfigPSO | None = None, margen: float = 1e-4,
                      penalizacion: float = 100.0) -> CambioMinimo:
    """PSO busca pesos w' lo más parecidos posible a ``pesos`` con los que
    ``objetivo`` obtiene el mayor puntaje (por al menos ``margen``).

    Minimiza  d(w', w) + penalización · max(0, max_{j≠obj} S_j(w') - S_obj(w') + margen)
    con w' = x / sum(x), x ∈ [0, 1]^6.
    """
    w0 = vector_pesos(pesos)
    X = norm[CLAVES].to_numpy()
    k = list(norm.index).index(objetivo)
    otros = np.arange(len(X)) != k

    def violacion(w):
        s = X @ w
        return max(0.0, s[otros].max() - s[k] + margen)

    def gana(w):
        s = X @ w
        return bool(s[k] > s[otros].max())

    def f(x):
        w = _a_simplex(x)
        return distancia_pesos(w, w0) + penalizacion * violacion(w)

    res = pso_minimizar(f, dim=len(CLAVES), config=config, x_inicial=w0)
    w = _a_simplex(res.mejor_x)
    factible = gana(w)
    return CambioMinimo(objetivo=objetivo, factible=factible,
                        distancia=distancia_pesos(w, w0) if factible else float("nan"),
                        pesos={k: float(v) for k, v in zip(CLAVES, w.round(4))}, pso=res)


def montecarlo_cambio_minimo(norm: pd.DataFrame, pesos, objetivo: str,
                             n: int = 50_000, semilla: int = 0, margen: float = 1e-4) -> tuple[float, int]:
    """Referencia para PSO: muestrea pesos al azar (mitad uniformes en el simplex,
    mitad perturbaciones locales de ``pesos``) y devuelve la menor distancia
    encontrada con la que ``objetivo`` gana. NaN si ninguna muestra lo logra."""
    rng = np.random.default_rng(semilla)
    w0 = vector_pesos(pesos)
    X = norm[CLAVES].to_numpy()
    k = list(norm.index).index(objetivo)
    uni = rng.dirichlet(np.ones(len(CLAVES)), size=n // 2)
    loc = np.clip(w0 + rng.normal(0, rng.uniform(0.01, 0.3, (n - n // 2, 1)), (n - n // 2, len(CLAVES))), 0, None)
    suma = loc.sum(axis=1, keepdims=True)
    loc = np.where(suma > 0, loc / np.where(suma > 0, suma, 1), w0)
    W = np.vstack([uni, loc])
    S = W @ X.T
    otros = np.delete(S, k, axis=1).max(axis=1)
    ok = S[:, k] > otros
    if not ok.any():
        return float("nan"), n
    d = 0.5 * np.abs(W[ok] - w0).sum(axis=1)
    return float(d.min()), n


def analisis_robustez(norm: pd.DataFrame, pesos, config: ConfigPSO | None = None,
                      n_montecarlo: int = 50_000) -> pd.DataFrame:
    """Para cada S-box distinta de la ganadora: cambio mínimo de pesos que la haría
    ganar (PSO) y el mismo valor estimado por Monte Carlo como referencia."""
    base = seleccion_exhaustiva(norm, pesos)
    dom = dominadas(norm)
    cfg = config or ConfigPSO()
    filas = []
    for alt in norm.index:
        if alt == base.ganador:
            continue
        cm = pso_cambio_minimo(norm, pesos, alt, cfg)
        mc, n_mc = montecarlo_cambio_minimo(norm, pesos, alt, n=n_montecarlo, semilla=cfg.semilla)
        filas.append({
            "sbox": alt,
            "dist_pso": cm.distancia,
            "dist_montecarlo": mc,
            "evals_pso": cm.pso.evaluaciones,
            "evals_montecarlo": n_mc,
            "dominada_por": ", ".join(dom[alt]) or "-",
            "pesos_pso": cm.pesos if cm.factible else None,
        })
    return pd.DataFrame(filas).set_index("sbox").sort_values("dist_pso", na_position="last")


def frecuencia_victoria(norm: pd.DataFrame, n: int = 100_000, semilla: int = 0) -> pd.Series:
    """Proporción de pesos aleatorios (uniformes en el simplex) en que gana cada S-box.
    Indica qué alternativas son competitivas sin depender de una elección de pesos."""
    rng = np.random.default_rng(semilla)
    W = rng.dirichlet(np.ones(len(CLAVES)), size=n)
    gan = np.argmax(W @ norm[CLAVES].to_numpy().T, axis=1)
    return pd.Series(np.bincount(gan, minlength=len(norm)) / n, index=norm.index)


@dataclass
class ResultadoPSODirecto:
    ganador: str
    coincide_con_exhaustiva: bool
    evaluaciones: int
    iteracion_hallazgo: int
    historial: list = field(default_factory=list)


def pso_seleccion_directa(norm: pd.DataFrame, pesos, config: ConfigPSO | None = None) -> ResultadoPSODirecto:
    """PSO sobre una variable continua x ∈ [0, n) decodificada como índice floor(x).
    Se incluye como contraste: con 7 alternativas la búsqueda exhaustiva es exacta
    y usa 7 evaluaciones; PSO necesita muchas más y no garantiza el óptimo."""
    cfg = config or ConfigPSO(n_particulas=10, iteraciones=20)
    w = vector_pesos(pesos)
    s = norm[CLAVES].to_numpy() @ w
    n = len(s)
    res = pso_minimizar(lambda x: -s[min(int(x[0]), n - 1)], dim=1,
                        limite_inf=0.0, limite_sup=n - 1e-9, config=cfg)
    ganador = norm.index[min(int(res.mejor_x[0]), n - 1)]
    return ResultadoPSODirecto(ganador=ganador,
                               coincide_con_exhaustiva=ganador == norm.index[int(np.argmax(s))],
                               evaluaciones=res.evaluaciones, iteracion_hallazgo=res.iteracion_mejor,
                               historial=[-h for h in res.historial])


def tasa_exito_pso_directo(norm: pd.DataFrame, pesos, semillas=range(30),
                           n_particulas: int = 10, iteraciones: int = 20) -> dict:
    """Repite ``pso_seleccion_directa`` con varias semillas y mide cuántas veces
    coincide con la búsqueda exhaustiva (que acierta siempre con 7 evaluaciones)."""
    res = [pso_seleccion_directa(norm, pesos, ConfigPSO(n_particulas=n_particulas,
                                 iteraciones=iteraciones, semilla=s)) for s in semillas]
    return {"corridas": len(res),
            "aciertos": sum(r.coincide_con_exhaustiva for r in res),
            "evaluaciones_por_corrida": res[0].evaluaciones}


# ------------------------------------------------------------- reproducción del artículo

def edas(matriz: np.ndarray, pesos: np.ndarray, beneficio: np.ndarray) -> dict:
    """EDAS (Keshavarz Ghorabaee et al., 2015) en su formulación estándar.

    Devuelve AV, PDA, NDA, SP, SN, NSP, NSN y AS para comparar con las
    Tablas 12-16 del artículo."""
    X = np.asarray(matriz, float)
    w = np.asarray(pesos, float)
    b = np.asarray(beneficio, bool)
    av = X.mean(axis=0)
    pda = np.where(b, np.maximum(0, X - av), np.maximum(0, av - X)) / av
    nda = np.where(b, np.maximum(0, av - X), np.maximum(0, X - av)) / av
    sp, sn = pda @ w, nda @ w
    nsp = sp / sp.max() if sp.max() > 0 else np.zeros_like(sp)
    nsn = 1 - sn / sn.max() if sn.max() > 0 else np.ones_like(sn)
    return {"AV": av, "PDA": pda, "NDA": nda, "SP": sp, "SN": sn,
            "NSP": nsp, "NSN": nsn, "AS": (nsp + nsn) / 2}


# Datos publicados que se usan para la verificación (Tablas 10 y 11 del artículo).
ART_TABLA10_COLUMNAS = ["NL", "BIC-SAC", "BIC-NL", "SAC", "DP", "LP"]
ART_TABLA10_BENEFICIO = np.array([True, True, True, True, False, False])
ART_TABLA11_PESOS = np.array([0.0053, 0.5027, 0.0034, 0.4873, 0.0009, 0.0004])


def matriz_tabla10(reportadas: pd.DataFrame) -> np.ndarray:
    """Matriz de decisión del artículo (Tabla 10) a partir de los valores reportados."""
    r = reportadas
    return np.column_stack([r["nl_prom"], r["bic_sac_prom"], r["bic_nl_prom"],
                            r["sac_prom"], r["du"] / 256, r["lp"]]).astype(float)
