"""Reproduce los experimentos de la demostración y escribe docs/resultados_experimentos.md.

Uso:  python experimentos.py
Todos los parámetros y semillas quedan registrados en el archivo generado.
"""
from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import selector as S
from src.pso import ConfigPSO
from src.sboxes import cargar_metricas_reportadas, cargar_sboxes

SEMILLAS_ROBUSTEZ = [1, 2, 3, 42, 123]
CFG = ConfigPSO()  # 30 partículas, 100 iteraciones, coeficientes de Clerc-Kennedy
SALIDA = Path(__file__).parent / "docs" / "resultados_experimentos.md"


def md(df: pd.DataFrame, dec: int = 4) -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.astype(object).iterrows():
        vals = [(f"{v:.{dec}f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else
                 ("—" if isinstance(v, (float, np.floating)) and np.isnan(v) else str(v))) for v in row]
        lines.append(f"| {idx} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    sboxes = cargar_sboxes()
    props = S.tabla_propiedades(sboxes)
    matriz = S.matriz_decision(props)
    norm = S.normalizar(matriz)
    rep = cargar_metricas_reportadas()
    out = [f"# Resultados de los experimentos\n",
           f"Generado por `python experimentos.py` el {date.today().isoformat()} "
           f"(Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}).\n",
           f"Configuración PSO: `{CFG.a_dict()}`. Semillas para la robustez: {SEMILLAS_ROBUSTEZ}. "
           "Monte Carlo: 50 000 muestras, semilla = semilla de PSO.\n",
           "## 1. Matriz de decisión (calculada por el prototipo)\n",
           md(matriz), "\n\nCriterios: nl = NL promedio (↑), sac = desvío SAC de 0.5 (↓), bic_nl = BIC-NL promedio (↑), "
           "bic_sac = desvío BIC-SAC de 0.5 (↓), dp = uniformidad diferencial (↓), lp = probabilidad lineal (↓).\n",
           "## 2. Matriz normalizada (1 = mejor, 0 = peor)\n", md(norm, 3),
           "\n\n**Dominancia de Pareto:** " + "; ".join(
               f"{a} dominada por {', '.join(p)}" for a, p in S.dominadas(norm).items() if p) + ".\n",
           "**Frecuencia de victoria con 100 000 pesos uniformes al azar (semilla 0):** " + ", ".join(
               f"{k} {100 * v:.2f} %" for k, v in S.frecuencia_victoria(norm).items()) + ".\n"]

    for nombre, pesos in [("Prueba A — prioridad en no linealidad", S.PESOS_PRUEBA_A),
                          ("Prueba B — prioridad en avalancha y resistencia diferencial", S.PESOS_PRUEBA_B),
                          ("Prueba C (control) — sólo avalancha", S.PESOS_PRUEBA_C),
                          ("Referencia — pesos iguales", S.PESOS_IGUALES)]:
        sel = S.seleccion_exhaustiva(norm, pesos)
        out += [f"## {nombre}\n", f"Pesos: `{sel.pesos}`\n",
                f"**Recomendada: {sel.ganador}** (puntaje {sel.ranking.puntaje.iloc[0]:.4f}); "
                f"segunda: {sel.segundo} (margen {sel.margen:.4f}).\n",
                md(sel.ranking[["puntaje", "posicion"]]), "\n"]
        rival = S.rival_alcanzable(norm, sel)
        if rival != sel.segundo:
            out.append(f"La segunda ({sel.segundo}) está dominada por "
                       f"{', '.join(S.dominadas(norm)[sel.segundo])}: no puede ganar con ningún peso. "
                       f"El rival alcanzable mejor ubicado es **{rival}**.\n")
        if rival:
            filas = []
            for s in SEMILLAS_ROBUSTEZ:
                cfg = ConfigPSO(**{**CFG.a_dict(), "semilla": s})
                cm = S.pso_cambio_minimo(norm, pesos, rival, cfg)
                mc, n = S.montecarlo_cambio_minimo(norm, pesos, rival, n=50_000, semilla=s)
                filas.append({"semilla": s, "PSO dist": cm.distancia, "PSO evals": cm.pso.evaluaciones,
                              "MC dist": mc, "MC evals": n,
                              "pesos PSO": ", ".join(f"{k}={v:.3f}" for k, v in cm.pesos.items() if v > 0.0005)})
            t = pd.DataFrame(filas).set_index("semilla")
            out += [f"\nCambio mínimo de pesos (variación total) para que **{rival}** pase a ganar:\n", md(t),
                    f"\n\nPSO: media {t['PSO dist'].mean():.4f}, mín {t['PSO dist'].min():.4f}, máx {t['PSO dist'].max():.4f}. "
                    f"Monte Carlo: media {t['MC dist'].mean():.4f}, mín {t['MC dist'].min():.4f}. "
                    f"PSO ≤ Monte Carlo en {int((t['PSO dist'] <= t['MC dist'] + 1e-9).sum())}/{len(t)} semillas, "
                    f"con {t['PSO evals'].iloc[0]} evaluaciones frente a {t['MC evals'].iloc[0]}.\n"]
        te = S.tasa_exito_pso_directo(norm, pesos)
        out.append(f"PSO aplicado directamente a elegir la S-box (10×20, semillas 0–29): acertó "
                   f"{te['aciertos']}/{te['corridas']} veces con {te['evaluaciones_por_corrida']} evaluaciones "
                   "por corrida; la búsqueda exhaustiva acierta siempre con 7.\n")

    ed = S.edas(S.matriz_tabla10(rep), S.ART_TABLA11_PESOS, S.ART_TABLA10_BENEFICIO)
    t = pd.DataFrame({"AS reproducido": ed["AS"], "AS artículo": rep["edas_score"].astype(float),
                      "SP reproducido": ed["SP"], "SN reproducido": ed["SN"]}, index=rep.index)
    out += ["## Reproducción del EDAS del artículo (Tablas 10–16)\n", md(t),
            "\n\nAV, PDA y NDA coinciden con las Tablas 12–14. SP, SN y AS no coinciden con las Tablas 15–16; "
            "ver `docs/referencia_cientifica.md`.\n"]
    SALIDA.write_text("\n".join(out), encoding="utf-8")
    print(f"Escrito {SALIDA}")


if __name__ == "__main__":
    main()
