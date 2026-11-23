"""Carga de las siete S-boxes publicadas en Khan (2025), Scientific Reports 15:41261.

Los valores provienen de las Tablas 1-7 del artículo (versión PMC12638994) y se
guardan en ``data/sboxes_articulo.json`` sin modificaciones. Las métricas que el
propio artículo reporta (Tablas 8-10, 16-18) están en ``data/metricas_reportadas.csv``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVO_SBOXES = DATA_DIR / "sboxes_articulo.json"
ARCHIVO_REPORTADAS = DATA_DIR / "metricas_reportadas.csv"


@dataclass(frozen=True)
class SBox:
    id: str
    tecnica: str
    tabla_articulo: int
    ref_articulo: int
    valores: tuple[int, ...]
    hex_publicado: tuple[str, ...]

    def como_array(self) -> np.ndarray:
        return np.array(self.valores, dtype=np.int64)

    def como_matriz_hex(self) -> pd.DataFrame:
        """Representación 16x16 como en el artículo (filas leídas de izquierda a derecha)."""
        m = np.array([f"{v:02X}" for v in self.valores]).reshape(16, 16)
        cols = [f"{c:X}" for c in range(16)]
        return pd.DataFrame(m, index=cols, columns=cols)


def cargar_sboxes(ruta: Path = ARCHIVO_SBOXES) -> list[SBox]:
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return [
        SBox(id=d["id"], tecnica=d["tecnica"], tabla_articulo=d["tabla_articulo"],
             ref_articulo=d["ref_articulo"], valores=tuple(d["valores"]),
             hex_publicado=tuple(d["hex_publicado"]))
        for d in datos["sboxes"]
    ]


def fuente(ruta: Path = ARCHIVO_SBOXES) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))["_fuente"]


def cargar_metricas_reportadas(ruta: Path = ARCHIVO_REPORTADAS) -> pd.DataFrame:
    """Valores tal como los publica el artículo (no calculados por este prototipo)."""
    return pd.read_csv(ruta, comment="#").set_index("id")
