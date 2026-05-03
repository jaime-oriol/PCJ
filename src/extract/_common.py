"""
_common - Helpers compartidos para los extractores a parquet.

Incluye:
  - Rutas estandar de input/output
  - Escritura segura de parquet con compression snappy
  - Round-trip check lossless (JSON original <-> parquet)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl

# -- Rutas ------------------------------------------------------------------

_REPO     = Path(__file__).resolve().parents[2]
DATA      = _REPO / "data"
DATA_PUB  = DATA / "public"
DATA_PFF  = _REPO / "data_mundial"
PARQUET   = DATA / "parquet"


def parquet_dir(source: str) -> Path:
    """Devuelve y crea data/parquet/{source}/."""
    p = PARQUET / source
    p.mkdir(parents=True, exist_ok=True)
    return p


def scan_glob(pattern: str) -> "pl.LazyFrame":
    """Scan lazy de parquets per-partido con schemas potencialmente distintos.

    Cada parquet se extrae de su JSON con su propio schema inferido,
    asi que cols opcionales (e.g. injury_stoppage en StatsBomb) aparecen
    en unos partidos y no en otros. polars.scan_parquet con glob falla;
    diagonal_relaxed une schemas anadiendo nulls donde falten cols.

    Args:
        pattern : Glob relativo a data/parquet (e.g. 'pff/events/*.parquet').

    Returns:
        LazyFrame con todos los parquets unidos.

    Uso:
        from src.extract import scan_glob
        df = scan_glob("pff/tracking/*.parquet").filter(
            pl.col("frameNum") < 1000
        ).collect()
    """
    files = sorted(PARQUET.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Sin matches: {PARQUET / pattern}")
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


# -- Escritura --------------------------------------------------------------

def write_parquet(df: pl.DataFrame, path: Path, overwrite: bool = False) -> Path:
    """Escribe df a parquet snappy. Crea dir padre si no existe.

    Args:
        df        : DataFrame a escribir.
        path      : Ruta destino (.parquet).
        overwrite : Si False y el fichero existe, lanza FileExistsError.

    Returns:
        Path escrito.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe: {path}. Usa overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="snappy", statistics=True)
    return path


# -- Round-trip lossless check ---------------------------------------------

def _normalize(obj: Any) -> Any:
    """Normaliza un valor para comparacion lossless robusta.

    - dicts: ordena claves recursivamente, DROP claves con valor None
             (polars unifica schema -> filas tienen todas las claves vistas
             en cualquier fila, con None donde el JSON original no las tenia.
             Semanticamente identico: null == ausente.)
    - listas: normaliza cada elem (mantiene orden)
    - floats NaN -> None (PFF a veces serializa NaN como null)
    - floats con valor entero -> int (polars guarda ints nullable como float)
    """
    if isinstance(obj, dict):
        return {
            k: _normalize(v)
            for k, v in sorted(obj.items())
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    # Strings que codifican enteros se tratan como int (Wyscout mezcla
    # int y str numerico arbitrariamente en area.id, currentTeamId, etc.;
    # polars unifica a String y se pierde el tipo Python pero NO el valor).
    if isinstance(obj, str):
        s = obj.strip()
        if s and s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                pass
    return obj


def deep_equal(a: Any, b: Any) -> bool:
    """Compara dos estructuras JSON de forma lossless ignorando orden de claves."""
    return _normalize(a) == _normalize(b)


# -- Limpieza de sentinelas Wyscout (compartido entre wyscout + audit) -------

_NULL_SENTINELS = {"", "null"}


def clean_empty_strings(obj: Any) -> Any:
    """Convierte sentinelas Wyscout de "ausente" a None recursivamente.

    Wyscout es inconsistente y usa "", "null" (string literal) y None
    como sinonimos para "valor ausente" en campos numericos. Si lo dejamos
    asi, polars infiere String para columnas que son int en 99% de las
    filas (e.g. subEventId, currentTeamId), perdiendo el tipo.
    Convertir todos a None preserva el lossless funcional.

    Vive en _common para evitar coupling cross-module (wyscout + audit).
    """
    if isinstance(obj, dict):
        return {k: clean_empty_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_empty_strings(x) for x in obj]
    if isinstance(obj, str) and obj in _NULL_SENTINELS:
        return None
    return obj


def roundtrip_check(
    original_json_path: Path,
    parquet_path: Path,
    n_sample: int | None = None,
) -> tuple[bool, list[str]]:
    """Verifica que parquet -> dicts == JSON original.

    Args:
        original_json_path : JSON crudo (lista de dicts).
        parquet_path       : Parquet generado.
        n_sample           : Si int, compara solo las primeras N filas (rapido).

    Returns:
        (ok, errores). ok=True si todo coincide; errores lista las diferencias.
    """
    original = json.load(open(original_json_path))
    if n_sample is not None:
        original = original[:n_sample]

    df = pl.read_parquet(parquet_path)
    if n_sample is not None:
        df = df.head(n_sample)
    reconstructed = df.to_dicts()

    if len(original) != len(reconstructed):
        return False, [f"len mismatch: {len(original)} vs {len(reconstructed)}"]

    errors = []
    for i, (a, b) in enumerate(zip(original, reconstructed)):
        if not deep_equal(a, b):
            errors.append(f"row {i} differs")
            if len(errors) >= 5:
                break
    return len(errors) == 0, errors
