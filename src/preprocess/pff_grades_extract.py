"""pff_grades_extract - Agrega los grades de PFF events por jugador.

PFF events trae un struct `grades` con ~25 campos (passerGrade, defenderGrade,
shooterGrade, ...). Para cada evento, calculamos el promedio horizontal de
los campos no-null como `event_grade_mean` (el grade representativo del rol
que el jugador del evento tuvo en ese momento). Despues agregamos por
jugador a lo largo de los 64 partidos WC22.

Output: data/parquet/derived/preprocess/pff_grades.parquet (~710 jugadores).
Schema: pff_player_id, pff_grade_mean, n_grades, player_name, team_name,
        position_group.

Usado por M14_cate.attach_pff_grades para priors informativos del random
effect mu_player (Gomes-Mendes-Neves 2025 estilo).

Uso:
    python -m src.preprocess.pff_grades_extract
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_OUT = _REPO / "data" / "parquet" / "derived" / "preprocess" / "pff_grades.parquet"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def extract_grades_for_match(match_id: int) -> pl.DataFrame:
    """Per evento del partido: mean horizontal de los grade fields no-null.

    Devuelve (player_id, event_grade_mean) filtrando filas sin player_id
    o sin grade no-null.
    """
    from M01_loader_pff import load_events
    ev = load_events(match_id)
    # Detect grade fields del struct dinamicamente (varian entre partidos:
    # PFF anade campos como closingDown2Grade en algunos partidos).
    grades_dtype = ev.schema["grades"]
    fields = [f.name for f in grades_dtype.fields]
    if not fields:
        return pl.DataFrame(schema={"player_id": pl.Int64,
                                      "event_grade_mean": pl.Float64})
    flat = ev.select([
        pl.col("gameEvents").struct.field("playerId").cast(pl.Int64).alias("player_id"),
        *[pl.col("grades").struct.field(f).alias(f) for f in fields],
    ]).filter(pl.col("player_id").is_not_null())
    flat = flat.with_columns(
        pl.mean_horizontal(fields).alias("event_grade_mean")
    ).filter(pl.col("event_grade_mean").is_not_null())
    return flat.select(["player_id", "event_grade_mean"])


def build() -> pl.DataFrame:
    """Agrega grades de los 64 partidos WC22 + joinea con rosters."""
    from M01_loader_pff import list_event_match_ids, load_rosters

    parts = []
    for mid in list_event_match_ids():
        try:
            parts.append(extract_grades_for_match(mid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
    if not parts:
        raise RuntimeError("no se extrajo ningun grade")
    long = pl.concat(parts)

    # Agrega per player: mean grade + count
    agg = long.group_by("player_id").agg([
        pl.col("event_grade_mean").mean().alias("pff_grade_mean"),
        pl.len().cast(pl.Int64).alias("n_grades"),
    ]).rename({"player_id": "pff_player_id"})

    # Joinea con rosters (player_name, team_name, position_group)
    ro = load_rosters().select([
        pl.col("player_id").alias("pff_player_id"),
        "player_name", "team_name", "position_group",
    ]).unique(subset=["pff_player_id"])
    out = agg.join(ro, on="pff_player_id", how="inner").sort(
        "n_grades", descending=True
    )
    return out


def main(overwrite: bool = False) -> Path:
    if _OUT.exists() and not overwrite:
        print(f"  pff_grades.parquet ya existe, skip (use overwrite=True)")
        return _OUT
    print("[pff_grades] extrayendo grades de los 64 partidos WC22...")
    df = build()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_OUT, compression="snappy")
    print(f"  pff_grades.parquet: {df.height} jugadores -> {_OUT}")
    print(f"  pff_grade_mean stats: mean={df['pff_grade_mean'].mean():+.4f}, "
          f"std={df['pff_grade_mean'].std():.4f}")
    return _OUT


if __name__ == "__main__":
    main(overwrite=True)
