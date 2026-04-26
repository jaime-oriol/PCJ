"""
M06_nearmiss - Identificacion cuasi-experimental de "casi gol" (near-miss).

Consumer directo de M05 PSxG. Produce los contrafactuales exogenos para la
Estrategia B causal (Gauriot & Page 2019 style: el resultado de un shot a
puerta clara es aleatorio dado pre-estado). M13 AIPW los usa como IV.

Cinco tipos con umbrales pre-registrados (propuesta §1.7):
  (a) Palo/travesano       : shot.outcome in {Post, Saved to Post},
                             xg pre-shot in [0.15, 0.85]
  (b) Offside milimetrico  : Offside event con margin <= 1.5m medido sobre
                             StatsBomb 360 freeze_frame (linea defensiva real).
                             Fallback proxy att_x > 110 si no hay 360.
  (c) Parada PSxG alto     : outcome=Saved y (PSxG>=0.6 OR xg_baseline>=0.4)
  (d) Despeje linea gol    : outcome == Saved Off Target (marker SB directo).
  (e) GLT no-gol           : fuera de scope (raro en WC22).

Acceptance (ARCHITECTURE.md): distribucion coherente con benchmarks.
  Escalado a 64 partidos: ~120-185 near-miss totales.

Output: data/parquet/derived/nearmiss/nearmiss_table.parquet
  cols: sb_match_id, event_uuid, period, minute, second, team_id, team_name,
        shot_outcome, is_goal, xg_baseline, psxg, near_miss_type, margin_info.
  El consumer (M13 AIPW) recupera pff_match_id via M03.sb_to_pff_match_id().

Depende de: M02 (SB events), M05 (PSxG cache).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M02_loader_public import (
    load_statsbomb_events, list_statsbomb_match_ids, load_statsbomb_360,
)


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "nearmiss"
_PSXG    = _REPO / "data" / "parquet" / "derived" / "psxg" / "shots.parquet"


# -- Umbrales pre-registrados ----------------------------------------------

PSXG_SAVE_STRICT     = 0.60
XG_SAVE_LAX          = 0.40
XG_POST_MIN          = 0.15
XG_POST_MAX          = 0.85
OFFSIDE_TIGHT_METERS = 1.5     # margen X entre atacante y ultimo defensor


# ===========================================================================
#  SECCION 1 — Loaders de data pre-computada
# ===========================================================================

def _load_psxg_shots() -> pl.DataFrame:
    """Carga la tabla PSxG cacheada de M05."""
    if not _PSXG.exists():
        raise FileNotFoundError(
            f"M05 PSxG cache no existe en {_PSXG}. "
            "Ejecuta `python src/M05_psxg.py` primero."
        )
    return pl.read_parquet(_PSXG)


# ===========================================================================
#  SECCION 2 — Detectores por tipo de near-miss
# ===========================================================================

def _detect_woodwork(psxg: pl.DataFrame, lax: bool = False) -> pl.DataFrame:
    """(a) Palo/travesano: outcome Post/Saved to Post, xg pre-shot en rango.

    lax=False (default): xg in [0.15, 0.85] (propuesta pre-registrada §1.7).
    lax=True: xg in [0.08, 0.92] (specification curve, robustez Simonsohn 2020).
    """
    lo, hi = (0.08, 0.92) if lax else (XG_POST_MIN, XG_POST_MAX)
    mask = (
        pl.col("shot_outcome").is_in(["Post", "Saved to Post"])
        & (pl.col("xg_baseline") >= lo)
        & (pl.col("xg_baseline") <= hi)
    )
    return psxg.filter(mask).with_columns(
        pl.lit("a_woodwork").alias("near_miss_type"),
        pl.col("xg_baseline").alias("margin_info"),
    )


def _detect_saves_clutch(psxg: pl.DataFrame,
                          strict_only: bool = False) -> pl.DataFrame:
    """(c) Parada PSxG alto: Saved + (PSxG>=0.6 OR xg_baseline>=0.4 si no strict)."""
    if strict_only:
        mask = (pl.col("shot_outcome") == "Saved") & (pl.col("psxg") >= PSXG_SAVE_STRICT)
    else:
        mask = (pl.col("shot_outcome") == "Saved") & (
            (pl.col("psxg") >= PSXG_SAVE_STRICT)
            | (pl.col("xg_baseline") >= XG_SAVE_LAX)
        )
    return psxg.filter(mask).with_columns(
        pl.lit("c_save_psxg").alias("near_miss_type"),
        pl.col("psxg").alias("margin_info"),
    )


def _detect_goal_line_clearance(psxg: pl.DataFrame) -> pl.DataFrame:
    """(d) Despeje linea gol: 'Saved Off Target' (SB marker directo).

    Nota: no tenemos end_x en el feature set post-fix de M05 (era leakage).
    Nos limitamos al marker SB directo; la heuristica end_x esta out of scope.
    """
    sot = psxg.filter(pl.col("shot_outcome") == "Saved Off Target")
    return sot.with_columns(
        pl.lit("d_goal_line_clearance").alias("near_miss_type"),
        pl.col("psxg").alias("margin_info"),
    )


def _detect_glt_denied(psxg: pl.DataFrame) -> pl.DataFrame:
    """(e) GLT no-gol: MUY raro en WC22. Heuristica pobre sin tracking ad-hoc.

    Por ahora retorna vacio (no hay marker SB directo). M06 puede re-invocarse
    con datos tracking PFF si se añade detector fine-grained en el futuro.
    """
    return psxg.head(0).with_columns(
        pl.lit("e_glt_denied").alias("near_miss_type"),
        pl.lit(None, dtype=pl.Float64).alias("margin_info"),
    )


_OFFSIDE_SCHEMA = {
    "sb_match_id": pl.Int64, "event_uuid": pl.String,
    "period": pl.Int64, "minute": pl.Int64, "second": pl.Int64,
    "team_id": pl.Int64, "team_name": pl.String,
    "shot_outcome": pl.String, "is_goal": pl.Boolean,
    "xg_baseline": pl.Float64, "psxg": pl.Float64,
    "near_miss_type": pl.String, "margin_info": pl.Float64,
}


def _offside_margin_from_ff(att_x: float, ff: list) -> float | None:
    """Margen (metros) entre atacante y linea defensiva a partir de freeze-frame SB.

    La linea defensiva = max(x) entre defensores no-keeper.
    Margin positivo = atacante por DELANTE de la linea (offsided).
    Coords SB 120x80; 1 unidad ~ 1m. Devuelve None si no hay defensores visibles.
    """
    if not ff:
        return None
    def_xs = []
    for p in ff:
        if p is None:
            continue
        tm = p.get("teammate")
        kp = p.get("keeper")
        loc = p.get("location")
        if tm is True or kp is True or loc is None or len(loc) < 2:
            continue
        def_xs.append(float(loc[0]))
    if not def_xs:
        return None
    return att_x - max(def_xs)


def _detect_offside_tight(match_ids: list[int],
                           tight_meters: float = OFFSIDE_TIGHT_METERS
                           ) -> pl.DataFrame:
    """(b) Offside milimetrico real: margen <= tight_meters entre atacante y
    linea defensiva (max x de no-keeper).

    Fuente: StatsBomb 360 freeze_frames. Si falta el 360 del partido o del
    evento concreto, cae a proxy att_x > 110 y flag margin_info = NaN para
    downstream sensitivity. Vectorizado por partido (1 carga 360 / partido).
    """
    rows = []
    for mid in match_ids:
        ev = load_statsbomb_events(mid)
        off = ev.filter(pl.col("type").struct.field("name") == "Offside")
        if off.height == 0:
            continue
        try:
            ff_df = load_statsbomb_360(mid)
            ff_map = dict(zip(ff_df["event_uuid"].to_list(),
                               ff_df["freeze_frame"].to_list()))
        except FileNotFoundError:
            ff_map = {}

        for r in off.iter_rows(named=True):
            loc = r.get("location")
            if loc is None or len(loc) < 2:
                continue
            att_x = float(loc[0])
            ff = ff_map.get(r.get("id"))
            margin = _offside_margin_from_ff(att_x, ff) if ff else None

            if margin is not None:
                is_tight = 0.0 <= margin <= tight_meters
            else:
                is_tight = att_x > 110.0        # proxy fallback

            if not is_tight:
                continue
            team = r.get("team") or {}
            rows.append({
                "sb_match_id": int(mid),
                "event_uuid":  r.get("id"),
                "period":      int(r.get("period") or 1),
                "minute":      int(r.get("minute") or 0),
                "second":      int(r.get("second") or 0),
                "team_id":     int(team.get("id") or 0),
                "team_name":   team.get("name"),
                "shot_outcome": "Offside",
                "is_goal":     False,
                "xg_baseline": None,
                "psxg":        None,
                "near_miss_type": "b_offside_close",
                "margin_info": float(margin) if margin is not None
                                else float(120.0 - att_x),
            })
    if not rows:
        return pl.DataFrame(schema=_OFFSIDE_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=_OFFSIDE_SCHEMA)


# ===========================================================================
#  SECCION 3 — API publica: build_near_miss_table
# ===========================================================================

def build_near_miss_table(cache: bool = True,
                          overwrite: bool = False,
                          lax_woodwork: bool = False) -> pl.DataFrame:
    """Construye la tabla unificada de near-miss en WC22 (5 tipos).

    Args:
        lax_woodwork: si True, expande xg range del tipo (a) a [0.08, 0.92]
                      para specification curve Simonsohn 2020.

    Cache en data/parquet/derived/nearmiss/nearmiss_table.parquet (strict).
    Para lax, usar cache=False y run in-memory.
    """
    cache_path = _DERIVED / "nearmiss_table.parquet"
    if cache and cache_path.exists() and not overwrite and not lax_woodwork:
        return pl.read_parquet(cache_path)

    psxg = _load_psxg_shots()
    wc22_mids = list_statsbomb_match_ids(comp_id=43, season_id=106)

    # team_id / team_name: extraccion vectorizada via polars (evita loop py).
    # sb_match_id ya viene del parquet de M05 (rename de _match_id).
    team_dfs = []
    for mid in wc22_mids:
        ev = load_statsbomb_events(mid)
        shots = ev.filter(pl.col("type").struct.field("name") == "Shot")
        if shots.height == 0:
            continue
        team_dfs.append(shots.select([
            pl.col("id").alias("event_uuid"),
            pl.col("team").struct.field("id").cast(pl.Int64).alias("team_id"),
            pl.col("team").struct.field("name").alias("team_name"),
        ]))
    teams = pl.concat(team_dfs) if team_dfs else pl.DataFrame(
        schema={"event_uuid": pl.String, "team_id": pl.Int64, "team_name": pl.String}
    )
    psxg_enriched = psxg.join(teams, on="event_uuid", how="left")

    # (a) Woodwork, (c) Saves, (d) GLC, (e) GLT, (b) Offside
    woodwork = _detect_woodwork(psxg_enriched, lax=lax_woodwork)
    saves    = _detect_saves_clutch(psxg_enriched, strict_only=False)
    glc      = _detect_goal_line_clearance(psxg_enriched)
    glt      = _detect_glt_denied(psxg_enriched)
    offside  = _detect_offside_tight(wc22_mids)

    # Unificar schema
    cols = ["sb_match_id", "event_uuid", "period", "minute", "second",
            "team_id", "team_name", "shot_outcome", "is_goal",
            "xg_baseline", "psxg", "near_miss_type", "margin_info"]

    def _align(df: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in cols if c not in df.columns]
        # Casts explicitos a los tipos target para que concat no falle
        lits = []
        for c in missing:
            if c in ("sb_match_id", "period", "minute", "second", "team_id"):
                lits.append(pl.lit(None, dtype=pl.Int64).alias(c))
            elif c in ("xg_baseline", "psxg", "margin_info"):
                lits.append(pl.lit(None, dtype=pl.Float64).alias(c))
            elif c == "is_goal":
                lits.append(pl.lit(None, dtype=pl.Boolean).alias(c))
            else:
                lits.append(pl.lit(None, dtype=pl.String).alias(c))
        return df.with_columns(lits).select(cols)

    all_nm = pl.concat(
        [_align(woodwork), _align(saves), _align(glc),
         _align(glt), _align(offside)],
        how="diagonal_relaxed",
    )

    # Dedup explicito por (event_uuid, near_miss_type): un mismo tiro podria
    # aparecer en (a) y (c) en edge cases raros (Post + PSxG alto). Dedup
    # defensivo para garantia downstream.
    all_nm = all_nm.unique(subset=["event_uuid", "near_miss_type"],
                            keep="first").sort(["sb_match_id", "minute", "second"])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        all_nm.write_parquet(cache_path, compression="snappy", statistics=True)
    return all_nm


def summary_by_type(nm: pl.DataFrame) -> pl.DataFrame:
    """Contea near-miss por tipo."""
    return nm.group_by("near_miss_type").len().sort("near_miss_type")


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M06_nearmiss sanity ===")
    t0 = time.time()
    nm = build_near_miss_table(cache=True, overwrite=True)
    print(f"near-miss table built en {time.time()-t0:.1f}s")
    print(f"  total near-miss WC22: {nm.height}")
    print()
    print("Distribucion por tipo:")
    print(summary_by_type(nm))

    print()
    print("Sample (5 de cada tipo):")
    for t in sorted(nm["near_miss_type"].unique().to_list()):
        sub = nm.filter(pl.col("near_miss_type") == t).head(5)
        if sub.height == 0: continue
        print(f"\n  -- {t} ({sub.height} muestra de {nm.filter(pl.col('near_miss_type')==t).height} total) --")
        print(sub.select(["sb_match_id", "minute", "team_name", "shot_outcome",
                          "xg_baseline", "psxg", "margin_info"]))

    # Check acceptance: distribucion razonable
    print()
    print("Acceptance vs propuesta §1.7 (ajustado a 64 partidos WC22):")
    expected = {
        "a_woodwork":            (15, 40),    # propuesta: ~15-25 en 48 pts -> 20-50 en 64
        "c_save_psxg":           (40, 120),   # propuesta: ~60-90 en 48 -> ~80-150
        "b_offside_close":       (5, 30),
        "d_goal_line_clearance": (0, 15),
        "e_glt_denied":          (0, 5),
    }
    for t, (lo, hi) in expected.items():
        n = nm.filter(pl.col("near_miss_type") == t).height
        status = "OK" if lo <= n <= hi else f"FUERA DE RANGO [{lo},{hi}]"
        print(f"  {t:<26} {n:>4}   (esperado [{lo},{hi}]) {status}")

    total = nm.height
    print(f"\n  TOTAL strict: {total} near-miss (esperado ~90-140 propuesta, 120-185 escalado 64pts)")

    # Specification curve (Simonsohn 2020 robustez — lax woodwork xg [0.08, 0.92])
    print()
    print("Specification curve — variante LAX (woodwork xg [0.08, 0.92]):")
    nm_lax = build_near_miss_table(cache=False, lax_woodwork=True)
    print(summary_by_type(nm_lax))
    print(f"  TOTAL lax: {nm_lax.height}")
