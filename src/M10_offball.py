"""
M10_offball - Canal Inteligencia Espacial Off-ball via OBSO + C-OBSO simplificado.

Fase 2 PCJ, canal 3 de 4. Valora el peligro off-ball que un jugador GENERA por
su posicionamiento (OBSO) y por su movimiento (C-OBSO).

Referencias SOTA:
  - Spearman (2018, MIT Sloan) "Beyond Expected Goals": OBSO = P(control) *
    P(ball_reaches) * P(shot_goal). Implementacion fisica sobre tracking.
  - Teranishi et al. (2022, MLSA LNCS): C-OBSO = OBSO - OBSO_counterfactual
    (si no se hubiera movido). Aisla contribucion del movimiento.

Implementacion pragmatica (factible con 64 partidos x 180k frames x 22 players):
  - **xG grid** pre-computado desde SB shots training (Euro20+Euro24+Bundes23)
    sobre media-campo atacante (60-120, 0-80 coords SB). Rejilla 10x7 =
    P(goal | zona).
  - **OBSO position-based** (simplificacion del 3-factor Spearman): tomamos
    solo S(r) = xG_grid_lookup(player_pos). P(control) y P(ball_reaches) se
    omiten — esto reduce expresividad pero mantiene la variabilidad principal
    (posicion en zona peligrosa). Para un refuerzo completo sobre tracking
    25fps usar Z02 pitch_control con grid completo (reservado para PCJ v2).
  - **C-OBSO delta**: diff entre OBSO actual y OBSO en frame t - 2s
    (aproxima contrafactual "si hubiera estado quieto"). Captura aporte del
    movimiento reciente.
  - Sample cada 25 frames (1 Hz) para factibility. 7200 samples/match x 64 =
    ~460k lookups.

Features output per (match_id, player_id, minute):
  - obso_mean      : OBSO promedio sobre frames del minuto con jugador en ataque
  - obso_max       : pico OBSO del minuto (posicion mas peligrosa alcanzada)
  - c_obso_mean    : movimiento off-ball promedio (delta positivo = se movio
                     hacia zona mas peligrosa)
  - attacking_frames: n frames en posesion propia (proxy de involvement)

Acceptance (ARCHITECTURE): top-decile correlaciona positivamente con assists
y secondary assists (proxyable aqui con n_actions ofensivas M08 en ventanas
post-shock).

Depende de: M01 (tracking, metadata), M03 (attacking_direction), M05 (training
shots para xG grid), M07 (shocks), M08 (mapping SB->PFF player).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import (load_metadata, load_rosters, scan_tracking,
                              list_event_match_ids)
from M03_preprocess import attacking_direction
from M07_shocks import build_shocks_table
import M08_ataque as atk


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "offball"


# -- Parametros -------------------------------------------------------------

_GRID_NX = 10      # x (half-pitch attacking: 60-120 SB)
_GRID_NY = 7       # y (0-80 SB)
_SAMPLE_EVERY_N_FRAMES = 25   # 1 Hz effective sampling
_COBSO_LAG_SEC = 2.0          # 2 sec atras para counterfactual


# ===========================================================================
#  SECCION 1 — xG grid (pre-computed)
# ===========================================================================

def build_xg_grid(nx: int = _GRID_NX, ny: int = _GRID_NY,
                  cache: bool = True) -> np.ndarray:
    """P(goal | shot desde celda) sobre media-campo atacante (60-120 x 0-80 SB).

    Fuente: SB shots training (Euro20+Euro24+Bundes23, sin WC22).
    Post-processing:
      1. Symmetrizar en Y (el campo es simetrico y=0 <-> y=80).
      2. Smoothing 3x3 uniform (reduce varianza por cells con pocos shots).
      3. Fallback 0.03 para cells sin data.
    Fix asimetria LW/RW detectada al samplear solo 3545 shots -> 50 shots/cell.
    """
    cache_path = _DERIVED / "xg_grid.npy"
    if cache and cache_path.exists():
        return np.load(cache_path)

    from M05_psxg import build_training_shots
    shots = build_training_shots()
    xg = np.zeros((nx, ny), dtype=np.float32)
    cnt = np.zeros((nx, ny), dtype=np.int32)
    for r in shots.iter_rows(named=True):
        x, y = r["x"], r["y"]
        if x < 60 or x > 120 or y < 0 or y > 80:
            continue
        ix = min(int((x - 60) / 60.0 * nx), nx - 1)
        iy = min(int(y / 80.0 * ny), ny - 1)
        cnt[ix, iy] += 1
        xg[ix, iy] += r["_label"]
    mask = cnt > 0
    xg[mask] /= cnt[mask]
    xg[~mask] = 0.03

    # Symmetrizar en Y (asumimos simetria del campo)
    xg = (xg + xg[:, ::-1]) / 2.0

    # Smoothing 3x3 uniform kernel (ignora bordes)
    smoothed = np.copy(xg)
    for i in range(nx):
        for j in range(ny):
            i0, i1 = max(0, i-1), min(nx, i+2)
            j0, j1 = max(0, j-1), min(ny, j+2)
            smoothed[i, j] = xg[i0:i1, j0:j1].mean()
    xg = smoothed

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, xg)
    return xg


def _xg_lookup_pff(x_pff: float, y_pff: float, attack_right: bool,
                    xg_grid: np.ndarray,
                    pitch_l: float = 105.0,
                    pitch_w: float = 68.0) -> float:
    """Lookup xG grid desde coords PFF (metros centrados en 0,0).

    Convierte a coords SB 120x80 media-campo atacante.
    Si el jugador no esta en media-campo atacante, retorna 0.
    """
    # Post-flip: transformar a "ataque hacia x_pff positivo" (normalizar)
    if not attack_right:
        x_pff = -x_pff
        y_pff = -y_pff
    # Atacar a la derecha: media-campo es x_pff > 0
    if x_pff <= 0:
        return 0.0
    # Convertir PFF->SB coords (solo media-campo)
    # PFF x in [0, pitch_l/2] -> SB x in [60, 120]
    x_sb = 60.0 + (x_pff / (pitch_l / 2.0)) * 60.0
    # PFF y in [-pitch_w/2, pitch_w/2] -> SB y in [0, 80]
    y_sb = (y_pff + pitch_w / 2.0) / pitch_w * 80.0
    ix = min(max(int((x_sb - 60) / 60.0 * xg_grid.shape[0]), 0),
             xg_grid.shape[0] - 1)
    iy = min(max(int(y_sb / 80.0 * xg_grid.shape[1]), 0),
             xg_grid.shape[1] - 1)
    return float(xg_grid[ix, iy])


# ===========================================================================
#  SECCION 2 — OBSO + C-OBSO per match
# ===========================================================================

def compute_obso_match(match_id: int, xg_grid: np.ndarray) -> pl.DataFrame:
    """Calcula OBSO + C-OBSO por (player, minute) en 1 partido.

    Estrategia:
      - Scan tracking lazy, sample cada 25 frames (1 Hz).
      - Para cada frame con posesion conocida, identificar equipo atacante.
      - Para cada jugador DEL EQUIPO ATACANTE (solo ellos tienen sentido OBSO):
          * OBSO = xG_grid_lookup(pos, attack_direction)
          * C-OBSO = OBSO - OBSO(pos hace 2s)  (si mismo jugador tenia entrada)
      - Agregar por (player_id, minute) con mean, max.
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    pitch_l = float(md.get("pitch_length") or 105.0)
    pitch_w = float(md.get("pitch_width") or 68.0)
    fps = float(md.get("fps") or 25.0)
    cobso_lag_frames = int(_COBSO_LAG_SEC * fps)

    # Mapping jersey -> player_id
    ro = load_rosters(match_id).select(["team_id", "player_id", "shirt_number"])
    home_map = {int(r["shirt_number"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id") == home_id).iter_rows(named=True)
                if r["shirt_number"] is not None}
    away_map = {int(r["shirt_number"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id") == away_id).iter_rows(named=True)
                if r["shirt_number"] is not None}

    # Attacking direction por (team, period)
    dirs = attacking_direction(match_id).to_dicts()
    dir_lookup = {(d["team_id"], d["period"]): d["direction"] for d in dirs}

    # Load tracking sampled
    lf = scan_tracking(match_id)
    frames = lf.select([
        pl.col("frameNum"),
        pl.col("period"),
        pl.col("game_event").struct.field("home_ball").alias("home_has_ball"),
        pl.col("homePlayersSmoothed").alias("home_players"),
        pl.col("awayPlayersSmoothed").alias("away_players"),
    ]).filter(
        (pl.col("home_has_ball").is_not_null()) &
        (pl.col("frameNum") % _SAMPLE_EVERY_N_FRAMES == 0)
    ).collect()

    if frames.height == 0:
        return pl.DataFrame(schema={
            "pff_match_id": pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
            "obso_mean": pl.Float64, "obso_max": pl.Float64,
            "c_obso_mean": pl.Float64, "attacking_frames": pl.Int64,
        })

    frames_per_min = fps * 60
    # Player history: (player_id, frameNum) -> (obso, x, y) for C-OBSO computation
    player_obso_buffer: dict[int, list[tuple[int, float, float, float]]] = {}

    rows = []
    for r in frames.iter_rows(named=True):
        frame_num = int(r["frameNum"])
        period = int(r["period"])
        home_has_ball = bool(r["home_has_ball"])
        minute = int(frame_num // frames_per_min)

        if home_has_ball and r["home_players"]:
            att_team, players = home_id, r["home_players"]
            pmap = home_map
        elif (not home_has_ball) and r["away_players"]:
            att_team, players = away_id, r["away_players"]
            pmap = away_map
        else:
            continue

        dir_att = dir_lookup.get((att_team, period), "R")
        attack_right = dir_att == "R"

        for p in players:
            x = p.get("x"); y = p.get("y"); jersey = p.get("jerseyNum")
            if x is None or y is None or jersey is None: continue
            try: jnum = int(jersey)
            except (ValueError, TypeError): continue
            pid = pmap.get(jnum)
            if pid is None: continue
            obso = _xg_lookup_pff(x, y, attack_right, xg_grid, pitch_l, pitch_w)
            # C-OBSO via buffer
            buf = player_obso_buffer.setdefault(pid, [])
            # clean old frames (older than cobso_lag_frames * 3)
            buf = [b for b in buf if frame_num - b[0] <= cobso_lag_frames * 3]
            player_obso_buffer[pid] = buf
            # find closest frame to frame_num - cobso_lag_frames
            target = frame_num - cobso_lag_frames
            lagged_obso = None
            for b in buf:
                if abs(b[0] - target) <= _SAMPLE_EVERY_N_FRAMES:
                    # OBSO hypothetical si se hubiera quedado quieto en pos anterior:
                    # recompute xg_lookup at lagged position, with current direction
                    lagged_obso = _xg_lookup_pff(b[1], b[2], attack_right, xg_grid,
                                                  pitch_l, pitch_w)
                    break
            c_obso = (obso - lagged_obso) if lagged_obso is not None else None
            buf.append((frame_num, x, y, obso))
            rows.append({
                "player_id": pid,
                "minute":    minute,
                "obso":      obso,
                "c_obso":    c_obso,
            })

    if not rows:
        return pl.DataFrame(schema={
            "pff_match_id": pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
            "obso_mean": pl.Float64, "obso_max": pl.Float64,
            "c_obso_mean": pl.Float64, "attacking_frames": pl.Int64,
        })

    # Forzar schema explicito para evitar falla de inferencia con Nulls iniciales.
    df = pl.DataFrame(rows, schema={
        "player_id": pl.Int64, "minute": pl.Int64,
        "obso": pl.Float64, "c_obso": pl.Float64,
    })
    agg = df.group_by(["player_id", "minute"]).agg([
        pl.col("obso").mean().alias("obso_mean"),
        pl.col("obso").max().alias("obso_max"),
        pl.col("c_obso").mean().alias("c_obso_mean"),
        pl.len().alias("attacking_frames"),
    ]).with_columns(
        pl.lit(match_id).cast(pl.Int64).alias("pff_match_id"),
    ).select(["pff_match_id", "player_id", "minute",
              "obso_mean", "obso_max", "c_obso_mean", "attacking_frames"])
    return agg


# ===========================================================================
#  SECCION 3 — Aggregate all matches + per shock window
# ===========================================================================

def aggregate_per_player_minute(cache: bool = True) -> pl.DataFrame:
    """Agrega OBSO + C-OBSO para los 64 partidos WC22."""
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    import time
    xg_grid = build_xg_grid(cache=True)
    dfs = []
    t0 = time.time()
    for i, mid in enumerate(list_event_match_ids()):
        try:
            dfs.append(compute_obso_match(mid, xg_grid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
        if (i+1) % 10 == 0:
            print(f"  {i+1}/64 en {time.time()-t0:.1f}s", flush=True)
    out = pl.concat(dfs) if dfs else pl.DataFrame()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


def aggregate_per_shock_window(cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma obso / c_obso en pre/post windows."""
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    per_min = aggregate_per_player_minute(cache=True)
    shocks = build_shocks_table(cache=True, overwrite=False)

    # per_min ya tiene pff_match_id y player_id (PFF). Alinear a shocks table.
    per_min = per_min.rename({"player_id": "pff_player_id",
                                "pff_match_id": "match_id"}).with_columns([
        pl.col("match_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64),
    ])

    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    joined = shocks_slim.join(per_min, on=["match_id", "pff_player_id"], how="left") \
                        .with_columns((pl.col("minute") * 60).alias("min_sec"))

    pre = joined.filter(
        (pl.col("min_sec") >= pl.col("window_pre_start")) &
        (pl.col("min_sec") < pl.col("window_pre_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("obso_mean").mean().alias("obso_pre"),
        pl.col("obso_max").max().alias("obso_max_pre"),
        pl.col("c_obso_mean").mean().alias("c_obso_pre"),
        pl.col("attacking_frames").sum().alias("att_frames_pre"),
    ])
    post = joined.filter(
        (pl.col("min_sec") >= pl.col("window_post_start")) &
        (pl.col("min_sec") <= pl.col("window_post_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("obso_mean").mean().alias("obso_post"),
        pl.col("obso_max").max().alias("obso_max_post"),
        pl.col("c_obso_mean").mean().alias("c_obso_post"),
        pl.col("attacking_frames").sum().alias("att_frames_post"),
    ])

    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    out = base.join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .join(post, on=["match_id","shock_id","pff_player_id","shock_type"], how="left")

    if cache:
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M10_offball sanity ===")

    # [1] xG grid
    t0 = time.time()
    grid = build_xg_grid(cache=True)
    print(f"\n[1] xG grid {grid.shape} en {time.time()-t0:.1f}s")
    print(f"  min/max: [{grid.min():.4f}, {grid.max():.4f}]")
    print(f"  cell central arco (ix=9, iy=3): xg={grid[9, 3]:.4f} (esperado >0.3)")
    print(f"  cell lejos (ix=0, iy=0): xg={grid[0, 0]:.4f} (esperado bajo)")

    # [2] Per-match OBSO
    t0 = time.time()
    print(f"\n[2] Computing OBSO for 64 matches...")
    per_min = aggregate_per_player_minute(cache=True)
    print(f"  filas: {per_min.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_min.columns}")
    print(f"  obso_mean range: [{per_min['obso_mean'].min():.4f}, {per_min['obso_mean'].max():.4f}]")
    if per_min.filter(pl.col("c_obso_mean").is_not_null()).height > 0:
        v = per_min.filter(pl.col("c_obso_mean").is_not_null())
        print(f"  c_obso_mean valido: {v.height}/{per_min.height}, "
              f"range [{v['c_obso_mean'].min():.4f}, {v['c_obso_mean'].max():.4f}]")

    # [3] Distribution por rol (acceptance: W + CF > CB)
    print(f"\n[3] Acceptance — OBSO por rol (W/CF > CB)")
    ro = load_rosters().select(["player_id","position_group"]).unique(subset=["player_id"])
    roles = per_min.join(ro, left_on="player_id", right_on="player_id", how="left")
    by_role = roles.group_by("position_group").agg([
        pl.col("obso_mean").mean().alias("mean_obso"),
        pl.col("obso_max").mean().alias("mean_obso_max"),
        pl.col("c_obso_mean").mean().alias("mean_c_obso"),
        pl.len().alias("n_minutes"),
    ]).sort("mean_obso", descending=True)
    print(by_role)

    # [4] Per shock window
    t0 = time.time()
    print(f"\n[4] Per shock window...")
    per_shock = aggregate_per_shock_window(cache=True)
    print(f"  filas: {per_shock.height:,} en {time.time()-t0:.1f}s")
    summary = per_shock.group_by("shock_type").agg([
        pl.col("obso_pre").mean().alias("obso_pre"),
        pl.col("obso_post").mean().alias("obso_post"),
        (pl.col("obso_post") - pl.col("obso_pre")).mean().alias("delta_obso"),
    ])
    print("  OBSO por shock_type:")
    print(summary)
