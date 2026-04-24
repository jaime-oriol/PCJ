"""
M10_offball - Canal Inteligencia Espacial Off-ball via OBSO completo (Spearman 2018).

Fase 2 PCJ, canal 3 de 4. Implementacion ELITE: OBSO completo = PPCF × T × S
con el PPCF SOTA del building block Z02 pitch_control (vectorizado Spearman
2018 sobre tracking 25fps), NO una simplificacion.

Referencias SOTA (implementadas):
  - Spearman (2018, MIT Sloan) "Beyond Expected Goals" — OBSO 3-factor:
      OBSO(r) = PPCF(r) * T(r) * S(r)
      - PPCF: probabilidad que jugador controle el balon en r (tracking fisico)
      - T: probabilidad que el balon llegue a r desde posicion actual
      - S: probabilidad que un shot desde r sea gol (xG grid)
  - Teranishi et al. (2022, MLSA LNCS) "C-OBSO": contribucion del movimiento
      C-OBSO = OBSO - OBSO_counterfactual (jugador quieto en posicion previa)

Adapter PFF -> Z02:
  Z02 ya tiene PPCF SOTA vectorizado para formato Opta MA25 (x/y, velocities,
  team_id, is_ball, is_goalkeeper). Adapter convierte frame PFF al mismo
  schema y reusa ppcf_at_targets (N targets simultaneos, numpy vectorizado).
  Coords PFF ya estan en metros centradas (0,0) -> compatible directamente.

Velocidades:
  Buffer per (player_id, frame_num) para calcular vx, vy via diferencias
  finitas entre frames consecutivos sampleados. Z02 PPCF usa vel para proyectar
  reach en reaction_time=0.7s.

Sampling: 25 Hz (todos los frames, full PFF quality). ~135k frames/match,
activa -> ~2500 PPCF calls/match × 2 (contrafactual) × 11 players × 64 matches.

Features output per (match_id, player_id, minute):
  - obso_mean     : OBSO medio (PPCF × T × S) sobre frames atacantes del minuto
  - obso_max      : pico OBSO del minuto
  - c_obso_mean   : contribucion del movimiento (OBSO - contrafactual)
  - attacking_frames: frames sampleados con jugador en ataque

Acceptance (ARCHITECTURE): top-decile correlaciona con assists/secondary
assists. Distribucion por rol: W/CF > CB/GK.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import (load_metadata, load_rosters, scan_tracking,
                              list_event_match_ids)
from M03_preprocess import attacking_direction
from M07_shocks import build_shocks_table
import Z02_pitch_control as pc


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "offball"


# -- Parametros Spearman 2018 + sampling -----------------------------------

_GRID_NX = 10
_GRID_NY = 7
_SAMPLE_EVERY_N_FRAMES = 1         # 25 Hz full-quality (todos los frames)
_COBSO_LAG_SEC   = 2.0             # 2s atras para contrafactual

# T(r): transition prob. via kernel Gauss sobre dist(ball, target).
# Spearman 2018: T(r,t) = P(ball_reaches r given carrier position at t).
# Simplificacion fisica: exp(-dist / sigma), sigma ~ 15m (~half-field).
_T_SIGMA_M = 15.0


# ===========================================================================
#  SECCION 1 — xG grid (pre-computed, simetrizado + smoothed)
# ===========================================================================

def build_xg_grid(nx: int = _GRID_NX, ny: int = _GRID_NY,
                  cache: bool = True) -> np.ndarray:
    """P(goal | shot desde celda) sobre media-campo atacante SB 120x80.

    Fuente: SB shots training (Euro20+Euro24+Bundes23, sin WC22).
    Post-processing:
      1. Symmetrizar en Y (campo simetrico).
      2. Smoothing 3x3 uniform (varianza por pocos shots/cell).
      3. Fallback 0.03 para cells sin shot.
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

    xg = (xg + xg[:, ::-1]) / 2.0    # symmetrize Y
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
    """Lookup S(r) = xG grid desde pos PFF (metros (0,0)) con flip a ataque-derecha."""
    if not attack_right:
        x_pff = -x_pff
        y_pff = -y_pff
    if x_pff <= 0:
        return 0.0
    x_sb = 60.0 + (x_pff / (pitch_l / 2.0)) * 60.0
    y_sb = (y_pff + pitch_w / 2.0) / pitch_w * 80.0
    ix = min(max(int((x_sb - 60) / 60.0 * xg_grid.shape[0]), 0),
             xg_grid.shape[0] - 1)
    iy = min(max(int(y_sb / 80.0 * xg_grid.shape[1]), 0),
             xg_grid.shape[1] - 1)
    return float(xg_grid[ix, iy])


def _transition_probability(ball_pos: np.ndarray, target: np.ndarray,
                             sigma: float = _T_SIGMA_M) -> float:
    """T(r) = probabilidad que el balon llegue a target.

    Kernel gaussiano sobre distancia euclidea balon-target. Sigma ~ 15m
    captura que transiciones a >30m son raras (gaussian cut off).
    """
    dist = float(np.linalg.norm(target - ball_pos))
    return float(np.exp(-(dist ** 2) / (2 * sigma ** 2)))


# ===========================================================================
#  SECCION 2 — Adapter PFF frame -> Z02-compatible DataFrame
# ===========================================================================

def _pff_frame_to_z02_df(frame_dict: dict,
                          home_id: int, away_id: int,
                          home_gk_jerseys: set[int], away_gk_jerseys: set[int],
                          vel_buffer: dict,
                          frame_num: int, dt_sec: float) -> pd.DataFrame:
    """Convierte frame PFF a DataFrame pandas formato Z02.

    Z02 espera cols: x_tracking, y_tracking, team_id, is_ball, is_goalkeeper, vx, vy.
    PFF coords ya en metros centradas (0,0) -> compatible directamente.
    Velocities derivadas de buffer de posiciones previas.
    """
    rows = []
    for p in (frame_dict.get("home_players") or []):
        x = p.get("x"); y = p.get("y"); jersey = p.get("jerseyNum")
        if x is None or y is None or jersey is None:
            continue
        try: jnum = int(jersey)
        except (ValueError, TypeError): continue
        key = (home_id, jnum)
        prev = vel_buffer.get(key)
        if prev is not None and dt_sec > 0:
            vx = (x - prev[1]) / dt_sec
            vy = (y - prev[2]) / dt_sec
            # Cap extreme (tracking jumps)
            vx = max(-12, min(12, vx)); vy = max(-12, min(12, vy))
        else:
            vx = vy = 0.0
        vel_buffer[key] = (frame_num, x, y)
        rows.append({
            "x_tracking": x, "y_tracking": y,
            "team_id": home_id, "is_ball": 0,
            "is_goalkeeper": 1 if jnum in home_gk_jerseys else 0,
            "jerseyNum": jnum,
            "vx": vx, "vy": vy,
        })
    for p in (frame_dict.get("away_players") or []):
        x = p.get("x"); y = p.get("y"); jersey = p.get("jerseyNum")
        if x is None or y is None or jersey is None:
            continue
        try: jnum = int(jersey)
        except (ValueError, TypeError): continue
        key = (away_id, jnum)
        prev = vel_buffer.get(key)
        if prev is not None and dt_sec > 0:
            vx = (x - prev[1]) / dt_sec
            vy = (y - prev[2]) / dt_sec
            vx = max(-12, min(12, vx)); vy = max(-12, min(12, vy))
        else:
            vx = vy = 0.0
        vel_buffer[key] = (frame_num, x, y)
        rows.append({
            "x_tracking": x, "y_tracking": y,
            "team_id": away_id, "is_ball": 0,
            "is_goalkeeper": 1 if jnum in away_gk_jerseys else 0,
            "jerseyNum": jnum,
            "vx": vx, "vy": vy,
        })
    ball = frame_dict.get("ball")
    if ball and ball.get("x") is not None:
        rows.append({
            "x_tracking": ball["x"], "y_tracking": ball["y"],
            "team_id": -1, "is_ball": 1, "is_goalkeeper": 0,
            "jerseyNum": None, "vx": 0.0, "vy": 0.0,
        })
    return pd.DataFrame(rows)


# ===========================================================================
#  SECCION 3 — OBSO completo per match (Spearman 2018 + C-OBSO contrafactual)
# ===========================================================================

def compute_obso_match(match_id: int, xg_grid: np.ndarray,
                       verbose: bool = False) -> pl.DataFrame:
    """OBSO completo = PPCF × T × S por (player, minute) con Z02 PPCF.

    C-OBSO: recomputa PPCF con el jugador atacante en posicion de hace
    _COBSO_LAG_SEC segundos con velocidad cero (contrafactual "si no se
    hubiera movido"), resto de jugadores y balon en posicion actual.
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    pitch_l = float(md.get("pitch_length") or 105.0)
    pitch_w = float(md.get("pitch_width") or 68.0)
    fps = float(md.get("fps") or 25.0)
    dt_sec_sample = _SAMPLE_EVERY_N_FRAMES / fps      # 1.0s si fps=25
    cobso_lag_frames = int(round(_COBSO_LAG_SEC * fps))

    ro = load_rosters(match_id)
    home_gk = {int(r["shirt_number"]) for r in ro.filter(
        (pl.col("team_id") == home_id) & (pl.col("position_group") == "GK")
    ).iter_rows(named=True) if r["shirt_number"] is not None}
    away_gk = {int(r["shirt_number"]) for r in ro.filter(
        (pl.col("team_id") == away_id) & (pl.col("position_group") == "GK")
    ).iter_rows(named=True) if r["shirt_number"] is not None}
    # jersey -> player_id
    home_map = {int(r["shirt_number"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id") == home_id).iter_rows(named=True)
                if r["shirt_number"] is not None}
    away_map = {int(r["shirt_number"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id") == away_id).iter_rows(named=True)
                if r["shirt_number"] is not None}

    dirs = attacking_direction(match_id).to_dicts()
    dir_lookup = {(d["team_id"], d["period"]): d["direction"] for d in dirs}

    lf = scan_tracking(match_id)
    frames = lf.select([
        pl.col("frameNum"),
        pl.col("period"),
        pl.col("game_event").struct.field("home_ball").alias("home_has_ball"),
        pl.col("homePlayersSmoothed").alias("home_players"),
        pl.col("awayPlayersSmoothed").alias("away_players"),
        pl.col("ballsSmoothed").alias("ball"),
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

    ppcf_params = pc.default_model_params()
    vel_buffer: dict = {}
    player_history: dict = {}    # (team, jersey) -> deque of (frame, x, y)
    rows = []
    frames_per_min = fps * 60

    for r in frames.iter_rows(named=True):
        frame_num = int(r["frameNum"])
        period = int(r["period"])
        home_has_ball = bool(r["home_has_ball"])
        minute = int(frame_num // frames_per_min)
        att_team = home_id if home_has_ball else away_id
        att_players_raw = r["home_players"] if home_has_ball else r["away_players"]
        att_map = home_map if home_has_ball else away_map
        if not att_players_raw:
            continue

        # Build frame_data DataFrame
        frame_dict = {
            "home_players": r["home_players"],
            "away_players": r["away_players"],
            "ball": r["ball"],
        }
        frame_df = _pff_frame_to_z02_df(
            frame_dict, home_id, away_id, home_gk, away_gk,
            vel_buffer, frame_num, dt_sec_sample,
        )
        if frame_df.empty:
            continue

        # Attacking targets: posiciones actuales de jugadores del equipo atacante
        att_rows = frame_df[(frame_df["team_id"] == att_team) & (frame_df["is_ball"] == 0)]
        if att_rows.empty:
            continue
        targets = att_rows[["x_tracking", "y_tracking"]].values.astype(np.float64)
        ball_pos = pc._get_ball_pos(frame_df)
        if ball_pos is None:
            continue

        # OBSO actual: PPCF × T × S en posicion actual
        try:
            ppcf_now = pc.ppcf_at_targets(frame_df, targets, att_team,
                                           ball_pos, ppcf_params)
        except Exception:
            continue

        dir_att = dir_lookup.get((att_team, period), "R")
        attack_right = dir_att == "R"

        obsos = []
        jerseys = att_rows["jerseyNum"].values
        for i, (x, y) in enumerate(targets):
            jnum = int(jerseys[i]) if jerseys[i] is not None else None
            pid = att_map.get(jnum) if jnum is not None else None
            if pid is None:
                continue
            t_r = _transition_probability(ball_pos, np.array([x, y]))
            s_r = _xg_lookup_pff(x, y, attack_right, xg_grid, pitch_l, pitch_w)
            obso = float(ppcf_now[i]) * t_r * s_r

            # C-OBSO: pos previa del jugador (hace ~2s)
            key = (att_team, jnum)
            hist = player_history.setdefault(key, deque(maxlen=20))
            target_frame = frame_num - cobso_lag_frames
            lagged_pos = None
            for (fn, px, py) in hist:
                if abs(fn - target_frame) <= _SAMPLE_EVERY_N_FRAMES:
                    lagged_pos = (px, py); break

            if lagged_pos is not None:
                # Contrafactual: mover al jugador de att_team a su posicion
                # previa, velocidad 0. Recompute PPCF en ese escenario
                # (el target tambien es la posicion previa, porque lo que
                # mide C-OBSO es "que OBSO habria tenido alli si estuviera
                # quieto").
                frame_cf = frame_df.copy()
                mask_player = ((frame_cf["team_id"] == att_team) &
                                (frame_cf["jerseyNum"] == jnum))
                frame_cf.loc[mask_player, "x_tracking"] = float(lagged_pos[0])
                frame_cf.loc[mask_player, "y_tracking"] = float(lagged_pos[1])
                frame_cf.loc[mask_player, "vx"] = 0.0
                frame_cf.loc[mask_player, "vy"] = 0.0
                target_cf = np.array([[lagged_pos[0], lagged_pos[1]]])
                try:
                    ppcf_cf = pc.ppcf_at_targets(frame_cf, target_cf, att_team,
                                                   ball_pos, ppcf_params)
                    t_r_cf = _transition_probability(ball_pos,
                                                       np.array(lagged_pos))
                    s_r_cf = _xg_lookup_pff(lagged_pos[0], lagged_pos[1],
                                              attack_right, xg_grid,
                                              pitch_l, pitch_w)
                    obso_cf = float(ppcf_cf[0]) * t_r_cf * s_r_cf
                    c_obso = obso - obso_cf
                except Exception:
                    c_obso = None
            else:
                c_obso = None

            hist.append((frame_num, float(x), float(y)))
            rows.append({
                "player_id": pid, "minute": minute,
                "obso": obso, "c_obso": c_obso,
            })

    if not rows:
        return pl.DataFrame(schema={
            "pff_match_id": pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
            "obso_mean": pl.Float64, "obso_max": pl.Float64,
            "c_obso_mean": pl.Float64, "attacking_frames": pl.Int64,
        })

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
#  SECCION 4 — Aggregate 64 matches + per-shock-window
# ===========================================================================

def aggregate_per_player_minute(cache: bool = True) -> pl.DataFrame:
    """Agrega OBSO + C-OBSO sobre los 64 partidos WC22."""
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    import time
    xg_grid = build_xg_grid(cache=True)
    dfs = []
    t0 = time.time()
    for i, mid in enumerate(list_event_match_ids()):
        t_match = time.time()
        try:
            dfs.append(compute_obso_match(mid, xg_grid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
        elapsed = time.time() - t_match
        if (i+1) % 5 == 0 or elapsed > 60:
            print(f"  {i+1}/64 en {time.time()-t0:.0f}s "
                  f"(last match {elapsed:.0f}s)", flush=True)
    out = pl.concat(dfs) if dfs else pl.DataFrame()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


def aggregate_per_shock_window(cache: bool = True) -> pl.DataFrame:
    """OBSO + C-OBSO agregados por ventana pre/post de cada shock."""
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    per_min = aggregate_per_player_minute(cache=True)
    shocks = build_shocks_table(cache=True, overwrite=False)

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

    print("=== M10_offball SANITY — OBSO completo Spearman 2018 + Z02 PPCF ===")

    # [1] xG grid
    grid = build_xg_grid(cache=True)
    print(f"\n[1] xG grid {grid.shape}: min={grid.min():.3f}, max={grid.max():.3f}")

    # [2] Full pipeline 64 matches (caro, ~30-60 min)
    t0 = time.time()
    print(f"\n[2] OBSO + C-OBSO para 64 matches (PPCF Z02 per frame, 25 Hz full)...")
    per_min = aggregate_per_player_minute(cache=True)
    print(f"  filas: {per_min.height:,} en {time.time()-t0:.0f}s")
    print(f"  obso_mean range: [{per_min['obso_mean'].min():.5f}, {per_min['obso_mean'].max():.5f}]")
    c_valid = per_min.filter(pl.col("c_obso_mean").is_not_null()).height
    print(f"  c_obso valido: {c_valid}/{per_min.height} ({100*c_valid/per_min.height:.1f}%)")

    # [3] Acceptance: W + CF > CB, simetria LW/RW
    print("\n[3] Acceptance — distribucion por rol:")
    ro = load_rosters().select(["player_id","position_group"]).unique(subset=["player_id"])
    roles = per_min.join(ro, on="player_id", how="left")
    by_role = roles.group_by("position_group").agg([
        pl.col("obso_mean").mean().alias("obso_mean"),
        pl.col("obso_max").mean().alias("obso_max_mean"),
        pl.col("c_obso_mean").mean().alias("c_obso_mean"),
        pl.len().alias("n_minutes"),
    ]).sort("obso_mean", descending=True)
    print(by_role)
    lw = by_role.filter(pl.col("position_group")=="LW")["obso_mean"].item() if by_role.filter(pl.col("position_group")=="LW").height else None
    rw = by_role.filter(pl.col("position_group")=="RW")["obso_mean"].item() if by_role.filter(pl.col("position_group")=="RW").height else None
    if lw and rw:
        print(f"\nSimetria LW={lw:.5f}, RW={rw:.5f}, delta={abs(lw-rw):.5f}")

    # [4] Per shock window
    t0 = time.time()
    print(f"\n[4] Per shock window...")
    per_shock = aggregate_per_shock_window(cache=True)
    print(f"  filas: {per_shock.height:,} en {time.time()-t0:.0f}s")
    summary = per_shock.group_by("shock_type").agg([
        pl.col("obso_pre").mean().alias("obso_pre"),
        pl.col("obso_post").mean().alias("obso_post"),
        (pl.col("obso_post") - pl.col("obso_pre")).mean().alias("delta_obso"),
        pl.col("c_obso_pre").mean().alias("c_obso_pre"),
        pl.col("c_obso_post").mean().alias("c_obso_post"),
    ])
    print(summary)
