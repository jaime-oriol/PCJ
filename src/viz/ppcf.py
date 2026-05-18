"""ppcf - Superficie Pitch Control (PPCF, Spearman 2018) sobre el campo.

Z02_pitch_control (auditado) computa el PPCF; aqui se monta la malla del
campo, el adapter del frame de tracking PFF 25/30 Hz al schema de Z02, y el
render con la identidad visual Diagonality (common.py).

Uso:
    python -m src.viz.ppcf            # render del gol de Messi (ARG-MEX)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import Z02_pitch_control as pc
from M01_loader_pff import scan_tracking, load_metadata, load_rosters

from viz.common import (ATT, BALL, DEF, GK, PE_S, PITCH_LENGTH,
                        PITCH_WIDTH, PPCF_CMAP, WHITE, make_pitch, save_fig)

# Lag (frames) para derivar velocidades por diferencias finitas (~0.5 s).
_VEL_LAG_FRAMES = 15
_SPEED_CAP_MPS  = 12.0          # cap anti-teleport del tracking


# ── Adapter: frame de tracking PFF -> schema Z02 ──────────────────────────

def _load_frame_z02(match_id: int, frame_num: int,
                    vel_lag: int = _VEL_LAG_FRAMES) -> tuple:
    """Construye el DataFrame schema-Z02 de un frame de tracking PFF.

    Velocidades por diferencia finita contra el frame `vel_lag` atras.
    Devuelve (frame_df, ball_pos, att_team_id, meta).
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id, away_id = int(md["home_team_id"]), int(md["away_team_id"])
    fps = float(md.get("fps") or 29.97)
    pitch_l = float(md.get("pitch_length") or PITCH_LENGTH)
    pitch_w = float(md.get("pitch_width") or PITCH_WIDTH)
    dt = vel_lag / fps

    # Jerseys de portero por equipo (rosters)
    gk: set[tuple[int, int]] = set()
    for r in load_rosters(match_id).iter_rows(named=True):
        if r["position_group"] == "GK" and r["shirt_number"] is not None:
            gk.add((int(r["team_id"]), int(r["shirt_number"])))

    tr = scan_tracking(match_id).select([
        "frameNum",
        pl.col("homePlayersSmoothed").alias("home"),
        pl.col("awayPlayersSmoothed").alias("away"),
        pl.col("ballsSmoothed").alias("ball"),
    ]).filter(pl.col("frameNum").is_in([frame_num, frame_num - vel_lag])).collect()
    frames = {int(r["frameNum"]): r for r in tr.iter_rows(named=True)}
    cur = frames.get(frame_num)
    if cur is None:
        raise ValueError(f"frame {frame_num} ausente en match {match_id}")
    prev = frames.get(frame_num - vel_lag)

    def _xy(frame, side: str) -> dict[int, tuple[float, float]]:
        out = {}
        for p in (frame[side] or []):
            j, x = p.get("jerseyNum"), p.get("x")
            if j is not None and x is not None:
                out[int(j)] = (float(x), float(p["y"]))
        return out

    prev_xy = {"home": _xy(prev, "home") if prev else {},
               "away": _xy(prev, "away") if prev else {}}

    rows = []
    for side, tid in (("home", home_id), ("away", away_id)):
        for p in (cur[side] or []):
            j, x = p.get("jerseyNum"), p.get("x")
            if j is None or x is None:
                continue
            j, x, y = int(j), float(x), float(p["y"])
            px, py = prev_xy[side].get(j, (x, y))
            vx, vy = (x - px) / dt, (y - py) / dt
            sp = float(np.hypot(vx, vy))
            if sp > _SPEED_CAP_MPS:                       # cap teleports
                vx, vy = vx * _SPEED_CAP_MPS / sp, vy * _SPEED_CAP_MPS / sp
            rows.append(dict(x_tracking=x, y_tracking=y, vx=vx, vy=vy,
                             team_id=tid, is_ball=0,
                             is_goalkeeper=int((tid, j) in gk), jersey=j))
    ball = cur["ball"]
    if ball and ball.get("x") is not None:
        rows.append(dict(x_tracking=float(ball["x"]), y_tracking=float(ball["y"]),
                          vx=0.0, vy=0.0, team_id=-1, is_ball=1,
                          is_goalkeeper=0, jersey=-1))
    df = pd.DataFrame(rows)

    ball_pos = pc.get_ball_pos(df)
    field = df[df["is_ball"] == 0]
    d = np.hypot(field["x_tracking"] - ball_pos[0], field["y_tracking"] - ball_pos[1])
    att_team_id = int(field.iloc[int(d.values.argmin())]["team_id"])
    return df, ball_pos, att_team_id, dict(pitch_l=pitch_l, pitch_w=pitch_w,
                                           home_id=home_id, away_id=away_id)


def frame_for_clock(match_id: int, period: int, clock_s: float) -> int:
    """frameNum cuyo periodGameClockTime es el mas cercano a clock_s (su periodo)."""
    tr = scan_tracking(match_id).select(
        ["frameNum", "period", "periodGameClockTime"]
    ).filter(pl.col("period") == period).collect()
    idx = int((tr["periodGameClockTime"] - clock_s).abs().arg_min())
    return int(tr["frameNum"][idx])


# ── Malla PPCF ─────────────────────────────────────────────────────────────

def compute_ppcf_grid(frame_df: pd.DataFrame, att_team_id: int,
                      ball_pos: np.ndarray, pitch_l: float, pitch_w: float,
                      n_x: int = 80, n_y: int = 52) -> np.ndarray:
    """PPCF del equipo atacante sobre una malla n_y x n_x del campo (Z02)."""
    xs = np.linspace(-pitch_l / 2, pitch_l / 2, n_x)
    ys = np.linspace(-pitch_w / 2, pitch_w / 2, n_y)
    XX, YY = np.meshgrid(xs, ys)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    ppcf = pc.ppcf_at_targets(frame_df, targets, att_team_id, ball_pos)
    return ppcf.reshape(n_y, n_x)


# ── Render ─────────────────────────────────────────────────────────────────

def plot_ppcf(match_id: int, frame_num: int, title: str = "",
              subtitle: str = "", save_path=None):
    """Render de un frame: superficie PPCF + jugadores + velocidades + balon."""
    df, ball_pos, att, meta = _load_frame_z02(match_id, frame_num)
    grid = compute_ppcf_grid(df, att, ball_pos, meta["pitch_l"], meta["pitch_w"])

    fig, ax = make_pitch(figsize=(16, 10.4),
                         pitch_length=meta["pitch_l"], pitch_width=meta["pitch_w"])
    L, W = meta["pitch_l"] / 2, meta["pitch_w"] / 2

    # Superficie: azul = control del equipo en posesion, rojo = del rival
    ax.imshow(grid, extent=[-L, L, -W, W], origin="lower", cmap=PPCF_CMAP,
              vmin=0, vmax=1, alpha=0.72, interpolation="spline36",
              zorder=1, aspect="auto")

    field = df[df["is_ball"] == 0]
    # Flechas de velocidad
    for is_att, color in ((True, ATT), (False, DEF)):
        sub = field[(field["team_id"] == att) == is_att]
        sub = sub[np.hypot(sub["vx"], sub["vy"]) > 0.6]
        if len(sub):
            ax.quiver(sub["x_tracking"], sub["y_tracking"], sub["vx"], sub["vy"],
                      color=color, scale=130, scale_units="width", width=0.003,
                      headwidth=3.5, headlength=4, alpha=0.6, zorder=3)
    # Jugadores + dorsales
    for _, p in field.iterrows():
        color = (GK if p["is_goalkeeper"]
                 else (ATT if p["team_id"] == att else DEF))
        ax.plot(p["x_tracking"], p["y_tracking"], "o", ms=22, color=color,
                markeredgecolor=WHITE, markeredgewidth=1.3, alpha=0.93, zorder=5)
        ax.text(p["x_tracking"], p["y_tracking"], str(int(p["jersey"])),
                color=WHITE, fontsize=8.5, ha="center", va="center",
                fontweight="bold", zorder=6, path_effects=PE_S)
    # Balon
    ax.plot(ball_pos[0], ball_pos[1], "o", ms=11, color=BALL,
            markeredgecolor="black", markeredgewidth=0.9, zorder=10)

    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", color=WHITE,
                 fontsize=16, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.928, subtitle, ha="center", va="top", color="#c8c8c8",
                 fontsize=11)
    # Leyenda
    fig.text(0.5, 0.045,
             "Superficie Pitch Control (Spearman 2018)   ·   "
             "azul = control del equipo en posesion   ·   rojo = control del rival",
             ha="center", color="#c8c8c8", fontsize=10)

    if save_path:
        save_fig(fig, save_path, logo=True)
    return fig


# ── Sanity / hero figure ───────────────────────────────────────────────────

if __name__ == "__main__":
    # ARG-MEX (3835), gol de Messi 1-0 — periodo 2, reloj ~3812 s.
    # Frame del buildup (~3 s antes del remate).
    MID, GOAL_CLOCK = 3835, 3812
    fnum = frame_for_clock(MID, period=2, clock_s=GOAL_CLOCK - 3)
    print(f"[ppcf] match {MID}  frame buildup = {fnum}")
    plot_ppcf(
        MID, fnum,
        title="Argentina 1 - 0 Mexico  ·  Pitch Control en la jugada del gol de Messi",
        subtitle="Mundial Qatar 2022 - Fase de grupos  ·  buildup 3 s antes del remate",
        save_path="outputs/viz/ppcf_messi_arg_mex.png",
    )
    print("OK -> outputs/viz/ppcf_messi_arg_mex.png")
