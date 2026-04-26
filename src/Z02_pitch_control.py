"""
Z02_pitch_control - PPCF (Spearman 2018) vectorizado, agnostico al proveedor.

Adapted from LaurieOnTracking (Friends-of-Tracking-Data-FoTD), reducido a lo
nuclear que necesita M10 sobre tracking PFF. El adapter PFF -> formato Z02
vive en M10 (`_pff_frame_to_z02_df`).

Core PPCF: numpy vectorizado (todos los targets se computan a la vez via
broadcasting). Identico matematicamente al per-target Euler integration de
Spearman 2018 Eq 2-4.

Convencion: posiciones en metros centradas en (0, 0).

Schema esperado en `frame_data` (1 fila por jugador + 1 fila por balon):
    x_tracking, y_tracking : float (metros)
    vx, vy                 : float (m/s)
    team_id                : int    (atacante / defensor)
    is_ball                : 0 / 1  (1 marca la fila del balon)
    is_goalkeeper          : 0 / 1  (1 marca al portero)

API publica:
    default_model_params  : parametros PPCF (Spearman 2018).
    ppcf_at_targets       : PPCF a posiciones especificas (vectorizado N targets).
    get_ball_pos          : extrae posicion del balon de un frame.

Reference: Spearman 2018 "Beyond Expected Goals".
"""

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------

def default_model_params(time_to_control_veto: int = 3) -> dict:
    """Return default PPCF model parameters (Spearman 2018).

    Args:
        time_to_control_veto: Ignore players with control probability < 10^-veto.

    Returns:
        Dict with all model parameters.
    """
    params = {
        "max_player_speed": 5.0,       # m/s
        "reaction_time": 0.7,          # seconds
        "tti_sigma": 0.45,             # uncertainty in arrival time (s)
        "kappa_def": 1.0,              # defending advantage factor
        "lambda_att": 4.3,             # attacking ball control rate
        "average_ball_speed": 15.0,    # m/s
        "int_dt": 0.04,                # integration timestep (s)
        "max_int_time": 10.0,          # max integration time (s)
        "model_converge_tol": 0.01,    # convergence at PPCF > 0.99
    }
    params["lambda_def"] = params["lambda_att"] * params["kappa_def"]
    params["lambda_gk"] = params["lambda_def"] * 3.0

    sigma_term = np.sqrt(3) * params["tti_sigma"] / np.pi
    params["time_to_control_att"] = (
        time_to_control_veto * np.log(10) * (sigma_term + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto * np.log(10) * (sigma_term + 1 / params["lambda_def"])
    )
    return params


# ---------------------------------------------------------------------
# Core PPCF (Spearman 2018 / LaurieOnTracking) - vectorizado
# ---------------------------------------------------------------------

def _ppcf_vectorized(
    targets: np.ndarray,
    att_pos: np.ndarray, att_vel: np.ndarray, att_gk: np.ndarray,
    def_pos: np.ndarray, def_vel: np.ndarray, def_gk: np.ndarray,
    ball_pos: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Vectorized PPCF for N targets simultaneously (Spearman 2018).

    Same Euler forward integration as the original per-target algorithm,
    but processes all grid/target positions in one pass using numpy
    broadcasting. Uses tau = t - ball_tt as integration variable so each
    target's integration is aligned to its own ball arrival time.

    Args:
        targets: Target positions in meters, shape (N, 2).
        att_pos, att_vel: Attacker positions/velocities, shape (Pa, 2).
        att_gk: Attacker goalkeeper mask, shape (Pa,).
        def_pos, def_vel: Defender positions/velocities, shape (Pd, 2).
        def_gk: Defender goalkeeper mask, shape (Pd,).
        ball_pos: Ball position in meters, shape (2,).
        params: PPCF model parameters.

    Returns:
        PPCF_att values, shape (N,).
    """
    N = len(targets)
    vmax = params["max_player_speed"]
    rt = params["reaction_time"]
    sigma = params["tti_sigma"]
    lam_a = params["lambda_att"]
    lam_d = params["lambda_def"]
    lam_gk = params["lambda_gk"]
    dt = params["int_dt"]
    max_int = params["max_int_time"]
    tol = params["model_converge_tol"]
    tc_a = params["time_to_control_att"]
    tc_d = params["time_to_control_def"]
    sig_c = np.pi / np.sqrt(3.0) / sigma
    Pa, Pd = len(att_pos), len(def_pos)

    # Ball travel time per target: (N,)
    if ball_pos is not None and not np.any(np.isnan(ball_pos)):
        ball_tt = np.linalg.norm(targets - ball_pos, axis=1) / params["average_ball_speed"]
    else:
        ball_tt = np.zeros(N)

    # Time-to-intercept: players × targets → (P, N)
    att_r = att_pos + att_vel * rt                                     # (Pa, 2)
    att_tti = rt + np.linalg.norm(
        targets[None] - att_r[:, None], axis=2,
    ) / vmax                                                           # (Pa, N)
    def_r = def_pos + def_vel * rt                                     # (Pd, 2)
    def_tti = rt + np.linalg.norm(
        targets[None] - def_r[:, None], axis=2,
    ) / vmax                                                           # (Pd, N)

    att_min = att_tti.min(axis=0) if Pa else np.full(N, np.inf)        # (N,)
    def_min = def_tti.min(axis=0) if Pd else np.full(N, np.inf)        # (N,)

    # Shortcuts: one team dominates → skip Euler (Spearman veto)
    result = np.full(N, np.nan)
    def_dom = (att_min - np.maximum(ball_tt, def_min)) >= tc_d
    att_dom = (def_min - np.maximum(ball_tt, att_min)) >= tc_a
    result[def_dom] = 0.0
    result[att_dom & ~def_dom] = 1.0

    eidx = np.where(np.isnan(result))[0]
    if len(eidx) == 0:
        return result

    # --- Euler integration for contested targets ---
    M = len(eidx)
    e_btt = ball_tt[eidx]                                              # (M,)
    e_att = att_tti[:, eidx]                                           # (Pa, M)
    e_def = def_tti[:, eidx]                                           # (Pd, M)

    # Active player masks per target: (P, M)
    a_act = (e_att - att_min[eidx]) < tc_a
    d_act = (e_def - def_min[eidx]) < tc_d

    # Defender lambda (GK gets higher rate): (Pd, 1)
    d_lam = np.where(def_gk, lam_gk, lam_d)[:, None]

    # Integrate in tau = t - ball_tt (aligned per target)
    tau_arr = np.arange(0, max_int, dt)
    att_tti_rel = e_att - e_btt[None, :]                               # (Pa, M)
    def_tti_rel = e_def - e_btt[None, :]                               # (Pd, M)

    pa_cum = np.zeros((Pa, M))
    pd_cum = np.zeros((Pd, M))
    tot_a = np.zeros(M)
    tot_d = np.zeros(M)
    conv = np.zeros(M, dtype=bool)

    for tau in tau_arr:
        if conv.all():
            break
        live = ~conv                                                   # (M,)
        rem = 1.0 - tot_a - tot_d                                      # (M,)
        with np.errstate(over='ignore'):
            p_a = 1.0 / (1.0 + np.exp(-sig_c * (tau - att_tti_rel)))   # (Pa, M)
            p_d = 1.0 / (1.0 + np.exp(-sig_c * (tau - def_tti_rel)))   # (Pd, M)
        pa_cum += (rem * p_a * lam_a * a_act * live) * dt
        pd_cum += (rem * p_d * d_lam * d_act * live) * dt
        tot_a = pa_cum.sum(axis=0)
        tot_d = pd_cum.sum(axis=0)
        conv = (tot_a + tot_d) > (1.0 - tol)

    result[eidx] = tot_a
    return result


# ---------------------------------------------------------------------
# Helpers de extraccion sobre frame_data (formato neutro)
# ---------------------------------------------------------------------

def _extract_teams(frame_data: pd.DataFrame, att_team_id):
    """Extract (pos, vel, is_gk) arrays for attacking and defending teams."""
    players = frame_data[frame_data["is_ball"] == 0]
    att = players[players["team_id"] == att_team_id]
    def_ = players[players["team_id"] != att_team_id]

    def _arrays(df):
        pos = df[["x_tracking", "y_tracking"]].values
        if "vx" in df.columns and "vy" in df.columns:
            vel = df[["vx", "vy"]].fillna(0).values
        else:
            vel = np.zeros_like(pos)
        is_gk = df["is_goalkeeper"].values.astype(bool)
        return pos, vel, is_gk

    return _arrays(att), _arrays(def_)


def get_ball_pos(frame_data: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract ball position in meters from a tracking frame.

    API publica usada por M10 para extraer la posicion del balon antes de
    invocar `ppcf_at_targets`.
    """
    ball = frame_data[frame_data["is_ball"] == 1]
    if ball.empty:
        return None
    return np.array([ball.iloc[0]["x_tracking"], ball.iloc[0]["y_tracking"]])


# ---------------------------------------------------------------------
# API publica: PPCF a targets especificos
# ---------------------------------------------------------------------

def ppcf_at_targets(
    frame_data: pd.DataFrame,
    targets_meters: np.ndarray,
    att_team_id,
    ball_pos: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> np.ndarray:
    """Compute PPCF at specific target positions (no grid).

    Mucho mas rapido que un grid completo cuando solo se necesitan unas pocas
    posiciones (e.g., posiciones de los jugadores atacantes para OBSO).

    Args:
        frame_data: Tracking rows for one frame (all players + ball).
                    Schema: x_tracking, y_tracking, vx, vy, team_id, is_ball,
                    is_goalkeeper. Coords en metros centradas en (0,0).
        targets_meters: Array of shape (N, 2) with positions in meters.
        att_team_id: Team ID of the attacking team.
        ball_pos: Ball position in meters [x, y]. Auto-detected si None.
        params: Model parameters (default si None).

    Returns:
        Array of shape (N,) with PPCF_att at each target.
    """
    if params is None:
        params = default_model_params()
    if ball_pos is None:
        ball_pos = get_ball_pos(frame_data)

    (att_pos, att_vel, att_gk), (def_pos, def_vel, def_gk) = _extract_teams(
        frame_data, att_team_id
    )

    return _ppcf_vectorized(
        targets_meters, att_pos, att_vel, att_gk,
        def_pos, def_vel, def_gk, ball_pos, params,
    )
