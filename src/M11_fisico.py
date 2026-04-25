"""
M11_fisico - Canal Pulso Fisico via metricas tracking + modelo bayesiano de fatiga.

Fase 2 PCJ, canal 4 de 4. Aisla "lo que el jugador APRIETA" (decision mental) de
"lo que el jugador PUEDE" (estado fisico) modelando la fatiga acumulada como
estado latente y reportando el RESIDUO sobre la prediccion fatiga-esperada.

Pipeline:
  1. Limpieza tracking + velocidades smoothed (Hampel + Butterworth fase-cero).
     Segmentacion por discontinuidades fisicas (camera switch / ID swap).
  2. Metricas frame-level agregadas por (player, match, minute) — Bradley 2024:
       - distance_m       : integral vel * dt (m).
       - hsr_s            : segundos a >= 19.8 km/h (Ju et al. 2022, FIFA reglas).
       - sprint_s         : segundos a >= 25 km/h.
       - sprint_count     : # eventos sprint distintos (Brad onset >=1s + recovery >=2s).
       - psv95            : peak speed velocity (p95 robusto del minuto, m/s).
       - n_high_accel     : segundos con |a| >= 3 m/s² (accel intenso).
       - n_high_decel     : segundos con a <= -3 m/s² (decel intenso, frenadas).
       - z1_m..z5_m       : distancia por zona Bradley (km/h):
                             Z1<7, Z2 7-13, Z3 13-19.8, Z4 19.8-25, Z5>25.
       - hmld_m           : High Metabolic Load Distance, Osgnach et al. 2010
                             (P_metabolic >= 25.5 W/kg).
  3. Modelo bayesiano state-space (numpyro SVI):
       fatiga[t] = (1-alpha) * fatiga[t-1] + load[t]
       log(metric) ~ alpha_player + beta * fatiga + minute_fe + epsilon
     score_phys_minute = residual (observado - prediccion fatiga-esperada).
  4. Agregacion per_shock_window (pre/post +-10min).

Acceptance (ARCHITECTURE.md + Bradley 2024 WC22):
  - Distancia top starters ~10-11 km/partido.
  - PSV95 top players ~32-34 km/h (NO cap saturado).
  - n_high_accel ~50-100 segundos/jugador-partido.
  - Residuos centrados en 0 (esperanza condicional 0).

Output:
  data/parquet/derived/fisico/
    raw_per_minute.parquet      # metricas raw frame-level agregadas.
    model/fatigue_state.pkl     # SVI fit (parametros posterior).
    per_minute.parquet          # score_phys (residuo) per (match, player, min).
    per_shock_window.parquet    # pre/post pm cada shock.

Depende de: M01 (tracking, rosters), M03 (player_minutes), M07 (shocks_table).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import (
    load_metadata, load_rosters, scan_tracking, list_event_match_ids,
)
from M03_preprocess import player_minutes
from M07_shocks import build_shocks_table


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "fisico"


# -- Parametros fisicos pre-registrados -------------------------------------

# Velocity thresholds (Bradley 2024 + Ju et al. 2022)
HSR_THRESHOLD_MPS    = 19.8 / 3.6     # 5.50 m/s = 19.8 km/h
SPRINT_THRESHOLD_MPS = 25.0 / 3.6     # 6.94 m/s = 25 km/h
MAX_HUMAN_SPEED_MPS  = 11.0           # cap antiruido (39.6 km/h)

# Bradley 2024 distance zones (km/h) -> m/s
_Z_BOUNDS_KMH = [0.0, 7.0, 13.0, 19.8, 25.0, np.inf]
_Z_BOUNDS_MPS = [k / 3.6 for k in _Z_BOUNDS_KMH]   # 5 zonas (Z1..Z5)

# Aceleraciones (Akenhead et al. 2013, Bradley 2024)
ACCEL_THRESHOLD_MPS2 = 3.0    # |a| >= 3 m/s² = high-intensity accel/decel

# Sprint event detection (Bradley 2024 onset/recovery)
SPRINT_MIN_DURATION_S  = 1.0   # un sprint debe durar >=1s
SPRINT_RECOVERY_GAP_S  = 2.0   # dos sprints separados por <2s = mismo evento

# Metabolic power (Osgnach et al. 2010)
HMLD_POWER_THRESHOLD_W_KG = 25.5

# Outlier rejection: detectar discontinuidades posicionales > velocidad fisica
# realista. 12 m/s × dt = max desplazamiento por frame en sprint humano elite.
_TELEPORT_SPEED_THRESHOLD_MPS = 12.0

# Hampel filter sobre velocidad (kill outliers residuales)
_HAMPEL_WINDOW_VEL = 9
_HAMPEL_NSIGMAS    = 3.0

# Butterworth fase-cero. Dos cutoffs paralelos:
#  - 1 Hz para velocidad (Buchheit standard, suaviza ruido alto-frecuencia).
#  - 2 Hz para aceleracion (frenazos reales 0.3-0.5s tienen energia 2-3 Hz, un
#    cutoff demasiado bajo atenuaria picos de accel/decel reales).
_BUTTER_ORDER          = 4
_BUTTER_CUTOFF_VEL_HZ  = 1.0
_BUTTER_CUTOFF_ACC_HZ  = 2.0
_FPS_DEFAULT           = 25.0

# Min frames para procesar un segmento. filtfilt default padlen = 3*max(len(a),len(b))
# con order=4 -> padlen=15. Necesitamos len(signal) > 15 -> minimo 16.
_MIN_SEGMENT_FRAMES = 16


# ===========================================================================
#  SECCION 1 — Limpieza tracking + velocidades smoothed
# ===========================================================================

def _butter_lowpass_filtfilt(signal: np.ndarray, fs: float = _FPS_DEFAULT,
                              cutoff: float = _BUTTER_CUTOFF_VEL_HZ,
                              order: int = _BUTTER_ORDER) -> np.ndarray:
    """Butterworth lowpass + filtfilt (fase cero, sin lag)."""
    if len(signal) < _MIN_SEGMENT_FRAMES:
        return signal
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="lowpass")
    return filtfilt(b, a, signal)


def _hampel_filter(x: np.ndarray, window: int = _HAMPEL_WINDOW_VEL,
                    n_sigmas: float = _HAMPEL_NSIGMAS) -> np.ndarray:
    """Hampel filter vectorizado: detecta y reemplaza outliers via median + MAD.

    Para cada punto: si |x[i] - median(window)| > n_sigmas * 1.4826 * MAD,
    se reemplaza por la mediana local. 1.4826 convierte MAD a sigma-equivalent
    para distribucion normal. Implementacion O(n) via scipy.ndimage.median_filter.
    """
    if len(x) < window:
        return x
    med = median_filter(x, size=window, mode="nearest")
    mad = median_filter(np.abs(x - med), size=window, mode="nearest")
    sigma = 1.4826 * mad
    threshold = n_sigmas * sigma
    return np.where((sigma > 0) & (np.abs(x - med) > threshold), med, x)


def _segment_by_teleports(px: np.ndarray, py: np.ndarray, dt: float,
                           max_speed: float = _TELEPORT_SPEED_THRESHOLD_MPS
                           ) -> list[tuple[int, int]]:
    """Detecta discontinuidades posicionales (camera switch / ID swap).

    Returns lista de (start_idx, end_idx_exclusive) de segmentos contiguos sin
    teleports. Un teleport = ||delta_pos|| > max_speed * dt entre frames
    consecutivos. Fisicamente, un humano no puede moverse mas de eso.
    """
    if len(px) < 2:
        return [(0, len(px))]
    max_step = max_speed * dt
    delta = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
    # Indices donde el segmento se rompe (i+1 es inicio de nuevo segmento)
    breaks = np.where(delta > max_step)[0] + 1
    boundaries = [0, *breaks.tolist(), len(px)]
    return [(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)]


def _process_segment(px_seg: np.ndarray, py_seg: np.ndarray,
                      fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """Procesa 1 segmento contiguo de posiciones, dual pipeline (vel 1Hz, acc 2Hz).

    Pipeline ELITE:
      1. Pipeline VEL (cutoff 1Hz): Butterworth lowpass sobre positions ->
         gradient -> vx, vy, speed. Cutoff 1Hz es Buchheit standard.
         Hampel filter sobre speed limpia outliers residuales.
      2. Pipeline ACC (cutoff 2Hz): Butterworth sobre positions con cutoff
         mas alto para preservar frenazos cortos (0.3-0.5s, energia ~2-3Hz);
         doble gradient -> ax, ay, |a|.
      3. Aceleracion tangencial (signed): gradient de speed (pipeline vel)
         para coherencia con HSR/sprint thresholds.
      4. Cap final sobre speed con re-scale proporcional a vx/vy.

    Hampel se aplica sobre VELOCIDAD (no posiciones) para no destruir dinamica
    de aceleracion cerca de teleports — la segmentacion previa ya elimina
    discontinuidades fisicas.

    Returns:
        (vx, vy, speed, signed_accel, accel_mod), todos shape (n,).
    """
    n = len(px_seg)
    if n < _MIN_SEGMENT_FRAMES:
        return (np.zeros(n),) * 5
    dt = 1.0 / fs

    # 1. Pipeline VEL (cutoff 1Hz)
    px_v = _butter_lowpass_filtfilt(px_seg, fs, _BUTTER_CUTOFF_VEL_HZ)
    py_v = _butter_lowpass_filtfilt(py_seg, fs, _BUTTER_CUTOFF_VEL_HZ)
    vx = np.gradient(px_v, dt)
    vy = np.gradient(py_v, dt)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    speed = _hampel_filter(speed)

    # 2. Pipeline ACC (cutoff 2Hz)
    px_a = _butter_lowpass_filtfilt(px_seg, fs, _BUTTER_CUTOFF_ACC_HZ)
    py_a = _butter_lowpass_filtfilt(py_seg, fs, _BUTTER_CUTOFF_ACC_HZ)
    vx_a = np.gradient(px_a, dt)
    vy_a = np.gradient(py_a, dt)
    ax = np.gradient(vx_a, dt)
    ay = np.gradient(vy_a, dt)
    accel_mod = np.sqrt(ax ** 2 + ay ** 2)

    # 3. Signed accel tangencial (sobre speed pipeline VEL)
    signed_accel = np.gradient(speed, dt)

    # 4. Cap final con re-scale proporcional
    cap_mask = speed > MAX_HUMAN_SPEED_MPS
    if cap_mask.any():
        scale = MAX_HUMAN_SPEED_MPS / speed[cap_mask]
        vx[cap_mask] *= scale
        vy[cap_mask] *= scale
        speed[cap_mask] = MAX_HUMAN_SPEED_MPS

    return vx, vy, speed, signed_accel, accel_mod


def _compute_velocities_clean(positions_x: np.ndarray, positions_y: np.ndarray,
                               fs: float = _FPS_DEFAULT
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """Pipeline ELITE: posiciones -> (vx, vy, speed, signed_accel, accel_mod).

    Segmenta por teleports y procesa cada segmento independientemente con
    `_process_segment` (dual cutoff 1Hz vel / 2Hz acc).
    """
    n = len(positions_x)
    vx = np.zeros(n)
    vy = np.zeros(n)
    speed = np.zeros(n)
    signed_a = np.zeros(n)
    accel_m = np.zeros(n)
    if n < 2:
        return vx, vy, speed, signed_a, accel_m

    dt = 1.0 / fs
    for (s, e) in _segment_by_teleports(positions_x, positions_y, dt):
        if e - s < _MIN_SEGMENT_FRAMES:
            continue
        seg_vx, seg_vy, seg_sp, seg_sa, seg_am = _process_segment(
            positions_x[s:e], positions_y[s:e], fs,
        )
        vx[s:e]      = seg_vx
        vy[s:e]      = seg_vy
        speed[s:e]   = seg_sp
        signed_a[s:e] = seg_sa
        accel_m[s:e]  = seg_am

    return vx, vy, speed, signed_a, accel_m


def _metabolic_power(speed: np.ndarray, signed_accel: np.ndarray) -> np.ndarray:
    """Potencia metabolica W/kg (Osgnach et al. 2010).

    Modelo simplificado de gait equivalente: P = (EM * a) * v + alpha_walk * v
    donde EM (equivalent mass) y coeficientes derivan del coste energetico de
    correr en pendiente equivalente.

    Implementacion del Eq 7-9 de Osgnach 2010:
      EC = (155.4*ES**5 - 30.4*ES**4 - 43.3*ES**3 + 46.3*ES**2 + 19.5*ES + 3.6) * EM
      ES = arctan(a/g)  (equivalent slope, g=9.81)
      EM = sqrt((a/g)**2 + 1)
      P  = EC * v
    """
    g = 9.81
    a_over_g = signed_accel / g
    es = np.arctan(a_over_g)
    em = np.sqrt(a_over_g ** 2 + 1)
    ec = (155.4 * es ** 5 - 30.4 * es ** 4 - 43.3 * es ** 3
          + 46.3 * es ** 2 + 19.5 * es + 3.6) * em
    p = ec * speed
    return np.maximum(p, 0)   # potencia metabolica no-negativa


def _count_sprint_events(speed: np.ndarray, fs: float = _FPS_DEFAULT) -> int:
    """Cuenta sprints distintos (Bradley 2024 onset/recovery rule).

    Un sprint = run de speed >= SPRINT_THRESHOLD durante >= 1s. Dos sprints
    consecutivos separados por < 2s de recovery cuentan como UN solo evento.
    """
    if len(speed) < int(SPRINT_MIN_DURATION_S * fs):
        return 0
    above = speed >= SPRINT_THRESHOLD_MPS
    if not above.any():
        return 0
    # Identificar runs consecutivos via diff
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]   # exclusive
    durations = (ends - starts) / fs
    valid = durations >= SPRINT_MIN_DURATION_S
    if not valid.any():
        return 0
    valid_starts = starts[valid]
    valid_ends = ends[valid]
    # Mergear sprints separados por < SPRINT_RECOVERY_GAP_S
    merged = 1
    for i in range(1, len(valid_starts)):
        gap = (valid_starts[i] - valid_ends[i - 1]) / fs
        if gap >= SPRINT_RECOVERY_GAP_S:
            merged += 1
    return merged


# ===========================================================================
#  SECCION 2 — Metricas fisicas per (player, minute)
# ===========================================================================

_PHYS_SCHEMA = {
    "pff_match_id":  pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
    "distance_m":    pl.Float64,
    "hsr_s":         pl.Float64,
    "sprint_s":      pl.Float64,
    "sprint_count":  pl.Int64,
    "psv95":         pl.Float64,
    "n_high_accel":  pl.Float64,
    "n_high_decel":  pl.Float64,
    "z1_m":          pl.Float64, "z2_m": pl.Float64, "z3_m": pl.Float64,
    "z4_m":          pl.Float64, "z5_m": pl.Float64,
    "hmld_m":        pl.Float64,
    "n_frames":      pl.Int64,
}


def _aggregate_minute_metrics(speed: np.ndarray, signed_accel: np.ndarray,
                               accel_mod: np.ndarray, p_metabolic: np.ndarray,
                               minutes: np.ndarray, dt: float) -> list[dict]:
    """Agrega metricas frame-level por minute. Devuelve lista de dicts."""
    rows = []
    for m in np.unique(minutes):
        mask = minutes == m
        if mask.sum() < 3:
            continue
        sp = speed[mask]
        sa = signed_accel[mask]
        am = accel_mod[mask]
        pm = p_metabolic[mask]

        distance = float(sp.sum() * dt)
        hsr      = float((sp >= HSR_THRESHOLD_MPS).sum() * dt)
        sprint   = float((sp >= SPRINT_THRESHOLD_MPS).sum() * dt)
        sprintc  = int(_count_sprint_events(sp))
        psv95    = float(np.percentile(sp, 95))   # p95 robusto vs cap-saturated
        accel_s  = float((sa >= ACCEL_THRESHOLD_MPS2).sum() * dt)
        decel_s  = float((sa <= -ACCEL_THRESHOLD_MPS2).sum() * dt)
        # Distance per zone
        zone_m = []
        for zi in range(5):
            lo, hi = _Z_BOUNDS_MPS[zi], _Z_BOUNDS_MPS[zi + 1]
            in_zone = (sp >= lo) & (sp < hi)
            zone_m.append(float(sp[in_zone].sum() * dt))
        hmld = float(sp[pm >= HMLD_POWER_THRESHOLD_W_KG].sum() * dt)

        rows.append({
            "minute":       int(m),
            "distance_m":   distance,
            "hsr_s":        hsr,
            "sprint_s":     sprint,
            "sprint_count": sprintc,
            "psv95":        psv95,
            "n_high_accel": accel_s,
            "n_high_decel": decel_s,
            "z1_m":         zone_m[0], "z2_m": zone_m[1], "z3_m": zone_m[2],
            "z4_m":         zone_m[3], "z5_m": zone_m[4],
            "hmld_m":       hmld,
            "n_frames":     int(mask.sum()),
        })
    return rows


def _phys_metrics_per_minute(match_id: int) -> pl.DataFrame:
    """Metricas fisicas per (player_id, minute) para 1 partido (vectorizado)."""
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    fs = float(md.get("fps") or _FPS_DEFAULT)
    dt = 1.0 / fs
    frames_per_min = fs * 60.0

    ro = load_rosters(match_id).select(["team_id", "player_id", "shirt_number"]) \
          .filter(pl.col("shirt_number").is_not_null()) \
          .with_columns(pl.col("shirt_number").cast(pl.Int64).alias("jersey_int"))
    home_map = {int(r["jersey_int"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id") == home_id).iter_rows(named=True)}
    away_map = {int(r["jersey_int"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id") == away_id).iter_rows(named=True)}

    lf = scan_tracking(match_id)
    frames = lf.select([
        pl.col("frameNum"), pl.col("period"),
        pl.col("homePlayersSmoothed").alias("home_players"),
        pl.col("awayPlayersSmoothed").alias("away_players"),
    ]).collect()

    if frames.height == 0:
        return pl.DataFrame(schema=_PHYS_SCHEMA)

    def _side_long(players_col: str, side_team_id: int) -> pl.DataFrame:
        # NO filtramos confidence: homePlayersSmoothed ya viene Kalman-filled,
        # los frames con conf=LOW son los rellenos validos que el smoother imputo.
        # La spec PFF recomienda filtrar LOW solo en raw homePlayers, no en smoothed.
        return frames.select([
            "frameNum", "period",
            pl.col(players_col).alias("p"),
        ]).explode("p").filter(pl.col("p").is_not_null()).with_columns([
            pl.col("p").struct.field("x").alias("x"),
            pl.col("p").struct.field("y").alias("y"),
            pl.col("p").struct.field("jerseyNum").cast(pl.Int64, strict=False).alias("jersey"),
            pl.lit(side_team_id, dtype=pl.Int64).alias("team_id"),
        ]).filter(
            pl.col("x").is_not_null() &
            pl.col("y").is_not_null() &
            pl.col("jersey").is_not_null()
        ).select(["frameNum", "period", "team_id", "jersey", "x", "y"])

    home_long = _side_long("home_players", home_id)
    away_long = _side_long("away_players", away_id)
    long = pl.concat([home_long, away_long])

    if long.height == 0:
        return pl.DataFrame(schema=_PHYS_SCHEMA)

    # Map jersey -> player_id ANTES del group_by, asi agrupamos por player_id
    # directamente (defensivo si un jugador cambiase dorsal entre periods).
    long = long.with_columns(
        pl.when(pl.col("team_id") == home_id)
          .then(pl.col("jersey").replace_strict(home_map, default=None))
          .otherwise(pl.col("jersey").replace_strict(away_map, default=None))
          .alias("player_id")
    ).filter(pl.col("player_id").is_not_null())

    rows_out = []
    for (player_id, period), group in long.sort("frameNum").group_by(
        ["player_id", "period"], maintain_order=True,
    ):
        if group.height < _MIN_SEGMENT_FRAMES:
            continue
        pid = int(player_id)

        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        fn = group["frameNum"].to_numpy()
        vx, vy, speed, signed_a, accel_mod = _compute_velocities_clean(x, y, fs)
        p_meta = _metabolic_power(speed, signed_a)

        minutes = (fn // frames_per_min).astype(np.int64)
        for r in _aggregate_minute_metrics(speed, signed_a, accel_mod,
                                            p_meta, minutes, dt):
            r["pff_match_id"] = match_id
            r["player_id"]    = pid
            rows_out.append(r)

    if not rows_out:
        return pl.DataFrame(schema=_PHYS_SCHEMA)
    out = pl.DataFrame(rows_out, schema_overrides=_PHYS_SCHEMA)
    # Si un (player, minute) aparece en >1 period (caso raro stoppage), sumar.
    return out.group_by(["pff_match_id", "player_id", "minute"]).agg([
        pl.col("distance_m").sum(),
        pl.col("hsr_s").sum(),
        pl.col("sprint_s").sum(),
        pl.col("sprint_count").sum(),
        pl.col("psv95").max(),
        pl.col("n_high_accel").sum(),
        pl.col("n_high_decel").sum(),
        pl.col("z1_m").sum(), pl.col("z2_m").sum(), pl.col("z3_m").sum(),
        pl.col("z4_m").sum(), pl.col("z5_m").sum(),
        pl.col("hmld_m").sum(),
        pl.col("n_frames").sum(),
    ]).sort(["pff_match_id", "player_id", "minute"])


def build_raw_per_minute(cache: bool = True, overwrite: bool = False) -> pl.DataFrame:
    """Aplica _phys_metrics_per_minute a los 64 partidos WC22."""
    cache_path = _DERIVED / "raw_per_minute.parquet"
    if cache and cache_path.exists() and not overwrite:
        return pl.read_parquet(cache_path)

    import time
    dfs = []
    t0 = time.time()
    mids = list_event_match_ids()
    for i, mid in enumerate(mids):
        try:
            dfs.append(_phys_metrics_per_minute(mid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(mids)} en {time.time()-t0:.1f}s", flush=True)

    out = pl.concat(dfs) if dfs else pl.DataFrame(schema=_PHYS_SCHEMA)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


# ===========================================================================
#  SECCION 3 — Modelo bayesiano state-space de fatiga (numpyro SVI)
# ===========================================================================
#
#  Idea: aislar "lo que el jugador APRIETA" (decision mental) del "puedo"
#  fisico modelando la fatiga acumulada y reportando el RESIDUO del log(psv95)
#  observado vs lo esperado dada la fatiga.
#
#  Pipeline:
#    1. Carga deterministica: load[p,m,t] = combinacion lineal estandarizada
#       de (distance, HSR, sprint, accel, decel) en el minuto t. Carga = "input
#       fisico" del jugador en ese minuto.
#    2. Estado latente fatiga: fatiga[p,m,t] = (1-alpha)*fatiga[p,m,t-1] + load.
#       Fatiga se acumula con la carga y decae linealmente. Reset a 0 al
#       inicio de cada partido (recovery completa entre matches).
#    3. Modelo bayesiano numpyro SVI:
#         log(psv95[p,m,t]) = mu_player[p] + beta_min*(minute/90)
#                              + beta_fat*fatiga + epsilon
#       Random effects por player (mu_player ~ Normal(mu_global, sigma_p)).
#       Coeficiente beta_fat NEGATIVO esperado: a mas fatiga, menor PSV95.
#    4. score_phys = (log(psv95) - log(psv95_predicho)) / sigma_eps.
#       Z-score del residuo. Positivo = "aprieta mas de lo esperado dado su
#       estado fisico". Negativo = "rinde por debajo de su capacidad".

_FAT_DECAY_ALPHA = 0.05   # recovery rate por minuto (half-life ~14 min)

# Pesos para combinar metricas en una "carga" single-scalar (estandarizada
# despues por z-score). Reflejan contribucion energetica relativa:
#   distance (m): base, peso 1
#   HSR (s):     mas costoso que jogging, peso 5
#   sprint (s):  alta intensidad anaerobica, peso 10
#   accel/decel (s): coste mecanico explosivo, peso 3 cada uno
_LOAD_WEIGHTS = {
    "distance_m":   1.0,
    "hsr_s":        5.0,
    "sprint_s":    10.0,
    "n_high_accel": 3.0,
    "n_high_decel": 3.0,
}


def _compute_load_and_fatigue(raw: pl.DataFrame,
                                alpha: float = _FAT_DECAY_ALPHA) -> pl.DataFrame:
    """Anade cols load_z + fatigue al DataFrame raw_per_minute.

    load = z-score por jugador-torneo de la suma ponderada (Σ peso * metric).
    Estandarizar por jugador hace que load sea comparable across players con
    diferentes baselines (un GK tiene carga mucho menor que un CF).

    fatigue: estado latente computado deterministicamente por (player, match)
    con el decay alpha. Reset a 0 al inicio de cada match.
    """
    # 1. Carga ponderada (raw)
    raw = raw.with_columns(
        sum(pl.col(k) * w for k, w in _LOAD_WEIGHTS.items()).alias("load_raw")
    )
    # 2. Z-score per player (mean/std globales del torneo por jugador)
    stats = raw.group_by("player_id").agg([
        pl.col("load_raw").mean().alias("load_mean"),
        pl.col("load_raw").std().fill_null(1.0).alias("load_std"),
    ]).with_columns(
        pl.when(pl.col("load_std") < 1e-6).then(1.0)
          .otherwise(pl.col("load_std")).alias("load_std")
    )
    raw = raw.join(stats, on="player_id", how="left").with_columns(
        ((pl.col("load_raw") - pl.col("load_mean")) / pl.col("load_std"))
        .alias("load_z")
    ).drop(["load_raw", "load_mean", "load_std"])
    # 3. Fatigue deterministico por (player, match) — loop O(N) en numpy
    #    porque depende secuencialmente del minuto anterior.
    raw_sorted = raw.sort(["pff_match_id", "player_id", "minute"])
    pmid = raw_sorted["pff_match_id"].to_numpy()
    pid  = raw_sorted["player_id"].to_numpy()
    mins = raw_sorted["minute"].to_numpy()
    load_z = raw_sorted["load_z"].to_numpy()

    fat = np.zeros(len(load_z))
    last_key = (None, None)
    f_prev = 0.0
    last_min = -1
    for i in range(len(load_z)):
        key = (pmid[i], pid[i])
        if key != last_key:
            f_prev = 0.0
            last_min = -1
            last_key = key
        # Decay desde last_min hasta mins[i] (gap-aware)
        gap = mins[i] - last_min
        if last_min >= 0 and gap > 1:
            f_prev = f_prev * ((1 - alpha) ** gap)
        f_curr = (1 - alpha) * f_prev + load_z[i]
        fat[i] = f_curr
        f_prev = f_curr
        last_min = mins[i]

    return raw_sorted.with_columns(pl.Series("fatigue", fat))


def fit_fatigue_model(df_with_fatigue: pl.DataFrame,
                       n_steps: int = 4000, seed: int = 42) -> dict:
    """Entrena state-space bayesiano numpyro SVI sobre psv95.

    Modelo:
      log(psv95[p,m,t]) ~ Normal(mu_player[p] + beta_min*(min/90)
                                  + beta_fat*fatigue, sigma_eps)
      mu_player[p] ~ Normal(mu_global, sigma_player)

    Returns dict con SVI params posterior (loc / scale).
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    # Filtrar filas con psv95 valido y > 0
    df = df_with_fatigue.filter(
        pl.col("psv95") > 0.5      # descartar minutos triviales (jugador parado)
    )
    if df.height == 0:
        raise ValueError("sin datos para fit_fatigue_model")

    log_psv95 = np.log(df["psv95"].to_numpy())
    fatigue = df["fatigue"].to_numpy()
    minute_norm = df["minute"].to_numpy() / 90.0
    player_ids = df["player_id"].to_numpy()
    uniq_p = np.unique(player_ids)
    p_to_idx = {p: i for i, p in enumerate(uniq_p)}
    p_idx = np.array([p_to_idx[p] for p in player_ids], dtype=np.int32)
    n_players = len(uniq_p)

    def model(p_idx_, mn_, fat_, y=None):
        sigma_p = numpyro.sample("sigma_p", dist.HalfNormal(0.5))
        mu_g = numpyro.sample("mu_global", dist.Normal(2.0, 1.0))
        mu_p = numpyro.sample("mu_player",
                              dist.Normal(mu_g, sigma_p).expand([n_players]).to_event(1))
        beta_min = numpyro.sample("beta_min", dist.Normal(0.0, 0.5))
        beta_fat = numpyro.sample("beta_fat", dist.Normal(0.0, 0.5))
        sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(0.3))
        pred = mu_p[p_idx_] + beta_min * mn_ + beta_fat * fat_
        with numpyro.plate("N", len(p_idx_)):
            numpyro.sample("obs", dist.Normal(pred, sigma_eps), obs=y)

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())

    state = svi.init(jax.random.PRNGKey(seed),
                      jnp.asarray(p_idx), jnp.asarray(minute_norm),
                      jnp.asarray(fatigue), jnp.asarray(log_psv95))
    for i in range(n_steps):
        state, loss = svi.update(state, jnp.asarray(p_idx),
                                  jnp.asarray(minute_norm),
                                  jnp.asarray(fatigue),
                                  jnp.asarray(log_psv95))
        if i % 500 == 0:
            print(f"  step {i:5d}  elbo_loss={float(loss):.2f}")
    params = svi.get_params(state)
    return {
        "params": params,
        "p_to_idx": p_to_idx,
        "n_obs": int(df.height),
        "n_players": n_players,
    }


def compute_score_phys(df_with_fatigue: pl.DataFrame, fit: dict) -> pl.DataFrame:
    """Computa score_phys = z-score del residuo log(psv95) - prediccion.

    Retorna df con cols (pff_match_id, player_id, minute, score_phys, ...).
    """
    p = fit["params"]
    p_to_idx = fit["p_to_idx"]

    df = df_with_fatigue.filter(pl.col("psv95") > 0.5).with_columns(
        pl.col("player_id").replace_strict(p_to_idx, default=-1).alias("p_idx")
    ).filter(pl.col("p_idx") >= 0)

    if df.height == 0:
        raise ValueError("sin filas validas para compute_score_phys")

    p_idx = df["p_idx"].to_numpy()
    mn = df["minute"].to_numpy() / 90.0
    fat = df["fatigue"].to_numpy()
    log_obs = np.log(df["psv95"].to_numpy())

    mu_p = np.array(p["mu_player_auto_loc"])
    beta_min = float(p["beta_min_auto_loc"])
    beta_fat = float(p["beta_fat_auto_loc"])
    sigma_eps = float(np.exp(p["sigma_eps_auto_loc"]))   # log-scale -> abs

    pred = mu_p[p_idx] + beta_min * mn + beta_fat * fat
    residual = log_obs - pred
    score_phys = residual / sigma_eps   # z-score

    return df.with_columns([
        pl.Series("log_psv95_pred", pred),
        pl.Series("log_psv95_obs", log_obs),
        pl.Series("score_phys", score_phys),
    ]).select([
        "pff_match_id", "player_id", "minute",
        "log_psv95_obs", "log_psv95_pred", "score_phys",
        "fatigue", "load_z",
    ])


def cache_score_phys(overwrite: bool = False, n_steps: int = 4000) -> pl.DataFrame:
    """Pipeline completa: load -> fatigue -> SVI fit -> score_phys per_minute."""
    import pickle
    cache_path = _DERIVED / "per_minute.parquet"
    fit_path   = _DERIVED / "model" / "fatigue_state.pkl"
    if cache_path.exists() and fit_path.exists() and not overwrite:
        return pl.read_parquet(cache_path)

    raw = build_raw_per_minute(cache=True)
    df_fat = _compute_load_and_fatigue(raw)
    print(f"  fitting SVI fatigue model ({df_fat.height:,} obs, {n_steps} steps)...")
    fit = fit_fatigue_model(df_fat, n_steps=n_steps)
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fit_path, "wb") as f:
        pickle.dump({k: (np.array(v) if hasattr(v, "shape") else v)
                     for k, v in fit.items() if k != "params"} |
                    {"params": {k: np.array(v) for k, v in fit["params"].items()}}, f)

    out = compute_score_phys(df_fat, fit)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache_path, compression="snappy")
    return out


# ===========================================================================
#  SECCION 4 — Aggregate per_shock_window (pre/post +-10min)
# ===========================================================================

def aggregate_per_shock_window(cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_phys + metricas raw en pre/post."""
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    pm = cache_score_phys()                                # (pff_match_id, player_id, minute, ...)
    shocks = build_shocks_table(cache=True, overwrite=False)

    pm = pm.rename({"pff_match_id": "match_id"})

    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ])

    joined = shocks_slim.join(pm, on=["match_id", "player_id"], how="left") \
                        .with_columns((pl.col("minute") * 60).alias("min_sec"))

    pre = joined.filter(
        (pl.col("min_sec") >= pl.col("window_pre_start")) &
        (pl.col("min_sec") < pl.col("window_pre_end"))
    ).group_by(["match_id", "shock_id", "player_id", "shock_type"]).agg([
        pl.col("score_phys").mean().alias("score_phys_pre"),
        pl.col("fatigue").mean().alias("fatigue_pre"),
        pl.col("load_z").sum().alias("load_z_pre"),
    ])
    post = joined.filter(
        (pl.col("min_sec") >= pl.col("window_post_start")) &
        (pl.col("min_sec") <= pl.col("window_post_end"))
    ).group_by(["match_id", "shock_id", "player_id", "shock_type"]).agg([
        pl.col("score_phys").mean().alias("score_phys_post"),
        pl.col("fatigue").mean().alias("fatigue_post"),
        pl.col("load_z").sum().alias("load_z_post"),
    ])

    base = shocks.select(["match_id", "shock_id", "player_id", "shock_type"]).unique()
    out = base.join(pre,  on=["match_id", "shock_id", "player_id", "shock_type"], how="left") \
              .join(post, on=["match_id", "shock_id", "player_id", "shock_type"], how="left")

    if cache:
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ----------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M11_fisico ELITE pipeline completo ===\n")

    # Paso 1: Metricas raw Bradley 2024 SOTA
    print("[1] Metricas raw fisicas (Butterworth dual cutoff + segmentacion teleports)")
    t0 = time.time()
    raw = build_raw_per_minute(cache=True, overwrite=False)
    print(f"  raw_per_minute: {raw.height:,} filas en {time.time()-t0:.1f}s")
    print(f"  matches: {raw['pff_match_id'].n_unique()}/64, players: {raw['player_id'].n_unique()}")

    # Acceptance Bradley 2024: filtrar SOLO STARTERS (>= 60 min jugados)
    per_pm = raw.group_by(["pff_match_id", "player_id"]).agg([
        pl.col("distance_m").sum().alias("dist_m"),
        pl.col("hsr_s").sum().alias("hsr_s"),
        pl.col("sprint_s").sum().alias("sprint_s"),
        pl.col("sprint_count").sum().alias("n_sprints"),
        pl.col("psv95").max().alias("peak_mps"),
        pl.col("n_high_accel").sum().alias("accel_s"),
        pl.col("n_high_decel").sum().alias("decel_s"),
        pl.col("hmld_m").sum().alias("hmld_m"),
        pl.col("n_frames").sum().alias("frames"),
    ])
    starters = per_pm.filter(pl.col("frames") >= 60 * 25 * 60)   # >=60min @ 25Hz
    print(f"\n[Acceptance Bradley 2024 — STARTERS solo (>=60 min jugados)]")
    print(f"jugador-partidos starter: {starters.height}")
    if starters.height > 0:
        for col, lbl, unit, scale in [
            ("dist_m",   "distance",  "km", 1/1000),
            ("hsr_s",    "HSR",       "s",  1.0),
            ("sprint_s", "sprint",    "s",  1.0),
            ("n_sprints","#sprints",  "",   1.0),
            ("peak_mps", "peak speed","km/h", 3.6),
            ("accel_s",  "high_accel","s",  1.0),
            ("decel_s",  "high_decel","s",  1.0),
            ("hmld_m",   "HMLD",      "m",  1.0),
        ]:
            arr = starters[col].to_numpy() * scale
            p25, p50, p75, p99 = np.percentile(arr, [25, 50, 75, 99])
            print(f"  {lbl:<12} {unit:<5}: p25={p25:>7.2f}, p50={p50:>7.2f}, "
                  f"p75={p75:>7.2f}, p99={p99:>7.2f}")

    # Top 10 starters distancia
    print(f"\n[Top 10 starters por distancia/partido]")
    print(starters.with_columns(
        (pl.col("dist_m")/1000).round(2).alias("km"),
        (pl.col("peak_mps")*3.6).round(1).alias("peak_kmh"),
    ).sort("km", descending=True).head(10).select(
        ["player_id","km","hsr_s","sprint_s","n_sprints","peak_kmh","accel_s","decel_s"]
    ))

    # Paso 3-4: Modelo bayesiano fatiga + score_phys + per_shock_window
    print("\n[3] Modelo bayesiano fatiga state-space + score_phys")
    t0 = time.time()
    pm = cache_score_phys(overwrite=True, n_steps=4000)
    print(f"  per_minute: {pm.height:,} filas en {time.time()-t0:.1f}s")
    print(f"  score_phys range: [{pm['score_phys'].min():.3f}, {pm['score_phys'].max():.3f}]")
    print(f"  score_phys mean (esperado ~0): {pm['score_phys'].mean():+.4f}")
    print(f"  score_phys std (esperado ~1):  {pm['score_phys'].std():.4f}")
    print(f"  fatigue range: [{pm['fatigue'].min():.2f}, {pm['fatigue'].max():.2f}]")

    print("\n[4] Aggregate per_shock_window")
    t0 = time.time()
    ps = aggregate_per_shock_window(cache=True)
    print(f"  per_shock_window: {ps.height:,} filas en {time.time()-t0:.1f}s")
    summary = ps.group_by("shock_type").agg([
        pl.col("score_phys_pre").mean().alias("phys_pre"),
        pl.col("score_phys_post").mean().alias("phys_post"),
        (pl.col("score_phys_post") - pl.col("score_phys_pre")).mean().alias("delta_phys"),
        pl.col("fatigue_pre").mean().alias("fat_pre"),
        pl.col("fatigue_post").mean().alias("fat_post"),
    ])
    print("  delta score_phys por shock_type (signo positivo = aprieta tras shock):")
    print(summary)
