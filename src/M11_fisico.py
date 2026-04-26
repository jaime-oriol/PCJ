"""
M11_fisico - Canal Pulso Fisico via metricas Bradley 2024 SOTA + residual bayesiano.

Fase 2 PCJ, canal 4 de 4. Aisla "lo que el jugador APRIETA" (decision mental)
de "lo que el jugador PUEDE" (estado fisico) reportando el RESIDUO de tres
metricas-rate observadas vs lo predicho por:
  baseline_jugador (capacidad personal) + curva_temporal_minuto (fatiga media).

Pipeline:
  1. Limpieza tracking + velocidades smoothed:
     - Segmentacion por discontinuidades fisicas (camera switch / ID swap).
     - Butterworth lowpass filtfilt (fase cero, cutoff 1 Hz, Buchheit standard).
     - Hampel filter sobre velocidad para outliers residuales.
     - Cap a 11 m/s con re-scale proporcional preservando direccion.
  2. Metricas frame-level agregadas por (player, match, minute) — Bradley 2024:
       - distance_m       : integral vel * dt (m).
       - hsr_s            : segundos a >= 19.8 km/h (Ju et al. 2022).
       - sprint_s         : segundos a >= 25 km/h.
       - sprint_count     : # eventos sprint distintos (onset>=1s + recovery>=2s).
       - psv95            : peak speed velocity (p95 robusto del minuto, m/s).
       - n_high_accel     : segundos con a >= 3 m/s²  (Akenhead 2013).
       - n_high_decel     : segundos con a <= -3 m/s² (Akenhead 2013).
       - z1_m..z5_m       : distancia por zona Bradley (Z1<7, Z2 7-13, Z3 13-19.8,
                             Z4 19.8-25, Z5>25 km/h).
       - hmld_m           : High Metabolic Load Distance, Osgnach et al. 2010
                             (P_metabolic >= 25.5 W/kg, Eq 7-9).
  3. Modelo bayesiano jerarquico multivariate (numpyro SVI, 3 RATES):
       log(rate_k[p,m,t]) ~ Normal(
           mu_player[p,k] + b1[k]*(t/90) + b2[k]*(t/90)^2,
           sigma_eps[k]
       )   k in {psv95, mean_speed, hsr_rate}
       mu_player[p,k] ~ Normal(mu_global[k], sigma_p[k])  random effects player
     score_phys = mean(z-score residuos sobre 3 targets).
     NO modelamos fatigue explicita: el baseline mu_player + curva temporal
     quadratic ya encierra la fatigue media-esperada. El residuo capta
     "se desvio del baseline esperado para EL en ESE minuto" — exactamente
     "APRIETA mas/menos de lo esperado". Targets son RATES (independientes de
     cobertura n_frames del minute), evitando contaminacion por subs / stoppage.
  4. Agregacion per_shock_window (pre/post +-10min) con score_phys + componentes.

Acceptance (ARCHITECTURE.md + Bradley 2024 WC22):
  - Distancia top starters ~10-12 km/partido.
  - PSV95 top players ~32-37 km/h (sin cap saturado).
  - n_high_accel ~50-100 s/jugador-partido, simetrico con n_high_decel.
  - score_phys mean ~0, std ~0.85 (z-scores correlados con mean(z)).

Output:
  data/parquet/derived/fisico/
    raw_per_minute.parquet      # metricas raw frame-level agregadas.
    model/phys_state.pkl        # SVI fit (parametros posterior).
    per_minute.parquet          # score_phys (residuo z-score) per (m, p, min).
    per_shock_window.parquet    # pre/post de cada shock.

Depende de: M01 (tracking, rosters), M07 (shocks_table).
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

# Butterworth fase-cero, cutoff 1 Hz (Buchheit standard biomecanica deportiva).
# Suaviza ruido alto-frecuencia preservando dinamica de movimiento real (gestos
# > ~1s). Aceleracion tangencial signed_accel se deriva del speed smoothed.
_BUTTER_ORDER     = 4
_BUTTER_CUTOFF_HZ = 1.0
_FPS_DEFAULT      = 25.0

# Min frames para procesar un segmento. filtfilt default padlen = 3*max(len(a),len(b))
# con order=4 -> padlen=15. Necesitamos len(signal) > 15 -> minimo 16.
_MIN_SEGMENT_FRAMES = 16


# ===========================================================================
#  SECCION 1 — Limpieza tracking + velocidades smoothed
# ===========================================================================

def _butter_lowpass_filtfilt(signal: np.ndarray, fs: float = _FPS_DEFAULT,
                              cutoff: float = _BUTTER_CUTOFF_HZ,
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
                                          np.ndarray]:
    """Procesa 1 segmento contiguo de posiciones.

    Pipeline (todo cutoff 1Hz, Buchheit standard):
      1. Butterworth lowpass filtfilt (fase cero) sobre px, py.
      2. np.gradient / dt -> vx, vy, speed.
      3. Hampel filter sobre speed: limpia outliers residuales en bordes
         de segmento (gradient artifacts).
      4. signed_accel = gradient(speed, dt): aceleracion tangencial signed
         (positive = acelera, negative = frena). Coherente con threshold
         HSR/sprint sobre la misma speed smoothed.
      5. Cap final sobre speed con re-scale proporcional a vx/vy
         (preserva direccion).

    La segmentacion previa por teleports elimina discontinuidades fisicas,
    asi que Hampel(speed) ataca solo gradient artifacts residuales.

    Returns:
        (vx, vy, speed, signed_accel), todos shape (n,) en m/s y m/s².
    """
    n = len(px_seg)
    if n < _MIN_SEGMENT_FRAMES:
        return (np.zeros(n),) * 4
    dt = 1.0 / fs

    # 1-2. Butterworth + gradient
    px_v = _butter_lowpass_filtfilt(px_seg, fs)
    py_v = _butter_lowpass_filtfilt(py_seg, fs)
    vx = np.gradient(px_v, dt)
    vy = np.gradient(py_v, dt)
    speed = np.sqrt(vx ** 2 + vy ** 2)

    # 3. Hampel sobre speed (outliers residuales)
    speed = _hampel_filter(speed)

    # 4. Signed accel tangencial
    signed_accel = np.gradient(speed, dt)

    # 5. Cap final con re-scale proporcional
    cap_mask = speed > MAX_HUMAN_SPEED_MPS
    if cap_mask.any():
        scale = MAX_HUMAN_SPEED_MPS / speed[cap_mask]
        vx[cap_mask] *= scale
        vy[cap_mask] *= scale
        speed[cap_mask] = MAX_HUMAN_SPEED_MPS

    return vx, vy, speed, signed_accel


def _compute_velocities_clean(positions_x: np.ndarray, positions_y: np.ndarray,
                               fs: float = _FPS_DEFAULT
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray]:
    """Pipeline ELITE: posiciones -> (vx, vy, speed, signed_accel).

    Segmenta por teleports y procesa cada segmento independientemente con
    `_process_segment` (Butterworth filtfilt + Hampel + cap proporcional).
    """
    n = len(positions_x)
    vx = np.zeros(n)
    vy = np.zeros(n)
    speed = np.zeros(n)
    signed_a = np.zeros(n)
    if n < 2:
        return vx, vy, speed, signed_a

    dt = 1.0 / fs
    for (s, e) in _segment_by_teleports(positions_x, positions_y, dt):
        if e - s < _MIN_SEGMENT_FRAMES:
            continue
        seg_vx, seg_vy, seg_sp, seg_sa = _process_segment(
            positions_x[s:e], positions_y[s:e], fs,
        )
        vx[s:e]       = seg_vx
        vy[s:e]       = seg_vy
        speed[s:e]    = seg_sp
        signed_a[s:e] = seg_sa

    return vx, vy, speed, signed_a


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

# Offsets para convertir (period, minute) <-> seconds-since-match-start.
# Coinciden con la convencion PFF/SB: period 1 [0, 45min), 2 [45, 90),
# 3 ET1 [90, 105), 4 ET2 [105, 120).
PERIOD_OFFSET_MIN = {1: 0, 2: 45, 3: 90, 4: 105}

_PHYS_SCHEMA = {
    "pff_match_id":     pl.Int64, "pff_player_id": pl.Int64,
    "period":           pl.Int64, "minute_in_period": pl.Int64,
    "distance_m":       pl.Float64,
    "hsr_s":            pl.Float64,
    "sprint_s":         pl.Float64,
    "sprint_count":     pl.Int64,
    "psv95":            pl.Float64,
    "n_high_accel":     pl.Float64,
    "n_high_decel":     pl.Float64,
    "z1_m":             pl.Float64, "z2_m": pl.Float64, "z3_m": pl.Float64,
    "z4_m":             pl.Float64, "z5_m": pl.Float64,
    "hmld_m":           pl.Float64,
    "n_frames":         pl.Int64,
}


def _aggregate_minute_metrics(speed: np.ndarray, signed_accel: np.ndarray,
                               p_metabolic: np.ndarray,
                               minutes: np.ndarray, dt: float) -> list[dict]:
    """Agrega metricas frame-level por minute. Devuelve lista de dicts."""
    rows = []
    for m in np.unique(minutes):
        mask = minutes == m
        if mask.sum() < 3:
            continue
        sp = speed[mask]
        sa = signed_accel[mask]
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
            "minute_in_period": int(m),
            "distance_m":       distance,
            "hsr_s":            hsr,
            "sprint_s":         sprint,
            "sprint_count":     sprintc,
            "psv95":            psv95,
            "n_high_accel":     accel_s,
            "n_high_decel":     decel_s,
            "z1_m":             zone_m[0], "z2_m": zone_m[1], "z3_m": zone_m[2],
            "z4_m":             zone_m[3], "z5_m": zone_m[4],
            "hmld_m":           hmld,
            "n_frames":         int(mask.sum()),
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
        per = int(period)

        x = group["x"].to_numpy()
        y = group["y"].to_numpy()
        fn = group["frameNum"].to_numpy()
        vx, vy, speed, signed_a = _compute_velocities_clean(x, y, fs)
        p_meta = _metabolic_power(speed, signed_a)

        # frameNum es absolute (no se resetea entre periods) -> reconstruir
        # minute desde el inicio del period (frame_period_start = min frameNum
        # del grupo). Asi minute es period-relative, comparable con M07
        # window_pre_start / window_post_end (que estan en period-relative seconds).
        period_start = fn.min()
        minutes_in_period = ((fn - period_start) // frames_per_min).astype(np.int64)
        for r in _aggregate_minute_metrics(speed, signed_a,
                                            p_meta, minutes_in_period, dt):
            r["pff_match_id"]  = match_id
            r["pff_player_id"] = pid
            r["period"]        = per
            rows_out.append(r)

    if not rows_out:
        return pl.DataFrame(schema=_PHYS_SCHEMA)
    out = pl.DataFrame(rows_out, schema_overrides=_PHYS_SCHEMA)
    # (player, period, minute_in_period) ya es unique por construccion.
    return out.sort(["pff_match_id", "pff_player_id",
                      "period", "minute_in_period"])


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
#  SECCION 3 — Modelo bayesiano jerarquico multivariate (numpyro SVI)
# ===========================================================================
#
#  Diseño causal honesto: NO modelamos fatigue explicitamente. La razon es
#  que cualquier proxy minute-level de fatigue (e.g., HMLD acumulada) esta
#  altamente correlado con la actividad reciente del propio jugador, que
#  a su vez esta autocorrelada con la actividad actual. Eso hace que
#  beta_fat capture autocorrelacion, no causalidad fatigue->rendimiento.
#
#  En lugar de eso, modelamos el rendimiento fisico como funcion de:
#      mu_player[p, k]               (random effect: capacidad personal del
#                                     jugador en target k)
#      b1[k] * (t/90) + b2[k] * (t/90)^2   (curva temporal del partido,
#                                     captura U-shape: warm-up + decline)
#  El RESIDUAL del modelo es exactamente lo que el TFM busca: "se desvio
#  del rendimiento esperado para EL en ESE momento del partido". Esto
#  encierra implicitamente la fatigue media-esperada (via curva temporal)
#  + capacidad personal (via mu_player). Los desvios son interpretables
#  causalmente como "APRIETA mas / menos de lo esperado".
#
#  Modelo:
#      log(rate_k[p,m,t]) ~ Normal(
#          mu_player[p,k] + b1[k]*(t/90) + b2[k]*(t/90)^2,
#          sigma_eps[k]
#      )
#      mu_player[p,k] ~ Normal(mu_global[k], sigma_p[k])
#      b1[k], b2[k] ~ Normal(0, 0.5)
#      sigma_p[k], sigma_eps[k] ~ HalfNormal(0.5)
#
#  Targets (RATES, independientes de cobertura n_frames del minute):
#      psv95         (m/s)  : peak speed velocity p95 robusto.
#      mean_speed    (m/s)  : distance_m / (n_frames * dt).
#      hsr_rate     [0,1]   : hsr_s / (n_frames * dt). Fraccion del minute
#                              en HSR (>=19.8 km/h).
#  Log-transformados: log(psv95), log(mean_speed), log(hsr_rate + 0.01).
#
#  score_phys = mean(z-score residuos sobre 3 targets).

_TARGETS = ["log_psv95", "log_mean_speed", "log_hsr_rate"]
_N_TARGETS = len(_TARGETS)
_HSR_RATE_FLOOR = 0.01    # evita log(0) cuando hsr_rate=0


def _add_rates(raw: pl.DataFrame, fps: float = _FPS_DEFAULT) -> pl.DataFrame:
    """Anade cols mean_speed (m/s), hsr_rate ([0,1]) — RATES, no integrales.

    distance_m, hsr_s son integrales sobre n_frames. Las RATES son
    independientes de cobertura del minuto (subs, stoppage, etc).
    """
    return raw.with_columns([
        (pl.col("distance_m") / (pl.col("n_frames") / fps))
            .alias("mean_speed_mps"),
        (pl.col("hsr_s") / (pl.col("n_frames") / fps))
            .alias("hsr_rate"),
    ])


def _prepare_targets(df: pl.DataFrame
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepara matriz Y (N, 3) de log(rates) + filtros minimos validos.

    minute_norm convierte (period, minute_in_period) a fraccion de partido
    en [0, 1.33] usando PERIOD_OFFSET_MIN: minute_match = offset + minute,
    minute_norm = minute_match / 90.

    Filtro laxo `n_frames >= 200` (8s): suficiente para rates estables sin
    perder muchos minutos validos (subs, stoppage).

    Returns: (y, minute_norm, player_ids).
    """
    df_clean = df.filter(
        (pl.col("psv95") > 0.5) &
        (pl.col("mean_speed_mps") > 0.1) &      # > 0.36 km/h promedio (no estatico)
        (pl.col("n_frames") >= 200)              # >=8s cobertura
    )
    log_psv = np.log(df_clean["psv95"].to_numpy())
    log_ms  = np.log(df_clean["mean_speed_mps"].to_numpy())
    log_hsr = np.log(df_clean["hsr_rate"].to_numpy() + _HSR_RATE_FLOOR)
    y = np.stack([log_psv, log_ms, log_hsr], axis=-1)
    period = df_clean["period"].to_numpy()
    minute = df_clean["minute_in_period"].to_numpy()
    offset = np.array([PERIOD_OFFSET_MIN.get(int(p), 0) for p in period])
    minute_norm = (offset + minute) / 90.0
    return y, minute_norm, df_clean["pff_player_id"].to_numpy()


def fit_phys_model(df_with_rates: pl.DataFrame,
                    n_steps: int = 4000, seed: int = 42) -> dict:
    """Entrena modelo bayesiano jerarquico multivariate (3 targets).

    Modelo limpio sin endogeneidad: solo mu_player + curva temporal del
    partido. Residual = "desviacion del baseline jugador-minuto esperado".

    Returns dict con SVI params posterior + indexer player_id -> idx.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    y, mn, player_ids = _prepare_targets(df_with_rates)
    if len(y) == 0:
        raise ValueError("sin datos validos para fit_phys_model")

    uniq_p = np.unique(player_ids)
    p_to_idx = {int(p): i for i, p in enumerate(uniq_p)}
    p_idx = np.array([p_to_idx[int(p)] for p in player_ids], dtype=np.int32)
    n_players = len(uniq_p)

    # Priors mu_global centrados en la log-mean observada (data-driven).
    # Escalas tipicas: log(psv95) ~ 1.8, log(mean_speed) ~ 0.2, log(hsr_rate+0.01) ~ -3.4.
    mu_g_prior = jnp.asarray(y.mean(axis=0))

    def model(p_idx_, mn_, y_=None):
        sigma_p = numpyro.sample(
            "sigma_p", dist.HalfNormal(0.5).expand([_N_TARGETS]).to_event(1)
        )
        mu_g = numpyro.sample(
            "mu_global", dist.Normal(mu_g_prior, 1.0).to_event(1)
        )
        mu_p = numpyro.sample(
            "mu_player",
            dist.Normal(mu_g[None, :], sigma_p[None, :])
                 .expand([n_players, _N_TARGETS]).to_event(2),
        )
        # Quadratic en minute_norm: captura U-shape (warm-up + decline)
        b1 = numpyro.sample("b1", dist.Normal(0.0, 0.5)
                                .expand([_N_TARGETS]).to_event(1))
        b2 = numpyro.sample("b2", dist.Normal(0.0, 0.5)
                                .expand([_N_TARGETS]).to_event(1))
        sigma_eps = numpyro.sample(
            "sigma_eps", dist.HalfNormal(0.5).expand([_N_TARGETS]).to_event(1)
        )

        mn2 = mn_ ** 2
        pred = (mu_p[p_idx_]
                + b1[None, :] * mn_[:, None]
                + b2[None, :] * mn2[:, None])
        with numpyro.plate("N", len(p_idx_)):
            numpyro.sample(
                "obs",
                dist.Normal(pred, sigma_eps[None, :]).to_event(1),
                obs=y_,
            )

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())

    state = svi.init(
        jax.random.PRNGKey(seed),
        jnp.asarray(p_idx), jnp.asarray(mn), jnp.asarray(y),
    )
    for i in range(n_steps):
        state, loss = svi.update(state, jnp.asarray(p_idx),
                                  jnp.asarray(mn), jnp.asarray(y))
        if i % 500 == 0:
            print(f"  step {i:5d}  elbo_loss={float(loss):.2f}")
    params = svi.get_params(state)
    return {
        "params": params,
        "p_to_idx": p_to_idx,
        "n_obs": int(len(y)),
        "n_players": n_players,
        "targets": _TARGETS,
    }


def compute_score_phys(df_with_rates: pl.DataFrame, fit: dict) -> pl.DataFrame:
    """Computa score_phys = mean z-score residuos sobre 3 targets.

    Filtros y transformaciones de target IDENTICOS a los del entrenamiento
    (`_prepare_targets`) -> evita disparidad train/predict.

    score_phys positivo = APRIETA mas de lo esperado en ese minuto.
    """
    p = fit["params"]
    p_to_idx = fit["p_to_idx"]

    # Mismo filter que entrenamiento + map pff_player_id -> idx
    df_full = df_with_rates.filter(
        (pl.col("psv95") > 0.5) &
        (pl.col("mean_speed_mps") > 0.1) &
        (pl.col("n_frames") >= 200)
    ).with_columns(
        pl.col("pff_player_id").replace_strict(p_to_idx, default=-1).alias("p_idx")
    ).filter(pl.col("p_idx") >= 0)

    if df_full.height == 0:
        raise ValueError("sin filas validas para compute_score_phys")

    p_idx = df_full["p_idx"].to_numpy()
    period = df_full["period"].to_numpy()
    minute = df_full["minute_in_period"].to_numpy()
    offset = np.array([PERIOD_OFFSET_MIN.get(int(pp), 0) for pp in period])
    mn = (offset + minute) / 90.0
    log_psv = np.log(df_full["psv95"].to_numpy())
    log_ms  = np.log(df_full["mean_speed_mps"].to_numpy())
    log_hsr = np.log(df_full["hsr_rate"].to_numpy() + _HSR_RATE_FLOOR)
    y_obs = np.stack([log_psv, log_ms, log_hsr], axis=-1)

    mu_p = np.array(p["mu_player_auto_loc"])           # (n_players, 3)
    b1 = np.array(p["b1_auto_loc"])                    # (3,)
    b2 = np.array(p["b2_auto_loc"])                    # (3,)
    sigma_eps = np.exp(np.array(p["sigma_eps_auto_loc"]))   # (3,)

    pred = (mu_p[p_idx]
            + b1[None, :] * mn[:, None]
            + b2[None, :] * (mn[:, None] ** 2))
    residual = y_obs - pred
    z_per_target = residual / sigma_eps[None, :]
    score_phys = z_per_target.mean(axis=1)

    # sec_abs derivado de PERIOD_OFFSET_MIN para alinear con M07 windows.
    # Resolucion del minuto, el residual del modelo es per-minuto.
    return df_full.with_columns([
        pl.Series("z_psv95",     z_per_target[:, 0]),
        pl.Series("z_meanspd",   z_per_target[:, 1]),
        pl.Series("z_hsr",       z_per_target[:, 2]),
        pl.Series("score_phys",  score_phys),
        ((pl.col("period").replace_strict(PERIOD_OFFSET_MIN, default=0)
          + pl.col("minute_in_period")) * 60).cast(pl.Int64).alias("sec_abs"),
    ]).select([
        "pff_match_id", "pff_player_id",
        "period", "minute_in_period", "sec_abs",
        "z_psv95", "z_meanspd", "z_hsr", "score_phys",
    ])


def cache_score_phys(overwrite: bool = False, n_steps: int = 4000) -> pl.DataFrame:
    """Pipeline completa: rates -> SVI multivariate -> score_phys."""
    import pickle
    cache_path = _DERIVED / "per_minute.parquet"
    fit_path   = _DERIVED / "model" / "phys_state.pkl"
    if cache_path.exists() and fit_path.exists() and not overwrite:
        return pl.read_parquet(cache_path)

    raw = build_raw_per_minute(cache=True)
    df_rates = _add_rates(raw)
    print(f"  fitting SVI multivariate phys model "
          f"({df_rates.height:,} obs, {n_steps} steps, 3 targets)...")
    fit = fit_phys_model(df_rates, n_steps=n_steps)
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fit_path, "wb") as f:
        pickle.dump({k: (np.array(v) if hasattr(v, "shape") else v)
                     for k, v in fit.items() if k != "params"} |
                    {"params": {k: np.array(v) for k, v in fit["params"].items()}}, f)

    out = compute_score_phys(df_rates, fit)
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

    pm = cache_score_phys()
    shocks = build_shocks_table(cache=True, overwrite=False)

    # Schema X3: M11 per_minute publica sec_abs ya alineado con M07 windows
    # (que tambien estan en sec_abs PFF). Join por (pff_match_id, pff_player_id)
    # y filter por window selecciona automaticamente period correcto.
    pm = pm.rename({"pff_match_id": "match_id"})
    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        pl.col("period").alias("shock_period"),
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    joined = shocks_slim.join(pm, on=["match_id", "pff_player_id"], how="left")

    # period == shock_period evita contaminacion cross-period (ver M08 doc)
    pre = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_pre_start")) &
        (pl.col("sec_abs") < pl.col("window_pre_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id", "shock_id", "pff_player_id", "shock_type"]).agg([
        pl.col("score_phys").mean().alias("score_phys_pre"),
        pl.col("z_psv95").mean().alias("z_psv95_pre"),
        pl.col("z_meanspd").mean().alias("z_meanspd_pre"),
        pl.col("z_hsr").mean().alias("z_hsr_pre"),
    ])
    post = joined.filter(
        (pl.col("sec_abs") >= pl.col("window_post_start")) &
        (pl.col("sec_abs") <= pl.col("window_post_end")) &
        (pl.col("period") == pl.col("shock_period"))
    ).group_by(["match_id", "shock_id", "pff_player_id", "shock_type"]).agg([
        pl.col("score_phys").mean().alias("score_phys_post"),
        pl.col("z_psv95").mean().alias("z_psv95_post"),
        pl.col("z_meanspd").mean().alias("z_meanspd_post"),
        pl.col("z_hsr").mean().alias("z_hsr_post"),
    ])

    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    # ids canonicos (X3): sb_match_id + sb_player_id via mappings publicos.
    from M03_preprocess import pff_to_sb_match_id
    import M08_ataque as atk
    pff2sb_match = pff_to_sb_match_id()
    pff_to_sb_pl = atk.build_sb_to_pff_player_map(cache=True).select([
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
        pl.col("sb_player_id").cast(pl.Int64),
    ]).filter(pl.col("pff_player_id").is_not_null()).unique(
        subset=["pff_player_id"], keep="first",
    )

    out = (
        base
        .join(pre,  on=["match_id", "shock_id", "pff_player_id", "shock_type"],
              how="left")
        .join(post, on=["match_id", "shock_id", "pff_player_id", "shock_type"],
              how="left")
        .rename({"match_id": "pff_match_id"})
        .join(pff_to_sb_pl, on="pff_player_id", how="left")
        .with_columns(
            pl.col("pff_match_id").replace_strict(pff2sb_match, default=None)
                                    .alias("sb_match_id")
        )
        .select([
            "pff_match_id", "sb_match_id",
            "shock_id", "shock_type",
            "pff_player_id", "sb_player_id",
            "score_phys_pre", "score_phys_post",
            "z_psv95_pre", "z_psv95_post",
            "z_meanspd_pre", "z_meanspd_post",
            "z_hsr_pre", "z_hsr_post",
        ])
    )

    if cache:
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ----------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M11_fisico ELITE pipeline completo ===\n")

    # Paso 1: Metricas raw Bradley 2024 SOTA
    print("[1] Metricas raw fisicas (Butterworth 1Hz + Hampel + segmentacion teleports)")
    t0 = time.time()
    raw = build_raw_per_minute(cache=True, overwrite=False)
    print(f"  raw_per_minute: {raw.height:,} filas en {time.time()-t0:.1f}s")
    print(f"  matches: {raw['pff_match_id'].n_unique()}/64, players: {raw['player_id'].n_unique()}")

    # Acceptance Bradley 2024: filtrar SOLO STARTERS (>= 60 min jugados)
    per_pm = raw.group_by(["pff_match_id", "pff_player_id"]).agg([
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
        ["pff_player_id","km","hsr_s","sprint_s","n_sprints","peak_kmh",
         "accel_s","decel_s"]
    ))

    # Paso 3-4: Modelo bayesiano residualizado + score_phys + per_shock_window
    print("\n[3] Modelo bayesiano jerarquico multivariate + score_phys")
    t0 = time.time()
    pm = cache_score_phys(overwrite=True, n_steps=4000)
    print(f"  per_minute: {pm.height:,} filas en {time.time()-t0:.1f}s")
    print(f"  score_phys range: [{pm['score_phys'].min():.3f}, {pm['score_phys'].max():.3f}]")
    print(f"  score_phys mean (esperado ~0): {pm['score_phys'].mean():+.4f}")
    print(f"  score_phys std (esperado ~1):  {pm['score_phys'].std():.4f}")
    print(f"  z_psv95 / z_meanspd / z_hsr means: "
          f"{pm['z_psv95'].mean():+.4f} / {pm['z_meanspd'].mean():+.4f} / {pm['z_hsr'].mean():+.4f}")

    print("\n[4] Aggregate per_shock_window")
    t0 = time.time()
    ps = aggregate_per_shock_window(cache=True)
    print(f"  per_shock_window: {ps.height:,} filas en {time.time()-t0:.1f}s")
    summary = ps.group_by("shock_type").agg([
        pl.col("score_phys_pre").mean().alias("phys_pre"),
        pl.col("score_phys_post").mean().alias("phys_post"),
        (pl.col("score_phys_post") - pl.col("score_phys_pre")).mean().alias("delta_phys"),
        pl.col("z_psv95_pre").mean().alias("psv_pre"),
        pl.col("z_psv95_post").mean().alias("psv_post"),
        pl.col("z_meanspd_pre").mean().alias("spd_pre"),
        pl.col("z_meanspd_post").mean().alias("spd_post"),
        pl.col("z_hsr_pre").mean().alias("hsr_pre"),
        pl.col("z_hsr_post").mean().alias("hsr_post"),
    ])
    print("  delta score_phys por shock_type (signo positivo = aprieta tras shock):")
    print(summary)
