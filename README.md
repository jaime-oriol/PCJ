# PCJ: Perfil Clutch del Jugador

TFM Master Big Data Aplicado al Scouting Deportivo (Sport Data Campus).

Estimacion causal del efecto del shock emocional (gol a favor / gol en contra) sobre el comportamiento del jugador en ventanas pre vs post de ±10 min, medido en cuatro canales (Empuje Ofensivo, Solidez Defensiva, Inteligencia Espacial Off-ball, Pulso Fisico) sobre PFF FC World Cup Qatar 2022.

Output: ranking bidireccional de jugadores clutch (Indice Remontador + Indice Cerrojo) con intervalos de credibilidad bayesianos.

## Pipeline

```text
src/
├── extract/                # extractores raw JSON -> parquet (lossless)
├── M01_loader_pff.py       # API PFF (events, tracking, metadata, rosters + vistas)
├── M02_loader_public.py    # API Wyscout + StatsBomb (polars nativo)
├── M03_preprocess.py       # direction, score state (SB ground truth), minutos, enrich_events
├── M04_wp.py               # Win Probability bayesiana (numpyro SVI, ordered-logistic
│                           #   tiempo-variables) + leverage + ET Poisson + tanda Tijms
│                           #   + Monte Carlo del grupo para elim_prox
├── M05_psxg.py             # Post-shot xG (LightGBM + Optuna 60 trials + isotonic
│                           #   + freeze-frame 360 + permutation importance)
│                           #   AUC OOF 0.974 / WC22 holdout 0.976 (vs SB baseline 0.827)
├── M06_nearmiss.py         # Near-miss 5 tipos (palo, offside milimetrico via 360,
│                           #   PSxG-save, GLC, GLT) + specification curve Simonsohn
├── M07_shocks.py           # 172 shocks-gol + ventanas ±10min por jugador en campo
├── M08_ataque.py           # Empuje Ofensivo: atomic-VAEP CatBoost + Optuna 30 + 5-fold
│                           #   CV by match + isotonic + mapping SB->PFF (74.9%)
├── M09_defensa.py          # Solidez Defensiva: score_def + vdep_minute (Toda 2022)
│                           #   + def_third_pct + pressing_intensity (Bekkers 2024)
│                           #   via tracking PFF 25Hz vectorizado polars
├── M10_offball.py          # Off-ball: OBSO + C-OBSO (Spearman 2018 + Teranishi 2022)
│                           #   PPCF Z02 + xG grid + tracking PFF 25Hz full quality
├── M11_fisico.py           # Pulso Fisico: metricas Bradley 2024 (HSR/sprint/PSV95/
│                           #   Z1-Z5/HMLD/accel/decel) con Hampel + Butterworth dual
│                           #   cutoff (1Hz vel, 2Hz acc) + state-space bayesiano de
│                           #   fatiga (numpyro SVI) -> score_phys = residuo z-score
├── (M12-M16)               # pipeline restante (DiD + AIPW + CATE + ranking final)
├── Z01_vaep.py             # building block atomic-VAEP wrapper (compute_features/labels
│                           #   + save_models/load_models, usado por M08/M09)
└── Z02_pitch_control.py    # building block PPCF Spearman 2018 vectorizado (core
│                           #   agnostico al proveedor, usado por M10)

notebooks/
└── M10_run.ipynb           # M10 ejecutado a 25 Hz partido a partido (resumable:
                            # cachea cada partido aparte, retoma tras interrupcion)
```

Estado: M01-M11 ejecutados sobre los 64 partidos WC22. M12-M16 pendientes.

Datos, documentacion interna del proyecto y outputs intermedios estan fuera del
repo (`.gitignore`).

## Stack

Python (polars, pyarrow, pandas) +
modelos (catboost, lightgbm, numpyro/jax, scikit-learn) +
hyperparam tuning (optuna) + acciones (socceraction atomic-VAEP).
