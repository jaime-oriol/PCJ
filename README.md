# PCJ: Perfil Clutch del Jugador

TFM Master Big Data Aplicado al Scouting Deportivo (Sport Data Campus).

Estimacion causal del efecto del shock emocional (gol a favor / gol en contra) sobre el comportamiento del jugador en ventanas pre vs post de ±10 min, medido en cuatro canales (Empuje Ofensivo, Solidez Defensiva, Inteligencia Espacial Off-ball, Pulso Fisico) sobre PFF FC World Cup Qatar 2022.

Output: ranking tridimensional de jugadores clutch (Indice Remontador post-GA + Indice Cerrojo post-GF + Pressure Response continuo en elim_prox) con intervalos de credibilidad bayesianos.

## Pipeline

```text
src/
├── extract/                # extractores raw JSON -> parquet (lossless)
├── preprocess/
│   └── pff_grades_extract.py  # priors PFF grades por jugador (input M14)
├── M01_loader_pff.py       # API PFF (events, tracking, metadata, rosters + vistas)
├── M02_loader_public.py    # API Wyscout + StatsBomb (polars nativo)
├── M03_preprocess.py       # direction, score state (SB ground truth), minutos, enrich_events
├── M04_wp.py               # Win Probability bayesiana (numpyro SVI, ordered-logistic
│                           #   tiempo-variables) + leverage + ET Poisson + tanda Tijms
│                           #   + Monte Carlo del grupo para elim_prox
├── M05_psxg.py             # Post-shot xG (LightGBM + Optuna 60 trials + isotonic
│                           #   + freeze-frame 360 + permutation importance)
│                           #   AUC OOF 0.974 / WC22 holdout 0.976 (vs SB baseline 0.827)
├── M05B_calibration.py     # PSxG calibration diagnostics (curve, ECE/MCE, Brier
│                           #   decomposition Murphy 1973, isotonic mapping)
├── M06_nearmiss.py         # Near-miss 5 tipos (palo, offside milimetrico via 360,
│                           #   PSxG-save, GLC, GLT via tracking PFF 25Hz)
├── M07_shocks.py           # 172 shocks-gol + ventanas ±10min por jugador en campo
│                           #   + composicion bloque LOO + helpers attach_team_loo /
│                           #   compute_team_loo_at_minute
├── M08_ataque.py           # Empuje Ofensivo: atomic-VAEP CatBoost + Optuna 30 + 5-fold
│                           #   CV by match + isotonic + un-xPass (Z06) -> score_atk_v2;
│                           #   mapping SB->PFF cascada 5 pases
├── M09_defensa.py          # Solidez Defensiva: score_def_v4 = vdep_strict (Z04) +
│                           #   xpress (Z03) + maejima (Z05) + def_third_pct + pressing
│                           #   intensity + press_value Maejima light via PFF events
├── M10_offball.py          # Off-ball: OBSO + C-OBSO (Spearman 2018 + Teranishi 2022)
│                           #   PPCF Z02 + xG grid + tracking PFF 25Hz full quality
├── M11_fisico.py           # Pulso Fisico: metricas Bradley 2024 (HSR/sprint/PSV95/
│                           #   Z1-Z5/HMLD/accel/decel) con Hampel + Butterworth +
│                           #   segmentacion teleports + modelo bayesiano jerarquico
│                           #   multivariate (numpyro SVI 3 RATES) -> score_phys
├── M12_did.py              # DiD within-player: ATE FE (player_shock + post,
│                           #   cluster player) + event-study Sun-Abraham 2021 +
│                           #   BJS imputation (Borusyak-Jaravel-Spiess 2024) +
│                           #   HonestDiD (Rambachan-Roth 2023) + pre-trends F-test
├── M12B_validation.py      # SOTA causal robustness suite:
│                           #   - placebo test 1000 perm + BH-FDR
│                           #   - statistical power (ICC + MDE@80% + observed)
│                           #   - baseline naive comparison
│                           #   - window sensitivity ±3/5/7/10/15 min
│                           #   - stage stratification (groups vs KO)
├── M13_aipw.py             # AIPW cuasi-experimento near-miss: DoubleMLIRM
│                           #   (Chernozhukov 2018, LightGBM cross-fit 5-fold by
│                           #   match) + DML PLR + DR-learner Kennedy 2023 + RDD
│                           #   local-lineal sobre PSxG (Imbens-Kalyanaraman) +
│                           #   spec curve Simonsohn 2020 + balance test
│                           #   Sant'Anna-Song-Xu 2022 + sensitivity Cinelli-
│                           #   Hazlett 2020 + comparison vs M12 ATE
├── M14_cate.py             # CATE multivariate jerarquico bayesiano (numpyro
│                           #   NUTS HMC, 4 chains, Multivariate BCF analog
│                           #   Hu 2025): jerarquia 3 niveles player ⊂ team
│                           #   ⊂ position + LKJCholesky cross-canal x5 +
│                           #   priors informativos PFF grades + 5 etas:
│                           #   eta_ga (post-GA), eta_gf (post-GF),
│                           #   eta_pressure (slope a elim_prox_z),
│                           #   eta_ga_x_teamdir + eta_gf_x_teamdir (slope
│                           #   individual a direccion del bloque) +
│                           #   R-hat/ESS + PPC KS-test
├── M15_pcj.py              # Perfil Clutch del Jugador ensamblaje scout-facing.
│                           #   ~234 jugadores >=270 min × ~200 cols integrando todo
│                           #   el pipeline: 5 perspectivas CATEs M14 (GA, GF,
│                           #   PRESSURE, GA_X_TEAMDIR, GF_X_TEAMDIR) + IC80/95 +
│                           #   16 cells contextualizados (4 canales x 2 shocks x
│                           #   2 directions equipo: team_attacks/team_defends) +
│                           #   3 indices bayesianos (chasing/protecting/pressure)
│                           #   + posterior probs P(idx>0|data) desde NUTS samples
│                           #   + 4-vec PCJ directional + tier labels global/in-position
│                           #   + tier_certain (Elite solo si IC80 excluye 0) + sig
│                           #   flags duales (0.85/0.95) + acute window CATE +-5min
│                           #   + intra-corr cross-canal per jugador + baselines
│                           #   absolutos M08-M11 + age/height + leverage/nearmiss
│                           #   exposure + channel_credibility + power_flag.
│                           #   Schema contract estable validado pre-write.
├── render_ficha.py         # ficha visual scout-facing por jugador. Consume
│                           #   pcj_table.parquet y printa barras unicode +
│                           #   frases scout (mapeo lenguaje del entrenador
│                           #   propuesta_final.md). Uso: python -m src.render_ficha
│                           #   <player_id|name> | --top {chasing,protecting,pressure} N
├── Z01_vaep.py             # building block atomic-VAEP wrapper (compute_features/
│                           #   labels + save_models/load_models, usado por M08/M09)
├── Z02_pitch_control.py    # building block PPCF Spearman 2018 vectorizado (core
│                           #   agnostico al proveedor, usado por M10)
├── Z03_xpress.py           # exPress Lee 2025 P(recovery<5s|press): LightGBM +
│                           #   Optuna + isotonic. 8 features events + 9 tracking
│                           #   25Hz (dist defensor-balon/carrier, n_def_within_5m,
│                           #   def_speed, ball_x_norm, def_ahead_of_carrier...).
├── Z04_vdep.py             # VDEP strict Toda 2022 PLOS ONE: 2 cabezas LightGBM
│                           #   (recovery + attacked) sobre acciones defensivas SPADL.
│                           #   C calibrado como ratio mean(att)/mean(rec).
├── Z05_maejima.py          # Maejima 2024 nearest-defender: atribuye -vaep_value si
│                           #   ataque exitoso o +|vaep_value| si fallado al defensor
│                           #   mas cercano al balon en frame del evento.
└── Z06_unxpass.py          # un-xPass Robberechts 2023 KDD: P(success|features)
                            #   LightGBM. unxpass_value = (success_obs - p_success) ×
                            #   vaep_value. Captura "creative decision rating".

notebooks/
└── regen_all.ipynb         # regen E2E completa (M03-M15 + Z03-Z06) en orden DAG;
                            # 1 celda por modulo + flags FORCE/RETRAIN granular.
```

Estado: M01-M15 + Z01-Z06 auditados modulo a modulo. Caches regenerables via `notebooks/regen_all.ipynb` (~6h por M10 OBSO 25Hz + M14 NUTS HMC).

## Validaciones empiricas

- **M09 (defensa) vs PFF defensive grades**: Spearman rho=+0.27 (n=264, p<0.001)
- **M10 c_obso vs PFF offensive grades**: Pearson r=+0.30 (n=610, p<10^-13);
  raw OBSO r=-0.21 confirma que el counterfactual es la metrica correcta
- **Placebo test 1000 perm + BH-FDR**: ataque-GF, offball-GF/GA, fisico-GA
  significativamente fuera del placebo null (z>2.4, p_FDR<0.025)
- **Window sensitivity ±3/5/7/10/15**: efectos ACUTOS (decay 7x de w3 a w10
  en fisico-GA y offball-GA) - el shock es de respuesta inmediata
- **Stage stratification**: fisico-GA en KO 4x magnitude vs groups
  (heterogeneidad oculta en pooled estimates)
- **PSxG WC22 holdout**: AUC 0.976 vs SB xG 0.844 (+13pp), Brier 0.037 vs
  0.083 (-55%), ECE 0.011 (calibracion casi perfecta)

Datos, documentacion interna del proyecto y outputs intermedios estan fuera del
repo (`.gitignore`).

## Stack

Python (polars, pyarrow, pandas) +
modelos (catboost, lightgbm, numpyro/jax, scikit-learn) +
hyperparam tuning (optuna) + acciones (socceraction atomic-VAEP) +
DiD moderno (pyfixest fixed-effects + Sun-Abraham event-study) +
DoubleML cross-fitted (doubleml IRM/PLR para AIPW Chernozhukov 2018) +
CATE bayesiano jerarquico via NUTS HMC + LKJCholesky cross-canal (numpyro).
