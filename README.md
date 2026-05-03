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
│                           #   Z1-Z5/HMLD/accel/decel) con Hampel + Butterworth +
│                           #   segmentacion teleports + modelo bayesiano jerarquico
│                           #   multivariate (numpyro SVI 3 RATES) -> score_phys =
│                           #   residuo z-score del baseline jugador-minuto esperado
├── M12_did.py              # DiD within-player: ATE FE (player_shock + post,
│                           #   cluster player) + event-study Sun-Abraham 2021
│                           #   (pyfixest) + BJS imputation (Borusyak-Jaravel-
│                           #   Spiess 2024, manual) + HonestDiD-style sensitivity
│                           #   (Rambachan-Roth 2023, M ∈ {0.5, 1, 2}) +
│                           #   pre-trends F-test sobre 4 canales x 2 shock_types
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
│                           #   ⊂ position + LKJCholesky cross-canal + priors
│                           #   informativos PFF grades + R-hat/ESS + PPC KS-
│                           #   test + Indice Remontador (atk_GA + off_GA) +
│                           #   Indice Cerrojo (def_GF + phys_GF) + ranking
│                           #   within position. 598 jugadores con (β_atk,
│                           #   β_def, β_off, β_phys) IC 80%/95%
├── M15_pcj.py              # Perfil Clutch del Jugador ensamblaje scout-facing:
│                           #   234 jugadores >=270 min x 71 cols (8 CATEs +
│                           #   IC80 + 4-vec PCJ summary + 2 indices + posterior
│                           #   probs + rankings + tier labels + sig flags
│                           #   bayesianos via samples NUTS) + 4 aux tables
│                           #   Top Cerrojo Sig: Keylor/Sommer/Neuer (3 GKs WC22)
├── (M16)                   # report PDF por jugador (pendiente)
├── M05B_calibration.py     # T4.12 PSxG calibration diagnostics (curve, ECE/MCE,
│                           #   Brier decomposition Murphy 1973, isotonic mapping)
│                           #   WC22 holdout AUC 0.976, Brier 0.037, ECE 0.011
├── M12B_validation.py      # T2 SOTA causal robustness suite:
│                           #   - placebo test 1000 perm + BH-FDR
│                           #   - statistical power (ICC + MDE@80% + observed)
│                           #   - baseline naive comparison
│                           #   - window sensitivity extendido +-3/5/7/10/15 min
│                           #   - stage stratification (groups vs KO)
├── Z01_vaep.py             # building block atomic-VAEP wrapper (compute_features/labels
│                           #   + save_models/load_models, usado por M08/M09)
└── Z02_pitch_control.py    # building block PPCF Spearman 2018 vectorizado (core
│                           #   agnostico al proveedor, usado por M10)

notebooks/
└── regen_all.ipynb         # regen completa M03-M12 con flags FORCE/RETRAIN
                            # por celda; 1 celda independiente por modulo
                            # para re-ejecucion granular
```

Estado: M01-M15 ejecutados sobre los 64 partidos WC22. M16 pendiente.

Validaciones SOTA (M05B + M12B):

- M09 (defensa) vs PFF defensive grades: Spearman rho=+0.27 (n=264, p<0.001)
- M10 c_obso vs PFF offensive grades: Pearson r=+0.30 (n=610, p<10^-13);
  raw OBSO -0.21 confirma que el counterfactual es la metrica correcta
- Placebo test 1000 perm + BH-FDR: ataque-GF, offball-GF/GA, fisico-GA
  significativamente fuera del placebo null (z>2.4, p_FDR<0.025)
- Window sensitivity +-3/5/7/10/15: efectos ACUTOS (decay 7x de w3 a w10
  en fisico-GA y offball-GA) -> el shock es de respuesta inmediata
- Stage stratification: fisico-GA en KO 4x magnitude vs groups
  (heterogeneidad oculta en pooled estimates)
- PSxG WC22 holdout: AUC 0.976 vs SB xG 0.844 (+13pp), Brier 0.037 vs
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
