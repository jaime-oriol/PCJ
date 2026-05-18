# PCJ: Perfil Clutch del Jugador

TFM Master Big Data Aplicado al Scouting Deportivo (Sport Data Campus).

Estimacion causal del efecto del shock emocional (gol a favor / gol en contra) sobre el comportamiento del jugador en ventanas pre vs post de ±10 min, medido en cuatro canales (Empuje Ofensivo, Solidez Defensiva, Inteligencia Espacial Off-ball, Pulso Fisico) sobre PFF FC World Cup Qatar 2022.

Output: ranking tridimensional de jugadores clutch (Indice Remontador post GA + Indice Cerrojo post GF + Pressure Response continuo en elim_prox) con intervalos de credibilidad bayesianos, agregado por bucket posicional (DEF/MED/ATA) y posicion granular (16 PFF labels).

## Estructura del repo

```text
TFM/
├── README.md                                      # este fichero
├── run_pipeline.sh                                # E2E orquestador (auto detect cores + GPU)
├── data/parquet/
│   ├── pff/                                       # versionado: events (64) + metadata + rosters
│   └── derived/                                   # versionado: caches M03 a M14
│       ├── preprocess/, wp/, psxg/, nearmiss/, shocks/
│       ├── ataque/, defensa/, offball/, fisico/
│       ├── did/, did_validation/, aipw/
│       └── cate/                                  # M14 outputs (cate_nuts.pkl ignorado, 409 MB regenerable)
├── cache/vaep/                                    # versionado: features + labels VAEP por partido
├── src/
│   ├── extract/                                   # extractores raw a parquet (lossless)
│   ├── preprocess/pff_grades_extract.py           # priors PFF grades (input M14)
│   ├── M01_loader_pff.py                          # API PFF (events, tracking, metadata, rosters)
│   ├── M02_loader_public.py                       # API Wyscout + StatsBomb (polars nativo)
│   ├── M03_preprocess.py                          # direction, score state (SB ground truth), minutos, enrich_events
│   ├── M04_wp.py                                  # Win Probability bayesiana (numpyro SVI ordered logistic)
│   │                                              #   + leverage + ET Poisson + tanda parametrica + MC group elim_prox
│   ├── M05_psxg.py                                # Post shot xG (LightGBM + Optuna 60 + isotonic + freeze 360)
│   │                                              #   AUC OOF 0.974, holdout WC22 0.976 (vs SB 0.827)
│   ├── M05B_calibration.py                        # PSxG calibration (curve, ECE/MCE, Brier Murphy 1973)
│   ├── M06_nearmiss.py                            # Near miss 5 tipos (palo, offside 360, PSxG save, GLC, GLT)
│   ├── M07_shocks.py                              # 172 shocks gol + ventanas ±10min + LOO team_members
│   ├── M08_ataque.py                              # Empuje Ofensivo: atomic VAEP CatBoost + un xPass (Z06)
│   ├── M09_defensa.py                             # Solidez Defensiva: vdep_strict (Z04) + xpress (Z03)
│   │                                              #   + maejima (Z05) + def3rd + press_value
│   ├── M10_offball.py                             # Off ball OBSO + C OBSO (Spearman 2018 + Teranishi 2022)
│   │                                              #   PPCF Z02 + xG grid + tracking PFF 25Hz
│   ├── M11_fisico.py                              # Pulso Fisico Bradley 2024 + bayesiano jerarquico SVI
│   ├── M12_did.py                                 # DiD within player: ATE FE + Sun Abraham + BJS + HonestDiD
│   ├── M12B_validation.py                         # placebo + power + window sensitivity + stage stratified
│   ├── M13_aipw.py                                # AIPW DoubleMLIRM + DML PLR + DR learner + RDD + spec curve
│   ├── M14_cate.py                                # CATE bayesiano NUTS HMC 4 chains + 5 etas + LKJ
│   ├── M15_pcj.py                                 # tabla scout final + 16 cells contextualizados + buckets
│   ├── render_ficha.py                            # ficha visual scout facing por jugador
│   ├── Z01_vaep.py                                # atomic VAEP wrapper
│   ├── Z02_pitch_control.py                       # PPCF Spearman 2018 vectorizado
│   ├── Z03_xpress.py                              # exPress Lee 2025 P(recovery<5s|press)
│   ├── Z04_vdep.py                                # VDEP strict Toda 2022 (recovery + attacked)
│   ├── Z05_maejima.py                             # Maejima 2024 nearest defender frame level
│   ├── Z06_unxpass.py                             # un xPass Robberechts 2023 creative decision
│   └── viz/                                       # capa de visualizacion (identidad propia)
│       ├── common.py                              # estilo, colores, draw_pitch, logo, helpers
│       ├── ppcf.py                                # superficie Pitch Control (Z02 + tracking PFF)
│       ├── radar.py                               # radar geometrico de las 8 dimensiones clutch
│       ├── scatter.py                             # diamond scatter Remontador x Cerrojo
│       ├── ficha.py                               # ficha jugador = radar + tabla percentil
│       └── figures.py                             # event-study causal (M12)
├── notebooks/
│   └── regen_all.ipynb                            # regen E2E completa M03-M15 + Z03-Z06 en orden DAG
└── outputs/
    ├── pcj_table.parquet                          # tabla scout final (234 jug x 277 cols)
    ├── viz/                                       # figuras PNG (PPCF, radar, ficha, scatter, event-study)
    └── pcj_aux/
        ├── top10_chasing_per_position.parquet     # 16 position_group granulares
        ├── top10_protecting_per_position.parquet
        ├── top10_pressure_per_position.parquet
        ├── top10_chasing_per_bucket.parquet       # 4 buckets (DEF/MED/ATA, GK aparte)
        ├── top10_protecting_per_bucket.parquet
        ├── top10_pressure_per_bucket.parquet
        ├── dual_clutch_top.parquet
        └── by_team.parquet
```

## Estado del pipeline

E2E ejecutado al 100%. Outputs versionados en repo. Caches regenerables via `notebooks/regen_all.ipynb` o `run_pipeline.sh`.

| Modulo | Output principal                                            | Sanity verificado                                              |
|--------|-------------------------------------------------------------|----------------------------------------------------------------|
| M03    | preprocess/events_enriched/{match_id}.parquet × 64          | 144,541 filas, 172 goles SB ground truth                       |
| M04    | wp/per_minute.parquet                                       | 5,910 filas (60 min reglamentarios x 99 partidos efectivos)    |
| M05    | psxg/{shots,model/psxg_lgb.pkl}                             | AUC OOF 0.974, holdout WC22 0.976 (vs SB 0.827)                |
| M05B   | psxg/calibration/{curve,brier,metrics,iso}.parquet          | ECE 0.011, Brier 0.037 (vs SB 0.083)                           |
| M06    | nearmiss/nearmiss_table.parquet                             | 70 near miss (12 woodw + 5 offs + 38 save + 2 GLC + 9 GLT)     |
| M07    | shocks/{shocks_table,shocks_team_members}.parquet           | 172 shocks x ~22 jug = 3,788 filas                             |
| M08    | ataque/{per_minute,per_shock_window,model}                  | atomic VAEP + un xPass; 57,520 filas; 234 jug clutch           |
| Z03    | defensa/xpress/per_minute.parquet                           | exPress Lee 2025; AUC 0.6178 (+24% baseline)                   |
| Z04    | defensa/vdep_strict/per_minute.parquet                      | VDEP Toda 2022; AUC rec 0.7950 / att 0.8308                    |
| Z05    | defensa/maejima/per_minute.parquet                          | Maejima nearest defender; 38,005 filas                         |
| Z06    | ataque/unxpass/per_minute.parquet                           | un xPass Robberechts 2023; AUC 0.8309                          |
| M09    | defensa/{per_minute,per_shock_window,press_value,ctx}       | score_def_v4 = vdep + xpress + maejima; 57,466 filas           |
| M10    | offball/{per_minute,per_shock_window,xg_grid}               | OBSO + C OBSO; 105,214 filas; 64 partidos a 25 Hz full         |
| M11    | fisico/{raw_per_minute,per_minute,per_shock_window,model}   | Bradley 2024 + SVI multivariate; 145,351 filas                 |
| M12    | did/{panel,ate_population,event_study,honest,diag}          | DiD within player + Sun Abraham + BJS; FE≈BJS (6.5% SE)        |
| M12B   | did_validation/{placebo,power,window,baseline_naive,stage}  | placebo 1000 perm + BH FDR; window decay 7x w3 a w10           |
| M13    | aipw/{panel_master,att_aipw,att_dml_plr,att_dr_learner}     | 193 shots; AIPW + DML + DR learner; same_sign vs M12           |
| M14    | cate/{panel_delta,posterior_player,indices,rankings,diag}   | NUTS 4x1000+1000 GPU; 0 div; 141/144 R-hat<1.05; PPC 8/8       |
| M15    | outputs/pcj_table.parquet + pcj_aux/                        | 234 jug x 277 cols + 4 buckets posicionales (GK/DEF/MED/ATA)   |

## Validaciones empiricas

* **M09 (defensa) vs PFF defensive grades**: Spearman rho=+0.27 (n=264, p<0.001)
* **M10 c_obso vs PFF offensive grades**: Pearson r=+0.30 (n=610, p<10^-13);
  raw OBSO r=-0.21 confirma que el counterfactual es la metrica correcta
* **Placebo test 1000 perm + BH FDR**: ataque-GF, offball-GF/GA, fisico-GA
  significativamente fuera del placebo null (z>2.4, p_FDR<0.025)
* **Window sensitivity ±3/5/7/10/15**: efectos AGUDOS (decay 7x de w3 a w10
  en fisico-GA y offball-GA), el shock es de respuesta inmediata
* **Stage stratification**: fisico-GA en KO 4x magnitude vs groups
  (heterogeneidad oculta en pooled estimates)
* **PSxG WC22 holdout**: AUC 0.976 vs SB xG 0.844 (+13pp), Brier 0.037 vs
  0.083 (-55%), ECE 0.011 (calibracion casi perfecta)
* **M14 NUTS HMC**: 4 chains x 1000+1000 secuencial T4 GPU (6.9 min);
  0 divergencias; 141/144 hyperparams convergidos (R-hat<1.05);
  PPC 8/8 channels calibrados (KS p>0.05)

Datos raw originales (PFF tracking 5 GB, StatsBomb, Wyscout) y documentacion interna del proyecto estan fuera del repo (`.gitignore`).

## Visualizaciones

Paquete `src/viz/` con identidad visual propia (fondo oscuro, paleta, logo).
Genera las figuras core del TFM a `outputs/viz/`:

| Figura      | Comando                           | Que muestra                                          |
|-------------|-----------------------------------|------------------------------------------------------|
| PPCF        | `python -m src.viz.ppcf`          | Superficie Pitch Control en un shock (Spearman 2018) |
| Radar       | `python -m src.viz radar "Messi"` | Radar de las 8 dimensiones clutch del jugador        |
| Ficha       | `python -m src.viz ficha "Messi"` | Radar + tabla de percentiles vs su posicion          |
| Scatter     | `python -m src.viz.scatter`       | Diamond Remontador x Cerrojo (234 jugadores)         |
| Event-study | `python -m src.viz.figures`       | Efecto causal del shock minuto a minuto (M12)        |

`python -m src.viz` renderiza todas de una.

## Reproducibilidad

```bash
# Clonar repo y arrancar pipeline E2E (cache hit en M03-M14 instantaneo)
git clone https://github.com/jaime-oriol/PCJ.git
cd PCJ
./run_pipeline.sh                # auto detect cores + GPU
# Outputs en outputs/pcj_table.parquet + pcj_aux/
```

Para regenerar desde cero (sin cache hit, requiere raw PFF + StatsBomb + Wyscout):

```bash
FORCE_CLEAN=1 ./run_pipeline.sh
```

## Stack

Python (polars, pyarrow, pandas<2.3) +
modelos (catboost, lightgbm, numpyro/jax cuda12, scikit-learn>=1.6) +
hyperparam tuning (optuna) + acciones (socceraction atomic VAEP) +
DiD moderno (pyfixest fixed effects + Sun Abraham event study) +
DoubleML cross fitted (doubleml IRM/PLR para AIPW Chernozhukov 2018) +
CATE bayesiano jerarquico via NUTS HMC + LKJCholesky cross canal (numpyro) +
visualizacion (matplotlib, mplsoccer, adjustText, Pillow).
