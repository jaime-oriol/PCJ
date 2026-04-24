# PCJ: Perfil Clutch del Jugador

TFM Master Big Data Aplicado al Scouting Deportivo (Sport Data Campus).

Estimacion causal del efecto del shock emocional (gol a favor / gol en contra) sobre el comportamiento del jugador en ventanas pre vs post de ±10 min, medido en cuatro canales (Empuje Ofensivo, Solidez Defensiva, Inteligencia Espacial Off-ball, Pulso Fisico) sobre PFF FC World Cup Qatar 2022.

Output: ranking bidireccional de jugadores clutch (Indice Remontador + Indice Cerrojo) con intervalos de credibilidad bayesianos.

## Estructura

```text
src/
├── extract/                # extractores raw JSON -> parquet (lossless)
├── M01_loader_pff.py       # API PFF (events, tracking, metadata, rosters + vistas)
├── M02_loader_public.py    # API Wyscout + StatsBomb (polars nativo)
├── M03_preprocess.py       # direction, score state, minutos, enrich_events
├── M04_wp.py               # Win Probability bayesiana (numpyro) + leverage + ET + pen
├── M05_psxg.py             # Post-shot xG (LightGBM + Optuna, AUC 0.976)
├── M06_nearmiss.py         # Near-miss identification (5 tipos, specification curve)
├── (M07-M16)               # pipeline restante (ver docs/ARCHITECTURE.md)
├── Z01_vaep.py             # B01 atomic-VAEP (building block)
└── Z02_pitch_control.py    # B02 PPCF (building block)
```

Datos, documentacion interna del proyecto y outputs intermedios estan fuera del repo (`.gitignore`).

## Stack

Python (polars, pyarrow, scikit-learn, xgboost, catboost, lightgbm).
