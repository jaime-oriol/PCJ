# PCJ: Perfil Clutch del Jugador

TFM Master Big Data Aplicado al Scouting Deportivo (Sport Data Campus).

Estimacion causal del efecto del shock emocional (gol a favor / gol en contra) sobre el comportamiento del jugador en ventanas pre vs post de ±10 min, medido en cuatro canales (Empuje Ofensivo, Solidez Defensiva, Inteligencia Espacial Off-ball, Pulso Fisico) sobre PFF FC World Cup Qatar 2022.

Output: ranking bidireccional de jugadores clutch (Indice Remontador + Indice Cerrojo) con intervalos de credibilidad bayesianos.

## Estructura

```text
src/
├── extract/            # extractores raw JSON -> parquet (lossless)
├── vaep.py             # B01 atomic-VAEP (building block)
├── pitch_control.py    # B02 PPCF (building block)
└── (M01-M16)           # pipeline modular del PCJ
```

Datos, documentacion interna del proyecto y outputs intermedios estan fuera del repo (`.gitignore`).

## Stack

Python (polars, pyarrow, scikit-learn, xgboost, catboost, lightgbm).
