"""Runner unico: renderiza las figuras core del PCJ a outputs/viz/.

Uso:
    python -m src.viz                 # PPCF + scatter + event-study + ficha
    python -m src.viz radar "Messi"   # solo el radar de un jugador
    python -m src.viz ficha "Messi"   # solo la ficha (radar + tabla)
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz import ficha, figures, ppcf, radar, scatter

_OUT = _SRC.parent / "outputs" / "viz"
_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"


def _render_radar(df: pl.DataFrame, query: str) -> Path:
    """Radar individual de un jugador (id o substring de nombre)."""
    pid = radar._find(df, query)
    r = df.filter(pl.col("pff_player_id") == pid).row(0, named=True)
    out = _OUT / f"radar_{pid}.png"
    radar.player_radar(
        df, pid,
        title=f"{r['player_name']}  ·  Perfil Clutch del Jugador",
        subtitle=f"{r['team_name']}  ·  {r['position_group']}  ·  "
                 f"{int(r['minutes_played'])} min  —  Mundial Qatar 2022",
        save_path=out)
    return out


def make_all() -> None:
    """Renderiza PPCF + scatter + event-study + ficha (jugador de portada)."""
    print("[viz] PPCF — gol de Messi (ARG-MEX)...")
    fnum = ppcf.frame_for_clock(3835, period=2, clock_s=3812 - 3)
    ppcf.plot_ppcf(
        3835, fnum,
        title="Argentina 1 - 0 Mexico  ·  Pitch Control en la jugada del gol de Messi",
        subtitle="Mundial Qatar 2022 - Fase de grupos  ·  buildup 3 s antes del remate",
        save_path=_OUT / "ppcf_messi_arg_mex.png")

    print("[viz] Scatter Remontador x Cerrojo...")
    scatter.diamond_scatter(pl.read_parquet(_TABLE),
                            save_path=_OUT / "scatter_remontador_cerrojo.png")

    print("[viz] Event-study causal (M12)...")
    figures.event_study(save_path=_OUT / "event_study.png")

    print("[viz] Ficha — jugador de portada (Messi)...")
    df = pl.read_parquet(_TABLE)
    ficha.player_ficha(df, ficha._find(df, "Messi"))

    print(f"[viz] OK — figuras en {_OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("radar", "ficha"):
        df = pl.read_parquet(_TABLE)
        if sys.argv[1] == "radar":
            print(f"OK -> {_render_radar(df, sys.argv[2])}")
        else:
            print(f"OK -> {ficha.player_ficha(df, ficha._find(df, sys.argv[2]))}")
    else:
        make_all()
