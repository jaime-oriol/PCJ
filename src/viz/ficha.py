"""ficha - Ficha PCJ completa: radar + tabla de percentiles lado a lado.

Tabla portada de jaime-oriol/footballdecoded (viz/stats_radar.py,
`create_stats_table`): valores + percentil coloreado por canal, leyenda
LOW->HIGH. Combinacion radar|tabla portada de `combine_radar_and_table`
(PIL, dimensiones alineadas). Adaptado a outputs/pcj_table.parquet +
identidad Diagonality.

Uso:
    python -m src.viz.ficha "Messi"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, PCT_CMAP, WHITE, add_logo
from viz.radar import PCJ_METRICS, PCJ_TITLES, _find, player_radar

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

# Tabla en orden de bloque: 4 canales post-GA, luego 4 post-GF.
TABLE_METRICS = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_defensa_GOAL_AGAINST_mean",
    "cate_offball_GOAL_AGAINST_mean", "cate_fisico_GOAL_AGAINST_mean",
    "cate_ataque_GOAL_FOR_mean",      "cate_defensa_GOAL_FOR_mean",
    "cate_offball_GOAL_FOR_mean",     "cate_fisico_GOAL_FOR_mean",
]
TABLE_TITLES = [
    "Ataque · post-GA", "Defensa · post-GA", "Off-ball · post-GA",
    "Fisico · post-GA", "Ataque · post-GF", "Defensa · post-GF",
    "Off-ball · post-GF", "Fisico · post-GF",
]


def _fmt(v: float) -> str:
    """Valor de celda: CATEs son pequenos con signo."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "0.000"
    return f"{v:+.3f}" if abs(v) < 1 else f"{v:.1f}"


def create_stats_table(df: pl.DataFrame, player_id: int,
                       metrics: list[str] = TABLE_METRICS,
                       metric_titles: list[str] = TABLE_TITLES,
                       save_path=None, logo: bool = True):
    """Tabla de stats con percentil coloreado (portado de create_stats_table).

    Modo single-player. Percentiles `{metric}_pct` calculados al vuelo
    (rank vs dataset) si no estan en df.
    """
    pdf = df.to_pandas()
    for m in metrics:
        if f"{m}_pct" not in pdf.columns:
            pdf[f"{m}_pct"] = pdf[m].rank(pct=True) * 100.0
    p = pdf[pdf["pff_player_id"] == player_id].iloc[0]

    norm = Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=(7.0, 9.0), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 15)
    ax.axis("off")

    # Cabecera: jugador + contexto
    y = 14.4
    ax.text(0.7, y, p.get("player_name", str(player_id)), fontweight="bold",
            fontsize=15, color=WHITE, ha="left", va="center", family="DejaVu Sans")
    ax.text(0.7, y - 0.5, f"{p.get('team_name','')}  ·  {p.get('position_group','')}"
            f"  ·  Mundial Qatar 2022", fontsize=10, color="#c8c8c8", ha="left")
    ax.plot([0.5, 8.0], [y - 0.95, y - 0.95], color="grey", lw=0.5, alpha=0.6)

    # Bloque de exposicion
    y = y - 1.45
    for lbl, val in (("Minutos jugados", int(p.get("minutes_played", 0))),
                     ("Partidos", int(p.get("n_matches_played", 0))),
                     ("Shocks vividos (GF / GA)",
                      f"{int(p.get('n_shocks_for',0))} / {int(p.get('n_shocks_against',0))}")):
        ax.text(0.7, y, lbl, fontsize=10, color=WHITE, fontweight="bold", va="center")
        ax.text(7.8, y, str(val), fontsize=10.5, color=WHITE, ha="right", va="center")
        y -= 0.42
    ax.plot([0.5, 8.0], [y + 0.07, y + 0.07], color="grey", lw=0.5, alpha=0.6)

    # Filas de metricas: valor + percentil coloreado
    y -= 0.55
    row_h = 0.92
    for idx, (m, t) in enumerate(zip(metrics, metric_titles)):
        yr = y - idx * row_h
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.5, yr - 0.38), 7.5, 0.76,
                                   facecolor="white", alpha=0.05))
        pct = p.get(f"{m}_pct", 0.0)
        pct = 0.0 if pct is None or np.isnan(pct) else float(pct)
        ax.text(0.7, yr, t, fontsize=10.5, color=WHITE, fontweight="bold", va="center")
        ax.text(6.7, yr, _fmt(p.get(m)), fontsize=10.5, color=WHITE,
                ha="right", va="center")
        ax.text(7.85, yr, f"{int(pct)}", fontsize=11, fontweight="bold",
                color=PCT_CMAP(norm(pct)), ha="right", va="center")

    # Cabeceras de columna
    ax.text(6.7, y + 0.62, "CATE", fontsize=8.5, color="#c8c8c8", ha="right",
            style="italic")
    ax.text(7.85, y + 0.62, "pct", fontsize=8.5, color="#c8c8c8", ha="right",
            style="italic")

    # Leyenda LOW -> HIGH
    leg_y = y - len(metrics) * row_h - 0.2
    for i, (lo, hi) in enumerate([(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]):
        xp = 1.4 + i * 1.15
        ax.plot([xp - 0.32, xp + 0.32], [leg_y, leg_y],
                color=PCT_CMAP(norm(i * 25)), lw=4, solid_capstyle="round")
        ax.text(xp, leg_y - 0.32, f"{lo}-{hi}", fontsize=8.5, color=WHITE,
                ha="center")
    ax.text(0.7, leg_y, "percentil", fontsize=9, color=WHITE, ha="left",
            va="center", style="italic")
    ax.text(4.25, leg_y - 0.78, "percentil vs los 234 jugadores del torneo",
            fontsize=8.5, color="#c8c8c8", ha="center", style="italic")

    if logo:
        add_logo(fig, width_frac=0.20)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


def _combine(radar_path: Path, table_path: Path, out_path: Path) -> None:
    """Pega radar (izq) + tabla (dcha) a igual altura sobre lienzo BG."""
    radar = Image.open(radar_path).convert("RGB")
    table = Image.open(table_path).convert("RGB")
    H = max(radar.height, table.height)
    rad = radar.resize((int(radar.width * H / radar.height), H), Image.LANCZOS)
    tab = table.resize((int(table.width * H / table.height), H), Image.LANCZOS)
    canvas = Image.new("RGB", (rad.width + tab.width, H), color=BG)
    canvas.paste(rad, (0, 0))
    canvas.paste(tab, (rad.width, 0))
    canvas.save(out_path, dpi=(300, 300))


def player_ficha(df: pl.DataFrame, player_id: int, save_path=None) -> Path:
    """Ficha PCJ completa: radar geometrico + tabla de percentiles."""
    import tempfile
    tmp = Path(tempfile.gettempdir())
    radar_p, table_p = tmp / f"_pcj_r_{player_id}.png", tmp / f"_pcj_t_{player_id}.png"

    # Sin titulo en el radar — la identidad va en la cabecera de la tabla.
    player_radar(df, player_id, PCJ_METRICS, PCJ_TITLES,
                 title="", subtitle="", logo=False, save_path=radar_p)
    create_stats_table(df, player_id, logo=True, save_path=table_p)

    if save_path is None:
        save_path = _SRC.parent / "outputs" / "viz" / f"ficha_{player_id}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _combine(radar_p, table_p, save_path)
    radar_p.unlink(missing_ok=True)
    table_p.unlink(missing_ok=True)
    return save_path


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    pid = _find(df, sys.argv[1] if len(sys.argv) > 1 else "Messi")
    out = player_ficha(df, pid)
    print(f"OK -> {out}")
