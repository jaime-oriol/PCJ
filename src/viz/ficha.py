"""ficha - Ficha PCJ completa: radar + tabla de percentiles lado a lado.

Tabla = port 1:1 de jaime-oriol/footballdecoded (viz/stats_radar.py,
`create_stats_table`): cabecera + Minutos/Partidos, filas metrica con valor
+ percentil coloreado (node_cmap), sombreado alterno, leyenda 5 tramos +
flecha BAJO->ALTO. Combinacion radar|tabla = `combine_radar_and_table`.

Adaptaciones minimas vs el original: datos pcj_table, percentil vs POSICION,
logo Diagonality (sin "Created by"), formato de valor para CATEs con signo.

Uso:
    python -m src.viz.ficha "Messi"
"""

from __future__ import annotations

import sys
import tempfile
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

# 8 dimensiones del radar, en orden de bloque post-GA / post-GF.
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

_NAME_COLOR = "#FF6B6B"   # team_colors[0] del original footballdecoded


def _short(name: str, max_len: int = 16) -> str:
    """'Lionel Messi' -> 'L. Messi' si es largo (idem _shorten_long_name)."""
    if len(name) <= max_len:
        return name
    parts = name.split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) >= 2 else name


def _fmt(v) -> str:
    """Formato de celda. Mantiene la logica del original + rama para los
    CATE (valores pequenos con signo, que el formato original aplastaria)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "0.000"
    if abs(v) < 1:
        return f"{v:+.3f}"
    if abs(v) < 10:
        return f"{v:.1f}"
    return f"{int(v)}"


def create_stats_table(df: pl.DataFrame, player_id: int,
                       metrics: list[str] = TABLE_METRICS,
                       metric_titles: list[str] = TABLE_TITLES,
                       footer_text: str = "percentil vs los jugadores de su posicion",
                       save_path=None):
    """Tabla de stats con percentil coloreado. Port 1:1 de create_stats_table.

    Percentil `{metric}_pct` calculado vs el mismo position_group.
    """
    pdf = df.to_pandas()
    for m in metrics:                                    # percentil POR POSICION
        pdf[f"{m}_pct"] = pdf.groupby("position_group")[m].rank(pct=True) * 100.0
    p1 = pdf[pdf["pff_player_id"] == player_id].iloc[0]

    node_cmap = PCT_CMAP
    percentile_norm = Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=(7.5, 8.5), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 15)
    ax.axis("off")

    y_start = 14.5
    text1_x, p1_value_x, p1_pct_x = 3.4, 4.1, 4.5

    # Cabecera: nombre + contexto
    name1 = _short(p1.get("player_name", str(player_id)))
    ax.text(text1_x, y_start, name1, fontweight="bold", fontsize=14,
            color=_NAME_COLOR, ha="left", va="center", family="DejaVu Sans")
    ax.text(text1_x, y_start - 0.425,
            f"{p1.get('team_name', '')}  ·  {p1.get('position_group', '')}"
            f"  ·  Mundial Qatar 2022",
            fontsize=10, color=WHITE, alpha=0.9, ha="left", family="DejaVu Sans")

    y_line = y_start - 0.7
    ax.plot([0.5, 8.5], [y_line, y_line], color="grey", linewidth=0.5, alpha=0.6)

    # Minutos / Partidos
    y_context = y_start - 1.2
    ax.text(0.7, y_context, "Minutos jugados", fontsize=10, color=WHITE,
            fontweight="bold", family="DejaVu Sans")
    ax.text(p1_value_x, y_context, f"{int(p1.get('minutes_played', 0))}",
            fontsize=11, color=WHITE, ha="right", family="DejaVu Sans")
    y_context -= 0.4
    ax.text(0.7, y_context, "Partidos jugados", fontsize=10, color=WHITE,
            fontweight="bold", family="DejaVu Sans")
    ax.text(p1_value_x, y_context, f"{int(p1.get('n_matches_played', 0))}",
            fontsize=11, color=WHITE, ha="right", family="DejaVu Sans")

    y_line = y_context - 0.3
    ax.plot([0.5, 8.5], [y_line, y_line], color="grey", linewidth=0.5, alpha=0.6)

    # Filas de metricas
    y_metrics = y_context - 0.7
    row_height = 1.0
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        y_pos = y_metrics - idx * row_height
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.5, y_pos - 0.4), 8.0, 0.8,
                                   facecolor="white", alpha=0.05))
        ax.text(0.7, y_pos, title, fontsize=10, color=WHITE, fontweight="bold",
                va="center", family="DejaVu Sans")
        pct = p1.get(f"{metric}_pct", 0)
        pct = 0 if pct is None or np.isnan(pct) else float(pct)
        ax.text(p1_value_x, y_pos, _fmt(p1.get(metric)), fontsize=11, color=WHITE,
                ha="right", va="center", family="DejaVu Sans")
        ax.text(p1_pct_x, y_pos, f"{int(pct)}", fontsize=10,
                color=node_cmap(percentile_norm(pct)), ha="left", va="center",
                family="DejaVu Sans")

    # Footer
    footer_y = y_metrics - len(metrics) * row_height
    if len(metrics) % 2 == 1:
        ax.add_patch(Rectangle((0.5, footer_y - 0.4), 8.0, 0.8,
                               facecolor="white", alpha=0.05))
    ax.text(0.7, footer_y, f"*{footer_text}", fontsize=10, color=WHITE,
            ha="left", style="italic", fontweight="bold", va="center",
            family="DejaVu Sans")

    # Leyenda: 5 tramos de percentil
    legend_y = footer_y - 0.8
    intervals = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    spacing = 0.8
    for i, (lo, hi) in enumerate(intervals):
        x_pos = 1.0 + i * spacing
        ax.plot([x_pos - 0.25, x_pos + 0.25], [legend_y, legend_y],
                color=node_cmap(percentile_norm(i * 25)), linewidth=3,
                solid_capstyle="round")
        ax.text(x_pos, legend_y - 0.3, f"{lo}-{hi}", fontsize=9, color=WHITE,
                ha="center", family="DejaVu Sans")

    # Flecha BAJO -> ALTO
    arrow_y = legend_y - 0.8
    ax.annotate("", xy=(4.0, arrow_y), xytext=(1.2, arrow_y),
                arrowprops=dict(arrowstyle="->", color=WHITE, lw=1))
    ax.text(1.1, arrow_y, "BAJO", fontsize=9, color=WHITE, ha="right",
            va="center", family="DejaVu Sans")
    ax.text(4.1, arrow_y, "ALTO", fontsize=9, color=WHITE, ha="left",
            va="center", family="DejaVu Sans")

    add_logo(fig, width_frac=0.22)
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
    tmp = Path(tempfile.gettempdir())
    radar_p, table_p = tmp / f"_pcj_r_{player_id}.png", tmp / f"_pcj_t_{player_id}.png"

    # Radar sin titulo ni logo: la identidad va en la tabla.
    player_radar(df, player_id, PCJ_METRICS, PCJ_TITLES,
                 title="", subtitle="", logo=False, save_path=radar_p)
    create_stats_table(df, player_id, save_path=table_p)

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
