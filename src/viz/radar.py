"""radar - Ficha-radar del Perfil Clutch del Jugador.

Radar geometrico portado tal cual de jaime-oriol/footballdecoded
(viz/swarm_radar.py, `_create_traditional_radar`): circulos concentricos,
rangos por percentil 1-99, poligono del jugador con anillos de color
alternos. Adaptado a outputs/pcj_table.parquet + identidad Diagonality.

Uso:
    python -m src.viz.radar 1234              # por pff_player_id
    python -m src.viz.radar "Messi"           # por nombre (substring)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import ATT, BG, DEF, WHITE, add_logo

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

# Ejes del radar PCJ: 4 canales x 2 contextos (post-GA / post-GF).
PCJ_METRICS = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_offball_GOAL_AGAINST_mean",
    "cate_defensa_GOAL_AGAINST_mean", "cate_fisico_GOAL_AGAINST_mean",
    "cate_fisico_GOAL_FOR_mean",      "cate_defensa_GOAL_FOR_mean",
    "cate_offball_GOAL_FOR_mean",     "cate_ataque_GOAL_FOR_mean",
]
PCJ_TITLES = [
    "Ataque\npost-GA", "Off-ball\npost-GA", "Defensa\npost-GA", "Fisico\npost-GA",
    "Fisico\npost-GF", "Defensa\npost-GF", "Off-ball\npost-GF", "Ataque\npost-GF",
]


def player_radar(df: pl.DataFrame, player_id: int,
                 metrics: list[str] = PCJ_METRICS,
                 metric_titles: list[str] = PCJ_TITLES,
                 colors: tuple[str, str] = (ATT, DEF),
                 title: str = "", subtitle: str = "",
                 logo: bool = True, save_path=None):
    """Radar geometrico de 1 jugador (anillos de color alternos).

    Portado de footballdecoded/viz/swarm_radar._create_traditional_radar.
    `df` debe traer las columnas `metrics` (valores crudos del dataset).
    """
    pdf = df.to_pandas()
    row = pdf[pdf["pff_player_id"] == player_id].iloc[0]

    # Mismo reordenado que el original: primer eje fijo, resto invertido
    reordered = [metrics[0]] + list(reversed(metrics[1:]))
    reordered_titles = [metric_titles[0]] + list(reversed(metric_titles[1:]))

    # Rangos por percentil 1-99 del dataset
    ranges = []
    for m in reordered:
        d = pdf[m].dropna()
        ranges.append((np.percentile(d, 1), np.percentile(d, 99)))

    fig, ax = plt.subplots(figsize=(9, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set(xlim=(-22, 22), ylim=(-23, 25))

    values = [row[m] for m in reordered]

    # Circulos concentricos
    radius_circles = [3, 5.5, 8, 10.5, 13, 15.5, 18, 20.5]
    for i, rad in enumerate(radius_circles):
        if i == 0:
            continue
        if i == len(radius_circles) - 1:
            color, lw, alpha = "white", 1.2, 1.0
        else:
            color, lw, alpha = "grey", 1, 0.4
        ax.add_patch(plt.Circle((0, 0), rad, fc="none", ec=color, lw=lw, alpha=alpha))

    n = len(reordered)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Etiquetas de eje
    for angle, t in zip(angles, reordered_titles):
        x, y = 21.5 * np.sin(angle), 21.5 * np.cos(angle)
        rot = -np.rad2deg(angle)
        if y < 0:
            rot += 180
        ax.text(x, y, t, rotation=rot, ha="center", va="center",
                fontsize=10, fontweight="bold", color=WHITE, family="DejaVu Sans")

    # Lineas radiales
    for angle in angles:
        ax.plot([0, 20.5 * np.sin(angle)], [0, 20.5 * np.cos(angle)],
                color="grey", linewidth=0.5, alpha=0.4)

    # Etiquetas de valor en los anillos
    for rad in [4.25, 6.75, 9.25, 11.75, 14.25, 16.75, 19.25]:
        for angle, (mn, mx) in zip(angles, ranges):
            rad_norm = (rad - 3) / (20.5 - 3)
            val = mn if mx == mn else mn + rad_norm * (mx - mn)
            if abs(val) < 0.01:
                label = f"{val:.3f}"
            elif abs(val) < 1:
                label = f"{val:.2f}"
            elif abs(val) < 10:
                label = f"{val:.1f}"
            else:
                label = f"{int(val)}"
            ax.text(rad * np.sin(angle), rad * np.cos(angle), label,
                    ha="center", va="center", size=7, color=WHITE,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                              edgecolor="none", alpha=0.9), family="DejaVu Sans")

    # Coordenadas polares del poligono del jugador
    vertices = []
    for value, (mn, mx) in zip(values, ranges):
        if mx == mn:
            nv = 11.75
        else:
            nv = 3 + (value - mn) / (mx - mn) * 17.5
        nv = max(3, min(20.5, nv))
        idx = len(vertices)
        vertices.append([nv * np.sin(angles[idx]), nv * np.cos(angles[idx])])

    # Poligono con anillos de color alternos recortados a su forma
    poly = Polygon(vertices, fc="none", alpha=1.0, zorder=1)
    ax.add_patch(poly)
    central = plt.Circle((0, 0), radius_circles[0], fc=colors[0], ec="none",
                         alpha=0.45, zorder=2)
    central.set_clip_path(poly)
    ax.add_patch(central)
    theta = np.linspace(0, 2 * np.pi, 100)
    for i in range(len(radius_circles) - 1):
        ri, ro = radius_circles[i], radius_circles[i + 1]
        cidx = (i + 1) % 2
        ring = list(zip(ro * np.cos(theta), ro * np.sin(theta))) + \
               list(zip(ri * np.cos(theta[::-1]), ri * np.sin(theta[::-1])))
        rp = Polygon(ring, fc=colors[cidx], alpha=0.45, zorder=2)
        rp.set_clip_path(poly)
        ax.add_patch(rp)
    closed = vertices + [vertices[0]]
    ax.plot([v[0] for v in closed], [v[1] for v in closed],
            color=colors[0], linewidth=3, zorder=10)

    ax.axis("off")
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", color=WHITE,
                 fontsize=17, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top", color="#c8c8c8",
                 fontsize=11)
    if logo:
        add_logo(fig, width_frac=0.13)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


def _find(df: pl.DataFrame, query: str):
    """Resuelve un jugador por id (entero) o substring del nombre."""
    try:
        return int(query)
    except ValueError:
        sub = df.filter(pl.col("player_name").str.contains(query, literal=False))
        if sub.height == 0:
            raise SystemExit(f"jugador no encontrado: {query!r}")
        return int(sub["pff_player_id"][0])


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    query = sys.argv[1] if len(sys.argv) > 1 else "Messi"
    pid = _find(df, query)
    r = df.filter(pl.col("pff_player_id") == pid).row(0, named=True)
    out = f"outputs/viz/radar_{pid}.png"
    player_radar(
        df, pid,
        title=f"{r['player_name']}  ·  Perfil Clutch del Jugador",
        subtitle=f"{r['team_name']}  ·  {r['position_group']}  ·  "
                 f"{int(r['minutes_played'])} min  —  Mundial Qatar 2022",
        save_path=out,
    )
    print(f"OK -> {out}")
