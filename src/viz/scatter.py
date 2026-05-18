"""scatter - Diamond scatter Remontador x Cerrojo del PCJ.

Portado de jaime-oriol/footballdecoded (viz/scatter.py, create_diamond_scatter):
ejes rotados 45 grados (floating_axes), puntos coloreados por percentil
combinado, top-10 etiquetado con adjustText, region sombreada P20-P80.

Adaptacion minima: los indices PCJ son CATEs con signo, asi que la
normalizacion es min-max `(v-min)/(max-min)` en vez de `v/max`.

Uso:
    python -m src.viz.scatter
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, PCT_CMAP, WHITE, add_logo

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

try:
    import adjustText
    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False

# 6 marcas por eje (no 11) — el diamante rotado se satura con mas.
_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def diamond_scatter(df: pl.DataFrame,
                    x_metric: str = "chasing_clutch_idx",
                    y_metric: str = "protecting_clutch_idx",
                    x_pct: str = "pct_chasing_global",
                    y_pct: str = "pct_protecting_global",
                    save_path=None):
    """Diamond scatter (ejes rotados 45 grados) Remontador x Cerrojo."""
    pdf = df.to_pandas()
    left = pdf[x_metric].fillna(0.0)      # eje izquierdo  (Remontador)
    right = pdf[y_metric].fillna(0.0)     # eje inferior   (Cerrojo)

    # Normalizacion min-max -> [0, 0.99]  (CATEs con signo)
    lmin, lmax = float(left.min()), float(left.max())
    rmin, rmax = float(right.min()), float(right.max())
    left_n = 0.99 * (left - lmin) / (lmax - lmin)
    right_n = 0.99 * (right - rmin) / (rmax - rmin)

    lq = left_n.quantile([0.2, 0.8]).tolist()
    rq = right_n.quantile([0.2, 0.8]).tolist()

    # Percentil 0-100 para seleccionar a los que mas destacan
    px, py = pdf[x_pct].to_numpy() * 100.0, pdf[y_pct].to_numpy() * 100.0
    pdf = pdf.assign(_px=px, _py=py)
    top = pdf[(pdf["_px"] >= 81) & (pdf["_py"] >= 81)]
    if len(top) < 5:
        top = pdf[(pdf["_px"] >= 75) & (pdf["_py"] >= 75)]
    top = top.assign(_tot=top["_px"] + top["_py"]).nlargest(10, "_tot")

    fig = plt.figure(figsize=(9.5, 10), facecolor=BG)

    # Marcas de eje: valor real (con signo) en 6 posiciones
    left_dict = {i: f"{lmin + (i / 0.99) * (lmax - lmin):+.3f}" for i in _TICKS}
    right_dict = {i: f"{rmin + (i / 0.99) * (rmax - rmin):+.3f}" for i in _TICKS}

    transform = Affine2D().rotate_deg(45)
    helper = floating_axes.GridHelperCurveLinear(
        transform, (0, 1.001, 0, 1.001),
        grid_locator1=FixedLocator(_TICKS), grid_locator2=FixedLocator(_TICKS),
        tick_formatter1=DictFormatter(right_dict),
        tick_formatter2=DictFormatter(left_dict))
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    ax.set_position([0.10, 0.10, 0.80, 0.70], which="both")
    aux = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux.patch = ax.patch

    ax.axis["left"].line.set_color(WHITE)
    ax.axis["bottom"].line.set_color(WHITE)
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["left"].major_ticklabels.set(rotation=0, ha="center", fontsize=8.5)
    ax.axis["bottom"].major_ticklabels.set(fontsize=8.5)
    ax.axis["bottom"].major_ticklabels.set_pad(6)
    for side, lbl in (("left",   "REMONTADOR  —  reaccion tras encajar un gol"),
                      ("bottom", "CERROJO  —  aguante tras marcar un gol")):
        ax.axis[side].set_label(lbl)
        ax.axis[side].label.set(color=WHITE, fontweight="bold", fontsize=11)
        ax.axis[side].LABELPAD += 9
    ax.axis["left"].label.set_rotation(0)
    ax.grid(alpha=0.18, color=WHITE)

    # Region sombreada: el 60% central de jugadores (percentil 20-80)
    aux.fill([rq[0], rq[0], rq[1], rq[1]], [0, 100, 100, 0],
             color="grey", alpha=0.13, zorder=0)
    aux.fill([0, rq[0], rq[0], 0], [lq[0], lq[0], lq[1], lq[1]],
             color="grey", alpha=0.13, zorder=0)
    aux.fill([rq[1], 100, 100, rq[1]], [lq[0], lq[0], lq[1], lq[1]],
             color="grey", alpha=0.13, zorder=0)
    aux.plot([0, 100], [0, 100], color=WHITE, lw=1.3, alpha=0.5,
             ls="--", zorder=1)
    aux.scatter(right_n, left_n, c=left_n + right_n, cmap=PCT_CMAP,
                edgecolor=WHITE, s=58, lw=0.5, zorder=2, alpha=0.8)

    # Los que mas destacan, etiquetados
    texts = []
    for i, p in top.iterrows():
        parts = str(p.get("player_name", i)).split()
        short = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else parts[0]
        texts.append(aux.annotate(
            short, xy=(right_n.loc[i], left_n.loc[i]), color="yellow",
            fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor="yellow",
                      alpha=0.95, linewidth=1)))
    if _HAS_ADJUST and texts:
        adjustText.adjust_text(texts, ax=aux, force_text=1.6,
                               expand_text=(2.1, 2.1),
                               arrowprops=dict(arrowstyle="-", color="yellow",
                                               alpha=0.9, linewidth=1.2))

    # Titulo + subtitulo (lenguaje simple)
    fig.text(0.5, 0.965, "Quien tira del equipo y quien lo sostiene",
             ha="center", va="top", color=WHITE, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.93,
             "Mundial Qatar 2022  ·  como cambia cada jugador tras un gol, "
             "aislando lo que aporta el resto del equipo",
             ha="center", va="top", color="#c8c8c8", fontsize=10.5)

    # Carteles laterales en lenguaje futbolero
    fig.text(0.205, 0.70, "Por este lado\nLOS QUE TIRAN DEL EQUIPO\ncuando toca remontar",
             ha="center", va="center", color="#e8e8e8", fontsize=10,
             linespacing=1.5)
    fig.text(0.795, 0.70, "Por este lado\nLOS QUE AGUANTAN EL RESULTADO\ncuando hay que cerrar",
             ha="center", va="center", color="#e8e8e8", fontsize=10,
             linespacing=1.5)
    fig.text(0.5, 0.795, "ARRIBA: los que hacen LAS DOS COSAS",
             ha="center", va="center", color="yellow", fontsize=10,
             fontweight="bold", style="italic")

    # Notas (en esquinas libres del lienzo, no sobre el diamante)
    fig.text(0.045, 0.06, "Zona gris: el 60% de jugadores mas normalitos\n"
             "(percentil 20-80 en cada indice).",
             ha="left", va="bottom", color="#9a9c9b", fontsize=8.5,
             style="italic", linespacing=1.5)
    add_logo(fig, width_frac=0.15)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    out = "outputs/viz/scatter_remontador_cerrojo.png"
    diamond_scatter(df, save_path=out)
    print(f"OK -> {out}  ({df.height} jugadores)")
