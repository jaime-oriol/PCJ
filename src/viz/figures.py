"""figures - Figuras analiticas del PCJ (capa causal).

Patron portado de jaime-oriol/Diagonality_3D (deliverable/make_figures.py):
ejes 'dark journal', grid-y suave, identidad Diagonality.

Figura principal: event-study (M12) — el efecto del shock minuto a minuto,
en lenguaje legible (antes / despues del gol).

Uso:
    python -m src.viz.figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import ATT, BG, DEF, WHITE, add_logo, style_ax

_DID = _SRC.parent / "data" / "parquet" / "derived" / "did"

_CH_LABEL = {"ataque": "Empuje ofensivo", "defensa": "Solidez defensiva",
             "offball": "Juego sin balon", "fisico": "Intensidad fisica"}
_CH_ORDER = ["ataque", "defensa", "offball", "fisico"]
_SH = [("GOAL_AGAINST", "tras ENCAJAR un gol", ATT),
       ("GOAL_FOR",     "tras MARCAR un gol",  DEF)]


def event_study(save_path=None):
    """Event-study Sun-Abraham: el efecto del shock minuto a minuto."""
    es = pl.read_parquet(_DID / "event_study.parquet")

    fig, axes = plt.subplots(4, 2, figsize=(13, 14.5), sharex=True)
    fig.set_facecolor(BG)

    for ri, ch in enumerate(_CH_ORDER):
        for ci, (sh, sh_lbl, color) in enumerate(_SH):
            ax = axes[ri, ci]
            style_ax(ax, ygrid=True)
            d = (es.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
                   .sort("relative_min"))
            x = d["relative_min"].to_numpy()
            b = d["beta"].to_numpy()
            lo, hi = d["ci_lo"].to_numpy(), d["ci_hi"].to_numpy()

            # Zona "despues del gol" sombreada + linea del gol en minuto 0
            ax.axvspan(0, 10, color=color, alpha=0.07, zorder=0)
            ax.axhline(0, color="#8a8c8b", lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.axvline(0, color=WHITE, lw=1.3, alpha=0.8, zorder=2)
            # Banda de incertidumbre + linea del efecto
            ax.fill_between(x, lo, hi, color=color, alpha=0.22, zorder=2)
            ax.plot(x, b, "-o", color=color, ms=4, lw=2.0, zorder=3)

            ax.set_title(f"{_CH_LABEL[ch]}   ·   {sh_lbl}",
                         color=WHITE, fontsize=11.5, fontweight="bold", pad=9)
            ax.set_xlim(-10.5, 10.5)
            ax.set_xticks(range(-10, 11, 5))
            ax.tick_params(labelsize=9)
            if ri == 3:
                ax.set_xlabel("minutos respecto al gol", fontsize=10)
            if ci == 0:
                ax.set_ylabel("cambio en el jugador", fontsize=10)
            # Etiquetas ANTES / DESPUES solo en el panel superior izquierdo
            if ri == 0 and ci == 0:
                yt = ax.get_ylim()[1]
                ax.text(-5, yt, "ANTES", ha="center", va="top", color="#9a9c9b",
                        fontsize=9, fontweight="bold", style="italic")
                ax.text(5, yt, "DESPUES", ha="center", va="top", color=WHITE,
                        fontsize=9, fontweight="bold", style="italic")

    fig.text(0.5, 0.977, "Cambia el jugador despues de un gol?",
             ha="center", va="top", color=WHITE, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.957,
             "El efecto del shock emocional minuto a minuto, en los 4 canales "
             "del juego  ·  Mundial Qatar 2022",
             ha="center", va="top", color="#c8c8c8", fontsize=10.5)
    fig.text(0.5, 0.022,
             "Como leerlo: cada punto es cuanto cambia el jugador medio ese "
             "minuto respecto al minuto previo al gol.  La banda es la "
             "incertidumbre.\nSi la linea se despega de la raya del 0 en la "
             "zona sombreada (despues del gol), el shock tuvo efecto.",
             ha="center", va="bottom", color="#9a9c9b", fontsize=9.5,
             style="italic", linespacing=1.6)

    fig.tight_layout(rect=[0, 0.055, 1, 0.945])
    add_logo(fig, width_frac=0.085)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    out = "outputs/viz/event_study.png"
    event_study(save_path=out)
    print(f"OK -> {out}")
