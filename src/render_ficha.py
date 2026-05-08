"""render_ficha - Render visual de la ficha PCJ scout-facing por jugador.

Lee outputs/pcj_table.parquet y produce un printout en terminal/markdown
con barras unicode + frases scout traducidas. NO genera PDF — solo console.

Uso:
    python -m src.render_ficha 1234              # por pff_player_id
    python -m src.render_ficha "Kylian Mbappé"   # por nombre (substring match)
    python -m src.render_ficha --top chasing 10  # top 10 chasing global
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_REPO  = Path(__file__).resolve().parents[1]
_TABLE = _REPO / "outputs" / "pcj_table.parquet"

CHANNEL_LABELS = {"atk": "Ataque", "def": "Defensa",
                   "off": "Off-ball", "phys": "Físico"}


def _bar(value: float, lo: float = -0.6, hi: float = +0.6, width: int = 10) -> str:
    """Barra unicode para un valor en escala bayesiana approx [-0.6, +0.6]."""
    if value is None:
        return "─" * width
    norm = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n_full = int(round(norm * width))
    return "█" * n_full + "░" * (width - n_full)


def _label_intensity(value: float) -> str:
    if value is None:        return "n/a"
    if value > 0.40:         return "TOP"
    if value > 0.20:         return "alto"
    if value > 0.05:         return "medio+"
    if value > -0.05:        return "neutro"
    if value > -0.20:        return "medio-"
    if value > -0.40:        return "bajo"
    return "MUY BAJO"


def _trend(value: float) -> str:
    if value is None:        return "─"
    if value > 0.30:         return "↑↑↑"
    if value > 0.15:         return "↑↑"
    if value > 0.05:         return "↑"
    if value > -0.05:        return "─"
    if value > -0.15:        return "↓"
    if value > -0.30:        return "↓↓"
    return "↓↓↓"


def _safe(row: dict, col: str, default=None):
    return row.get(col, default) if col in row else default


def render_ficha(player_row: dict) -> str:
    """Construye string-ficha legible desde 1 fila pcj_table."""
    r = player_row
    name = r.get("player_name") or f"player_id={r['pff_player_id']}"
    team = r.get("team_name", "?")
    pos = r.get("position_group", "?")
    age = r.get("age_years")
    age_s = f"{int(age)} años" if age is not None else "edad ?"
    minutes = r.get("minutes_played", 0)
    n_matches = r.get("n_matches_played", 0)
    n_gf = r.get("n_shocks_for", 0); n_ga = r.get("n_shocks_against", 0)
    n_ko = r.get("n_shocks_ko", 0); n_high_lev = r.get("n_high_leverage_shocks", 0)
    n_elim = r.get("n_elimination_shocks", 0)

    out = []
    out.append("═" * 72)
    out.append(f"{name:<40} {team:>15} · {pos} · {age_s}")
    out.append(f"{minutes:>6} min · {n_matches} partidos · "
                f"{n_gf} GF · {n_ga} GA · {n_ko} KO · {n_high_lev} high-lev")
    out.append("═" * 72)

    # Bloques GA / GF base + escenarios
    for shock_label, shock_key in [("CUANDO SU EQUIPO ENCAJA (post-GA)", "GOAL_AGAINST"),
                                     ("CUANDO SU EQUIPO MARCA (post-GF)", "GOAL_FOR")]:
        out.append("")
        out.append(shock_label)
        out.append(f"  {'canal':<10} {'jugador (base)':<28} "
                    f"{'equipo ataca':<16} {'equipo defiende':<16}")
        for ch in ("atk", "def", "off", "phys"):
            base_mean = r.get(f"cate_{CHANNEL_LABELS_REV[ch]}_{shock_key}_mean")
            sc_atk = r.get(f"clutch_{ch}_{shock_key}_team_attacks_mean")
            sc_def = r.get(f"clutch_{ch}_{shock_key}_team_defends_mean")
            base_str = f"{_bar(base_mean):<10} {_label_intensity(base_mean):<10}"
            sc_atk_str = f"{_label_intensity(sc_atk):<6} {_trend(sc_atk):>6}"
            sc_def_str = f"{_label_intensity(sc_def):<6} {_trend(sc_def):>6}"
            out.append(f"  {CHANNEL_LABELS[ch]:<10} {base_str:<28} "
                        f"{sc_atk_str:<16} {sc_def_str:<16}")

    # Pressure response (3a dimension)
    out.append("")
    out.append("EN MOMENTOS DE ELIMINACIÓN (elim_prox alto)")
    pri = r.get("pressure_response_idx")
    pri_p = r.get("p_pressure_clutch_positive", 0.0)
    out.append(f"  índice global: {_bar(pri):<10} {_label_intensity(pri):<10} "
                f"P(>0)={pri_p:.2f}  → {r.get('sig_pressure', '?')}")

    # Indices agregados
    out.append("")
    out.append("ÍNDICES AGREGADOS")
    chasing = r.get("chasing_clutch_idx"); p_chasing = r.get("p_chasing_positive", 0.0)
    prot = r.get("protecting_clutch_idx"); p_prot = r.get("p_protecting_positive", 0.0)
    out.append(f"  Empuje post-GA:  {_bar(chasing)} {_label_intensity(chasing):<10} "
                f"P(>0)={p_chasing:.2f}  → {r.get('sig_chasing','?')}")
    out.append(f"  Solidez post-GF: {_bar(prot)} {_label_intensity(prot):<10} "
                f"P(>0)={p_prot:.2f}  → {r.get('sig_protecting','?')}")
    out.append(f"  Pressure resp:   {_bar(pri)} {_label_intensity(pri):<10} "
                f"P(>0)={pri_p:.2f}  → {r.get('sig_pressure','?')}")

    # Frases scout traducidas (mapeo lenguaje del entrenador, propuesta_final)
    out.append("")
    out.append("LO QUE EL SCOUT LEE EN UNA FRASE")
    frases = [
        ("Aparece cuando importa", chasing, p_chasing),
        ("Aguanta cuando hay que aguantar", prot, p_prot),
        ("Carga con el equipo a la espalda", chasing, p_chasing),
        ("Pressure clutch (final eliminatoria)", pri, pri_p),
    ]
    for frase, val, p in frases:
        check = "SI" if (val is not None and val > 0.10 and p >= 0.85) else "NO"
        out.append(f"  {frase:<42} {check:<4}  P={p:.2f}")

    # Ranking within position
    out.append("")
    out.append(f"COMPARADO CON SU ROL ({pos})")
    for label, col_rank in [("Empuje post-GA", "rank_chasing_in_position"),
                              ("Solidez post-GF", "rank_protecting_in_position"),
                              ("Pressure resp.", "rank_pressure_in_position")]:
        rk = r.get(col_rank)
        tier_col = col_rank.replace("rank_", "tier_")
        tier = r.get(tier_col, "?")
        out.append(f"  {label:<22} #{rk if rk is not None else '?':<4} · {tier}")

    out.append("═" * 72)
    return "\n".join(out)


# Reverse map para CHANNEL_LABELS (cate uses full names: ataque/defensa/...)
CHANNEL_LABELS_REV = {"atk": "ataque", "def": "defensa",
                       "off": "offball", "phys": "fisico"}


def find_player(query: str, df: pl.DataFrame) -> dict | None:
    """Busca jugador por id (entero) o por substring del nombre."""
    try:
        pid = int(query)
        sub = df.filter(pl.col("pff_player_id") == pid)
    except ValueError:
        sub = df.filter(pl.col("player_name").str.contains(query, literal=False))
    if sub.height == 0:
        return None
    if sub.height > 1:
        print(f"  varios matches ({sub.height}); usando el primero. Otros:")
        for r in sub.head(5).iter_rows(named=True):
            print(f"    {r['pff_player_id']} {r['player_name']} ({r['team_name']})")
    return sub.row(0, named=True)


def render_top(idx: str, n: int = 10) -> None:
    df = pl.read_parquet(_TABLE)
    sort_col = {"chasing": "chasing_clutch_idx",
                "protecting": "protecting_clutch_idx",
                "pressure": "pressure_response_idx"}[idx]
    top = df.sort(sort_col, descending=True).head(n)
    print(f"\nTop {n} por {idx}:\n")
    print(top.select(["pff_player_id", "player_name", "team_name",
                       "position_group", sort_col,
                       sort_col.replace("_idx", "_lo80")
                         if idx != "pressure" else "pressure_response_lo80"]))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--top":
        idx = sys.argv[2] if len(sys.argv) > 2 else "chasing"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        render_top(idx, n)
        return
    df = pl.read_parquet(_TABLE)
    row = find_player(sys.argv[1], df)
    if row is None:
        print(f"  jugador no encontrado: {sys.argv[1]!r}")
        sys.exit(1)
    print(render_ficha(row))


if __name__ == "__main__":
    main()
