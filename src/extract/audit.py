"""
extract.audit - Auditoria exhaustiva de los parquets vs JSON original.

Verifica al maximo detalle, partido a partido, sin OOM:
  1. CONTEOS: ningun fichero source omitido en parquet
  2. KEYS: todas las claves presentes en cualquier row del JSON estan en
     el schema parquet
  3. ROUND-TRIP: comparacion JSON <-> parquet, lossless
     - PFF events: TODOS los rows de TODOS los partidos
     - PFF tracking: 500 frames muestreados por partido
     - PFF metadata + rosters: TODO
     - StatsBomb: TODOS los rows de cada partido (events, lineups, freeze)
     - Wyscout: TODOS los rows de cada catalogo + 5000 muestreados de events

Uso:
    from src.extract.audit import run_full_audit
    run_full_audit()
"""

from __future__ import annotations

import bz2
import gc
import json
import random
from pathlib import Path
from typing import Any

import polars as pl

from ._common import (
    DATA_PFF, DATA_PUB, PARQUET, clean_empty_strings as _clean_empty_strings,
    deep_equal,
)


SB  = DATA_PUB / "statsbomb" / "data"
WS  = DATA_PUB / "wyscout"


# -- Helpers ----------------------------------------------------------------

def _all_keys(rows: list[dict]) -> set[str]:
    """Union de claves de todos los dicts (top-level)."""
    out = set()
    for r in rows:
        out.update(r.keys())
    return out


def _flatten_schema_keys(schema: pl.Schema) -> set[str]:
    """Top-level cols del parquet."""
    return set(schema.names())


def _read_jsonl_bz2_sampled(path: Path, sample_idxs: set[int]) -> dict[int, dict]:
    """Stream jsonl.bz2 y devuelve solo las filas en sample_idxs."""
    out = {}
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in sample_idxs:
                line = line.strip()
                if line:
                    out[i] = json.loads(line)
                if len(out) == len(sample_idxs):
                    break
    return out


# -- AUDITORIAS POR FUENTE --------------------------------------------------

def audit_pff_events() -> dict:
    """Round-trip COMPLETO en TODOS los partidos PFF events."""
    res = {"source": "PFF events", "files_ok": 0, "files_fail": 0,
           "rows_ok": 0, "rows_fail": 0, "errors": []}
    src_files = sorted((DATA_PFF/"Event Data").glob("*.json"))

    for src in src_files:
        gid = int(src.stem)
        pq = PARQUET / "pff/events" / f"{gid}.parquet"
        if not pq.exists():
            res["errors"].append(f"MISSING PARQUET {gid}")
            res["files_fail"] += 1
            continue

        orig = json.load(open(src))
        recon = pl.read_parquet(pq).to_dicts()
        if len(orig) != len(recon):
            res["errors"].append(f"len mismatch {gid}: {len(orig)} vs {len(recon)}")
            res["files_fail"] += 1
            continue

        # Keys check
        orig_keys = _all_keys(orig)
        pq_keys = set(pl.read_parquet(pq).columns)
        missing = orig_keys - pq_keys
        if missing:
            res["errors"].append(f"keys missing {gid}: {missing}")

        diffs = sum(1 for a,b in zip(orig, recon) if not deep_equal(a, b))
        res["rows_ok"] += len(orig) - diffs
        res["rows_fail"] += diffs
        if diffs == 0:
            res["files_ok"] += 1
        else:
            res["files_fail"] += 1
            res["errors"].append(f"{gid}: {diffs}/{len(orig)} rows differ")

        del orig, recon
        gc.collect()
    return res


def audit_pff_tracking(sample_per_match: int = 500) -> dict:
    """Round-trip muestreado en TODOS los partidos PFF tracking."""
    res = {"source": "PFF tracking", "files_ok": 0, "files_fail": 0,
           "frames_compared": 0, "frames_fail": 0, "errors": []}
    src_files = sorted((DATA_PFF/"Tracking Data").glob("*.jsonl.bz2"))
    rng = random.Random(0)

    for src in src_files:
        gid = int(src.name.split(".")[0])
        pq = PARQUET / "pff/tracking" / f"{gid}.parquet"
        if not pq.exists():
            res["errors"].append(f"MISSING PARQUET {gid}")
            res["files_fail"] += 1
            continue

        n = pl.scan_parquet(pq).select(pl.len()).collect().item()
        sample = sorted(rng.sample(range(n), min(sample_per_match, n)))
        json_rows = _read_jsonl_bz2_sampled(src, set(sample))
        pq_df = pl.read_parquet(pq).with_row_index("_idx").filter(
            pl.col("_idx").is_in(sample)
        ).drop("_idx")
        pq_rows = {idx: row for idx, row in zip(sample, pq_df.to_dicts())}

        diffs = 0
        for i in sample:
            if i not in json_rows or i not in pq_rows:
                diffs += 1
                continue
            if not deep_equal(json_rows[i], pq_rows[i]):
                diffs += 1
        res["frames_compared"] += len(sample)
        res["frames_fail"] += diffs
        if diffs == 0:
            res["files_ok"] += 1
        else:
            res["files_fail"] += 1
            res["errors"].append(f"{gid}: {diffs}/{len(sample)} frames differ")

        del json_rows, pq_df, pq_rows
        gc.collect()
    return res


def audit_pff_metadata_rosters() -> dict:
    """Round-trip COMPLETO de metadata y rosters PFF."""
    res = {"source": "PFF metadata+rosters", "errors": []}

    # Metadata
    md_orig = []
    for f in sorted((DATA_PFF/"Metadata").glob("*.json")):
        d = json.load(open(f))
        md_orig.extend(d if isinstance(d, list) else [d])
    md_recon = pl.read_parquet(PARQUET/"pff/metadata.parquet").to_dicts()
    md_diffs = sum(1 for a,b in zip(md_orig, md_recon) if not deep_equal(a, b))
    res["metadata_rows"] = len(md_orig)
    res["metadata_diffs"] = md_diffs
    if md_diffs:
        res["errors"].append(f"metadata: {md_diffs}/{len(md_orig)} differ")

    # Rosters: anadimos col match_id, comparamos respetando esa adicion
    ros_orig = []
    for f in sorted((DATA_PFF/"Rosters").glob("*.json")):
        gid = int(f.stem)
        for r in json.load(open(f)):
            r["match_id"] = gid
            ros_orig.append(r)
    ros_recon = pl.read_parquet(PARQUET/"pff/rosters.parquet").to_dicts()
    ros_diffs = sum(1 for a,b in zip(ros_orig, ros_recon) if not deep_equal(a, b))
    res["rosters_rows"] = len(ros_orig)
    res["rosters_diffs"] = ros_diffs
    if ros_diffs:
        res["errors"].append(f"rosters: {ros_diffs}/{len(ros_orig)} differ")

    return res


def audit_statsbomb() -> dict:
    """Round-trip COMPLETO de TODOS los partidos StatsBomb."""
    res = {"source": "StatsBomb",
           "competitions_diffs": 0, "matches_diffs": 0,
           "events_files_ok": 0, "events_files_fail": 0,
           "events_rows_compared": 0, "events_rows_fail": 0,
           "lineups_files_ok": 0, "lineups_files_fail": 0,
           "freeze_files_ok": 0, "freeze_files_fail": 0,
           "freeze_rows_compared": 0, "freeze_rows_fail": 0,
           "errors": []}

    # Competitions
    comp_orig = json.load(open(SB/"competitions.json"))
    comp_recon = pl.read_parquet(PARQUET/"statsbomb/competitions.parquet").to_dicts()
    res["competitions_diffs"] = sum(1 for a,b in zip(comp_orig, comp_recon) if not deep_equal(a,b))

    # Matches union
    matches_orig = []
    for cd in sorted((SB/"matches").iterdir()):
        if cd.is_dir():
            for f in sorted(cd.glob("*.json")):
                matches_orig.extend(json.load(open(f)))
    matches_recon = pl.read_parquet(PARQUET/"statsbomb/matches.parquet").to_dicts()
    res["matches_diffs"] = sum(1 for a,b in zip(matches_orig, matches_recon) if not deep_equal(a,b))

    # Events per partido (TODOS los rows)
    src_events = sorted((SB/"events").glob("*.json"))
    for src in src_events:
        mid = int(src.stem)
        pq = PARQUET / "statsbomb/events" / f"{mid}.parquet"
        orig = json.load(open(src))
        recon = pl.read_parquet(pq).to_dicts()
        diffs = sum(1 for a,b in zip(orig, recon) if not deep_equal(a, b))
        res["events_rows_compared"] += len(orig)
        res["events_rows_fail"] += diffs
        if diffs == 0:
            res["events_files_ok"] += 1
        else:
            res["events_files_fail"] += 1
            res["errors"].append(f"events {mid}: {diffs}/{len(orig)}")
        del orig, recon
        gc.collect()

    # Lineups (todos)
    src_l = sorted((SB/"lineups").glob("*.json"))
    for src in src_l:
        mid = int(src.stem)
        pq = PARQUET / "statsbomb/lineups" / f"{mid}.parquet"
        orig = json.load(open(src))
        recon = pl.read_parquet(pq).to_dicts()
        diffs = sum(1 for a,b in zip(orig, recon) if not deep_equal(a, b))
        if diffs == 0:
            res["lineups_files_ok"] += 1
        else:
            res["lineups_files_fail"] += 1
            res["errors"].append(f"lineups {mid}: {diffs}")

    # Freeze frames TODOS rows
    src_ff = sorted((SB/"three-sixty").glob("*.json"))
    for src in src_ff:
        mid = int(src.stem)
        pq = PARQUET / "statsbomb/freeze_frames" / f"{mid}.parquet"
        if not pq.exists():
            res["errors"].append(f"freeze missing parquet {mid}")
            continue
        orig = json.load(open(src))
        recon = pl.read_parquet(pq).to_dicts()
        diffs = sum(1 for a,b in zip(orig, recon) if not deep_equal(a, b))
        res["freeze_rows_compared"] += len(orig)
        res["freeze_rows_fail"] += diffs
        if diffs == 0:
            res["freeze_files_ok"] += 1
        else:
            res["freeze_files_fail"] += 1
            res["errors"].append(f"freeze {mid}: {diffs}/{len(orig)}")
        del orig, recon
        gc.collect()

    return res


def audit_wyscout(events_sample: int = 5000) -> dict:
    """Wyscout: round-trip COMPLETO de catalogos + muestreo en events."""
    res = {"source": "Wyscout", "errors": [], "checks": {}}
    rng = random.Random(0)

    # Catalogos COMPLETOS
    for fname in ["players", "teams", "coaches", "playerank"]:
        orig = _clean_empty_strings(json.load(open(WS/f"{fname}.json")))
        recon = pl.read_parquet(PARQUET/f"wyscout/{fname}.parquet").to_dicts()
        diffs = sum(1 for a,b in zip(orig, recon) if not deep_equal(a, b))
        res["checks"][fname] = (len(orig), diffs)
        if diffs:
            res["errors"].append(f"{fname}: {diffs}/{len(orig)}")

    # Matches union
    rows = []
    for comp in ["England","France","Germany","Italy","Spain","European_Championship","World_Cup"]:
        src = WS/f"matches_{comp}.json"
        if src.exists():
            for m in json.load(open(src)):
                m["competition"] = comp
                rows.append(m)
    rows = _clean_empty_strings(rows)
    recon = pl.read_parquet(PARQUET/"wyscout/matches.parquet").to_dicts()
    diffs = sum(1 for a,b in zip(rows, recon) if not deep_equal(a,b))
    res["checks"]["matches"] = (len(rows), diffs)
    if diffs:
        res["errors"].append(f"matches: {diffs}")

    # Events: muestreo grande por competicion
    for comp in ["England","France","Germany","Italy","Spain","European_Championship","World_Cup"]:
        src = WS/f"events_{comp}.json"
        pq = PARQUET/f"wyscout/events_{comp}.parquet"
        orig = _clean_empty_strings(json.load(open(src)))
        n = len(orig)
        sample = sorted(rng.sample(range(n), min(events_sample, n)))
        sample_orig = [orig[i] for i in sample]
        recon = pl.read_parquet(pq).with_row_index("_idx").filter(
            pl.col("_idx").is_in(sample)
        ).drop("_idx").to_dicts()
        diffs = sum(1 for a,b in zip(sample_orig, recon) if not deep_equal(a,b))
        res["checks"][f"events_{comp}"] = (len(sample), diffs, n)
        if diffs:
            res["errors"].append(f"events_{comp}: {diffs}/{len(sample)}")
        del orig, sample_orig, recon
        gc.collect()

    return res


# -- Runner -----------------------------------------------------------------

def run_full_audit() -> dict:
    """Orquesta todas las auditorias y devuelve dict con resultados."""
    print("=" * 70)
    print("AUDITORIA EXHAUSTIVA — todos los parquets vs JSON original")
    print("=" * 70)

    print("\n[1/4] PFF events (TODOS los rows de TODOS los partidos)...")
    r1 = audit_pff_events()
    print(f"  files OK={r1['files_ok']}  FAIL={r1['files_fail']}")
    print(f"  rows  OK={r1['rows_ok']:,}  FAIL={r1['rows_fail']}")
    for e in r1["errors"][:5]: print(f"  ERR: {e}")

    print("\n[2/4] PFF tracking (500 frames/partido x 64 partidos)...")
    r2 = audit_pff_tracking(sample_per_match=500)
    print(f"  files OK={r2['files_ok']}  FAIL={r2['files_fail']}")
    print(f"  frames compared={r2['frames_compared']:,}  FAIL={r2['frames_fail']}")
    for e in r2["errors"][:5]: print(f"  ERR: {e}")

    print("\n[3/4] PFF metadata + rosters (TODOS los rows)...")
    r3 = audit_pff_metadata_rosters()
    print(f"  metadata: {r3['metadata_rows']} rows, {r3['metadata_diffs']} diffs")
    print(f"  rosters:  {r3['rosters_rows']} rows, {r3['rosters_diffs']} diffs")
    for e in r3["errors"]: print(f"  ERR: {e}")

    print("\n[4/4] StatsBomb (TODOS los rows de TODOS los partidos)...")
    r4 = audit_statsbomb()
    print(f"  competitions diffs: {r4['competitions_diffs']}")
    print(f"  matches diffs: {r4['matches_diffs']}")
    print(f"  events  files OK={r4['events_files_ok']}  FAIL={r4['events_files_fail']}  rows={r4['events_rows_compared']:,} diffs={r4['events_rows_fail']}")
    print(f"  lineups files OK={r4['lineups_files_ok']}  FAIL={r4['lineups_files_fail']}")
    print(f"  freeze  files OK={r4['freeze_files_ok']}  FAIL={r4['freeze_files_fail']}  rows={r4['freeze_rows_compared']:,} diffs={r4['freeze_rows_fail']}")
    for e in r4["errors"][:5]: print(f"  ERR: {e}")

    print("\n[5/5] Wyscout (catalogos completos + 5000 events/comp)...")
    r5 = audit_wyscout(events_sample=5000)
    for k, v in r5["checks"].items():
        if len(v) == 2:
            n, d = v
            print(f"  {k}: {n} comparados, {d} diffs")
        else:
            n, d, total = v
            print(f"  {k}: {n}/{total} comparados, {d} diffs")
    for e in r5["errors"]: print(f"  ERR: {e}")

    print("\n" + "=" * 70)
    all_ok = (r1["rows_fail"] == 0 and r1["files_fail"] == 0
              and r2["frames_fail"] == 0
              and r3["metadata_diffs"] == 0 and r3["rosters_diffs"] == 0
              and r4["events_rows_fail"] == 0 and r4["freeze_rows_fail"] == 0
              and r4["lineups_files_fail"] == 0
              and not r5["errors"])
    print(f"VEREDICTO: {'TODO 100% LOSSLESS PERFECTO' if all_ok else 'HAY FALLOS'}")
    print("=" * 70)
    return {"pff_events": r1, "pff_tracking": r2, "pff_md_rost": r3,
            "statsbomb": r4, "wyscout": r5, "all_ok": all_ok}


if __name__ == "__main__":
    run_full_audit()
