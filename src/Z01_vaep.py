"""
Z01_vaep - Atomic-VAEP feature/label extraction + persistencia de modelos.

Wrapper minimo sobre `socceraction.atomic.vaep` para exponer SOLO lo que M08
necesita (M08 hace su propio train + Optuna + isotonic + apply, asi que aqui
NO replicamos esa logica). Reduce Z01 a:

  1. compute_features(actions, atomic=True, ...) -> features dataframe
  2. compute_labels(actions, atomic=True, ...)   -> (y_scores, y_concedes)
  3. formula_mod(atomic)                          -> modulo formula (usado en M08
                                                     para `value(actions, p_s, p_c)`)
  4. save_models / load_models                    -> CatBoost .cbm helpers

Cache granular por partido en `cache/vaep/{features,labels}/{tag}/{game_id}.parquet`.
Tag = "atomic_{provider}_{prev/horizon}" para evitar colisiones entre datasets.

Soporta tambien VAEP clasico (atomic=False) por compatibilidad con codigo que
quiera el formato no-atomic, pero M08 no lo usa.

Uso:
    import Z01_vaep as vaep_mod
    X = vaep_mod.compute_features(actions, atomic=True, provider="statsbomb_atk")
    y_s, y_c = vaep_mod.compute_labels(actions, atomic=True, provider="statsbomb_atk")
    # ... entrena fuera (M08 usa Optuna+CatBoost) ...
    values = vaep_mod.formula_mod(atomic=True).value(actions, p_s, p_c)
    vaep_mod.save_models(model_s, model_c, "model/vaep_atk")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

# socceraction VAEP clasico
from socceraction.vaep import features as _vf
from socceraction.vaep import labels as _vl
from socceraction.vaep import formula as _vfm

# socceraction Atomic-VAEP
from socceraction.atomic.vaep import features as _af
from socceraction.atomic.vaep import labels as _al
from socceraction.atomic.vaep import formula as _afm


# -- Constantes -------------------------------------------------------------

_CACHE = Path(__file__).resolve().parents[1] / "cache" / "vaep"

NB_PREV_ACTIONS = 3   # ventana de acciones previas para features
NR_ACTIONS = 10       # horizonte para labels (gol/encajar en 10 acciones)

_VAEP_FEAT_FNS = [
    _vf.actiontype_onehot, _vf.bodypart_onehot, _vf.result_onehot,
    _vf.goalscore, _vf.startlocation, _vf.endlocation,
    _vf.movement, _vf.space_delta, _vf.time, _vf.time_delta, _vf.team,
]
_ATOMIC_FEAT_FNS = [
    _af.actiontype_onehot, _af.bodypart_onehot,
    _af.goalscore, _af.location, _af.polar,
    _af.movement_polar, _af.direction, _af.team, _af.time, _af.time_delta,
]


# -- Helpers privados -------------------------------------------------------

def _feat_fns(atomic: bool) -> list:
    return _ATOMIC_FEAT_FNS if atomic else _VAEP_FEAT_FNS


def _label_mod(atomic: bool):
    return _al if atomic else _vl


def _feat_mod(atomic: bool):
    return _af if atomic else _vf


def formula_mod(atomic: bool):
    """Modulo formula VAEP (atomic o classic). API publica usada por M08."""
    return _afm if atomic else _vfm


def _mode_tag(atomic: bool) -> str:
    return "atomic" if atomic else "vaep"


# -- Features ---------------------------------------------------------------

def compute_features(
    actions: pd.DataFrame,
    atomic: bool = False,
    nb_prev: int = NB_PREV_ACTIONS,
    provider: str = "default",
) -> pd.DataFrame:
    """Extrae features VAEP de las acciones, partido a partido.

    Procesa por game_id porque gamestates() agrupa por (game_id, period_id).
    Cache: un parquet por partido en cache/vaep/features/{tag}/{game_id}.parquet.

    Args:
        actions  : DataFrame SPADL (con type_name, etc. de add_names).
        atomic   : Si True, usa features Atomic-VAEP (10 fns, 148 cols).
                   Si False, VAEP clasico (11 fns, 142 cols).
        nb_prev  : Numero de acciones previas en la ventana (default: 3).
        provider : Etiqueta de provider ("statsbomb_atk", "statsbomb_wc22", ...)
                   para evitar colisiones de cache entre datasets.

    Returns:
        DataFrame con las features. Mismo indice y orden que actions.
    """
    mode = _mode_tag(atomic)
    tag = f"{mode}_{provider}_prev{nb_prev}"
    cache_dir = _CACHE / "features" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    fmod = _feat_mod(atomic)
    fns = _feat_fns(atomic)

    parts = []
    for gid, group in actions.groupby("game_id", sort=False):
        cache_path = cache_dir / f"{gid}.parquet"
        if cache_path.exists():
            X = pd.read_parquet(cache_path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gamestates = fmod.gamestates(group, nb_prev_actions=nb_prev)
                X = pd.concat([fn(gamestates) for fn in fns], axis=1)
            X.to_parquet(cache_path, index=False)
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


# -- Labels -----------------------------------------------------------------

def compute_labels(
    actions: pd.DataFrame,
    atomic: bool = False,
    nr_actions: int = NR_ACTIONS,
    provider: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computa labels de scores y concedes, partido a partido.

    Cache: un parquet por partido en cache/vaep/labels/{tag}/{game_id}.parquet.

    Args:
        actions    : DataFrame SPADL.
        atomic     : Si True, usa labels Atomic-VAEP.
        nr_actions : Horizonte de acciones (default: 10).
        provider   : Etiqueta de provider para evitar colisiones de cache.

    Returns:
        (y_scores, y_concedes) cada uno DataFrame con 1 columna.
    """
    mode = _mode_tag(atomic)
    tag = f"{mode}_{provider}_h{nr_actions}"
    cache_dir = _CACHE / "labels" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    lmod = _label_mod(atomic)

    scores_parts, concedes_parts = [], []
    for gid, group in actions.groupby("game_id", sort=False):
        cache_path = cache_dir / f"{gid}.parquet"
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            y_s = cached[["scores"]]
            y_c = cached[["concedes"]]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_s = lmod.scores(group, nr_actions=nr_actions)
                y_c = lmod.concedes(group, nr_actions=nr_actions)
            pd.concat([y_s, y_c], axis=1).to_parquet(cache_path, index=False)
        scores_parts.append(y_s)
        concedes_parts.append(y_c)

    return (pd.concat(scores_parts, ignore_index=True),
            pd.concat(concedes_parts, ignore_index=True))


# -- Save / Load modelos CatBoost -------------------------------------------

def save_models(
    model_scores: CatBoostClassifier,
    model_concedes: CatBoostClassifier,
    path: str | Path,
) -> Path:
    """Guarda los dos modelos CatBoost en disco (formato nativo .cbm).

    Crea {path}_scores.cbm y {path}_concedes.cbm.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_scores.save_model(str(path) + "_scores.cbm")
    model_concedes.save_model(str(path) + "_concedes.cbm")
    return path.parent


def load_models(path: str | Path) -> tuple[CatBoostClassifier, CatBoostClassifier]:
    """Carga los dos modelos CatBoost desde disco. Mismo prefijo que save_models."""
    model_s = CatBoostClassifier()
    model_c = CatBoostClassifier()
    model_s.load_model(str(path) + "_scores.cbm")
    model_c.load_model(str(path) + "_concedes.cbm")
    return model_s, model_c
