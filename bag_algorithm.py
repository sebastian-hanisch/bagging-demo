"""Bagging: B volle CART-Bäume auf Bootstrap-Stichproben (mit Zurücklegen), gemittelt (Regression) bzw. per Mehrheit/Wahrscheinlichkeits-Mittel (Klassifikation). Der Baumkern selbst steht in `bag_tree.py` (wortgleich aus cart-demo).
Jeder Baum sieht im Mittel nur 1 - 1/e ≈ 63.2 % der Trainingszeilen; die übrigen ("out-of-bag", OOB) liefern eine eingebaute Testschätzung ohne eigene Testdaten zu verbrauchen."""

from dataclasses import dataclass

import numpy as np

import bag_tree as T


@dataclass(frozen=True)
class Bagging:
    trees: tuple                 # Tupel von T.Tree, ein Baum je Bootstrap-Stichprobe
    in_bag: np.ndarray            # (B, n) bool: True, wo die Zeile in dieser Stichprobe vorkam (mit Zurücklegen, Zählung egal)
    task: str
    n_train: int


def bootstrap_indices(n, seed):
    """n Zeilennummern mit Zurücklegen (eine Bootstrap-Stichprobe)."""
    return np.random.default_rng(seed).integers(0, n, n)


def fit(X, y, task, criterion=None, min_leaf=1, n_trees=30, seed=0):
    """Wächst n_trees volle Bäume auf unabhängigen Bootstrap-Stichproben (Seeds seed, seed+1, ... - deterministisch, damit einzelne Bäume reproduzierbar sind)."""
    n = len(y)
    trees, in_bag = [], np.zeros((n_trees, n), dtype=bool)
    for b in range(n_trees):
        idx = bootstrap_indices(n, seed * 1_000_003 + b)
        trees.append(T.grow(X[idx], y[idx], task, criterion, None, min_leaf))
        in_bag[b, idx] = True
    return Bagging(tuple(trees), in_bag, task, n)


def _tree_values(bagging, X, upto=None):
    """(B, m) Matrix der Blattwerte jedes Baums (Anteil bzw. Mittelwert) für X; `upto` beschränkt auf die ersten Bäume."""
    trees = bagging.trees[:upto] if upto else bagging.trees
    return np.array([T.predict_value(t, X) for t in trees])


def predict_value(bagging, X, upto=None):
    """Mittelwert der Blattwerte über die (ersten `upto`) Bäume - bei Klassifikation eine gemittelte Wahrscheinlichkeit, bei Regression den Mittelwert der Vorhersagen."""
    return _tree_values(bagging, X, upto).mean(axis=0)


def predict(bagging, X, upto=None):
    v = predict_value(bagging, X, upto)
    return (v > 0.5).astype(int) if bagging.task == "class" else v


def oob_predict(bagging, X):
    """Für jede Trainingszeile der Mittelwert nur der Bäume, die sie NICHT gesehen haben (out-of-bag). NaN, wenn nie out-of-bag (bei wenigen Bäumen möglich)."""
    vals = _tree_values(bagging, X)                            # (B, n)
    oob = ~bagging.in_bag                                       # (B, n)
    count = oob.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, (vals * oob).sum(axis=0) / np.maximum(count, 1), np.nan)
    return np.where(count > 0, mean, np.nan)


def scores_from_values(task, y, v):
    """Gütemaße aus Blattwerten `v` (Wahrscheinlichkeit bzw. Zahl) gegen `y`, wie in cart-demo. NaN-Zeilen (nie OOB) werden ausgeschlossen."""
    ok = ~np.isnan(v)
    y, v = np.asarray(y)[ok], v[ok]
    if task == "class":
        pred = (v > 0.5).astype(int)
        return {"error": 1.0 - T.accuracy(y, pred), "accuracy": T.accuracy(y, pred), "auc": T.auc(y, v), "logloss": T.log_loss(y, v), "n": int(ok.sum())}
    return {"error": T.rmse(y, v), "rmse": T.rmse(y, v), "mae": T.mae(y, v), "r2": T.r2(y, v), "n": int(ok.sum())}


# --- Wald-Kennzahlen (Korrelation, Wurzeln) -------------------------------------------------------------------------------------------------------

def root_shares(bagging):
    """Häufigkeit jedes Wurzelmerkmals über die Bäume, absteigend: [(Merkmal, Anzahl), ...]."""
    feats = [int(t.feature[0]) for t in bagging.trees]
    vals, counts = np.unique(feats, return_counts=True)
    order = np.argsort(-counts)
    return [(int(vals[i]), int(counts[i])) for i in order]


def tree_correlation(bagging, X, max_pairs=None):
    """Mittlere paarweise Ähnlichkeit der Einzelbaum-Vorhersagen auf X: Korrelationskoeffizient (Regression) bzw. Anteil gleicher Vorhersagen (Klassifikation), über alle Paare der Bäume. None bei nur einem Baum (keine Paare)."""
    vals = _tree_values(bagging, X)                             # (B, m)
    B = len(vals)
    if B < 2:
        return None
    if bagging.task == "reg":
        c = np.corrcoef(vals)
        iu = np.triu_indices(B, k=1)
        return float(np.nanmean(c[iu]))
    preds = (vals > 0.5).astype(int)
    iu = np.triu_indices(B, k=1)
    agree = np.array([np.mean(preds[i] == preds[j]) for i, j in zip(*iu)])
    return float(agree.mean())


def importances_mean(bagging):
    """Mittlere Wichtigkeit je Merkmal über alle Bäume (jeder Baum einzeln normiert wie in cart-demo)."""
    imps = np.array([T.importances(t) for t in bagging.trees])
    return imps.mean(axis=0)
