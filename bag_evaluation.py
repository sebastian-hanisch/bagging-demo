"""Messungen an Bagging: Testfehler und OOB-Fehler gegen die Zahl der Bäume, Bias-Varianz-Zerlegung gegen einen Einzelbaum, Korrelation der Bäume mit und ohne dominantes Merkmal."""

import numpy as np
from dataclasses import dataclass

import bag_algorithm as bag
import bag_constants as C
import bag_scenario as S
import bag_tree as T

SIX = (C.DEFAULT_SEED,) + C.SWEEP_SEEDS


def error_of(bagging, X, y, upto=None):
    return bag.scores_from_values(bagging.task, y, bag.predict_value(bagging, X, upto))["error"]


def baseline_error(ds, task):
    """Fehler ohne Wald: immer die häufigere Trainingsklasse bzw. der Trainings-Mittelwert."""
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


@dataclass
class Analysis:
    ds: object
    task: str
    criterion: str
    leaf: int
    n_trees: int
    bagging: object
    single: object                # der erste Baum des Walds allein (Bootstrap-Stichprobe 0) - Vergleichsmaßstab
    train: dict
    test: dict
    single_test: dict
    oob: dict
    baseline: float
    verdict: str
    imp: np.ndarray
    root_shares: list
    correlation: float


def analyse(task, criterion, leaf, n_trees, n, n_noise, label_noise, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    criterion = criterion if criterion in C.CRITERIA[task] else C.DEFAULT_CRITERION[task]
    Xtr, ytr, Xte, yte = S.split(ds, task)
    bagging = bag.fit(Xtr, ytr, task, criterion, leaf, n_trees, seed=0)
    single = bagging.trees[0]
    train = bag.scores_from_values(task, ytr, bag.predict_value(bagging, Xtr))
    test = bag.scores_from_values(task, yte, bag.predict_value(bagging, Xte))
    single_test = bag.scores_from_values(task, yte, T.predict_value(single, Xte))
    oob = bag.scores_from_values(task, ytr, bag.oob_predict(bagging, Xtr))
    baseline = baseline_error(ds, task)
    imp = bag.importances_mean(bagging)
    root_shares = bag.root_shares(bagging)
    correlation = bag.tree_correlation(bagging, Xte)
    a = Analysis(ds, task, criterion, leaf, n_trees, bagging, single, train, test, single_test, oob, baseline, "", imp, root_shares, correlation)
    a.verdict = verdict(a)
    return a


def verdict(a):
    """'stump' (ein einziger Baum, kein Ensemble), 'better' (Wald deutlich besser als sein erster Einzelbaum), 'similar' (kaum Unterschied, z. B. sehr wenige Bäume), sonst 'worse' (sollte praktisch nie vorkommen)."""
    if a.n_trees <= 1:
        return "stump"
    gap = a.single_test["error"] - a.test["error"]
    rel = gap / max(a.single_test["error"], 1e-9)
    if rel > 0.03:
        return "better"
    return "worse" if rel < -0.01 else "similar"


# --- Sättigung: Testfehler und OOB-Fehler gegen die Zahl der Bäume ------------------------------------------------------------------------------------

def saturation_rows(a, ks=None):
    ks = ks or sorted(set(np.unique(np.round(np.geomspace(1, a.n_trees, min(24, a.n_trees))).astype(int))))
    Xtr, ytr, Xte, yte = S.split(a.ds, a.task)
    rows = []
    for k in ks:
        sub = bag.Bagging(a.bagging.trees[:k], a.bagging.in_bag[:k], a.task, a.bagging.n_train)
        rows.append({"k": int(k), "test": error_of(sub, Xte, yte), "oob": bag.scores_from_values(a.task, ytr, bag.oob_predict(sub, Xtr))["error"], "train": error_of(sub, Xtr, ytr)})
    return rows


# --- Bias-Varianz-Zerlegung gegen einen Einzelbaum -----------------------------------------------------------------------------------------------------

def bias_variance_rows(task, criterion, leaf, n_trees, n, n_noise, seeds=SIX):
    """Fixe Testpunkte (die 30-%-Testlieferungen von Seed 7); je Trainingsseed ein Einzelbaum und ein Wald, beide auf denselben Testpunkten ausgewertet.
    bias² = (Mittel der Vorhersagen über die Trainingsseeds - Wahrheit)², Varianz = Streuung der Vorhersagen über die Trainingsseeds, je Testpunkt und dann gemittelt."""
    ds0 = S.generate_dataset(n, n_noise, 0, C.DEFAULT_SEED)
    Xeval = ds0.X[ds0.test]
    yeval = (ds0.y_true if task == "class" else ds0.y_reg)[ds0.test]
    single_preds, bag_preds = [], []
    for sd in seeds:
        ds = S.generate_dataset(n, n_noise, 0, sd)
        Xtr, ytr, _, _ = S.split(ds, task)
        single_preds.append(T.predict_value(T.grow(Xtr, ytr, task, criterion, None, leaf), Xeval))
        bag_preds.append(bag.predict_value(bag.fit(Xtr, ytr, task, criterion, leaf, n_trees, seed=0), Xeval))

    def decompose(preds):
        preds = np.array(preds)
        mean = preds.mean(axis=0)
        bias2 = float(np.mean((mean - yeval) ** 2))
        var = float(np.mean(preds.var(axis=0)))
        return {"bias2": bias2, "variance": var, "total": bias2 + var}

    return {"single": decompose(single_preds), "bagging": decompose(bag_preds)}


# --- Korrelation der Bäume mit und ohne dominantes Merkmal ---------------------------------------------------------------------------------------------

DOMINANCE_STEPS = (100, 80, 60, 40, 20, 0)


def dominance_rows(task, criterion, leaf, n_trees, n, n_noise, seed, strengths=DOMINANCE_STEPS):
    ds = S.generate_dataset(n, n_noise, 0, seed)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    rows = []
    for s in strengths:
        Xw = S.weaken_dominant(Xtr, task, s, seed)
        Xtw = S.weaken_dominant(Xte, task, s, seed)
        bagging = bag.fit(Xw, ytr, task, criterion, leaf, n_trees, seed=0)
        shares = bag.root_shares(bagging)
        rows.append({"strength": s, "correlation": bag.tree_correlation(bagging, Xtw), "test": error_of(bagging, Xtw, yte), "root_share": shares[0][1] / n_trees, "n_roots": len(shares), "root_feature": shares[0][0]})
    return rows
