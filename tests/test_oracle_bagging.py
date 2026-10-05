"""Unabhängiges Orakel für Bagging: (1) der CART-Kern gegen eine Brute-Force-Split-Suche je Knoten (jede Schwelle von Hand, Gini, Entropie, Varianz, Mindestblattgröße, Gleichstände in den Werten),
(2) Baumwichtigkeiten gegen scikit-learn, (3) Bootstrap, Out-of-Bag-Vorhersage, Mittel, `upto`, Wurzel-Häufigkeiten und Baumkorrelation gegen Schleifen-Rechnungen,
(4) die Bias-Varianz-Zerlegung gegen die Identität bias² + Varianz = mittlerer quadratischer Fehler über die Trainingsseeds."""

import math
from collections import Counter

import numpy as np
import pytest

import bag_algorithm as bag
import bag_evaluation as ev
import bag_scenario as S
import bag_tree as T


def _impurity(y, criterion):
    p = y.mean()
    if criterion == "variance":
        return float(np.mean((y - p) ** 2))
    if criterion == "gini":
        return 2 * p * (1 - p)
    return -sum(q * math.log2(q) for q in (p, 1 - p) if q > 0)


def _best_gain(X, y, criterion, min_leaf):
    best = None
    for f in range(X.shape[1]):
        vals = sorted(set(X[:, f]))
        for a, b in zip(vals[:-1], vals[1:]):
            left = X[:, f] <= (a + b) / 2
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            gain = _impurity(y, criterion) - (left.sum() * _impurity(y[left], criterion) + (~left).sum() * _impurity(y[~left], criterion)) / len(y)
            best = gain if best is None else max(best, gain)
    return best


def _check_node(X, y, tree, t, criterion, min_leaf, counter):
    counter[0] += 1
    assert tree.n[t] == len(y) and tree.value[t] == pytest.approx(y.mean(), abs=1e-12) and tree.impurity[t] == pytest.approx(_impurity(y, criterion), abs=1e-10)
    best = _best_gain(X, y, criterion, min_leaf) if _impurity(y, criterion) > 1e-12 and len(y) >= 2 * min_leaf else None
    assert (best is None) == (tree.feature[t] < 0)                                    # geteilt genau dann, wenn ein zulässiger Split existiert
    if best is None:
        return
    left = X[:, tree.feature[t]] <= tree.threshold[t]
    assert left.sum() >= min_leaf and (~left).sum() >= min_leaf
    gain = _impurity(y, criterion) - (left.sum() * _impurity(y[left], criterion) + (~left).sum() * _impurity(y[~left], criterion)) / len(y)
    assert gain == pytest.approx(best, abs=1e-9)                                       # der gewählte Split ist ein optimaler (bei Gleichstand nicht eindeutig)
    _check_node(X[left], y[left], tree, tree.left[t], criterion, min_leaf, counter)
    _check_node(X[~left], y[~left], tree, tree.right[t], criterion, min_leaf, counter)


def test_every_split_of_the_tree_is_a_brute_force_optimum_also_with_ties_in_the_values():
    rng = np.random.default_rng(777)
    counter = [0]
    for it in range(90):
        task = "reg" if it % 3 == 2 else "class"
        criterion = "variance" if task == "reg" else ("gini", "entropy")[it % 2]
        n, d, min_leaf = int(rng.integers(8, 45)), int(rng.integers(1, 4)), int(rng.integers(1, 5))
        X = rng.normal(size=(n, d))
        if it % 4 == 3:
            X = np.round(X * 2) / 2                                                    # viele gleiche Werte
        y = ((X[:, 0] + rng.normal(size=n)) > 0).astype(float) if task == "class" else 2 * X[:, 0] + rng.normal(size=n)
        _check_node(X, y, T.grow(X, y, task, criterion, None, min_leaf), 0, criterion, min_leaf, counter)
    assert counter[0] > 1000


def _same_structure(tree, ref, t, r):
    """Gleiche Knotenstruktur wie scikit-learn (Merkmal, Schwelle auf float32-Genauigkeit), rekursiv von der Wurzel."""
    if (tree.feature[t] < 0) != (ref.feature[r] < 0):
        return False
    if tree.feature[t] < 0:
        return True
    return (tree.feature[t] == ref.feature[r] and abs(tree.threshold[t] - ref.threshold[r]) < 1e-5
            and _same_structure(tree, ref, tree.left[t], ref.children_left[r]) and _same_structure(tree, ref, tree.right[t], ref.children_right[r]))


def test_tree_importances_equal_scikit_learn_when_the_trees_agree():
    tree_mod = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(3)
    compared = 0
    for it in range(40):
        n, d = int(rng.integers(20, 60)), int(rng.integers(2, 5))
        X = rng.normal(size=(n, d))
        y = ((X[:, 0] + rng.normal(size=n)) > 0).astype(float)
        tree = T.grow(X, y, "class", "gini", None, 2)
        ref = tree_mod.DecisionTreeClassifier(min_samples_leaf=2, random_state=0).fit(X, y)
        Xt = rng.normal(size=(80, d))
        if _same_structure(tree, ref.tree_, 0, 0):                                                          # gleiche Splits an gleicher Stelle (bei gleicher Zeilenaufteilung zweier Merkmale entscheiden die Bibliotheken verschieden)
            assert np.allclose(T.importances(tree), ref.feature_importances_, atol=1e-9)
            compared += 1
    assert compared >= 15


def test_bagging_bootstrap_oob_mean_correlation_and_root_shares_equal_loop_computations():
    rng = np.random.default_rng(11)
    for it in range(40):
        task = ("class", "reg")[it % 2]
        n, d, B, min_leaf = int(rng.integers(15, 50)), int(rng.integers(1, 4)), int(rng.integers(1, 9)), int(rng.integers(1, 5))
        X = rng.normal(size=(n, d))
        y = ((X[:, 0] + rng.normal(size=n)) > 0).astype(float) if task == "class" else 2 * X[:, 0] + rng.normal(size=n)
        seed = int(rng.integers(0, 5))
        ens = bag.fit(X, y, task, None, min_leaf, B, seed)
        for b in range(B):
            idx = np.random.default_rng(seed * 1_000_003 + b).integers(0, n, n)
            mask = np.zeros(n, bool)
            mask[idx] = True
            assert np.array_equal(ens.in_bag[b], mask)
            fresh = T.grow(X[idx], y[idx], task, None, None, min_leaf)
            assert np.array_equal(fresh.feature, ens.trees[b].feature) and np.allclose(fresh.value, ens.trees[b].value)
        Xt = rng.normal(size=(25, d))
        vals = np.array([t.value[T.apply(t, Xt)] for t in ens.trees])
        assert np.allclose(bag.predict_value(ens, Xt), vals.mean(axis=0))
        k = int(rng.integers(1, B + 1))
        assert np.allclose(bag.predict_value(ens, Xt, upto=k), vals[:k].mean(axis=0))
        oob = bag.oob_predict(ens, X)
        for i in range(n):
            seen = [t.value[T.apply(t, X[i:i + 1])[0]] for b, t in enumerate(ens.trees) if not ens.in_bag[b, i]]
            assert (np.isnan(oob[i]) and not seen) or oob[i] == pytest.approx(np.mean(seen), abs=1e-12)
        corr = bag.tree_correlation(ens, Xt)
        if B < 2:
            assert corr is None
        else:
            pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
            if task == "class":
                assert corr == pytest.approx(np.mean([np.mean((vals[i] > 0.5) == (vals[j] > 0.5)) for i, j in pairs]), abs=1e-12)
            else:
                cs = [np.corrcoef(vals[i], vals[j])[0, 1] for i, j in pairs]
                assert np.isnan(corr) if np.all(np.isnan(cs)) else corr == pytest.approx(np.nanmean(cs), abs=1e-9)
        roots = bag.root_shares(ens)
        assert dict(roots) == dict(Counter(int(t.feature[0]) for t in ens.trees)) and [c for _, c in roots] == sorted((c for _, c in roots), reverse=True)


@pytest.mark.parametrize("task, criterion", [("class", "gini"), ("reg", "variance")])
def test_bias_variance_decomposition_adds_up_to_the_mean_squared_error_over_the_training_seeds(task, criterion):
    result = ev.bias_variance_rows(task, criterion, 5, 6, 200, 3)
    ds0 = S.generate_dataset(200, 3, 0, 7)
    X_eval, y_eval = ds0.X[ds0.test], (ds0.y_true if task == "class" else ds0.y_reg)[ds0.test]
    single, bagged = [], []
    for seed in ev.SIX:
        Xtr, ytr, _, _ = S.split(S.generate_dataset(200, 3, 0, seed), task)
        single.append(T.predict_value(T.grow(Xtr, ytr, task, criterion, None, 5), X_eval))
        bagged.append(bag.predict_value(bag.fit(Xtr, ytr, task, criterion, 5, 6, seed=0), X_eval))
    for name, preds in (("single", single), ("bagging", bagged)):
        P = np.array(preds)
        got = result[name]
        assert got["total"] == pytest.approx(np.mean((P - y_eval) ** 2), rel=1e-9)
        assert got["bias2"] == pytest.approx(np.mean((P.mean(axis=0) - y_eval) ** 2), rel=1e-9)
        assert got["variance"] == pytest.approx(np.mean(P.var(axis=0)), rel=1e-9)


@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.parametrize("task", ["class", "reg"])
def test_forest_mean_and_oob_equal_scikit_learn_bagging_on_the_same_bootstrap_samples(task):
    ens_mod = pytest.importorskip("sklearn.ensemble")
    tree_mod = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(31)
    for it in range(30):
        n, B = int(rng.integers(20, 120)), int(rng.integers(3, 12))
        X = rng.normal(size=(n, 1))                                                    # ein Merkmal: keine Gleichstände zwischen Merkmalen, die Bäume stimmen überein
        y = ((X[:, 0] + rng.normal(size=n)) > 0).astype(float) if task == "class" else 2 * X[:, 0] + rng.normal(size=n)
        if task == "class":
            sk = ens_mod.BaggingClassifier(tree_mod.DecisionTreeClassifier(), n_estimators=B, oob_score=True, random_state=it).fit(X, y)
        else:
            sk = ens_mod.BaggingRegressor(tree_mod.DecisionTreeRegressor(), n_estimators=B, oob_score=True, random_state=it).fit(X, y)
        trees, in_bag = [], np.zeros((B, n), bool)
        for b in range(B):
            idx = np.asarray(sk.estimators_samples_[b])                                # Stichprobe mit Wiederholungen, wie sklearn sie gezogen hat
            trees.append(T.grow(X[idx], y[idx], task, "gini" if task == "class" else "variance", None, 1))
            in_bag[b, idx] = True
        ens = bag.Bagging(tuple(trees), in_bag, task, n)
        Xt = rng.normal(size=(40, 1))
        theirs = sk.predict_proba(Xt)[:, 1] if task == "class" else sk.predict(Xt)
        assert np.allclose(bag.predict_value(ens, Xt), theirs, atol=1e-9)
        oob_theirs = sk.oob_decision_function_[:, 1] if task == "class" else sk.oob_prediction_
        oob = bag.oob_predict(ens, X)
        seen = (~in_bag).any(axis=0)                                                   # sklearn setzt nie-OOB-Zeilen auf 0/NaN, die Demo auf NaN
        assert np.array_equal(~np.isnan(oob), seen)
        assert np.allclose(oob[seen], oob_theirs[seen], atol=1e-9)
