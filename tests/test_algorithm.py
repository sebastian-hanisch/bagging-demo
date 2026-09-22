"""Bagging gegen unabhängige Referenzen: mit denselben Bootstrap-Indizes liefert scikit-learn (DecisionTree*, nicht BaggingClassifier - das lässt keine eigenen Indizes zu) dieselben Einzelbäume und damit dasselbe Mittel;
OOB-Vorhersage wird zusätzlich mit einer naiven Python-Schleife nachgerechnet. Der Baumkern selbst ist in cart-demo geprüft (siehe dortige Tests) - hier geht es um Bootstrap, Mittelung und OOB."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import bag_algorithm as bag
import bag_scenario as S
import bag_tree as T


def _continuous(n=400, d=5, seed=0, task="class", noise=1.6):
    """Wie in cart-demo: stetige Merkmale, stark verrauschtes Ziel - keine Gleichstände zwischen Splits."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = X[:, 0] + 0.7 * np.sin(2 * X[:, 1]) + 0.5 * (X[:, 2] > 0.3) * X[:, 3]
    y = (signal + rng.normal(0, noise, n) > 0).astype(float) if task == "class" else signal * 3 + 10 + rng.normal(0, 1.0, n)
    return X, y


def _first_divergence(ours, X, y, ref):
    """Läuft beide Bäume parallel (unsere Breiten-, sklearns Tiefenreihenfolge - deshalb ein eigener Lauf statt Feldvergleich) und gibt (Zeilen an der Stelle, unser Merkmal, sklearns Merkmal) der ersten Abweichung zurück, oder None."""
    t = ref.tree_
    stack = [(0, 0, np.arange(len(y)))]
    while stack:
        a, b, ix = stack.pop()
        fa, fb = int(ours.feature[a]), int(t.feature[b])
        if (fa < 0) != (t.children_left[b] < 0) or (fa >= 0 and (fa != fb or abs(ours.threshold[a] - t.threshold[b]) > 1e-6)):
            return ix, fa, fb
        if fa >= 0:
            m = X[ix, fa] <= ours.threshold[a]
            stack.append((int(ours.left[a]), int(t.children_left[b]), ix[m]))
            stack.append((int(ours.right[a]), int(t.children_right[b]), ix[~m]))
    return None


def _naive_oob(bagging, X, y):
    """OOB-Vorhersage per Python-Schleife: für jede Zeile das Mittel der Bäume, die sie nicht gesehen haben."""
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(n):
        vals = [T.predict_value(t, X[i:i + 1])[0] for b, t in enumerate(bagging.trees) if not bagging.in_bag[b, i]]
        if vals:
            out[i] = float(np.mean(vals))
    return out


@pytest.mark.parametrize("task,crit,cls", [("class", "gini", DecisionTreeClassifier), ("reg", "variance", DecisionTreeRegressor)])
def test_most_individual_trees_match_scikit_learn_exactly_on_the_same_bootstrap_rows(task, crit, cls):
    """Eine Bootstrap-Stichprobe zieht mit Zurücklegen: bei n=300 sind im Mittel nur ~63 % der Zeilen einzigartig (siehe unten), der Rest sind exakte Duplikate. Das erhöht die Gleichstandsrate gegenüber frischen,
    stetigen Daten (cart-demo) deutlich - tief im Baum bleiben oft nur noch wenige, teils doppelte Zeilen übrig, und zwei Merkmale können exakt denselben Gain erzielen (unten nachgerechnet). Trotzdem stimmt bei
    min_samples_leaf 10 die deutliche Mehrheit der Bäume exakt überein; wo nicht, ist der Baum genauso groß und der Gain an der abweichenden Stelle exakt gleich (echter Gleichstand, kein Fehler)."""
    X, y = _continuous(300, 5, 1, task)
    n_trees, leaf = 20, 10
    bagging = bag.fit(X, y, task, crit, leaf, n_trees, seed=3)
    sk_kw = {"criterion": crit} if task == "class" else {"criterion": "squared_error"}
    exact = 0
    for b in range(n_trees):
        idx = bag.bootstrap_indices(len(y), 3 * 1_000_003 + b)
        ref = cls(**sk_kw, min_samples_leaf=leaf, random_state=0).fit(X[idx], y[idx])
        ours = bagging.trees[b]
        assert ours.n_leaves == ref.get_n_leaves(), b                                              # gleich groß, auch wenn nicht exakt gleich aufgebaut
        exact += _first_divergence(ours, X[idx], y[idx], ref) is None
    assert exact >= 0.5 * n_trees


def test_a_tie_between_individual_trees_and_scikit_learn_is_a_genuine_equal_gain():
    """Nachweis für den Befund oben: an der ersten Abweichungsstelle erzielen unser Merkmal und das von scikit-learn exakt denselben Gain (kein Rundungsfehler, kein Bug)."""
    X, y = _continuous(300, 5, 1, "class")
    leaf = 10
    idx = bag.bootstrap_indices(len(y), 3 * 1_000_003 + 1)                                          # bekanntlich eine abweichende Ziehung
    Xb, yb = X[idx], y[idx]
    ours = T.grow(Xb, yb, "class", None, None, leaf)
    ref = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(Xb, yb)
    div = _first_divergence(ours, Xb, yb, ref)
    assert div is not None
    ix, fa, fb = div
    g, _, _ = T.gain_matrix(Xb[ix], yb[ix], "gini", leaf)
    assert g[:, fa].max() == pytest.approx(g[:, fb].max())


def test_the_ensemble_mean_is_close_to_scikit_learns_ensemble_mean_despite_the_ties():
    X, y = _continuous(300, 5, 1, "class")
    Xt, _ = _continuous(200, 5, 2, "class")
    n_trees, leaf = 20, 10
    bagging = bag.fit(X, y, "class", None, leaf, n_trees, seed=3)
    sk_preds = []
    for b in range(n_trees):
        idx = bag.bootstrap_indices(len(y), 3 * 1_000_003 + b)
        ref = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(X[idx], y[idx])
        sk_preds.append(ref.predict_proba(Xt)[:, 1])
    diff = np.abs(bag.predict_value(bagging, Xt) - np.mean(sk_preds, axis=0))
    assert diff.mean() < 0.02 and diff.max() < 0.15


def test_bootstrap_samples_have_about_two_thirds_unique_rows():
    shares = [len(np.unique(bag.bootstrap_indices(300, s))) / 300 for s in range(30)]
    assert np.mean(shares) == pytest.approx(1 - 1 / np.e, abs=0.02)


def test_in_bag_marks_exactly_the_rows_drawn_by_the_bootstrap():
    X, y = _continuous(100, 4, 0, "class")
    bagging = bag.fit(X, y, "class", None, 1, 5, seed=1)
    for b in range(5):
        idx = bag.bootstrap_indices(100, 1 * 1_000_003 + b)
        assert np.array_equal(np.nonzero(bagging.in_bag[b])[0], np.unique(idx))
        assert bagging.trees[b].n_total == 100                                                     # der Baum sieht 100 Zeilen (mit Wiederholungen), auch wenn weniger EINDEUTIGE Zeilen


def test_oob_predictions_match_a_naive_python_loop():
    X, y = _continuous(150, 4, 5, "reg")
    bagging = bag.fit(X, y, "reg", None, 1, 20, seed=2)
    mine = bag.oob_predict(bagging, X)
    naive = _naive_oob(bagging, X, y)
    both = ~np.isnan(mine) & ~np.isnan(naive)
    assert both.sum() > 0.9 * len(y) and np.allclose(mine[both], naive[both])
    assert np.array_equal(np.isnan(mine), np.isnan(naive))


def test_oob_share_is_close_to_one_minus_one_over_e():
    """Im Mittel sieht ein Bootstrap-Baum 1 - 1/e ≈ 63.2 % der Zeilen; bei genug Bäumen ist fast jede Zeile mindestens einmal out-of-bag."""
    X, y = _continuous(500, 4, 0, "class")
    bagging = bag.fit(X, y, "class", None, 1, 60, seed=4)
    share_in_bag = bagging.in_bag.mean()
    assert share_in_bag == pytest.approx(1 - 1 / np.e, abs=0.01)
    never_oob = np.all(bagging.in_bag, axis=0).mean()
    assert never_oob < 0.02


def test_more_trees_use_more_of_the_training_rows_as_oob():
    X, y = _continuous(200, 4, 0, "class")
    shares = []
    for n_trees in (3, 10, 40):
        bagging = bag.fit(X, y, "class", None, 1, n_trees, seed=6)
        shares.append(np.mean(~np.isnan(bag.oob_predict(bagging, X))))
    assert shares[0] < shares[1] < shares[2] and shares[2] > 0.99


def test_predict_upto_k_trees_matches_a_fresh_ensemble_of_k_trees():
    """`upto` beim Mittel muss dieselben Bäume treffen wie ein Wald, der nur bis dahin gewachsen ist (gleiche Bootstrap-Seeds)."""
    X, y = _continuous(200, 4, 0, "reg")
    Xt, _ = _continuous(100, 4, 1, "reg")
    full = bag.fit(X, y, "reg", None, 1, 15, seed=9)
    for k in (1, 5, 10, 15):
        small = bag.fit(X, y, "reg", None, 1, k, seed=9)
        assert np.allclose(bag.predict_value(full, Xt, upto=k), bag.predict_value(small, Xt))


def test_root_shares_and_tree_correlation_are_consistent():
    X, y = _continuous(300, 4, 0, "class")
    bagging = bag.fit(X, y, "class", None, 1, 25, seed=0)
    shares = bag.root_shares(bagging)
    assert sum(c for _, c in shares) == 25 and shares == sorted(shares, key=lambda t: -t[1])
    corr = bag.tree_correlation(bagging, X)
    assert 0.0 <= corr <= 1.0                                                                       # Anteil gleicher Vorhersagen bei Klassifikation
    ident = bag.tree_correlation(bag.Bagging((bagging.trees[0],) * 5, np.tile(bagging.in_bag[:1], (5, 1)), "class", bagging.n_train), X)
    assert ident == pytest.approx(1.0)


def test_importances_mean_is_the_plain_average_and_sums_near_one():
    X, y = _continuous(300, 4, 0, "class")
    bagging = bag.fit(X, y, "class", None, 1, 10, seed=0)
    mine = bag.importances_mean(bagging)
    by_hand = np.mean([T.importances(t) for t in bagging.trees], axis=0)
    assert np.allclose(mine, by_hand) and mine.sum() == pytest.approx(1.0, abs=1e-9)


# --- Gütemaße --------------------------------------------------------------------------------------------------------------------------------------

def test_scores_from_values_ignores_nan_rows_and_matches_hand_computation():
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    v = np.array([0.2, 0.9, np.nan, 0.6, np.nan])
    s = bag.scores_from_values("class", y, v)
    assert s["n"] == 3 and s["accuracy"] == pytest.approx(T.accuracy(y[[0, 1, 3]], v[[0, 1, 3]] > 0.5))


def test_scores_on_the_delivery_data_are_never_worse_than_a_single_deep_tree_in_the_mean():
    """Nicht als Behauptung im Text, aber eine Invariante, die jeder Bagging-Aufruf erfüllen sollte: das Ensemble-Mittel ist auf den echten Lieferdaten nicht schlechter als ein zufällig herausgegriffener Einzelbaum (im Mittel über viele Ziehungen)."""
    ds = S.generate_dataset(600, 3, 0, 7)
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    bagging = bag.fit(Xtr, ytr, "class", None, 1, 30, seed=0)
    ens = bag.scores_from_values("class", yte, bag.predict_value(bagging, Xte))
    single_errs = [bag.scores_from_values("class", yte, T.predict_value(t, Xte))["error"] for t in bagging.trees]
    assert ens["error"] <= np.mean(single_errs) + 1e-9


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_a_single_tree_ensemble_equals_that_tree():
    X, y = _continuous(150, 4, 0, "reg")
    bagging = bag.fit(X, y, "reg", None, 1, 1, seed=0)
    assert np.allclose(bag.predict_value(bagging, X), T.predict_value(bagging.trees[0], X))


def test_different_seeds_give_different_bootstrap_draws():
    X, y = _continuous(80, 4, 0, "class")
    b1 = bag.fit(X, y, "class", None, 1, 5, seed=1)
    b2 = bag.fit(X, y, "class", None, 1, 5, seed=2)
    assert not np.array_equal(b1.in_bag, b2.in_bag)


def test_constant_target_gives_a_constant_ensemble():
    X = np.random.default_rng(0).normal(size=(50, 3))
    bagging = bag.fit(X, np.ones(50), "class", None, 1, 5, seed=0)
    assert np.allclose(bag.predict_value(bagging, X), 1.0)
