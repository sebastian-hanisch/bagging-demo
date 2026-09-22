"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-22, Toleranzen fangen Rundung ab). Alle Zahlen kommen aus deterministischen Bootstrap-Seeds (bag.fit(..., seed=0) intern) auf festen
Datensätzen - keine Zufallsstreuung zwischen Testläufen."""

import functools

import numpy as np
import pytest

import bag_algorithm as bag
import bag_constants as C
import bag_evaluation as ev
import bag_scenario as S

PRESET = {"one": "🌲 Ein Baum", "small": "🌳 Kleiner Wald", "big": "🌲🌳 Großer Wald", "dom": "🎯 Dominantes Merkmal", "reg": "📈 Regression"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["criterion"], p["leaf"], p["n_trees"], p["n"], p["n_noise"], p["label_noise"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


@functools.lru_cache(maxsize=None)
def _default(task):
    return ev.analyse(task, None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_one_tree_preset():
    a = _preset("one")
    assert a.verdict == "stump" and a.n_trees == 1
    assert (a.test["error"], a.single_test["error"]) == pytest.approx((0.2556, 0.2556), abs=0.0005)
    assert a.oob["error"] == pytest.approx(0.2236, abs=0.0005) and a.oob["n"] < len(S.split(a.ds, "class")[1])
    _help("one", "25.6 %", "22.4 %")


def test_small_forest_preset():
    a = _preset("small")
    assert a.n_trees == 10 and a.test["error"] == pytest.approx(0.1806, abs=0.0005) and a.verdict == "better"
    _help("small", "10 Bäume", "18.1 %", "25.6 %")


def test_big_forest_preset_shows_saturation():
    a = _preset("big")
    small = _default("class")                                                                       # 30 Bäume, derselbe Datensatz
    assert a.n_trees == 100 and a.test["error"] == pytest.approx(0.1611, abs=0.0005) and a.oob["error"] == pytest.approx(0.1500, abs=0.0005)
    assert a.test["error"] > small.test["error"]                                                     # 100 Bäume sind hier NICHT besser als 30 - genau der Punkt (Sättigung)
    _help("big", "100 Bäume", "16.1 %", "15.0 %", "30 Bäume", "15.0 %")


def test_dominant_feature_preset():
    a = _preset("dom")
    assert a.task == "reg" and a.n_trees == C.DEFAULT_N_TREES
    dr = ev.dominance_rows("reg", None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE, C.DEFAULT_SEED)
    at20 = next(r for r in dr if r["strength"] == 20)
    assert at20["correlation"] == pytest.approx(0.35, abs=0.005) and at20["root_share"] == pytest.approx(0.5333, abs=0.001)
    _help("dom", "0.80", "0.35", "100 % -> 53 %")


def test_regression_preset():
    a = _preset("reg")
    assert a.task == "reg" and a.n_trees == C.DEFAULT_N_TREES
    assert (a.test["error"], a.single_test["error"], a.baseline) == pytest.approx((10.2796, 17.0170, 27.1543), abs=0.005)
    _help("reg", "10.3 min", "17.0 min", "27.2 min")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["criterion"] in C.CRITERIA[p["task"]] and C.LEAF_MIN <= p["leaf"] <= C.LEAF_MAX and C.N_TREES_MIN <= p["n_trees"] <= C.N_TREES_MAX
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.DOMINANT_MIN <= p["dominant"] <= C.DOMINANT_MAX and 0 <= p["fx"] < C.N_BASE + p["n_noise"] and 0 <= p["fy"] < C.N_BASE + p["n_noise"] and name in C.PRESET_HELP


# --- Sidebar-Hilfen ---------------------------------------------------------------------------------------------------------------------------------

def test_n_trees_help_numbers():
    a1 = ev.analyse("class", None, 1, 1, C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)
    a10 = ev.analyse("class", None, 1, 10, C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)
    a100 = ev.analyse("class", None, 1, 100, C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)
    assert (a1.test["error"], a10.test["error"], a100.test["error"]) == pytest.approx((0.2556, 0.1806, 0.1611), abs=0.0005)


def test_dominant_slider_help_number():
    dr = ev.dominance_rows("reg", None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE, C.DEFAULT_SEED)
    at20 = next(r for r in dr if r["strength"] == 20)
    assert at20["correlation"] == pytest.approx(0.35, abs=0.005)


# --- Standardansicht (Klassifikation und Regression) ------------------------------------------------------------------------------------------------

def test_default_view_numbers():
    a = _default("class")
    assert (a.test["error"], a.single_test["error"], a.oob["error"], a.baseline) == pytest.approx((0.1500, 0.2556, 0.1798, 0.4639), abs=0.0005)
    assert a.oob["n"] == 840 and a.correlation == pytest.approx(0.7725, abs=0.0005) and a.root_shares == [(1, 30)]
    assert a.verdict == "better"
    ar = _default("reg")
    assert (ar.test["error"], ar.single_test["error"], ar.oob["error"], ar.baseline) == pytest.approx((10.2796, 17.0170, 11.0394, 27.1543), abs=0.005)
    assert ar.correlation == pytest.approx(0.8006, abs=0.0005) and ar.root_shares == [(0, 30)]


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------------

def test_bias_variance_numbers():
    bv = ev.bias_variance_rows("class", None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert bv["single"]["bias2"] == pytest.approx(0.1083, abs=0.0005) and bv["single"]["variance"] == pytest.approx(0.1037, abs=0.0005)
    assert bv["bagging"]["bias2"] == pytest.approx(0.1019, abs=0.0005) and bv["bagging"]["variance"] == pytest.approx(0.0161, abs=0.0005)
    var_cut = 1 - bv["bagging"]["variance"] / bv["single"]["variance"]
    bias_cut = 1 - bv["bagging"]["bias2"] / bv["single"]["bias2"]
    assert var_cut == pytest.approx(0.845, abs=0.003) and bias_cut == pytest.approx(0.059, abs=0.003)

    bvr = ev.bias_variance_rows("reg", None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert bvr["single"]["bias2"] == pytest.approx(101.97, abs=0.05) and bvr["single"]["variance"] == pytest.approx(112.00, abs=0.05)
    assert bvr["bagging"]["bias2"] == pytest.approx(91.50, abs=0.05) and bvr["bagging"]["variance"] == pytest.approx(19.00, abs=0.05)
    var_cut_r = 1 - bvr["bagging"]["variance"] / bvr["single"]["variance"]
    assert var_cut_r == pytest.approx(0.830, abs=0.003)


def test_dominance_numbers_class_and_reg():
    dc = ev.dominance_rows("class", None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE, C.DEFAULT_SEED)
    assert [r["correlation"] for r in dc] == pytest.approx([0.7725, 0.7180, 0.6697, 0.6421, 0.6492, 0.6526], abs=0.001)
    assert dc[0]["root_feature"] == 1 and dc[-1]["root_feature"] == 3 and dc[0]["n_roots"] == 1 and dc[-1]["n_roots"] == 1              # Ladegewicht -> Verkehr, in beiden Fällen einheitlich

    dr = ev.dominance_rows("reg", None, 1, C.DEFAULT_N_TREES, C.DEFAULT_N, C.DEFAULT_NOISE, C.DEFAULT_SEED)
    assert [r["correlation"] for r in dr] == pytest.approx([0.8006, 0.6276, 0.5116, 0.3597, 0.3477, 0.3286], abs=0.001)
    assert dr[0]["root_feature"] == 0 and dr[0]["n_roots"] == 1 and dr[-1]["n_roots"] > 1                                              # Distanz -> verteilt sich auf mehrere Merkmale
    assert dr[0]["correlation"] > dr[-1]["correlation"]                                                                                 # monoton fallend an den Enden


def test_saturation_rows_trend_toward_zero_training_error():
    a = _default("class")
    rows = ev.saturation_rows(a)
    assert rows[-1]["k"] == a.n_trees and rows[0]["k"] == 1
    assert rows[-1]["train"] < rows[0]["train"] and rows[-1]["train"] < 0.01                          # nicht streng monoton (Bootstrap-Zufall), aber der Trend ist eindeutig fallend
    assert rows[-1]["test"] == pytest.approx(a.test["error"])


# --- Erzeuger und Baumkern (geteilt mit cart-demo) ---------------------------------------------------------------------------------------------------

def test_readme_bootstrap_unique_share_number():
    shares = [len(np.unique(bag.bootstrap_indices(300, s))) / 300 for s in range(30)]
    assert np.mean(shares) == pytest.approx(0.629, abs=0.001)


def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150


def test_weaken_dominant_is_a_no_op_at_full_strength_and_changes_only_the_dominant_column():
    ds = S.generate_dataset(300, 2, 0, 7)
    X = ds.X
    assert np.array_equal(S.weaken_dominant(X, "class", 100, 7), X)
    Xw = S.weaken_dominant(X, "class", 0, 7)
    other_cols = [c for c in range(X.shape[1]) if c != S.DOMINANT_FEATURE["class"]]
    assert np.array_equal(Xw[:, other_cols], X[:, other_cols]) and not np.array_equal(Xw[:, S.DOMINANT_FEATURE["class"]], X[:, S.DOMINANT_FEATURE["class"]])
    assert sorted(Xw[:, S.DOMINANT_FEATURE["class"]]) == pytest.approx(sorted(X[:, S.DOMINANT_FEATURE["class"]]))     # dieselbe Verteilung, nur permutiert
