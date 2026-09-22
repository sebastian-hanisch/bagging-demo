"""Bagging - viele Bäume auf Bootstrap-Stichproben - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Bagging - und lässt stattdessen das Beispiel wachsen.
Zweites Stück der Baumbasierten Linie der "Konzepte"-Reihe: der Nachfolger von CART (cart-demo). Derselbe Baumkern (bag_tree.py, wortgleich aus cart-demo) wächst hier B-mal auf
Bootstrap-Stichproben; gemittelt sinkt die Varianz stark, ohne den Bias groß zu ändern - außer, ein einzelnes Merkmal dominiert jeden Baum.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import bag_algorithm as bag
import bag_constants as C
import bag_evaluation as ev
from bag_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bag_visualization import (
    build_bias_variance,
    build_dominance,
    build_importance,
    build_map,
    build_saturation,
    build_tree,
    feature_label,
    value_text,
)

st.set_page_config(page_title="Bagging – Sebastian Hanisch", layout="wide")

VERDICT_TEXT = {
    "stump": "ℹ️ Nur ein Baum: kein Mittel, keine Varianzsenkung - das ist der Vergleichsmaßstab.",
    "better": "✅ **Deutlich besser als der Einzelbaum**",
    "similar": "ℹ️ **Kaum besser als der Einzelbaum** - bei so wenigen Bäumen ist das normal.",
    "worse": "⚠️ **Schlechter als der Einzelbaum** (sollte selten vorkommen - kleine Stichprobe, Zufall)",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=8)
def _saturation(task, criterion, leaf, n_trees, n, n_noise, label_noise, seed):
    a = _analysis(task, criterion, leaf, n_trees, n, n_noise, label_noise, seed)
    return ev.saturation_rows(a)


@st.cache_data(show_spinner=False, max_entries=6)
def _bias_variance(task, criterion, leaf, n_trees, n, n_noise):
    return ev.bias_variance_rows(task, criterion, leaf, n_trees, n, n_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _dominance(task, criterion, leaf, n_trees, n, n_noise, seed):
    return ev.dominance_rows(task, criterion, leaf, n_trees, n, n_noise, seed)


st.title("🌳🌳 Bagging – viele Bäume auf Bootstrap-Stichproben")
st.markdown(
    """
Ein einzelner Entscheidungsbaum (cart-demo) hat zwei Schwächen: er lernt das Rauschen mit, und eine kleine Änderung der Trainingsdaten kann einen ganz anderen Baum ergeben. **Bagging** (Bootstrap **Agg**regat**ing**, Breiman 1996) macht aus der zweiten Schwäche eine Stärke:
es zieht **B Bootstrap-Stichproben** (mit Zurücklegen, dieselbe Größe wie das Original) aus den Trainingsdaten, wächst auf jeder einen **vollen** CART-Baum (derselbe Kern wie in cart-demo) und **mittelt** die Vorhersagen.
Da die Bäume unterschiedliche, aber verzerrungsfreie Stichproben derselben Daten sehen, hebt sich ihr Rauschen beim Mitteln zu einem großen Teil weg - die **Varianz** sinkt stark, der **Bias** bleibt fast gleich (gemessen unten).
Als Nebenprodukt gibt es die **Out-of-Bag-Schätzung**: jeder Baum sieht im Mittel nur 63,2 % der Zeilen, die übrigen liefern eine eingebaute Testschätzung ohne eigene Testdaten zu verbrauchen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweites Stück der Baumbasierten Linie der \"Konzepte\"-Reihe, Nachfolger von CART (cart-demo) - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Verfahren geht auf Breiman (1996) zurück; alle Lieferungen, Merkmale und Zahlen dieser Demo sind erzeugt und gemessen - keine echten Daten. Derselbe Baumkern wie in cart-demo (`bag_tree.py`, wortgleich übernommen), scikit-learn kommt nur in den Tests als Gegenprobe vor."
)
st.caption(
    "**Bezug zu OR:** ein gemitteltes Ensemble liefert nicht nur eine Vorhersage, sondern über die Streuung der Einzelbäume auch eine **Unsicherheit** - ein Eingang für robuste oder stochastische Planung (z. B. Pufferzeiten in der Tourenplanung), nicht nur ein Punktwert."
)

with st.expander("So funktioniert Bagging", expanded=True):
    st.markdown(
        """
1. **Bootstrap-Stichprobe:** aus n Trainingszeilen werden n Zeilen **mit Zurücklegen** gezogen - manche Zeilen mehrfach, andere gar nicht. Im Mittel bleiben **1 − 1/e ≈ 63,2 %** der Zeilen mindestens einmal drin; der Rest ("out-of-bag", OOB) fehlt diesem Baum ganz.
2. **B volle Bäume:** auf jeder Stichprobe wächst ein CART-Baum bis zur gewählten Mindestblattgröße (klassisch bis zum Ende, Blatt = 1) - **ohne** Beschneiden, denn kein Einzelbaum muss für sich gut sein.
3. **Mitteln:** die Vorhersage ist der Mittelwert der Blattwerte aller B Bäume - bei Klassifikation eine gemittelte Wahrscheinlichkeit, bei Regression der Mittelwert der Zahlen.
4. **Out-of-Bag:** für jede Trainingszeile gibt es Bäume, die sie nie gesehen haben (im Mittel 63,2 % der Bäume). Ihr Mittel ist eine eingebaute Testschätzung, ganz ohne eigene Testdaten.
5. **Voraussetzung:** der Gewinn kommt aus der **Streuung zwischen** den Bäumen. Ein Merkmal, das jeden Baum genauso teilt (weil es viel stärker ist als alle anderen), lässt die Bäume ähnlich bleiben - dann bringt Bagging wenig (Experiment unten, Hook für Random Forest).
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
if st.session_state["criterion_select"] not in ("gini", "entropy"):
    st.session_state["criterion_select"] = "gini"

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k],
                        help="Klassifikation: gemittelte Wahrscheinlichkeit, dass die Lieferung zu spät kommt. Regression: gemittelte Dauer in Minuten. Derselbe Baumkern wie in cart-demo, nur das Ziel wechselt.")
    if task == "class":
        crit = st.selectbox("Schnittkriterium", C.CRITERIA["class"], key="criterion_select", format_func=lambda k: C.CRITERION_LABELS[k], help="Wie in cart-demo: Gini und Entropie liegen fast immer beieinander.")
        st.session_state[KEPT["criterion_select"]] = crit
    else:
        crit = "variance"
        st.caption("Schnittkriterium: Varianz - bei einem Zahlenziel gibt es keine Wahl.")
    leaf = st.slider("Mindestgröße eines Blatts", *bounds("leaf_slider"), key="leaf_slider",
                     help="Wie in cart-demo, gilt für jeden Baum des Walds. 1 = Bäume wachsen voll (klassisches Bagging) - kein Beschneiden, weil kein Einzelbaum für sich gut sein muss.")
    n_trees = st.slider("Zahl der Bäume", *bounds("n_trees_slider"), key="n_trees_slider",
                        help="Testfehler und Out-of-Bag-Fehler sinken mit mehr Bäumen und sättigen dann: im Standarddatensatz bringt der Sprung von 1 auf 10 Bäume viel (25,6 % → 18,1 % Testfehler), 30 auf 100 kaum noch etwas.")
    st.markdown("**Experiment: Stärke des dominanten Merkmals**")
    dominant = st.slider("Stärke des dominanten Merkmals [%]", *bounds("dominant_slider"), key="dominant_slider",
                         help="100 % = unveränderte Daten. Darunter wird das Merkmal, das in jedem Baum die Wurzel bildet (Ladegewicht bei Klassifikation, Distanz bei Regression), in einem Teil der Zeilen durch eine "
                              "zufällige Permutation seiner selbst ersetzt - derselbe Wertebereich, aber ohne Bezug zum Ziel. Zeigt, was mit der Korrelation der Bäume passiert, wenn nichts mehr dominiert (Regression: Korrelation 0,80 → 0,35 bei 20 %). Kein realistischer Datenregler, nur zum Beobachten.")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100, help="Zahl der erzeugten Lieferungen; 70 % zum Lernen, 30 % zum Testen.")
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", help="Zusätzliche Merkmale ohne Bezug zum Ziel (wie in cart-demo).")
    if task == "class":
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider", help="Anteil vertauschter Trainingsetiketten; der Test bleibt sauber (wie in cart-demo).")
        st.session_state[KEPT["label_noise_slider"]] = label_noise
    else:
        label_noise = int(st.session_state.get(KEPT["label_noise_slider"], C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Erzeugt einen anderen Datensatz mit denselben Regeln.")
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed.")

base_params = (task, crit, int(leaf), int(n_trees))
data_params = (int(n), int(n_noise), int(label_noise), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
Xtr, ytr, Xte, yte = ds.X[ds.train], ds.y(task)[ds.train], ds.X[ds.test], (ds.y_true if task == "class" else ds.y_reg)[ds.test]
names = ds.names
n_feat = len(names)
n_test = len(ds.test)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f), help="Die Karte zeigt das Wald-Mittel über zwei Merkmale; alle anderen Merkmale stehen dabei auf ihrem Median im Training.")
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider", help="Eine Lieferung aus dem Test: ihre Vorhersage steht unten, in der Karte als Stern.")
sync_query_params({"task_select": task, "criterion_select": crit if task == "class" else st.session_state.get(KEPT["criterion_select"], "gini"), "leaf_slider": int(leaf), "n_trees_slider": int(n_trees),
                   "dominant_slider": int(dominant), "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise), "seed_input": int(seed), "map_x_select": int(fx), "map_y_select": int(fy)})

view_key = (base_params, data_params)
if st.session_state.get("bag_owner") != view_key:
    st.session_state["bag_owner"] = view_key
    st.session_state["bag_step"] = int(n_trees)

# --- Bagging in Aktion ------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Bagging in Aktion")
st.caption("Der Wald wächst Baum für Baum; links der zuletzt hinzugekommene Einzelbaum (grob, nur zur Einordnung), rechts das Mittel aller Bäume bis dahin über zwei Merkmale.")
if n_trees > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Bäume im Mittel", 1, int(n_trees), key="bag_step", help="Wie viele Bäume schon im Mittel stecken - 1 = der erste Bootstrap-Baum allein, ganz rechts alle Bäume.")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur ein Baum eingestellt - kein Mittel zu bilden. Mehr Bäume in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()
sample_x = Xte[sample_idx]
sample_y = yte[sample_idx]
sat_rows_full = _saturation(*base_params, *data_params)


def _render(current):
    sub = bag.Bagging(a.bagging.trees[:current], a.bagging.in_bag[:current], task, a.bagging.n_train)
    latest = a.bagging.trees[current - 1]
    with view_slot.container():
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(build_tree(latest, task, (float(ytr.min()), float(ytr.max()))), width="stretch", key=f"tree_chart_{current}")
            st.caption(f"Baum {current}: Wurzel = **{names[latest.feature[0]]}**, {latest.n_leaves} Blätter.")
        with c2:
            st.plotly_chart(build_map(sub, ds, fx, fy, upto=None, sample=sample_x), width="stretch", key=f"map_chart_{current}")
        pred_here = bag.predict_value(sub, sample_x.reshape(1, -1))[0]
        truth = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.0f} min"
        st.markdown(f"**Testlieferung {sample_idx}:** Mittel der ersten {current} Bäume = **{value_text(task, pred_here)}**; tatsächlich: **{truth}**.")
        st.plotly_chart(build_saturation(sat_rows_full, task, a.single_test["error"], current), width="stretch", key=f"sat_chart_{current}")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, int(n_trees), min(12, int(n_trees)))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = int(n_trees)
else:
    _render(step)

st.markdown("---")

# --- Was der Wald gelernt hat -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was der Wald gelernt hat – und wie gut er auf neuen Lieferungen ist")
st.caption("**Training** = die Lieferungen, aus denen der Wald gebaut wurde; **Test** = die zurückgehaltenen 30 %; **Out-of-Bag (OOB)** = für jede Trainingszeile das Mittel nur der Bäume, die sie nicht gesehen haben - eine eingebaute Testschätzung.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bäume", int(n_trees), delta=f"1 Baum: {_err(task, a.single_test['error'])}", delta_color="off", help="Zahl der Bäume im Wald gegen den Testfehler eines einzelnen Baums davon.")
m2.metric("Trainingsfehler", _err(task, a.train["error"]), help="Fehler auf den Trainingsdaten (Wald-Mittel).")
m3.metric("Testfehler", _err(task, a.test["error"]), delta=f"Raten: {_err(task, a.baseline)}", delta_color="off", help="Fehler auf den zurückgehaltenen Testdaten.")
oob_n = a.oob["n"]
m4.metric("Out-of-Bag-Fehler", _err(task, a.oob["error"]), delta=f"{oob_n} von {len(ytr)} Zeilen", delta_color="off", help="OOB-Fehler und die Zahl der Trainingszeilen, die mindestens einmal out-of-bag waren (nicht alle bei sehr wenigen Bäumen).")
st.markdown(VERDICT_TEXT[a.verdict])
if a.correlation is not None:
    root_txt = f"{a.root_shares[0][1]} von {int(n_trees)} Bäumen beginnen mit **{names[a.root_shares[0][0]]}**" if len(a.root_shares) == 1 or a.root_shares[0][1] == int(n_trees) else f"{a.root_shares[0][1]} von {int(n_trees)} Bäumen beginnen mit {names[a.root_shares[0][0]]}, der Rest verteilt sich auf {len(a.root_shares) - 1} weitere Merkmale"
    st.caption(f"Korrelation der Bäume (Anteil gleicher Vorhersagen bzw. Korrelationskoeffizient) auf dem Test: **{a.correlation:.2f}**. {root_txt}.")

st.markdown("**Wichtigkeit der Merkmale (Mittel über die Bäume)**")
st.plotly_chart(build_importance(names, a.imp), width="stretch", key="importance_chart")
st.caption("Wie in cart-demo, aber über alle Bäume gemittelt - stabiler als bei einem Einzelbaum, aber noch immer nicht die unverzerrte Permutationswichtigkeit (kommt mit Random Forest).")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Bias-Varianz-Zerlegung: Einzelbaum gegen Wald")
if st.button("Bias und Varianz über sechs Datensätze messen (dauert einen Moment)", key="bv_start"):
    st.session_state["bv_on"] = True
if st.session_state.get("bv_on"):
    with st.spinner("Wachse Einzelbäume und Wälder auf sechs Datensätzen ..."):
        bv = _bias_variance(task, crit, int(leaf), int(n_trees), int(n), int(n_noise))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_bias_variance(bv), width="stretch", key="bv_chart")
    c2.table({"": ["Bias²", "Varianz", "Gesamt"], "Einzelbaum": [f"{bv['single']['bias2']:.4g}", f"{bv['single']['variance']:.4g}", f"{bv['single']['total']:.4g}"],
              "Wald": [f"{bv['bagging']['bias2']:.4g}", f"{bv['bagging']['variance']:.4g}", f"{bv['bagging']['total']:.4g}"]})
    var_cut = 1 - bv["bagging"]["variance"] / bv["single"]["variance"]
    bias_cut = 1 - bv["bagging"]["bias2"] / max(bv["single"]["bias2"], 1e-9)
    st.caption(f"Sechs Datensätze mit denselben Regeln (nur der Seed ändert sich), je ein Einzelbaum und ein Wald auf denselben festen Testpunkten (Seed {C.DEFAULT_SEED}). Die **Varianz** (wie stark die Vorhersage vom Trainingsdatensatz abhängt) sinkt um **{var_cut:.0%}**, "
               f"der **Bias²** kaum ({bias_cut:.0%}) - Bagging senkt die Varianz, nicht den Bias. Ein Baum, der systematisch falschliegt, bleibt auch im Mittel falsch.")

st.markdown("---")

st.subheader("🔬 Korrelation der Bäume: mit und ohne dominantes Merkmal")
if st.button("Korrelation gegen die Stärke des dominanten Merkmals messen (dauert einen Moment)", key="dom_start"):
    st.session_state["dom_on"] = True
if st.session_state.get("dom_on"):
    with st.spinner("Wachse Wälder bei sechs Stärken ..."):
        drows = _dominance(task, crit, int(leaf), int(n_trees), int(n), int(n_noise), int(seed))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_dominance(drows), width="stretch", key="dom_chart")
    c2.table({"Stärke": [f"{r['strength']} %" for r in drows], "Korrelation": [f"{r['correlation']:.2f}" for r in drows], "Testfehler": [_err(task, r["test"]) for r in drows]})
    top, bottom = drows[0], drows[-1]
    if bottom["n_roots"] > 1:
        root_txt = f"die Wurzel verteilt sich auf {bottom['n_roots']} statt {top['n_roots']} Merkmal"
    elif bottom["root_feature"] != top["root_feature"]:
        root_txt = f"die Wurzel wechselt vollständig zu **{names[bottom['root_feature']]}**"
    else:
        root_txt = "die Wurzel bleibt trotzdem einheitlich (ein anderes Merkmal übernimmt die Rolle)"
    st.caption(f"{'Regression' if task == 'reg' else 'Klassifikation'}, {int(n_trees)} Bäume, Mittel über die Bäume dieses einen Datensatzes. Bei voller Stärke (100 %) beginnen alle Bäume mit **{names[top['root_feature']]}**, Korrelation {top['correlation']:.2f}. "
               f"Wird das Merkmal geschwächt (0 %), fällt die Korrelation auf {bottom['correlation']:.2f}; {root_txt}. Der Testfehler steigt dabei - das Merkmal trug echte Information, "
               "der Regler dient nur der Beobachtung. Bei einem wirklich dominanten Merkmal bringt Bagging deshalb weniger, als die Bias-Varianz-Rechnung oben verspricht - genau das behebt Random Forest im nächsten Stück.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Bäume unterscheiden sich** | Der Gewinn kommt aus der Streuung zwischen den Bäumen. Dominiert ein Merkmal jeden Baum (jede Bootstrap-Stichprobe wählt dieselbe Wurzel), bleiben die Bäume stark korreliert - Mitteln hilft weniger (Experiment oben). | Random Forest: zufällige Merkmalsteilmenge je Schnitt |
| **Bagging senkt die Varianz, nicht den Bias** | Ein Baum, der systematisch danebenliegt (zu wenige Merkmale, zu flach, falsches Kriterium), bleibt im Mittel genauso verzerrt (gemessen oben). | Boosting (nächster Ast der Linie): korrigiert Fehler gezielt |
| **Volle Bäume, kein Beschneiden** | Kein Einzelbaum muss für sich gut sein - das ist Absicht, nicht vergessen. Ein beschnittener Baum je Stichprobe hätte weniger Varianz zum Wegmitteln und im Mittel selten einen kleineren Testfehler. | - |
| **Mehr Bäume kosten nur noch Rechenzeit** | Ab einigen Dutzend Bäumen sättigt der Testfehler (Messwert oben); mehr Bäume ändern kaum noch etwas, machen die Vorhersage aber teurer. | fixe Baumzahl, früh stoppen |
| **Wichtigkeit bleibt verzerrt** | Gemittelt über die Bäume ist sie stabiler als beim Einzelbaum, aber weiterhin die Gini-Wichtigkeit - sie bevorzugt Merkmale mit vielen möglichen Schwellen. | Permutationswichtigkeit (Random Forest) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Bootstrap.** Aus den Trainingsdaten $(x_i,y_i)_{i=1}^n$ werden $n$ Indizes mit Zurücklegen gezogen: $P(\text{Zeile } i \text{ nie gezogen}) = (1-\tfrac1n)^n \to e^{-1} \approx 0{,}368$ für große $n$ - im Mittel bleiben $1-e^{-1}\approx 63{,}2\%$ der Zeilen mindestens einmal drin.

**Bagging-Vorhersage.** $B$ Bäume $T_1,\dots,T_B$, je auf einer Bootstrap-Stichprobe gewachsen: $\hat f_{\text{bag}}(x) = \frac1B\sum_{b=1}^B T_b(x)$ (Blattwert: Anteil bzw. Mittelwert).

**Out-of-Bag.** $\text{OOB}(i) = \{b : \text{Zeile } i \text{ war nicht in Stichprobe } b\}$; $\hat f_{\text{oob}}(x_i) = \frac{1}{|\text{OOB}(i)|}\sum_{b\in \text{OOB}(i)} T_b(x_i)$ - eine eingebaute Kreuzvalidierung ohne eigene Testdaten.

**Bias-Varianz-Zerlegung** (quadratischer Verlust, auch für Wahrscheinlichkeiten sinnvoll): über Trainingsdatensätze $D_1,\dots,D_m$ derselben Verteilung, an festen Testpunkten $x$ mit Wahrheit $y(x)$:
$$\overline{f}(x) = \frac1m\sum_k \hat f_{D_k}(x), \quad \text{Bias}^2 = \frac1{|X|}\sum_x (\overline f(x)-y(x))^2, \quad \text{Varianz} = \frac1{|X|}\sum_x \frac1m\sum_k (\hat f_{D_k}(x)-\overline f(x))^2.$$
Mitteln über $B$ **unabhängige** Modelle mit Varianz $\sigma^2$ ergäbe Varianz $\sigma^2/B$; Bootstrap-Bäume sind aber nicht unabhängig (dieselben Trainingsdaten), mit Korrelation $\rho$ zwischen Baumpaaren gilt eher $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ - bei $\rho$ nahe 1 (dominantes Merkmal) bleibt die Varianz hoch, egal wie groß $B$ ist.

**Baumkorrelation** (gemessen, nicht $\rho$ der Formel direkt): mittlere paarweise Ähnlichkeit der $B$ Einzelbaum-Vorhersagen auf denselben Punkten - Korrelationskoeffizient (Regression) bzw. Anteil gleicher Vorhersagen (Klassifikation).

Implementiert in `bag_tree.py` (Baumkern, wortgleich aus cart-demo), `bag_algorithm.py` (Bootstrap, Mitteln, OOB, Korrelation), `bag_evaluation.py` (Sättigung, Bias-Varianz, Dominanz-Experiment).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
