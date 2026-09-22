# Bagging – viele Bäume auf Bootstrap-Stichproben – Streamlit-Demo

Zweites Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Nachfolger von [CART](../cart-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Bagging**, Breimans Bootstrap-Aggregating (1996) – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo (acht echte Merkmale, einstellbare Rauschmerkmale, Klassifikation "kommt die Lieferung zu spät?" und Regression "wie lange dauert sie?"). Der Baumkern ist **wortgleich aus cart-demo übernommen** (`bag_tree.py`, dort ausführlich geprüft); dieses Stück fügt Bootstrap, Mittelung und Out-of-Bag hinzu.
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, scikit-learn nur in den Tests als Gegenprobe.

**Bezug zu OR:** ein gemitteltes Ensemble liefert nicht nur eine Vorhersage, sondern über die Streuung der Einzelbäume auch eine **Unsicherheit** – ein Eingang für robuste oder stochastische Planung (z. B. Pufferzeiten in der Tourenplanung), nicht nur ein Punktwert.

**Einordnung in die Reihe:** CART wählt Schnitte gierig und ist **instabil** (kleine Datenänderung, anderer Baum). Bagging macht aus dieser Instabilität eine Stärke – B Bootstrap-Bäume mitteln das Rauschen weg, die Varianz sinkt stark. Die Kehrseite: dominiert ein Merkmal jeden Baum, bleiben die Bäume korreliert und Mitteln hilft weniger – **das behebt Random Forest** im nächsten Stück (zufällige Merkmalsteilmenge je Schnitt).

```
CART → Bagging (dieses Stück) → Random Forest → Extra Trees
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale, 70 % Training / 30 % Test, Seed 7; Klassifikation "zu spät", 30 Bäume falls nicht anders angegeben) |
|---|---|
| Testfehler gegen die Zahl der Bäume | ✅ 1 Baum **25,6 %** → 10 Bäume **18,1 %** → 30 Bäume **15,0 %** → 100 Bäume **16,1 %**: der größte Sprung kommt von den ersten Bäumen, ab einigen Dutzend sättigt der Gewinn. |
| Out-of-Bag gegen Test | ✅ Bei 30 Bäumen OOB **18,0 %** gegen Testfehler **15,0 %** – schon mit einem Baum ist OOB verfügbar (**22,4 %**, aus den ~37 % nie gezogenen Zeilen) und nahe am Testfehler, ganz ohne eigene Testdaten. |
| **Bias-Varianz-Zerlegung** (Einzelbaum gegen Wald, sechs Datensätze, feste Testpunkte) | ✅ Klassifikation: Varianz **−84,5 %** (0,104 → 0,016), Bias² fast gleich (**−5,9 %**). Regression: Varianz **−83,0 %** (112,0 → 19,0), Bias² **−10,3 %** (102,0 → 91,5). **Bagging senkt die Varianz, nicht den Bias.** |
| Baumkorrelation (Standarddatensatz) | ➖ 0,77 (Klassifikation) bzw. 0,80 (Regression) – die Bäume sind trotz unterschiedlicher Bootstrap-Stichproben deutlich ähnlich; alle 30 Bäume beginnen mit demselben Merkmal (Ladegewicht bzw. Distanz). |
| **Dominantes Merkmal** (Regler, Regression, das dominante Merkmal zu einem Anteil durch Rauschen ersetzt) | ❌ Korrelation **0,80 → 0,33** (100 % → 0 % Stärke), die Wurzel verteilt sich von 1 auf bis zu 3 Merkmale. Der Testfehler steigt dabei (echte Information geht verloren) – der Regler dient nur der Beobachtung: **ein wirklich dominantes Merkmal begrenzt den Bagging-Gewinn**, genau der Hook für Random Forest. |
| Bootstrap-Anteil einzigartiger Zeilen | ➖ **63,2 %** (1 − 1/e), unabhängig von n – über 30 Ziehungen bei n = 300 gemessen: Mittel 62,9 %. |
| Kreuzprobe mit scikit-learn | ➖ Mit denselben Bootstrap-Indizes und großen Blättern (`min_samples_leaf ≥ 10`) stimmt die **Mehrheit** der Einzelbäume exakt überein; Bootstrap-Duplikate (nur ~63 % einzigartige Zeilen) erhöhen die Gleichstandsrate gegenüber cart-demo deutlich – ein gemessener, kein Implementierungsfehler (siehe Verifikation). |

## Was die Demo zeigt

- **Bagging in Aktion:** der Wald wächst Baum für Baum mit Schritt-Regler und Abspielen: links der zuletzt hinzugekommene Einzelbaum (grob, zur Einordnung), rechts das Wald-Mittel über zwei wählbare Merkmale, darunter die Sättigungskurve (Test- und OOB-Fehler gegen die Zahl der Bäume) mit einer Marke beim aktuellen Stand.
- **Was der Wald gelernt hat:** Bäume, Trainings-, Test- und Out-of-Bag-Fehler, ein Urteil (deutlich besser / kaum besser als der Einzelbaum), Baumkorrelation und Wurzel-Anteile, gemittelte Wichtigkeit der Merkmale.
- **Regler:** Aufgabe (Klassifikation | Regression), Kriterium (Gini | Entropie; bei Regression fest Varianz), Mindestblattgröße, Zahl der Bäume, **Stärke des dominanten Merkmals** (Experiment-Regler), Lieferungen, Rauschmerkmale, falsche Etiketten, Seed.
- **Experimente auf Knopfdruck:** Bias-Varianz-Zerlegung (Einzelbaum gegen Wald, sechs Datensätze) und Korrelation der Bäume gegen die Stärke des dominanten Merkmals.

## Modell und Verfahren

- **Bootstrap:** n Zeilen mit Zurücklegen gezogen, deterministisch je Baum (`seed·10⁶ + b`), damit einzelne Bäume reproduzierbar sind.
- **B volle Bäume:** derselbe CART-Kern wie in cart-demo, ohne Beschneiden (klassisch `min_samples_leaf = 1`, hier auch einstellbar).
- **Mitteln:** Blattwerte über alle Bäume gemittelt; `upto` erlaubt das Mittel der ersten k Bäume (für die Sättigungskurve und die Schrittanimation).
- **Out-of-Bag:** je Trainingszeile das Mittel nur der Bäume, die sie nicht gesehen haben.
- **Dominantes Merkmal (Experiment):** das Merkmal, das in jedem Baum die Wurzel bildet, wird in einem Anteil der Zeilen durch eine zufällige Permutation seiner selbst ersetzt (dieselbe Verteilung, kein Bezug mehr zum Ziel) – kein realistischer Datenregler, nur zum Beobachten der Korrelation.

## Was nicht funktioniert hat / Grenzen

- **Erste Idee für den Dominanz-Regler war falsch:** das dominante Merkmal Richtung Mittelwert zu stauchen ändert an den Bäumen **gar nichts** – Entscheidungsbäume sind invariant gegenüber monotonen Transformationen einzelner Merkmale (dieselbe Rangfolge, derselbe Schnitt), solange die Streuung nicht exakt auf null fällt. Erst das **teilweise Vertauschen** der Zeilen (Bernoulli-Maske) erzeugt einen echten, graduellen Effekt auf Korrelation und Testfehler.
- **Kreuzprobe mit scikit-learn ist auf Bootstrap-Daten seltener exakt als bei cart-demo:** eine Bootstrap-Stichprobe von n = 300 behält im Mittel nur ~63 % einzigartige Zeilen – Duplikate erhöhen die Zahl echter Gleichstände (zwei Merkmale mit exakt demselben Gewinn) deutlich. Verifiziert über den first-divergence-Vergleich: an der Abweichungsstelle ist der Gewinn beider Merkmale bis auf Gleitkomma-Genauigkeit identisch. Die Tests prüfen deshalb Blattzahl-Gleichheit für jeden Baum und **Mehrheit statt Alle** für exakte Struktur-Gleichheit (siehe Verifikation).
- **100 Bäume sind nicht zuverlässig besser als 30:** im Standarddatensatz ist der Testfehler bei 100 Bäumen (16,1 %) sogar leicht schlechter als bei 30 (15,0 %) – Stichprobenrauschen, kein Bug; die Sättigungskurve zeigt den Trend über viele k-Werte, nicht ein einzelner Endpunkt.
- **Bias sinkt bei Regression stärker als erwartet** (−10,3 %, nicht "praktisch null"): der volle Regressionsbaum hat durch Rauschen in der Dauer (7 min Standardabweichung) eine kleine, aber messbare Verzerrung, die das Mitteln teilweise mitkorrigiert – die Hauptaussage (Varianz sinkt viel stärker als Bias) bleibt klar.

## Verifikation

`tests/test_algorithm.py` (17 Tests): Bootstrap-Indizes gegen `bag.in_bag`; Out-of-Bag-Vorhersage gegen eine naive Python-Schleife; **Mehrheit der Einzelbäume exakt gleich scikit-learn** auf denselben Bootstrap-Zeilen (`min_samples_leaf = 10`), abweichende Bäume mit gleicher Blattzahl und nachgewiesenem Gleichstand (gleicher Gewinn an der Abweichungsstelle); Bootstrap-Anteil einzigartiger Zeilen ≈ 1 − 1/e; `upto` gegen einen frisch gewachsenen kleineren Wald; Wurzel-Häufigkeit, Baumkorrelation, gemittelte Wichtigkeit; Grenzfälle (ein Baum = der Baum selbst, konstantes Ziel).
`tests/test_claims.py` hält **jede Zahl** aus App und README fest. `tests/test_app.py` prüft die Oberfläche per AppTest (jedes Preset, Aufgabenwechsel, ausgeblendete Regler, Abspielen mit mehreren Bildern und schrittspezifischen Diagramm-Schlüsseln, Permalink, Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `bag_tree.py` | Baumkern (wortgleich aus cart-demo) |
| `bag_algorithm.py` | Bootstrap, Mitteln, Out-of-Bag, Baumkorrelation |
| `bag_scenario.py` | Lieferdaten (wie cart-demo) + Dominanz-Experiment |
| `bag_evaluation.py` | Analyse, Sättigung, Bias-Varianz, Dominanz-Experiment |
| `bag_visualization.py` | Baumdiagramm, Karte, Sättigungs-, Bias-Varianz- und Dominanz-Kurven |
| `bag_presets.py`, `bag_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
