"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

# Merkmale der Lieferungen: (Name, Einheit) - wortgleich aus cart-demo
FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"
CRITERIA = {"class": ("gini", "entropy"), "reg": ("variance",)}
CRITERION_LABELS = {"gini": "Gini-Unreinheit", "entropy": "Entropie", "variance": "Varianz (Fehlerquadrate)"}
DEFAULT_CRITERION = {"class": "gini", "reg": "variance"}

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3               # Rauschmerkmale (zufällig, ohne Bezug zum Ziel)
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0     # Prozent falsche Etiketten (nur Klassifikation)
TEST_SHARE = 0.3
DEFAULT_SEED = 7

SWEEP_SEEDS = tuple(range(100000, 100005))

# Bagging-eigene Regler
N_TREES_MIN, N_TREES_MAX, DEFAULT_N_TREES = 1, 150, 30
LEAF_MIN, LEAF_MAX, DEFAULT_LEAF = 1, 50, 1                 # Bagging-Bäume wachsen voll (Blatt = 1) und werden gemittelt, nicht beschnitten
DEPTH_MAX = 16                                              # Obergrenze der Baumtiefe (kein Regler; volle Bäume, siehe cart-demo)
BOOTSTRAPS = 30                                              # Wiederholungen der Instabilitätsmessung (aus cart-demo übernommen)
DOMINANT_MIN, DOMINANT_MAX, DEFAULT_DOMINANT = 0, 100, 100   # "Stärke des dominanten Merkmals" (Experiment): 100 = unveränderte Daten

DEFAULT_MAP = (0, 3)          # Kartenausschnitt: Distanz x Verkehr

COLORS = {"train": "#1f77b4", "test": "#d62728", "single": "#9aa0a6", "bag": "#2ca02c", "oob": "#ff7f0e"}

PRESETS = {
    "🌲 Ein Baum": dict(task="class", criterion="gini", leaf=1, n_trees=1, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, dominant=DEFAULT_DOMINANT, fx=0, fy=3),
    "🌳 Kleiner Wald": dict(task="class", criterion="gini", leaf=1, n_trees=10, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, dominant=DEFAULT_DOMINANT, fx=0, fy=3),
    "🌲🌳 Großer Wald": dict(task="class", criterion="gini", leaf=1, n_trees=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, dominant=DEFAULT_DOMINANT, fx=0, fy=3),
    "🎯 Dominantes Merkmal": dict(task="reg", criterion="variance", leaf=1, n_trees=DEFAULT_N_TREES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, dominant=20, fx=0, fy=3),
    "📈 Regression": dict(task="reg", criterion="variance", leaf=1, n_trees=DEFAULT_N_TREES, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, dominant=DEFAULT_DOMINANT, fx=0, fy=3),
}
PRESET_HELP = {
    "🌲 Ein Baum": "Nur ein Baum (= der volle Baum aus cart-demo): Testfehler 25.6 % - der OOB-Fehler ist mit nur einem Baum schon verfügbar (22.4 %, aus den rund 37 % nie gezogenen Zeilen) und nahe am Testfehler.",
    "🌳 Kleiner Wald": "10 Bäume gemittelt: Testfehler 18.1 % statt 25.6 % beim Einzelbaum - der größte Sprung kommt von den ersten Bäumen.",
    "🌲🌳 Großer Wald": "100 Bäume: Testfehler 16.1 %, OOB 15.0 % - kaum besser als 30 Bäume (15.0 %). Ab einigen Dutzend Bäumen sättigt der Gain; mehr Bäume kosten nur noch Rechenzeit.",
    "🎯 Dominantes Merkmal": "Regression, das dominante Merkmal (Distanz) ist zu 80 % durch Rauschen ersetzt: die Korrelation der Bäume fällt von 0.80 (unverändert) auf 0.35, und die Wurzel ist nicht mehr einheitlich (Anteil des häufigsten Wurzelmerkmals 100 % -> 53 %). Der Testfehler steigt dabei auch (Distanz trägt echte Information) - dieser Regler ist zum Beobachten der Korrelation da, nicht empfohlen.",
    "📈 Regression": "Regression mit 30 Bäumen: Testfehler 10.3 min gegen 17.0 min beim Einzelbaum (Raten: 27.2 min) - fast eine Halbierung des Fehlers über dem Mittelwert.",
}
