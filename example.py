"""
GeoCross – Beispiele

Dieses Skript zeigt die grundlegende Verwendung.
Abhängigkeiten: numpy, scikit-learn (nur für Testdaten und Vergleich)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from geocross import GeoCross

sc = MinMaxScaler()

# ══════════════════════════════════════════════
# BEISPIEL 1: Binäre Klassifikation (Cancer)
# ══════════════════════════════════════════════
print("=" * 60)
print("  Beispiel 1: Binäre Klassifikation – Cancer (30D)")
print("=" * 60)

cancer = load_breast_cancer()
X = sc.fit_transform(cancer.data)
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = GeoCross(top_k=8)
model.fit(X_train, y_train, feat_namen=list(cancer.feature_names))

acc = model.score(X_test, y_test)
print(f"\n  Accuracy: {acc:.1%}")
print(f"\n  Modell-Info:")
print(model.info())
print(f"\n  Erklärung für Testpunkt 0:")
print(model.explain(X_test[0]))

# ══════════════════════════════════════════════
# BEISPIEL 2: Mehrklassen (Iris)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Beispiel 2: Mehrklassen – Iris (4D, 3 Klassen)")
print("=" * 60)

iris = load_iris()
X = sc.fit_transform(iris.data)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model3 = GeoCross(top_k=6)
model3.fit(X_train, y_train, feat_namen=list(iris.feature_names))
acc3 = model3.score(X_test, y_test)
print(f"\n  Accuracy: {acc3:.1%}  (3 Klassen, kein One-vs-Rest)")

# ══════════════════════════════════════════════
# BEISPIEL 3: auto_tune
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Beispiel 3: auto_tune – findet zg_klein automatisch")
print("=" * 60)

from sklearn.datasets import load_wine
wine = load_wine()
X = sc.fit_transform(wine.data)
y = (wine.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model_at = GeoCross(top_k=8, auto_tune=True)
model_at.fit(X_train, y_train)
acc_at = model_at.score(X_test, y_test)
print(f"\n  Accuracy:    {acc_at:.1%}")
print(f"  zg_klein:    {model_at.zg_k}  (automatisch gewählt)")

# ══════════════════════════════════════════════
# BEISPIEL 4: Speichern und Laden
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Beispiel 4: Speichern und Laden")
print("=" * 60)

# Frisches Cancer-Modell für diesen Test
cancer2 = load_breast_cancer()
X4 = sc.fit_transform(cancer2.data)
y4 = cancer2.target
X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.3, random_state=42)
model4 = GeoCross(top_k=8)
model4.fit(X4_train, y4_train)

model4.save("cancer_model.json")
model4.save("cancer_model.pkl")

model_json = GeoCross.load("cancer_model.json")
model_pkl  = GeoCross.load("cancer_model.pkl")

print(f"\n  Original:   {model4.score(X4_test, y4_test):.1%}")
print(f"  Nach JSON:  {model_json.score(X4_test, y4_test):.1%}")
print(f"  Nach Pkl:   {model_pkl.score(X4_test, y4_test):.1%}")

# ══════════════════════════════════════════════
# BEISPIEL 5: Speed-Vergleich
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Beispiel 5: Speed vs Random Forest")
print("=" * 60)

import time, os

cancer = load_breast_cancer()
X = sc.fit_transform(cancer.data)
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# GeoCross
gc = GeoCross(top_k=8)
gc.fit(X_train, y_train)

lat_gc = []
for _ in range(3000):
    t0 = time.perf_counter()
    gc.predict_one(X_test[0])
    lat_gc.append((time.perf_counter() - t0) * 1e6)

# Random Forest
rf = RandomForestClassifier(100, random_state=42)
rf.fit(X_train, y_train)

lat_rf = []
for _ in range(3000):
    t0 = time.perf_counter()
    rf.predict(X_test[0:1])
    lat_rf.append((time.perf_counter() - t0) * 1e6)

gc.save("speed_test.json")
kb_gc = os.path.getsize("speed_test.json") / 1024

print(f"\n  {'':20s}  {'GeoCross':12s}  {'RF (100 Bäume)':14s}")
print(f"  {'Accuracy':20s}  {gc.score(X_test,y_test):.1%}         "
      f"{rf.score(X_test,y_test):.1%}")
print(f"  {'Latenz (Ø)':20s}  {np.mean(lat_gc):6.1f} µs      "
      f"{np.mean(lat_rf):6.1f} µs")
print(f"  {'Modellgröße':20s}  {kb_gc:6.1f} KB       ~300 KB")
print(f"  {'Erklärbar':20s}  {'ja':12s}  nein")
print(f"  {'Nur numpy':20s}  {'ja':12s}  nein")

# Aufräumen
import os
for f in ["cancer_model.json", "cancer_model.pkl", "speed_test.json"]:
    if os.path.exists(f):
        os.remove(f)

print("\n  Fertig.")
