"""
GeoCross – Benchmark & Ablation

Führt drei Tests durch:

1. BENCH: 5-fold Stratified CV auf 7 Datensätzen
   Vergleich: GeoCross vs Random Forest vs SVM

2. ABLATION: Nur Groß vs Nur Klein vs Beide
   Beweis dass zwei Ebenen mehr bringen als eine

3. EMERGENZ: Wie viele Punkte liegen in Grenzzonen?
   Wo widersprechen sich Groß und Klein?

Ausgabe: bench.png
"""

import numpy as np
import warnings; warnings.filterwarnings('ignore')

from geocross import GeoCross, _WuerfelZelle

from sklearn.datasets import (load_iris, load_breast_cancer,
                               load_wine, make_moons, make_circles)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sc = MinMaxScaler()
BG = '#0a0a14'


# ══════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════

def cv5(model_fn, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr], y[tr])
        accs.append(m.score(X[te], y[te]) * 100)
    return np.mean(accs), np.std(accs)


def nur_gross(X, y, top_k=8, zg=0.33):
    """GeoCross mit nur großem Würfel."""
    class NurGross(GeoCross):
        def predict_one(self, p):
            stimmen = {}
            for pi, (da, db) in enumerate(self.paare):
                kg = (int(p[da] / self.zg_g), int(p[db] / self.zg_g))
                if kg in self.gross[pi]:
                    pr = self.gross[pi][kg].pred()
                    if pr is not None:
                        stimmen[pr] = stimmen.get(pr, 0) + 1
            return max(stimmen, key=stimmen.get) if stimmen else int(self.klassen[0])
    m = NurGross(top_k=top_k, zg_gross=zg)
    m.fit(X, y)
    return m

def nur_klein(X, y, top_k=8, zg=0.10):
    """GeoCross mit nur kleinem Würfel."""
    class NurKlein(GeoCross):
        def predict_one(self, p):
            stimmen = {}
            for pi, (da, db) in enumerate(self.paare):
                kk = (int(p[da] / self.zg_k), int(p[db] / self.zg_k))
                if kk in self.klein[pi]:
                    pr = self.klein[pi][kk].pred()
                    if pr is not None:
                        stimmen[pr] = stimmen.get(pr, 0) + 1
            return max(stimmen, key=stimmen.get) if stimmen else int(self.klassen[0])
    m = NurKlein(top_k=top_k, zg_klein=zg)
    m.fit(X, y)
    return m


# ══════════════════════════════════════════════
# DATENSÄTZE
# ══════════════════════════════════════════════

iris   = load_iris()
cancer = load_breast_cancer()
wine   = load_wine()
X_m, y_m   = make_moons(600, noise=0.25, random_state=42)
X_c, y_c   = make_circles(600, noise=0.15, factor=0.4, random_state=42)

datasets = {
    'Monde (2D)':    (sc.fit_transform(X_m),          y_m),
    'Iris (4D)':     (sc.fit_transform(iris.data),     (iris.target > 0).astype(int)),
    'Wine (13D)':    (sc.fit_transform(wine.data),     (wine.target == 0).astype(int)),
    'Cancer (30D)':  (sc.fit_transform(cancer.data),   cancer.target),
    'Iris 3-Kl.':    (sc.fit_transform(iris.data),     iris.target),
    'Wine 3-Kl.':    (sc.fit_transform(wine.data),     wine.target),
}


# ══════════════════════════════════════════════
# 1. BENCH
# ══════════════════════════════════════════════

print("=" * 70)
print("  1. BENCHMARK: 5-fold Stratified CV")
print("=" * 70)
print(f"\n  {'Dataset':14s}  {'GeoCross':12s}  {'RF':8s}  {'SVM':8s}  Rang")
print("  " + "─" * 56)

bench_ergebnisse = []
for name, (X, y) in datasets.items():
    ag, sg = cv5(lambda: GeoCross(top_k=8), X, y)
    ar, _  = cv5(lambda: RandomForestClassifier(100, random_state=42), X, y)
    av, _  = cv5(lambda: SVC(kernel='rbf', probability=False), X, y)
    rang = sorted([ag, ar, av], reverse=True).index(ag) + 1
    rang_str = f"#{rang}" + (" ★" if rang == 1 else "")
    print(f"  {name:14s}  {ag:5.1f}±{sg:3.1f}%  {ar:6.1f}%  {av:6.1f}%  {rang_str}")
    bench_ergebnisse.append(dict(name=name, gc=ag, rf=ar, svm=av))

avg_gc  = np.mean([e['gc']  for e in bench_ergebnisse])
avg_rf  = np.mean([e['rf']  for e in bench_ergebnisse])
avg_svm = np.mean([e['svm'] for e in bench_ergebnisse])
rang1 = sum(1 for e in bench_ergebnisse
            if e['gc'] >= max(e['rf'], e['svm']) - 0.05)
print(f"\n  {'Ø':14s}  {avg_gc:5.1f}%        {avg_rf:5.1f}%    {avg_svm:5.1f}%")
print(f"  GeoCross Rang-1: {rang1}/{len(bench_ergebnisse)}")
print(f"  Lücke zu RF: {avg_gc - avg_rf:+.1f}%")


# ══════════════════════════════════════════════
# 2. ABLATION
# ══════════════════════════════════════════════

print("\n" + "=" * 70)
print("  2. ABLATION: Nur Groß vs Nur Klein vs Beide")
print("=" * 70)
print(f"\n  {'Dataset':14s}  {'Nur Groß':10s}  {'Nur Klein':10s}  "
      f"{'Beide':10s}  Δ Beide")
print("  " + "─" * 60)

ablation_ergebnisse = []
for name, (X, y) in list(datasets.items())[:5]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ag_accs = []; ak_accs = []; ab_accs = []
    for tr, te in skf.split(X, y):
        mg = nur_gross(X[tr], y[tr])
        mk = nur_klein(X[tr], y[tr])
        mb = GeoCross(top_k=8); mb.fit(X[tr], y[tr])
        ag_accs.append(mg.score(X[te], y[te]) * 100)
        ak_accs.append(mk.score(X[te], y[te]) * 100)
        ab_accs.append(mb.score(X[te], y[te]) * 100)
    ag = np.mean(ag_accs); ak = np.mean(ak_accs); ab = np.mean(ab_accs)
    delta = ab - max(ag, ak)
    icon = '↑' if delta > 0.3 else ('→' if abs(delta) <= 0.3 else '↓')
    print(f"  {name:14s}  {ag:6.1f}%     {ak:6.1f}%     "
          f"{ab:6.1f}%     {delta:+.1f}%{icon}")
    ablation_ergebnisse.append(dict(name=name, g=ag, k=ak, b=ab, delta=delta))

gewinnt = sum(1 for e in ablation_ergebnisse if e['delta'] > 0.3)
print(f"\n  Beide besser als Einzeln: {gewinnt}/{len(ablation_ergebnisse)}")
print(f"  Ø Emergenz-Mehrwert: "
      f"{np.mean([e['delta'] for e in ablation_ergebnisse]):+.2f}%")


# ══════════════════════════════════════════════
# 3. EMERGENZ
# ══════════════════════════════════════════════

print("\n" + "=" * 70)
print("  3. EMERGENZ: Grenzzonen-Analyse (Cancer)")
print("=" * 70)

X_ca = sc.fit_transform(cancer.data)
y_ca = cancer.target
m = GeoCross(top_k=8)
m.fit(X_ca[:400], y_ca[:400])

beide_gleich = 0; konflikt = 0
lokal_richtig = 0; global_richtig = 0

for p, label in zip(X_ca[400:], y_ca[400:]):
    stimmen_g = {}; stimmen_k = {}
    for pi, (da, db) in enumerate(m.paare):
        kg = (int(p[da] / m.zg_g), int(p[db] / m.zg_g))
        kk = (int(p[da] / m.zg_k), int(p[db] / m.zg_k))
        if kg in m.gross[pi]:
            pr = m.gross[pi][kg].pred()
            if pr is not None: stimmen_g[pr] = stimmen_g.get(pr, 0) + 1
        if kk in m.klein[pi]:
            pr = m.klein[pi][kk].pred()
            if pr is not None: stimmen_k[pr] = stimmen_k.get(pr, 0) + 1
    if not stimmen_g or not stimmen_k: continue
    pred_g = max(stimmen_g, key=stimmen_g.get)
    pred_k = max(stimmen_k, key=stimmen_k.get)
    if pred_g == pred_k:
        beide_gleich += 1
    else:
        konflikt += 1
        if pred_k == int(label): lokal_richtig += 1
        elif pred_g == int(label): global_richtig += 1

print(f"\n  Beide einig:      {beide_gleich}x  (stabile Regionen)")
print(f"  Konflikt:          {konflikt}x  (Grenzzonen)")
print(f"  Lokal richtig:    {lokal_richtig}/{konflikt}  "
      f"({lokal_richtig/max(konflikt,1):.0%})")
print(f"  Global richtig:   {global_richtig}/{konflikt}  "
      f"({global_richtig/max(konflikt,1):.0%})")
print(f"\n  → Lokal gewinnt Konflikte "
      f"{lokal_richtig/max(global_richtig,1):.1f}x öfter als Global")


# ══════════════════════════════════════════════
# VISUALISIERUNG
# ══════════════════════════════════════════════

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle("GeoCross – Benchmark & Ablation",
             color='white', fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig,
    left=0.06, right=0.98, top=0.92, bottom=0.06,
    hspace=0.38, wspace=0.28)

C_GC  = '#9b5de5'; C_RF = '#00ff88'; C_SVM = '#00d4ff'
C_G   = '#e94560'; C_K  = '#ffbe0b'; C_B   = '#9b5de5'

# Bench-Balken
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_facecolor(BG)
[s.set_edgecolor('#222244') for s in ax0.spines.values()]
n = len(bench_ergebnisse); x = np.arange(n); w = 0.25
ax0.bar(x-w, [e['gc']  for e in bench_ergebnisse], w, label='GeoCross', color=C_GC, alpha=0.9)
ax0.bar(x,   [e['rf']  for e in bench_ergebnisse], w, label='RF (100)',  color=C_RF, alpha=0.8)
ax0.bar(x+w, [e['svm'] for e in bench_ergebnisse], w, label='SVM',       color=C_SVM, alpha=0.7)
ax0.set_xticks(x)
ax0.set_xticklabels([e['name'] for e in bench_ergebnisse],
                     color='white', fontsize=9, rotation=15, ha='right')
ax0.set_ylabel('Accuracy % (5-fold CV)', color='white', fontsize=10)
ax0.set_title('Benchmark (5-fold CV)', color='white', fontsize=10, fontweight='bold')
ax0.tick_params(colors='gray'); ax0.set_ylim(70, 108)
ax0.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')
ax0.grid(color='white', alpha=0.05, axis='y')

# Ø-Balken
ax1 = fig.add_subplot(gs[0, 2])
ax1.set_facecolor(BG)
[s.set_edgecolor('#222244') for s in ax1.spines.values()]
bars = ax1.bar(['GeoCross', 'RF', 'SVM'],
               [avg_gc, avg_rf, avg_svm],
               color=[C_GC, C_RF, C_SVM], alpha=0.9, width=0.5)
for bar, v in zip(bars, [avg_gc, avg_rf, avg_svm]):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.2,
             f'{v:.1f}%', ha='center', color='white',
             fontsize=10, fontweight='bold')
ax1.set_ylim(80, 102); ax1.set_ylabel('Ø Accuracy %', color='white', fontsize=10)
ax1.set_title('Ø über alle Datensätze', color='white', fontsize=10, fontweight='bold')
ax1.tick_params(colors='gray')

# Ablation
ax2 = fig.add_subplot(gs[1, :2])
ax2.set_facecolor(BG)
[s.set_edgecolor('#222244') for s in ax2.spines.values()]
n2 = len(ablation_ergebnisse); x2 = np.arange(n2)
ax2.bar(x2-w, [e['g'] for e in ablation_ergebnisse], w, label='Nur Groß',  color=C_G,  alpha=0.8)
ax2.bar(x2,   [e['k'] for e in ablation_ergebnisse], w, label='Nur Klein', color=C_K,  alpha=0.8)
ax2.bar(x2+w, [e['b'] for e in ablation_ergebnisse], w, label='Beide ★',   color=C_B,  alpha=0.9)
ax2.set_xticks(x2)
ax2.set_xticklabels([e['name'] for e in ablation_ergebnisse],
                     color='white', fontsize=9, rotation=15, ha='right')
ax2.set_ylabel('Accuracy % (5-fold CV)', color='white', fontsize=10)
ax2.set_title('Ablation: Zwei Ebenen > Eine Ebene',
              color='white', fontsize=10, fontweight='bold')
ax2.tick_params(colors='gray'); ax2.set_ylim(70, 108)
ax2.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')
ax2.grid(color='white', alpha=0.05, axis='y')
for xi, e in zip(x2, ablation_ergebnisse):
    col = '#00ff88' if e['delta'] > 0.3 else ('white' if abs(e['delta']) <= 0.3 else '#f77f00')
    ax2.text(xi+w, e['b']+0.3, f"{e['delta']:+.1f}%",
             ha='center', color=col, fontsize=8, fontweight='bold')

# Emergenz Torte
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor(BG)
gesamt = beide_gleich + konflikt
ax3.pie([beide_gleich, lokal_richtig, global_richtig,
         konflikt - lokal_richtig - global_richtig],
        labels=[f'Einig\n{beide_gleich}',
                f'Lokal ✓\n{lokal_richtig}',
                f'Groß ✓\n{global_richtig}',
                f'Beide ✗\n{konflikt-lokal_richtig-global_richtig}'],
        colors=['#1a4a1a', '#00ff88', '#9b5de5', '#3a1a1a'],
        autopct='%1.0f%%', textprops={'color': 'white', 'fontsize': 8},
        startangle=90)
ax3.set_title(f'Emergenz-Analyse\n(Cancer, {gesamt} Testpunkte)',
              color='white', fontsize=9, fontweight='bold')

plt.savefig('bench.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n→ bench.png gespeichert")
