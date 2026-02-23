"""
GeoCross – Zellgröße-Analyse (zg_klein)

Testet zg_klein von 0.50 (sehr grob) bis 0.005 (extrem fein).

Zwei Effekte:
  Zu groß: alle Punkte in einer Zelle → keine lokale Info
  Zu klein: jede Zelle hat nur 1 Punkt → kein Voting möglich

Gibt an welches zg_klein für verschiedene Datensätze optimal ist
und wo der Kollaps einsetzt (Abdeckung < 50%).

Ausgabe: zg_grenze.png
"""

import numpy as np
import warnings; warnings.filterwarnings('ignore')

from geocross import GeoCross

from sklearn.datasets import load_breast_cancer, load_wine, make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sc  = MinMaxScaler()
BG  = '#0a0a14'

ZG_WERTE = [0.50, 0.40, 0.33, 0.25, 0.20, 0.15,
            0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]


def teste_zg(X, y, zg_k, zg_g=0.33):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []; n_zellen = []; abdeckung = []
    for tr, te in skf.split(X, y):
        m = GeoCross(top_k=8, zg_gross=zg_g, zg_klein=zg_k)
        m.fit(X[tr], y[tr])
        accs.append(m.score(X[te], y[te]) * 100)
        nk = sum(len(m.klein[pi]) for pi in range(len(m.paare)))
        n_zellen.append(nk)
        treffer = sum(
            1 for p in X[te]
            if any(
                (int(p[da] / zg_k), int(p[db] / zg_k)) in m.klein[pi]
                for pi, (da, db) in enumerate(m.paare)
            )
        )
        abdeckung.append(treffer / len(te) * 100)
    return (np.mean(accs), np.std(accs),
            np.mean(n_zellen), np.mean(abdeckung))


X_m,  y_m  = make_moons(600, noise=0.20, random_state=42)
X_c2, y_c2 = make_circles(600, noise=0.12, factor=0.4, random_state=42)
cancer = load_breast_cancer()
wine   = load_wine()

datasets = {
    'Monde (2D)':   (sc.fit_transform(X_m),         y_m),
    'Kreise (2D)':  (sc.fit_transform(X_c2),         y_c2),
    'Cancer (30D)': (sc.fit_transform(cancer.data),  cancer.target),
    'Wine (13D)':   (sc.fit_transform(wine.data),
                     (wine.target == 0).astype(int)),
}

print("=" * 72)
print("  ZG-GRENZE: Optimales zg_klein + Kollaps-Analyse")
print("=" * 72)

alle = {}
for name, (X, y) in datasets.items():
    print(f"\n  {name}:")
    print(f"    {'zg_k':6s}  {'Acc':10s}  {'Zellen':8s}  "
          f"{'Abdeckung':10s}  Status")
    print(f"    " + "─" * 50)
    ergs = []
    for zg_k in ZG_WERTE:
        acc, std, nz, abk = teste_zg(X, y, zg_k)
        if abk > 95:    status = "✓ gut"
        elif abk > 70:  status = "~ ok"
        elif abk > 40:  status = "! dünn"
        else:            status = "✗ kollaps"
        print(f"    {zg_k:6.3f}  {acc:5.1f}±{std:3.1f}%  "
              f"{nz:8.0f}  {abk:8.1f}%  {status}")
        ergs.append(dict(zg=zg_k, acc=acc, std=std, nz=nz, abk=abk))
    alle[name] = ergs
    best = max(ergs, key=lambda e: e['acc'])
    print(f"    → Optimum: zg_k={best['zg']}  Acc={best['acc']:.1f}%")


# Visualisierung
fig = plt.figure(figsize=(20, 12), facecolor=BG)
fig.suptitle(
    "ZG-Grenze  •  Optimales zg_klein  •  Wo kollabiert Emergenz?",
    color='white', fontsize=13, fontweight='bold')

gs_f = gridspec.GridSpec(2, 4, figure=fig,
    left=0.06, right=0.98, top=0.92, bottom=0.05,
    hspace=0.45, wspace=0.28)

farben = ['#00d4ff', '#00ff88', '#9b5de5', '#e94560']
C3 = '#ffbe0b'

for col, (name, farbe) in enumerate(zip(list(alle.keys()), farben)):
    ergs  = alle[name]
    zgs   = [e['zg']  for e in ergs]
    accs  = [e['acc'] for e in ergs]
    stds  = [e['std'] for e in ergs]
    abks  = [e['abk'] for e in ergs]
    nzs   = [e['nz']  for e in ergs]
    best_i = np.argmax(accs)

    # Accuracy
    ax = fig.add_subplot(gs_f[0, col])
    ax.set_facecolor(BG)
    [s.set_edgecolor('#222244') for s in ax.spines.values()]
    ax.plot(range(len(zgs)), accs, color=farbe, lw=2.5, marker='o', markersize=7)
    ax.fill_between(range(len(zgs)),
                    [a - s for a, s in zip(accs, stds)],
                    [a + s for a, s in zip(accs, stds)],
                    color=farbe, alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(range(len(zgs)), abks, color=C3, lw=1.5,
             ls='--', marker='s', markersize=4, alpha=0.7)
    ax2.set_ylim(0, 110); ax2.tick_params(colors=C3, labelsize=6)
    ax2.set_ylabel('Abdeckung %', color=C3, fontsize=7)
    ax.axvline(best_i, color='white', lw=1, ls=':', alpha=0.5)
    ax.annotate(f"★{zgs[best_i]}", xy=(best_i, accs[best_i]),
                xytext=(best_i + 0.4, accs[best_i] - 5),
                color='white', fontsize=7,
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
    kollaps_i = next((i for i, e in enumerate(ergs) if e['abk'] < 50), None)
    if kollaps_i:
        ax.axvspan(kollaps_i - 0.5, len(zgs) - 0.5, color='#f77f00', alpha=0.07)
    ax.set_xticks(range(len(zgs)))
    ax.set_xticklabels([str(z) for z in zgs],
                        rotation=45, ha='right', color='white', fontsize=6)
    ax.set_ylabel('Accuracy %', color='white', fontsize=9)
    ax.set_title(name, color=farbe, fontsize=9, fontweight='bold')
    ax.tick_params(colors='gray', labelsize=6)
    ax.grid(color='white', alpha=0.05); ax.set_ylim(40, 108)

    # Zellen
    ax3 = fig.add_subplot(gs_f[1, col])
    ax3.set_facecolor(BG)
    [s.set_edgecolor('#222244') for s in ax3.spines.values()]
    ax3.bar(range(len(zgs)), nzs, color=farbe, alpha=0.7)
    ax3.axvline(best_i, color='white', lw=1.5, ls='--', alpha=0.7)
    ax3.set_xticks(range(len(zgs)))
    ax3.set_xticklabels([str(z) for z in zgs],
                         rotation=45, ha='right', color='white', fontsize=6)
    ax3.set_ylabel('Klein-Zellen', color='white', fontsize=9)
    ax3.set_title('Zellen-Wachstum', color='white', fontsize=8, fontweight='bold')
    ax3.tick_params(colors='gray', labelsize=6)

plt.savefig('zg_grenze.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n→ zg_grenze.png gespeichert")
