"""
╔══════════════════════════════════════════════════════════════════╗
║  GeoCross  –  Geometrischer Klassifikator                        ║
║                                                                  ║
║  Nur numpy. Kein Gradient. Kein Black-Box.                       ║
║                                                                  ║
║  Wie es funktioniert:                                            ║
║    1. Fisher-Score findet die relevanten Dimension-Paare         ║
║    2. Zwei Würfel-Ebenen: groß (global) + klein (lokal)          ║
║    3. Lokal schlägt global – Konflikte = emergente Grenzzonen    ║
║    4. Voting über alle Paare → Vorhersage                        ║
║    5. Mehrklassen: beliebig viele Labels                         ║
║                                                                  ║
║  Warum es funktioniert:                                          ║
║    Großer Würfel lernt die grobe Struktur.                       ║
║    Kleiner Würfel korrigiert an den Grenzen.                     ║
║    Niemand hat "Grenze" programmiert – sie entsteht.             ║
║                                                                  ║
║  API:                                                            ║
║    model = GeoCross(top_k=8)                                     ║
║    model.fit(X_train, y_train)                                   ║
║    preds  = model.predict(X_test)                                ║
║    acc    = model.score(X_test, y_test)                          ║
║    erkl   = model.explain(X_test[0])                             ║
║    info   = model.info()                                         ║
║    model.save("model.json")   # oder .pkl                        ║
║    model2  = GeoCross.load("model.json")                         ║
║                                                                  ║
║  Abhängigkeiten: numpy (+ json, pickle – stdlib)                 ║
║                                                                  ║
║  Edge-Computing (C-Pseudocode):                                  ║
║    key_k = (int)(x/0.1)*1000 + (int)(y/0.1)                     ║
║    key_g = (int)(x/0.33)*100 + (int)(y/0.33)                    ║
║    if lookup(table_klein, key_k): return label                   ║
║    if lookup(table_gross, key_g): return label                   ║
║    → 2 Integer-Divisionen + 2 Lookups. Fertig.                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import json
import pickle


# ══════════════════════════════════════════════════════════════════
# INTERN: Würfel-Zelle
# ══════════════════════════════════════════════════════════════════

class _WuerfelZelle:
    """
    Einfachste mögliche Zelle: Mehrheitsvoting.
    Kein Kreuz. Kein Arm. Keine Lernregel.
    Nur: welches Label hat diese Zelle am häufigsten gesehen?
    """
    __slots__ = ['stimmen', 'n']

    def __init__(self):
        self.stimmen = {}
        self.n = 0

    def lern(self, label):
        self.stimmen[label] = self.stimmen.get(label, 0) + 1
        self.n += 1

    def pred(self):
        if not self.stimmen:
            return None
        return max(self.stimmen, key=self.stimmen.get)

    def sicherheit(self):
        if not self.stimmen or self.n == 0:
            return 0.0
        return max(self.stimmen.values()) / self.n

    def to_dict(self):
        return {'stimmen': {int(k): v for k, v in self.stimmen.items()}, 'n': self.n}

    @classmethod
    def from_dict(cls, d):
        z = cls()
        z.stimmen = {int(k): v for k, v in d['stimmen'].items()}
        z.n = d['n']
        return z


# ══════════════════════════════════════════════════════════════════
# HAUPTKLASSE: GeoCross
# ══════════════════════════════════════════════════════════════════

class GeoCross:
    """
    GeoCross – Geometrischer Klassifikator.

    Zwei verschachtelte Würfel-Ebenen pro Dimension-Paar:
      Groß (global): zg_gross  →  grobe Struktur, voller Fallback
      Klein (lokal): zg_klein  →  feine Korrektur an Grenzen

    Vorhersage-Logik:
      1. Kleiner Würfel hat Daten → lokale Entscheidung (Vorfahrt)
      2. Nur großer Würfel        → globaler Fallback
      3. Beide leer               → Default-Klasse

    Parameter:
        top_k     : Anzahl der besten Dim-Paare (default: 8)
        zg_gross  : Große Würfel-Größe, global (default: 0.33)
        zg_klein  : Kleine Würfel-Größe, lokal  (default: 0.10)
        epochen   : Trainings-Epochen (default: 5)
        auto_tune : Optimales zg_klein automatisch finden (default: False)
                    Testet [0.25, 0.20, 0.15, 0.10, 0.07, 0.05] via 3-fold CV
                    Kein Gradient. Reine geometrische Grid-Search.

    Beispiel:
        model = GeoCross()
        model.fit(X_train, y_train)
        print(model.score(X_test, y_test))
        print(model.explain(X_test[0]))

        # Mit Auto-Tune:
        model = GeoCross(auto_tune=True)
        model.fit(X_train, y_train)
        print(f"Optimales zg_klein: {model.zg_k}")
    """

    # Kandidaten für auto_tune (geometrisch sinnvoll)
    _ZG_KANDIDATEN = [0.25, 0.20, 0.15, 0.10, 0.07, 0.05]

    def __init__(self, top_k=8, zg_gross=0.33, zg_klein=0.10,
                 epochen=5, auto_tune=False):
        self.top_k     = top_k
        self.zg_g      = zg_gross
        self.zg_k      = zg_klein
        self.epochen   = epochen
        self.auto_tune = auto_tune
        self.D = None; self.klassen = None
        self.paare = None; self.fisher_scores = None
        self.gross = None; self.klein = None
        self.feat_namen = None

    # ── Fisher-Score ──────────────────────────────────────────────

    def _fisher(self, X, y, da, db):
        klassen = np.unique(y)
        pts = np.column_stack([X[:, da], X[:, db]])
        sp  = {k: pts[y == k].mean(0) for k in klassen}
        st  = sum(pts[y == k].var(0).sum() + 1e-9 for k in klassen)
        ab  = sum(
            np.sum((sp[a] - sp[b]) ** 2)
            for i, a in enumerate(klassen)
            for b in klassen[i + 1:]
        )
        return ab / st

    def _vorauswahl(self, X, y):
        scores = {}
        for da in range(self.D):
            for db in range(da + 1, self.D):
                scores[(da, db)] = self._fisher(X, y, da, db)
        sortiert = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n = min(self.top_k, len(sortiert))
        self.paare          = [p for p, _ in sortiert[:n]]
        self.fisher_scores  = scores

    # ── Training ──────────────────────────────────────────────────

    def fit(self, X, y, feat_namen=None):
        """
        Trainiere GeoCross.
        X: numpy array (n, features), normiert auf [0, 1]
        y: numpy array (n,), beliebige Integer-Labels
        feat_namen: optional Liste der Feature-Namen
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.D = X.shape[1]; self.klassen = np.unique(y)
        self.feat_namen = feat_namen

        # Auto-Tune: finde optimales zg_klein via 3-fold CV
        if self.auto_tune and len(X) >= 30:
            self.zg_k = self._tune_zg(X, y)

        self._vorauswahl(X, y)
        self.gross = [{} for _ in self.paare]
        self.klein = [{} for _ in self.paare]
        for _ in range(self.epochen):
            for p, label in zip(X, y):
                self._lern_punkt(p, int(label))
        return self

    def _tune_zg(self, X, y):
        """
        Finde optimales zg_klein via 3-fold CV.
        Reine geometrische Grid-Search – kein Gradient.
        Schnell: nur 6 Kandidaten × 3 Folds.
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        beste_zg = self.zg_k
        beste_acc = -1.0

        for zg_k in self._ZG_KANDIDATEN:
            accs = []
            for tr, te in skf.split(X, y):
                m = GeoCross(top_k=self.top_k,
                             zg_gross=self.zg_g,
                             zg_klein=zg_k,
                             epochen=self.epochen,
                             auto_tune=False)
                m.fit(X[tr], y[tr])
                accs.append(m.score(X[te], y[te]))
            acc = np.mean(accs)
            if acc > beste_acc:
                beste_acc = acc
                beste_zg  = zg_k

        return beste_zg

    def _lern_punkt(self, p, label):
        for pi, (da, db) in enumerate(self.paare):
            kg = (int(p[da] / self.zg_g), int(p[db] / self.zg_g))
            kk = (int(p[da] / self.zg_k), int(p[db] / self.zg_k))
            if kg not in self.gross[pi]: self.gross[pi][kg] = _WuerfelZelle()
            if kk not in self.klein[pi]: self.klein[pi][kk] = _WuerfelZelle()
            self.gross[pi][kg].lern(label)
            self.klein[pi][kk].lern(label)

    # ── Vorhersage ────────────────────────────────────────────────

    def predict_one(self, p):
        """Vorhersage für einen Punkt. Klein hat Vorfahrt vor Groß."""
        stimmen = {}
        for pi, (da, db) in enumerate(self.paare):
            kg = (int(p[da] / self.zg_g), int(p[db] / self.zg_g))
            kk = (int(p[da] / self.zg_k), int(p[db] / self.zg_k))
            if kk in self.klein[pi]:
                pr = self.klein[pi][kk].pred()
            elif kg in self.gross[pi]:
                pr = self.gross[pi][kg].pred()
            else:
                continue
            if pr is not None:
                stimmen[pr] = stimmen.get(pr, 0) + 1
        return max(stimmen, key=stimmen.get) if stimmen else int(self.klassen[0])

    def predict(self, X):
        """Vorhersage für mehrere Punkte. Gibt numpy array zurück."""
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(p) for p in X])

    def score(self, X, y):
        """Accuracy (0.0 – 1.0)."""
        return float((self.predict(X) == np.asarray(y)).mean())

    # ── Erklärung ─────────────────────────────────────────────────

    def explain(self, p):
        """Erkläre die Vorhersage: welche Ebene, welche Dims, wie sicher."""
        p = np.asarray(p, dtype=float)
        stimmen_detail = {}
        for pi, (da, db) in enumerate(self.paare):
            fs = self.fisher_scores.get((da, db), 0)
            kg = (int(p[da] / self.zg_g), int(p[db] / self.zg_g))
            kk = (int(p[da] / self.zg_k), int(p[db] / self.zg_k))
            n0 = self.feat_namen[da] if self.feat_namen else f"Dim{da}"
            n1 = self.feat_namen[db] if self.feat_namen else f"Dim{db}"
            if kk in self.klein[pi]:
                z = self.klein[pi][kk]
                ebene = f"LOKAL  (klein zg={self.zg_k})"
            elif kg in self.gross[pi]:
                z = self.gross[pi][kg]
                ebene = f"GLOBAL (groß  zg={self.zg_g})"
            else:
                continue
            pr = z.pred()
            if pr is None: continue
            erkl = (
                f"  ({n0} × {n1})  Fisher={fs:.3f}\n"
                f"    Ebene:      {ebene}\n"
                f"    Sicherheit: {z.sicherheit():.0%}  ({z.n} Punkte)\n"
                f"    {n0}: {p[da]:.4f}\n"
                f"    {n1}: {p[db]:.4f}\n"
                f"    → Label {pr}"
            )
            if pr not in stimmen_detail: stimmen_detail[pr] = []
            stimmen_detail[pr].append((fs, erkl))
        if not stimmen_detail:
            return "Keine Zellen getroffen – Standardvorhersage."
        vorhersage = max(stimmen_detail, key=lambda k: len(stimmen_detail[k]))
        stimmen_detail[vorhersage].sort(reverse=True)
        votes = {k: len(v) for k, v in stimmen_detail.items()}
        return "\n".join([
            f"Vorhersage: Label {vorhersage}",
            "Stimmen:    " + "  ".join(
                f"Label {k}: {v}x" for k, v in sorted(votes.items())),
            "",
            "Stärkste geometrische Begründung:",
            stimmen_detail[vorhersage][0][1],
        ])

    # ── Info ──────────────────────────────────────────────────────

    def info(self):
        """Übersicht über das trainierte Modell."""
        if self.paare is None: return "Nicht trainiert."
        n_g = sum(len(self.gross[pi]) for pi in range(len(self.paare)))
        n_k = sum(len(self.klein[pi]) for pi in range(len(self.paare)))
        zeilen = [
            "GeoCross – Modell-Info",
            f"  Dimensionen   : {self.D}",
            f"  Klassen       : {list(self.klassen)}",
            f"  Dim-Paare     : {len(self.paare)}",
            f"  Würfel groß   : zg={self.zg_g}  →  {n_g} Zellen",
            f"  Würfel klein  : zg={self.zg_k}  →  {n_k} Zellen"
            + (" ★ (auto-tuned)" if self.auto_tune else ""),
            f"  Zellen gesamt : {n_g + n_k}",
            "", "  Top Dim-Paare (Fisher-Score):",
        ]
        for score, (da, db) in sorted(
            [(self.fisher_scores[p], p) for p in self.paare], reverse=True)[:5]:
            n0 = self.feat_namen[da] if self.feat_namen else f"Dim{da}"
            n1 = self.feat_namen[db] if self.feat_namen else f"Dim{db}"
            zeilen.append(f"    {n0} × {n1}  →  {score:.4f}")
        return "\n".join(zeilen)

    # ── Emergenz-Analyse ──────────────────────────────────────────

    def emergenz(self, X, y):
        """
        Wo groß ≠ klein → emergente Grenzzonen.
        Gibt dict zurück: beide_gleich, konflikt, lokal_korrekt, global_korrekt.
        """
        beide = kon = lok = gl = 0
        for p, label in zip(X, y):
            label = int(label)
            for pi, (da, db) in enumerate(self.paare):
                kg = (int(p[da] / self.zg_g), int(p[db] / self.zg_g))
                kk = (int(p[da] / self.zg_k), int(p[db] / self.zg_k))
                pg = self.gross[pi][kg].pred() if kg in self.gross[pi] else None
                pk = self.klein[pi][kk].pred() if kk in self.klein[pi] else None
                if pg is not None and pk is not None:
                    if pg == pk: beide += 1
                    else:
                        kon += 1
                        if pk == label: lok += 1
                        if pg == label: gl  += 1
        return {'beide_gleich': beide, 'konflikt': kon,
                'lokal_korrekt': lok, 'global_korrekt': gl}

    # ── Speichern & Laden ─────────────────────────────────────────

    def save(self, pfad):
        """Speichere Modell als .json oder .pkl."""
        if pfad.endswith('.json'):
            daten = {
                'version': 2, 'top_k': self.top_k,
                'zg_gross': self.zg_g, 'zg_klein': self.zg_k,
                'epochen': self.epochen, 'auto_tune': self.auto_tune,
                'D': self.D,
                'klassen': [int(k) for k in self.klassen],
                'feat_namen': self.feat_namen,
                'paare': [[int(da), int(db)] for da, db in self.paare],
                'fisher_scores': {
                    f"{da},{db}": float(s)
                    for (da, db), s in self.fisher_scores.items()
                },
                'gross': [
                    {f"{k[0]},{k[1]}": z.to_dict()
                     for k, z in self.gross[pi].items()}
                    for pi in range(len(self.paare))
                ],
                'klein': [
                    {f"{k[0]},{k[1]}": z.to_dict()
                     for k, z in self.klein[pi].items()}
                    for pi in range(len(self.paare))
                ],
            }
            with open(pfad, 'w', encoding='utf-8') as f:
                json.dump(daten, f, indent=2, ensure_ascii=False)
        elif pfad.endswith('.pkl'):
            with open(pfad, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Nur .json oder .pkl unterstützt.")
        print(f"Modell gespeichert → {pfad}")

    @classmethod
    def load(cls, pfad):
        """Lade gespeichertes Modell."""
        if pfad.endswith('.json'):
            with open(pfad, 'r', encoding='utf-8') as f:
                d = json.load(f)
            m = cls(top_k=d['top_k'], zg_gross=d['zg_gross'],
                    zg_klein=d['zg_klein'], epochen=d['epochen'],
                    auto_tune=d.get('auto_tune', False))
            m.D = d['D']; m.klassen = np.array(d['klassen'])
            m.feat_namen = d.get('feat_namen')
            m.paare = [tuple(p) for p in d['paare']]
            m.fisher_scores = {
                tuple(int(x) for x in k.split(',')): v
                for k, v in d['fisher_scores'].items()
            }
            m.gross = [
                {tuple(int(x) for x in k.split(',')): _WuerfelZelle.from_dict(z)
                 for k, z in d['gross'][pi].items()}
                for pi in range(len(m.paare))
            ]
            m.klein = [
                {tuple(int(x) for x in k.split(',')): _WuerfelZelle.from_dict(z)
                 for k, z in d['klein'][pi].items()}
                for pi in range(len(m.paare))
            ]
            return m
        elif pfad.endswith('.pkl'):
            with open(pfad, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Nur .json oder .pkl unterstützt.")


# ══════════════════════════════════════════════════════════════════
# SCHNELLTEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("GeoCross v2 – Schnelltest")
    print("Architektur: Würfel-Hierarchie (groß=global, klein=lokal)\n")

    try:
        from sklearn.datasets import load_breast_cancer, load_wine, load_iris
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split

        sc = MinMaxScaler()
        tests = [
            ("Iris  (4D,  binär)", load_iris(),         True),
            ("Wine  (13D, binär)", load_wine(),          True),
            ("Cancer(30D, binär)", load_breast_cancer(), False),
            ("Iris  (4D,  3-Kl.)", load_iris(),         False),
            ("Wine  (13D, 3-Kl.)", load_wine(),          False),
        ]
        print(f"  {'Dataset':22s}  {'Kl.':4s}  {'Train':7s}  {'Test':7s}  {'Gap':6s}")
        print("  " + "─" * 55)
        for name, ds, binaer in tests:
            X = sc.fit_transform(ds.data)
            y = (ds.target > 0).astype(int) if binaer else ds.target
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=42)
            fn = list(ds.feature_names) if hasattr(ds, 'feature_names') else None
            m = GeoCross(top_k=8)
            m.fit(X_tr, y_tr, feat_namen=fn)
            tr = m.score(X_tr, y_tr) * 100
            te = m.score(X_te, y_te) * 100
            print(f"  {name:22s}  {len(np.unique(y)):4d}  "
                  f"{tr:6.1f}%  {te:6.1f}%  {tr-te:+5.1f}%")

        # Erklärung
        print("\n  Erklärung (Cancer):")
        ds = load_breast_cancer()
        X  = sc.fit_transform(ds.data); y = ds.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        m  = GeoCross(top_k=8)
        m.fit(X_tr, y_tr, feat_namen=list(ds.feature_names))
        print(m.explain(X_te[0]))

        # Emergenz
        em = m.emergenz(X_te, y_te)
        print(f"\n  Emergenz (Cancer Testset):")
        print(f"    Beide gleich : {em['beide_gleich']:4d}x")
        print(f"    Konflikt     : {em['konflikt']:4d}x  ← Grenzzonen!")
        print(f"    Lokal korrekt: {em['lokal_korrekt']:4d}x bei Konflikt")

        # Info
        print(f"\n{m.info()}")

        # Speichern
        m.save("model.json")
        m2 = GeoCross.load("model.json")
        print(f"\n  Nach Laden: {m2.score(X_te, y_te)*100:.1f}%  ✓")
        import os
        print(f"  Dateigröße: {os.path.getsize('model.json')/1024:.1f} KB")

    except ImportError:
        X = np.random.rand(100, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        m = GeoCross(top_k=3)
        m.fit(X[:70], y[:70])
        print(f"Score: {m.score(X[70:], y[70:]):.3f}")
        print(m.explain(X[0]))
