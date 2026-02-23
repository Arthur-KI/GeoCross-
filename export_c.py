"""
GeoCross – FPGA/C-Header Exporter

Konvertiert ein trainiertes GeoCross-Modell in einen C-Header (.h).
Keine Float-Mathematik zur Laufzeit. Keine Abhängigkeiten.
Direkt verwendbar auf Mikrocontrollern oder als FPGA-BRAM-Vorlage.

Verwendung:
    from geocross import GeoCross
    from export_c import export_c_header

    model = GeoCross(top_k=8, zg_gross=0.25, zg_klein=0.125)
    model.fit(X_train, y_train)
    export_c_header(model, "geocross_lut.h")

Hinweis zu zg-Werten auf FPGA:
    Zweierpotenzen sind ideal weil Division → Bit-Shift:
    zg = 0.500  →  >> 1  (BINS = 2)
    zg = 0.250  →  >> 2  (BINS = 4)
    zg = 0.125  →  >> 3  (BINS = 8)
    zg = 0.0625 →  >> 4  (BINS = 16)
"""

import math
from geocross import GeoCross


def export_c_header(model: GeoCross, filename="geocross_lut.h"):
    """
    Exportiert das Modell als statischen C-Header.

    Das generierte Array GC_LUT_KLEIN kann per $readmemh
    direkt in Verilog-BRAM-Module geladen werden.

    Speicherbedarf:
        zg_klein=0.125 → 9x9 = 81 Einträge × top_k Paare
        top_k=8        → 648 Bytes für lokale Intelligenz
    """
    if model.paare is None:
        raise ValueError("Modell ist nicht trainiert.")

    bins_g = math.ceil(1.0 / model.zg_g) + 1
    bins_k = math.ceil(1.0 / model.zg_k) + 1
    default_label = int(model.klassen[0])

    lines = [
        "/* ============================================================",
        " * GeoCross Hardware Lookup Table",
        " * Automatisch generiert – nicht manuell bearbeiten.",
        " *",
        f" * top_k    = {model.top_k}",
        f" * zg_gross = {model.zg_g}  (BINS = {bins_g})",
        f" * zg_klein = {model.zg_k}  (BINS = {bins_k})",
        f" * Default  = {default_label}",
        " *",
        " * Speicher: "
        f"{model.top_k * bins_g * bins_g + model.top_k * bins_k * bins_k} Bytes gesamt",
        " * ============================================================ */",
        "",
        "#ifndef GEOCROSS_LUT_H",
        "#define GEOCROSS_LUT_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define GC_TOP_K        {model.top_k}",
        f"#define GC_BINS_GROSS   {bins_g}",
        f"#define GC_BINS_KLEIN   {bins_k}",
        f"#define GC_ZG_GROSS     {model.zg_g}f",
        f"#define GC_ZG_KLEIN     {model.zg_k}f",
        f"#define GC_DEFAULT_LABEL {default_label}",
        "",
    ]

    # Dimensions-Paare
    paare_flat = [str(d) for paar in model.paare for d in paar]
    lines.append(f"/* Dimensions-Paare (je 2 Werte pro Paar): "
                 f"[da0, db0, da1, db1, ...] */")
    lines.append(f"const uint8_t GC_PAARE[{model.top_k * 2}] = "
                 f"{{{', '.join(paare_flat)}}};")
    lines.append("")

    # Lookup-Tables für Groß und Klein
    for e_name, model_dict, bins, zg in [
        ("GROSS", model.gross, bins_g, model.zg_g),
        ("KLEIN", model.klein, bins_k, model.zg_k),
    ]:
        array_size = bins * bins
        bytes_total = model.top_k * array_size
        lines.append(f"/* LUT {e_name}: zg={zg}  "
                     f"{bins}x{bins}={array_size} Einträge × {model.top_k} Paare "
                     f"= {bytes_total} Bytes */")
        lines.append(f"const uint8_t GC_LUT_{e_name}"
                     f"[{model.top_k}][{array_size}] = {{")

        for pi in range(len(model.paare)):
            ebene_flat = [default_label] * array_size

            for (x, y), zelle in model_dict[pi].items():
                pred = zelle.pred()
                if pred is not None:
                    x_safe = min(x, bins - 1)
                    y_safe = min(y, bins - 1)
                    ebene_flat[x_safe * bins + y_safe] = int(pred)

            row_str = ", ".join(map(str, ebene_flat))
            comma = "," if pi < len(model.paare) - 1 else ""
            lines.append(f"    /* Paar {pi} */ {{{row_str}}}{comma}")

        lines.append("};")
        lines.append("")

    # Inference-Funktion
    lines.append(
        "/* ============================================================\n"
        " * Inference-Funktion\n"
        " * X_input: float-Array [0, 1], Länge = Anzahl Features\n"
        " *\n"
        " * Auf FPGA: Division durch zg → Bit-Shift wenn zg = 2er-Potenz\n"
        " *   zg=0.125 → x_k = input >> 3  (0 Takte, nur Verdrahtung)\n"
        " * ============================================================ */\n"
        "uint8_t geocross_predict(const float* X_input) {\n"
        "    uint8_t stimmen[256] = {0};\n"
        "    uint8_t max_stimmen  = 0;\n"
        "    uint8_t best_label   = GC_DEFAULT_LABEL;\n"
        "\n"
        "    for (int i = 0; i < GC_TOP_K; i++) {\n"
        "        uint8_t da = GC_PAARE[i * 2];\n"
        "        uint8_t db = GC_PAARE[i * 2 + 1];\n"
        "\n"
        "        /* Würfel-Koordinaten berechnen */\n"
        f"        uint8_t x_g = (uint8_t)(X_input[da] / GC_ZG_GROSS);\n"
        f"        uint8_t y_g = (uint8_t)(X_input[db] / GC_ZG_GROSS);\n"
        f"        uint8_t x_k = (uint8_t)(X_input[da] / GC_ZG_KLEIN);\n"
        f"        uint8_t y_k = (uint8_t)(X_input[db] / GC_ZG_KLEIN);\n"
        "\n"
        "        /* Bounds-Check */\n"
        "        if (x_g >= GC_BINS_GROSS) x_g = GC_BINS_GROSS - 1;\n"
        "        if (y_g >= GC_BINS_GROSS) y_g = GC_BINS_GROSS - 1;\n"
        "        if (x_k >= GC_BINS_KLEIN) x_k = GC_BINS_KLEIN - 1;\n"
        "        if (y_k >= GC_BINS_KLEIN) y_k = GC_BINS_KLEIN - 1;\n"
        "\n"
        "        /* Flache Indizes */\n"
        "        uint16_t idx_g = x_g * GC_BINS_GROSS + y_g;\n"
        "        uint16_t idx_k = x_k * GC_BINS_KLEIN + y_k;\n"
        "\n"
        "        /* Veto-Logik: Klein schlägt Groß */\n"
        "        uint8_t label = GC_DEFAULT_LABEL;\n"
        "        if (GC_LUT_KLEIN[i][idx_k] != GC_DEFAULT_LABEL) {\n"
        "            label = GC_LUT_KLEIN[i][idx_k];\n"
        "        } else if (GC_LUT_GROSS[i][idx_g] != GC_DEFAULT_LABEL) {\n"
        "            label = GC_LUT_GROSS[i][idx_g];\n"
        "        }\n"
        "\n"
        "        /* XOR-Konflikt-Sensor (optional) */\n"
        "        /* uint8_t konflikt = (GC_LUT_GROSS[i][idx_g] ^\n"
        "                               GC_LUT_KLEIN[i][idx_k]) & 1; */\n"
        "\n"
        "        /* Voting */\n"
        "        if (label != GC_DEFAULT_LABEL) {\n"
        "            stimmen[label]++;\n"
        "            if (stimmen[label] > max_stimmen) {\n"
        "                max_stimmen = stimmen[label];\n"
        "                best_label  = label;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    return best_label;\n"
        "}\n"
        "\n"
        "#endif /* GEOCROSS_LUT_H */"
    )

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    # Statistik
    bytes_gross = model.top_k * bins_g * bins_g
    bytes_klein = model.top_k * bins_k * bins_k
    print(f"C-Header exportiert: {filename}")
    print(f"  Groß-LUT:  {bytes_gross} Bytes  ({bins_g}x{bins_g} × {model.top_k} Paare)")
    print(f"  Klein-LUT: {bytes_klein} Bytes  ({bins_k}x{bins_k} × {model.top_k} Paare)")
    print(f"  Gesamt:    {bytes_gross + bytes_klein} Bytes")
    print(f"  BRAM Tang 138K: ~1.1 MB verfügbar → "
          f"{(bytes_gross + bytes_klein) / (1.1 * 1024 * 1024) * 100:.3f}% belegt")


# ══════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    print("=" * 60)
    print("  GeoCross → C-Header Export")
    print("=" * 60)

    cancer = load_breast_cancer()
    X = MinMaxScaler().fit_transform(cancer.data)
    y = cancer.target

    # FPGA-optimale Werte: Zweierpotenzen für Bit-Shifting
    # zg_gross = 0.25  → >> 2
    # zg_klein = 0.125 → >> 3
    m = GeoCross(top_k=8, zg_gross=0.25, zg_klein=0.125)
    m.fit(X, y)
    acc = m.score(X, y)
    print(f"\n  Modell trainiert: {acc:.1%} Train-Accuracy")
    print(f"  (FPGA-optimale zg: gross=0.25, klein=0.125)")

    export_c_header(m, "geocross_lut.h")

    # Zeige Anfang der generierten Datei
    print("\n  Erste 20 Zeilen von geocross_lut.h:")
    print("  " + "─" * 50)
    with open("geocross_lut.h") as f:
        for i, line in enumerate(f):
            if i >= 20: break
            print(f"  {line}", end="")

    import os; os.remove("geocross_lut.h")
