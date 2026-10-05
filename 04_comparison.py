"""
04_comparison.py — Tableau comparatif final des deux architectures CNN.

Ce script charge les métriques calculées par 03_train_evaluate.py
et produit un tableau comparatif visuel (PNG) ainsi qu'une analyse
synthétique des résultats pour le rapport académique.

Exécution :
    python 04_comparison.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RESULTS_DIR, FIGURES_DIR


def charger_metriques() -> pd.DataFrame:
    """
    Charge le fichier CSV des métriques produit par 03_train_evaluate.py.
    """
    chemin = os.path.join(RESULTS_DIR, "metriques_comparaison.csv")
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            "Veuillez d'abord exécuter : python 03_train_evaluate.py"
        )
    df = pd.read_csv(chemin)
    print(f"[INFO] Métriques chargées : {df.shape[0]} modèles")
    return df


def tracer_tableau_comparatif(df: pd.DataFrame) -> None:
    """
    Génère un tableau comparatif visuel des deux architectures.

    Le tableau affiche pour chaque modèle :
    - Accuracy, Precision, Recall, F1-score (en %)
    - RMSE (valeur brute)
    - Stratégie anti-déséquilibre utilisée

    Pourquoi visualiser sous forme de tableau ?
    Un tableau formaté est plus lisible dans un rapport académique
    qu'un simple fichier CSV. Il permet une comparaison directe
    des performances sur chaque critère.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metriques_pct = ["accuracy", "precision", "recall", "f1_score"]
    metriques_raw = ["rmse"]

    # ── Préparation des données d'affichage ───────────────────────
    strategies = {
        "AlexNet-1D": "class_weight",
        "ResNet-1D" : "SMOTE"
    }

    rows = []
    for _, row in df.iterrows():
        r = {
            "Modèle"    : row["modele"],
            "Stratégie" : strategies.get(row["modele"], "-"),
            "Accuracy"  : f"{row['accuracy']*100:.2f}%",
            "Precision" : f"{row['precision']*100:.2f}%",
            "Recall"    : f"{row['recall']*100:.2f}%",
            "F1-score"  : f"{row['f1_score']*100:.2f}%",
            "RMSE"      : f"{row['rmse']:.4f}",
        }
        rows.append(r)

    df_display = pd.DataFrame(rows)

    # ── Figure tableau ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                              gridspec_kw={"height_ratios": [2, 3]})
    fig.suptitle("Comparaison des Architectures CNN — Prédiction de Feux de Forêt",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Sous-figure 1 : Tableau ────────────────────────────────────
    axes[0].axis("off")
    colonnes  = list(df_display.columns)
    valeurs   = df_display.values.tolist()

    couleurs_entete = [["#2c3e50"] * len(colonnes)]
    couleurs_lignes = [
        ["#d5e8f0", "#d5e8f0", "#d5e8f0", "#d5e8f0", "#d5e8f0", "#d5e8f0", "#d5e8f0"],
        ["#fde8d8", "#fde8d8", "#fde8d8", "#fde8d8", "#fde8d8", "#fde8d8", "#fde8d8"],
    ]
    couleurs_cellules = couleurs_entete + couleurs_lignes

    table = axes[0].table(
        cellText   = [colonnes] + valeurs,
        cellLoc    = "center",
        loc        = "center",
        cellColours= couleurs_cellules,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Texte blanc pour l'en-tête
    for j in range(len(colonnes)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    # ── Sous-figure 2 : Barres comparatives ───────────────────────
    axes[1].set_title("Comparaison visuelle des métriques (%)", fontsize=12, pad=10)

    metriques_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(metriques_labels))
    width = 0.35

    couleurs_modeles = ["#3498db", "#e74c3c"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row["accuracy"]*100, row["precision"]*100,
                row["recall"]*100,   row["f1_score"]*100]
        bars = axes[1].bar(x + i * width, vals, width,
                           color=couleurs_modeles[i], alpha=0.85,
                           edgecolor="black", linewidth=0.7,
                           label=row["modele"])
        for bar, v in zip(bars, vals):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9
            )

    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(metriques_labels, fontsize=11)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chemin = os.path.join(FIGURES_DIR, "10_comparaison_finale.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


def afficher_analyse(df: pd.DataFrame) -> None:
    """
    Affiche une analyse textuelle des résultats pour le rapport.

    Compare les deux modèles et identifie le meilleur selon le F1-score
    (métrique la plus robuste pour les classes déséquilibrées).
    """
    print("\n" + "="*60)
    print("  ANALYSE COMPARATIVE")
    print("="*60)

    meilleur = df.loc[df["f1_score"].idxmax(), "modele"]
    pire     = df.loc[df["f1_score"].idxmin(), "modele"]

    for _, row in df.iterrows():
        print(f"\n  {row['modele']} :")
        print(f"    Accuracy  = {row['accuracy']*100:.2f}%")
        print(f"    Precision = {row['precision']*100:.2f}%")
        print(f"    Recall    = {row['recall']*100:.2f}%")
        print(f"    F1-score  = {row['f1_score']*100:.2f}%")
        print(f"    RMSE      = {row['rmse']:.4f}")

    diff_f1 = abs(df.loc[df["modele"] == meilleur, "f1_score"].values[0] -
                  df.loc[df["modele"] == pire,    "f1_score"].values[0]) * 100

    print(f"\n  Meilleur F1-score  : {meilleur}")
    print(f"  Ecart F1-score     : +{diff_f1:.2f}% en faveur de {meilleur}")
    print("\n  Interpretation :")
    print("  Le F1-score (weighted) est la metrique de reference sur des")
    print("  donnees desequilibrees car il equilibre precision et rappel.")
    print("  Le RMSE mesure l'amplitude des erreurs de classe : plus il")
    print("  est faible, plus les confusions entre classes sont mineures.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ÉTAPE 6 — TABLEAU COMPARATIF FINAL")
    print("="*60)

    df_metriques = charger_metriques()
    tracer_tableau_comparatif(df_metriques)
    afficher_analyse(df_metriques)

    print("\n" + "="*60)
    print("  PROJET TERMINÉ")
    print("="*60)
    print("\nFigure finale : outputs/figures/10_comparaison_finale.png")
    print("Métriques CSV : outputs/results/metriques_comparaison.csv")
