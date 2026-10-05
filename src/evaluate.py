"""
evaluate.py — Calcul des métriques et visualisation des résultats.

Métriques calculées pour chaque modèle :
  - Accuracy  : taux de bonne classification global
  - Precision : parmi les prédictions positives, combien sont correctes ?
  - Recall    : parmi les vrais positifs, combien sont détectés ?
  - F1-score  : moyenne harmonique de Precision et Recall
  - RMSE      : racine de l'erreur quadratique moyenne sur les labels

Pourquoi ces métriques sur des données déséquilibrées ?
L'accuracy seule est trompeuse : un modèle qui prédit toujours la classe 0
atteindrait 92.9% d'accuracy sans rien apprendre.
Le F1-score (pondéré/macro) est plus représentatif du vrai comportement.
Le RMSE est inhabituel en classification mais permet de mesurer l'amplitude
des erreurs de prédiction de classe (utile pour le rapport comparatif).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from src.config import FIGURES_DIR, RESULTS_DIR


# ═══════════════════════════════════════════════════════════════════════
# CALCUL DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════

def calculer_metriques(y_true: np.ndarray, y_pred: np.ndarray,
                       nom_modele: str = "modele") -> dict:
    """
    Calcule toutes les métriques de classification + RMSE.

    Paramètres average='weighted' pour Precision/Recall/F1 :
    Chaque classe contribue proportionnellement à son effectif.
    C'est le choix le plus adapté aux classes déséquilibrées.

    Le RMSE est calculé sur les indices de classes (0, 2, 3)
    traités comme des valeurs numériques ordonnées.
    """
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    rmse      = float(np.sqrt(np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2)))

    metriques = {
        "modele"    : nom_modele,
        "accuracy"  : round(acc,       4),
        "precision" : round(precision,  4),
        "recall"    : round(recall,     4),
        "f1_score"  : round(f1,         4),
        "rmse"      : round(rmse,       4),
    }

    print(f"\n[METRIQUES] {nom_modele}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {precision:.4f}  (weighted)")
    print(f"   Recall    : {recall:.4f}  (weighted)")
    print(f"   F1-score  : {f1:.4f}  (weighted)")
    print(f"   RMSE      : {rmse:.4f}")
    print(f"\n[RAPPORT COMPLET]\n{classification_report(y_true, y_pred, zero_division=0)}")

    return metriques


# ═══════════════════════════════════════════════════════════════════════
# MATRICE DE CONFUSION
# ═══════════════════════════════════════════════════════════════════════

def tracer_matrice_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                             nom_modele: str, nom_fichier: str) -> None:
    """
    Trace et sauvegarde la matrice de confusion normalisée et brute.

    Pourquoi deux versions (brute + normalisée) ?
    - Brute      : montre les effectifs réels → utile pour voir les classes rares
    - Normalisée : montre les taux par ligne (recall par classe) → comparable
                   entre classes même avec des effectifs très différents
    """
    classes      = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels_noms  = {0: "Végétation\n(0)", 2: "Volcan\n(2)", 3: "Autre\n(3)"}
    noms_classes = [labels_noms.get(c, str(c)) for c in classes]

    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Matrice de confusion — {nom_modele}", fontsize=13, fontweight="bold")

    for ax, data, titre, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Valeurs brutes", "Normalisée (recall par classe)"],
        ["d", ".2f"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=noms_classes, yticklabels=noms_classes,
            linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(titre, fontsize=11)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    chemin = os.path.join(FIGURES_DIR, nom_fichier)
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


# ═══════════════════════════════════════════════════════════════════════
# COURBES D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════

def tracer_courbes_entrainement(history, nom_modele: str, nom_fichier: str) -> None:
    """
    Trace les courbes de perte (loss) et d'accuracy pendant l'entraînement.

    Pourquoi ces courbes ?
    Elles permettent de détecter :
    - Sur-apprentissage (overfitting) : train loss ↘, val loss ↗
    - Sous-apprentissage (underfitting) : les deux restent élevées
    - Convergence correcte : les deux descendent et se stabilisent
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Courbes d'entraînement — {nom_modele}", fontsize=13, fontweight="bold")

    # Perte
    axes[0].plot(history.history["loss"],     label="Train",      color="#e74c3c")
    axes[0].plot(history.history["val_loss"], label="Validation", color="#3498db", linestyle="--")
    axes[0].set_title("Perte (Loss)")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train",      color="#e74c3c")
    axes[1].plot(history.history["val_accuracy"], label="Validation", color="#3498db", linestyle="--")
    axes[1].set_title("Précision (Accuracy)")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    chemin = os.path.join(FIGURES_DIR, nom_fichier)
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


# ═══════════════════════════════════════════════════════════════════════
# SAUVEGARDE DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════

def sauvegarder_metriques(liste_metriques: list, nom_fichier: str = "metriques.csv") -> None:
    """
    Sauvegarde un tableau de métriques en CSV pour le rapport.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(liste_metriques)
    chemin = os.path.join(RESULTS_DIR, nom_fichier)
    df.to_csv(chemin, index=False)
    print(f"[INFO] Métriques sauvegardées : {chemin}")
    return df
