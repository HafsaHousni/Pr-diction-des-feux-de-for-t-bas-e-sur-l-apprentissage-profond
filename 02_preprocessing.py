"""
02_preprocessing.py — Script exécutable de prétraitement.

Ce script orchestre toute la chaîne de prétraitement et produit
deux versions des données :
  - Version A : sans SMOTE  (déséquilibre géré par class_weight)
  - Version B : avec SMOTE  (déséquilibre géré par sur-échantillonnage)

Les deux seront utilisées pour entraîner les modèles CNN en parallèle,
ce qui permettra de comparer l'impact de la stratégie anti-déséquilibre.

Exécution :
    python 02_preprocessing.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    pipeline_class_weight,
    pipeline_smote,
    sauvegarder_donnees,
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ÉTAPE 2 — PRÉTRAITEMENT DES DONNÉES")
    print("="*60)

    # ── Pipeline A : class_weight (sans modification des données) ──
    print("\n--- Pipeline A : class_weight (sans SMOTE) ---")
    X_train_cw, X_test_cw, y_train_cw, y_test, class_weights, feature_names, scaler_cw = \
        pipeline_class_weight()

    sauvegarder_donnees(
        X_train_cw, X_test_cw, y_train_cw, y_test,
        feature_names, suffix="_cw"
    )

    print(f"\n[RESUME] Pipeline class_weight :")
    print(f"   X_train : {X_train_cw.shape} | X_test : {X_test_cw.shape}")
    print(f"   Poids de classes : {class_weights}")
    print(f"   Features ({len(feature_names)}) : {feature_names}")

    # ── Pipeline B : SMOTE (rééquilibrage par sur-échantillonnage) ──
    print("\n--- Pipeline B : SMOTE ---")
    X_train_sm, X_test_sm, y_train_sm, y_test_sm, feature_names_sm, scaler_sm = \
        pipeline_smote()

    sauvegarder_donnees(
        X_train_sm, X_test_sm, y_train_sm, y_test_sm,
        feature_names_sm, suffix="_smote"
    )

    print(f"\n[RESUME] Pipeline SMOTE :")
    print(f"   X_train : {X_train_sm.shape} | X_test : {X_test_sm.shape}")

    # ── Résumé final ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PRÉTRAITEMENT TERMINÉ")
    print("="*60)
    print("\nFichiers générés dans outputs/results/ :")
    print("   X_train_cw.npy / X_test_cw.npy / y_train_cw.npy / y_test_cw.npy")
    print("   X_train_smote.npy / X_test_smote.npy / y_train_smote.npy / y_test_smote.npy")
    print("   feature_names.csv")
    print("   scaler.pkl")
    print("\nFigure générée dans outputs/figures/ :")
    print("   07_smote_comparison.png")
