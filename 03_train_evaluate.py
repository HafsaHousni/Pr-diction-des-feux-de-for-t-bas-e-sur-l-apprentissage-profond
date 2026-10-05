"""
03_train_evaluate.py — Entraînement et évaluation des deux CNN.

Ce script :
  1. Charge les données prétraitées (sorties du script 02)
  2. Applique le reshape (n, 12) → (n, 12, 1) pour les Conv1D
  3. Affiche les architectures des deux modèles
  4. Entraîne AlexNet-1D  (avec class_weight — données non augmentées)
  5. Entraîne ResNet-1D   (avec SMOTE — données augmentées)
  6. Évalue chaque modèle : métriques + matrice de confusion + courbes

Exécution :
    python 03_train_evaluate.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.config   import RESULTS_DIR, MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.models   import construire_alexnet_1d, construire_resnet_1d, reshape_pour_cnn, afficher_resume_modeles
from src.evaluate import (
    calculer_metriques,
    tracer_matrice_confusion,
    tracer_courbes_entrainement,
    sauvegarder_metriques,
)

os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ═══════════════════════════════════════════════════════════════════════

def charger_donnees(suffix: str):
    """
    Charge les tableaux numpy produits par 02_preprocessing.py.
    suffix = '_cw' (class_weight) ou '_smote'
    """
    X_train = np.load(os.path.join(RESULTS_DIR, f"X_train{suffix}.npy"))
    X_test  = np.load(os.path.join(RESULTS_DIR, f"X_test{suffix}.npy"))
    y_train = np.load(os.path.join(RESULTS_DIR, f"y_train{suffix}.npy"))
    y_test  = np.load(os.path.join(RESULTS_DIR, f"y_test{suffix}.npy"))
    print(f"[INFO] Données '{suffix}' chargées — train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════
# CALLBACKS D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════

def construire_callbacks(nom_modele: str):
    """
    Configure les callbacks Keras pour un entraînement robuste.

    - EarlyStopping : arrête l'entraînement si val_loss ne s'améliore
      plus pendant 'patience' époques → évite le sur-apprentissage
      et réduit le temps de calcul.
      restore_best_weights=True : restaure les poids de la meilleure époque.

    - ReduceLROnPlateau : divise le taux d'apprentissage par 'factor'
      si val_loss stagne → permet de sortir des plateaux de convergence.

    - ModelCheckpoint : sauvegarde le meilleur modèle automatiquement
      (basé sur val_loss minimal) → permet de réutiliser le modèle
      sans ré-entraîner.
    """
    chemin_sauvegarde = os.path.join(MODELS_DIR, f"{nom_modele}_best.keras")
    return [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=chemin_sauvegarde,
            monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
# ENCODAGE DES LABELS POUR CLASS_WEIGHT
# ═══════════════════════════════════════════════════════════════════════

def remap_labels(y: np.ndarray) -> tuple:
    """
    Remappe les labels {0, 2, 3} → {0, 1, 2} pour Keras.

    Pourquoi ?
    Keras/TensorFlow avec sparse_categorical_crossentropy attend des
    labels contigus de 0 à n_classes-1. Les labels {0, 2, 3} sautent
    l'indice 1, ce qui provoquerait une erreur ou un comportement incorrect.

    Mapping appliqué : 0→0, 2→1, 3→2
    """
    mapping  = {0: 0, 2: 1, 3: 2}
    inv_map  = {0: 0, 1: 2, 2: 3}
    y_remapped = np.array([mapping[v] for v in y])
    return y_remapped, inv_map


# ═══════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT + ÉVALUATION D'UN MODÈLE
# ═══════════════════════════════════════════════════════════════════════

def entrainer_et_evaluer(model, X_train, y_train, X_test, y_test,
                         nom_modele: str, class_weight_dict=None):
    """
    Entraîne le modèle et calcule toutes les métriques sur le test set.

    Paramètres :
        class_weight_dict : dict {0: w0, 1: w1, 2: w2} ou None (si SMOTE)
    """
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT : {nom_modele}")
    print(f"{'='*60}")
    print(f"[INFO] X_train={X_train.shape}, X_test={X_test.shape}")

    # Remap des labels {0,2,3} → {0,1,2}
    y_train_r, inv_map = remap_labels(y_train)
    y_test_r,  _       = remap_labels(y_test)

    # Ajustement du class_weight si fourni (remap les clés aussi)
    cw_final = None
    if class_weight_dict is not None:
        mapping = {0: 0, 2: 1, 3: 2}
        cw_final = {mapping[k]: v for k, v in class_weight_dict.items()}
        print(f"[INFO] class_weight (après remap) : {cw_final}")

    callbacks = construire_callbacks(nom_modele)

    # ── Entraînement ──────────────────────────────────────────────
    history = model.fit(
        X_train, y_train_r,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_split= 0.15,          # 15% du train pour la validation
        class_weight    = cw_final,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── Prédictions ───────────────────────────────────────────────
    proba   = model.predict(X_test, verbose=0)
    y_pred_r = np.argmax(proba, axis=1)

    # Inverse remap → labels originaux {0, 2, 3}
    y_pred  = np.array([inv_map[v] for v in y_pred_r])

    # ── Métriques ─────────────────────────────────────────────────
    metriques = calculer_metriques(y_test, y_pred, nom_modele)

    # ── Figures ───────────────────────────────────────────────────
    tracer_courbes_entrainement(
        history, nom_modele,
        nom_fichier=f"08_courbes_{nom_modele.lower().replace('-','_').replace(' ','_')}.png"
    )
    tracer_matrice_confusion(
        y_test, y_pred, nom_modele,
        nom_fichier=f"09_confusion_{nom_modele.lower().replace('-','_').replace(' ','_')}.png"
    )

    return metriques, history, y_pred


# ═══════════════════════════════════════════════════════════════════════
# SCRIPT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ÉTAPES 3-4-5 : RESHAPE, ARCHITECTURES CNN, ENTRAÎNEMENT")
    print("="*60)

    N_FEATURES = 12
    N_CLASSES  = 3

    # ── Étape 3 : Affichage du reshape et des architectures ──────
    print("\n--- Étape 3 : Reshape (12,) → (12, 1) pour Conv1D ---")
    print("Les données tabulaires (n, 12) sont transformées en (n, 12, 1)")
    print("pour être traitées par des couches Conv1D Keras.")
    afficher_resume_modeles(N_FEATURES, N_CLASSES)

    # ── Chargement des données ────────────────────────────────────
    print("\n--- Chargement des données prétraitées ---")

    # AlexNet-1D utilisera class_weight (données non augmentées)
    X_train_cw, X_test_cw, y_train_cw, y_test_cw = charger_donnees("_cw")

    # ResNet-1D utilisera SMOTE (données augmentées)
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = charger_donnees("_smote")

    # ── Étape 3 : Reshape pour Conv1D ─────────────────────────────
    X_train_cw_3d = reshape_pour_cnn(X_train_cw)
    X_test_cw_3d  = reshape_pour_cnn(X_test_cw)
    X_train_sm_3d = reshape_pour_cnn(X_train_sm)
    X_test_sm_3d  = reshape_pour_cnn(X_test_sm)

    print(f"[INFO] Forme après reshape : {X_train_cw_3d.shape}  (batch, steps, channels)")

    # ── Chargement des poids de classe ────────────────────────────
    import joblib, ast
    # Reconstruit class_weight depuis les données cw
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as _np
    classes_cw   = _np.unique(y_train_cw)
    weights_vals = compute_class_weight("balanced", classes=classes_cw, y=y_train_cw)
    class_weight_dict = dict(zip(classes_cw.tolist(), weights_vals.tolist()))

    # ── Étape 4 & 5 : AlexNet-1D + class_weight ──────────────────
    alexnet = construire_alexnet_1d(N_FEATURES, N_CLASSES, LEARNING_RATE)
    metriques_alex, hist_alex, y_pred_alex = entrainer_et_evaluer(
        alexnet,
        X_train_cw_3d, y_train_cw,
        X_test_cw_3d,  y_test_cw,
        nom_modele="AlexNet-1D",
        class_weight_dict=class_weight_dict,
    )

    # ── Étape 4 & 5 : ResNet-1D + SMOTE ──────────────────────────
    resnet = construire_resnet_1d(N_FEATURES, N_CLASSES, LEARNING_RATE)
    metriques_res, hist_res, y_pred_res = entrainer_et_evaluer(
        resnet,
        X_train_sm_3d, y_train_sm,
        X_test_sm_3d,  y_test_sm,
        nom_modele="ResNet-1D",
        class_weight_dict=None,   # SMOTE a déjà rééquilibré
    )

    # ── Sauvegarde des métriques ───────────────────────────────────
    sauvegarder_metriques(
        [metriques_alex, metriques_res],
        nom_fichier="metriques_comparaison.csv"
    )

    print("\n" + "="*60)
    print("  ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print("\nFigures générées dans outputs/figures/")
    print("  08_courbes_alexnet_1d.png")
    print("  08_courbes_resnet_1d.png")
    print("  09_confusion_alexnet_1d.png")
    print("  09_confusion_resnet_1d.png")
    print("\nModèles sauvegardés dans outputs/models/")
    print("  AlexNet-1D_best.keras")
    print("  ResNet-1D_best.keras")
    print("\nMétriques sauvegardées dans outputs/results/")
    print("  metriques_comparaison.csv")
    print("\nProchaine étape : python 04_comparison.py")
