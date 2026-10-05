"""
preprocessing.py — Prétraitement complet du dataset MODIS.

Rôle de ce module :
Transformer les données brutes en données prêtes pour l'entraînement
des modèles de deep learning. Cette étape est critique : des données
mal préparées produisent systématiquement de mauvais modèles,
même avec les architectures les plus sophistiquées.

Etapes réalisées :
    1. Suppression des colonnes constantes (sans pouvoir prédictif)
    2. Encodage des variables catégorielles
    3. Feature engineering (extraction du mois, normalisation de l'heure)
    4. Gestion du déséquilibre des classes (class_weight ET SMOTE)
    5. Normalisation des features (StandardScaler)
    6. Split stratifié train/test 80/20
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from src.config import (
    RAW_CSV, FIGURES_DIR, RESULTS_DIR,
    TARGET_COL, COLS_TO_DROP,
    TEST_SIZE, RANDOM_STATE,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET NETTOYAGE
# ═══════════════════════════════════════════════════════════════════════

def charger_et_nettoyer() -> pd.DataFrame:
    """
    Charge le CSV et supprime les colonnes sans pouvoir prédictif.

    Pourquoi supprimer 'instrument' et 'version' ?
    Ces deux colonnes ont une variance nulle (une seule valeur unique
    dans tout le dataset). Un modèle ne peut pas apprendre à partir
    d'une variable constante : elle n'apporte aucune information
    discriminante entre les classes.
    """
    df = pd.read_csv(RAW_CSV)
    print(f"[INFO] Dataset chargé : {df.shape}")

    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    print(f"[INFO] Après suppression de {COLS_TO_DROP} : {df.shape}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. ENCODAGE DES VARIABLES CATÉGORIELLES
# ═══════════════════════════════════════════════════════════════════════

def encoder_categoriques(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles en valeurs numériques.

    Pourquoi encoder ?
    Les réseaux de neurones (et la plupart des algorithmes ML) ne peuvent
    pas traiter directement des chaînes de caractères. Il faut les
    convertir en représentations numériques.

    Stratégie choisie : encodage ordinal binaire (Label Encoding)
    - 'satellite'  : 'A' -> 0, 'B' -> 1
      (seulement 2 satellites TERRA/AQUA, donc 1 bit suffit)
    - 'daynight'   : 'D' -> 0, 'N' -> 1
      (variable binaire naturelle)

    Note : on évite le One-Hot Encoding ici car les deux variables
    sont strictement binaires, ce qui ne crée pas de biais ordinal.
    """
    df = df.copy()

    # Encodage du satellite (Terra=A -> 0, Aqua=B -> 1)
    satellite_map = {val: idx for idx, val in enumerate(sorted(df["satellite"].unique()))}
    df["satellite"] = df["satellite"].map(satellite_map)
    print(f"[INFO] satellite encodé : {satellite_map}")

    # Encodage jour/nuit (Day -> 0, Night -> 1)
    daynight_map = {val: idx for idx, val in enumerate(sorted(df["daynight"].unique()))}
    df["daynight"] = df["daynight"].map(daynight_map)
    print(f"[INFO] daynight encodé : {daynight_map}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING TEMPOREL
# ═══════════════════════════════════════════════════════════════════════

def feature_engineering_temporel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait de nouvelles features à partir des colonnes temporelles.

    Pourquoi le feature engineering sur les dates ?
    Les dates brutes (ex: '2020-07-15') ne sont pas directement
    exploitables par un réseau de neurones. En revanche, le mois
    encode la saisonnalité, information clé pour les feux de forêt
    (été = risque maximal en Turquie). On extrait donc des features
    numériques interprétables.

    Features créées :
    - 'mois'       : mois de la détection (1 à 12), encode la saisonnalité
    - 'heure'      : heure locale approchée (acq_time / 100), simplifie
                     l'heure HHMM en valeur continue 0-23

    La colonne 'acq_date' originale est ensuite supprimée car elle
    est de type chaîne de caractères et sa granularité (jour exact)
    serait trop spécifique (risque de sur-apprentissage).
    """
    df = df.copy()

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["mois"]     = df["acq_date"].dt.month          # saisonnalité (1-12)
    df["heure"]    = (df["acq_time"] // 100).astype(int)  # heure locale (0-23)

    # Supprimer les colonnes brutes remplacées
    df = df.drop(columns=["acq_date", "acq_time"])
    print("[INFO] Features temporelles créées : 'mois', 'heure'")
    print(f"[INFO] Colonnes finales ({df.shape[1]}) : {list(df.columns)}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. GESTION DU DÉSÉQUILIBRE DES CLASSES
# ═══════════════════════════════════════════════════════════════════════

def calculer_class_weights(y: np.ndarray) -> dict:
    """
    Calcule les poids de classe pour compenser le déséquilibre.

    ─── APPROCHE 1 : class_weight ──────────────────────────────────────
    Principe :
    On attribue un poids inversement proportionnel à la fréquence de
    chaque classe. La fonction de perte est multipliée par ce poids lors
    de l'entraînement : une erreur sur une classe rare 'compte plus'.

    Formule : w_i = N_total / (N_classes * N_i)
        - N_total = nombre total d'exemples
        - N_classes = nombre de classes
        - N_i = nombre d'exemples de la classe i

    Avantages :
    + Simple à mettre en oeuvre (1 paramètre dans model.fit)
    + Ne modifie pas les données d'entraînement
    + Pas de risque de sur-apprentissage sur des exemples synthétiques

    Inconvénients :
    - Peut instabiliser l'entraînement si les déséquilibres sont extrêmes
    - N'apporte pas de nouvelles informations (pas de nouveaux exemples)
    ─────────────────────────────────────────────────────────────────────
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weight_dict = dict(zip(classes, weights))
    print(f"[INFO] class_weight calcule : {class_weight_dict}")
    return class_weight_dict


def appliquer_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Applique SMOTE pour rééquilibrer les classes par sur-échantillonnage.

    ─── APPROCHE 2 : SMOTE ─────────────────────────────────────────────
    SMOTE = Synthetic Minority Over-sampling TEchnique (Chawla et al., 2002)

    Principe :
    Pour chaque exemple d'une classe minoritaire, SMOTE :
    1. Trouve ses k plus proches voisins (dans l'espace des features)
    2. Crée un nouvel exemple SYNTHÉTIQUE sur le segment reliant
       l'exemple à l'un de ses voisins (interpolation linéaire)

    Exemple :
        x_nouveau = x_i + lambda * (x_voisin - x_i),  lambda ~ Uniform(0,1)

    SMOTE est appliqué UNIQUEMENT sur l'ensemble d'entraînement
    (jamais sur le test set) pour éviter la fuite de données (data leakage).

    Avantages :
    + Crée de nouveaux exemples variés (pas de simple duplication)
    + Améliore la frontière de décision pour les classes rares
    + Compatible avec tout algorithme d'apprentissage

    Inconvénients :
    - Peut générer des exemples dans des zones peu réalistes de l'espace
    - Augmente la taille du dataset (temps d'entraînement plus long)
    - Risque de sur-apprentissage si k est trop petit
    ─────────────────────────────────────────────────────────────────────

    Paramètre k_neighbors=2 (au lieu de 5 par défaut) car la classe 3
    ne contient que 7 exemples dans le train set (~5-6 après split).
    Avec k=5, SMOTE échouerait car il n'y aurait pas assez de voisins.
    """
    print("\n[INFO] Distribution AVANT SMOTE :")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   Classe {cls} : {cnt} exemples")

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=2)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("[INFO] Distribution APRES SMOTE :")
    unique, counts = np.unique(y_res, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   Classe {cls} : {cnt} exemples")

    return X_res, y_res


def tracer_comparaison_smote(y_before: np.ndarray, y_after: np.ndarray) -> None:
    """
    Visualise la distribution des classes avant et après SMOTE.

    Pourquoi cette figure ?
    Elle illustre concrètement l'effet du rééquilibrage et permet
    de vérifier que les trois classes sont bien équilibrées après SMOTE.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Effet de SMOTE sur la distribution des classes", fontsize=13, fontweight="bold")

    colors = ["#e74c3c", "#f39c12", "#3498db"]

    for ax, y, title in zip(axes, [y_before, y_after], ["Avant SMOTE", "Après SMOTE"]):
        unique, counts = np.unique(y, return_counts=True)
        bars = ax.bar([str(c) for c in unique], counts,
                      color=[colors[i % len(colors)] for i in range(len(unique))],
                      edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Classe")
        ax.set_ylabel("Nombre d'exemples")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(cnt), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "07_smote_comparison.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


# ═══════════════════════════════════════════════════════════════════════
# 5. NORMALISATION
# ═══════════════════════════════════════════════════════════════════════

def normaliser(X_train: np.ndarray, X_test: np.ndarray):
    """
    Normalise les features avec StandardScaler (z-score).

    Pourquoi normaliser ?
    Les features du dataset ont des échelles très différentes :
    - 'brightness' : ~300-500 K
    - 'confidence' : 0-100
    - 'frp'        : 2-2787 MW/sr/m²
    - 'mois'       : 1-12

    Sans normalisation, les features à grande échelle dominent
    le gradient et les couches de neurones convergent mal.

    Formule StandardScaler : x' = (x - mu) / sigma
        - mu : moyenne calculée SUR LE TRAIN SET
        - sigma : écart-type calculé SUR LE TRAIN SET

    IMPORTANT : le scaler est ajusté (fit) uniquement sur X_train,
    puis appliqué (transform) sur X_train ET X_test.
    Ajuster sur X_test introduirait une fuite de données (data leakage).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[INFO] Normalisation appliquée — mean[0]={scaler.mean_[0]:.3f}, std[0]={scaler.scale_[0]:.3f}")

    # Sauvegarde du scaler pour réutilisation en inférence
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chemin_scaler = os.path.join(RESULTS_DIR, "scaler.pkl")
    joblib.dump(scaler, chemin_scaler)
    print(f"[INFO] Scaler sauvegardé : {chemin_scaler}")

    return X_train_scaled, X_test_scaled, scaler


# ═══════════════════════════════════════════════════════════════════════
# 6. SPLIT TRAIN / TEST
# ═══════════════════════════════════════════════════════════════════════

def split_train_test(df: pd.DataFrame):
    """
    Sépare le dataset en ensembles d'entraînement (80%) et de test (20%).

    Pourquoi un split stratifié ?
    Un split aléatoire standard ne garantit pas que les proportions
    de chaque classe sont respectées dans train et test.
    Avec stratify=y, scikit-learn s'assure que chaque sous-ensemble
    conserve les mêmes proportions que le dataset complet.
    C'est indispensable avec des données déséquilibrées comme ici.
    """
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # preserve la distribution des classes
    )

    print(f"[INFO] Split : X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"[INFO] Noms des features ({X_train.shape[1]}) : {list(df.drop(columns=[TARGET_COL]).columns)}")

    return X_train, X_test, y_train, y_test, list(df.drop(columns=[TARGET_COL]).columns)


# ═══════════════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLET — SANS SMOTE (class_weight)
# ═══════════════════════════════════════════════════════════════════════

def pipeline_class_weight():
    """
    Exécute le pipeline complet de prétraitement SANS SMOTE.
    Le déséquilibre sera géré via class_weight lors de l'entraînement.

    Retourne :
        X_train_s, X_test_s : features normalisées
        y_train, y_test     : labels
        class_weights       : dict {classe: poids}
        feature_names       : liste des noms de colonnes
        scaler              : objet StandardScaler ajusté
    """
    df = charger_et_nettoyer()
    df = encoder_categoriques(df)
    df = feature_engineering_temporel(df)
    X_train, X_test, y_train, y_test, feature_names = split_train_test(df)
    X_train_s, X_test_s, scaler = normaliser(X_train, X_test)
    class_weights = calculer_class_weights(y_train)

    return X_train_s, X_test_s, y_train, y_test, class_weights, feature_names, scaler


# ═══════════════════════════════════════════════════════════════════════
# 8. PIPELINE COMPLET — AVEC SMOTE
# ═══════════════════════════════════════════════════════════════════════

def pipeline_smote():
    """
    Exécute le pipeline complet de prétraitement AVEC SMOTE.
    SMOTE est appliqué APRÈS le split pour éviter le data leakage,
    et AVANT la normalisation (certains auteurs le font après,
    mais avant est plus courant et évite les interactions entre
    les exemples synthétiques et le scaler).

    Ordre : split -> SMOTE sur train -> normalisation

    Retourne :
        X_train_s, X_test_s : features normalisées
        y_train_s           : labels après SMOTE (train uniquement)
        y_test              : labels test (inchangés)
        feature_names       : liste des noms de colonnes
        scaler              : objet StandardScaler ajusté
    """
    df = charger_et_nettoyer()
    df = encoder_categoriques(df)
    df = feature_engineering_temporel(df)
    X_train, X_test, y_train, y_test, feature_names = split_train_test(df)

    # SMOTE appliqué sur le train AVANT normalisation
    y_train_before_smote = y_train.copy()
    X_train_s_raw, y_train_s = appliquer_smote(X_train, y_train)

    # Normalisation APRÈS SMOTE (fit sur le train augmenté)
    X_train_s, X_test_s, scaler = normaliser(X_train_s_raw, X_test)

    # Visualisation de l'effet SMOTE
    tracer_comparaison_smote(y_train_before_smote, y_train_s)

    return X_train_s, X_test_s, y_train_s, y_test, feature_names, scaler


# ═══════════════════════════════════════════════════════════════════════
# 9. SAUVEGARDE DES DONNÉES PRÉTRAITÉES
# ═══════════════════════════════════════════════════════════════════════

def sauvegarder_donnees(X_train, X_test, y_train, y_test,
                        feature_names, suffix: str = "") -> None:
    """
    Sauvegarde les données prétraitées en format numpy pour réutilisation.

    Pourquoi sauvegarder ?
    Le prétraitement (surtout SMOTE) est coûteux en calcul.
    Sauvegarder les données permet de relancer les scripts
    d'entraînement sans refaire toute la chaîne de prétraitement.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.save(os.path.join(RESULTS_DIR, f"X_train{suffix}.npy"), X_train)
    np.save(os.path.join(RESULTS_DIR, f"X_test{suffix}.npy"),  X_test)
    np.save(os.path.join(RESULTS_DIR, f"y_train{suffix}.npy"), y_train)
    np.save(os.path.join(RESULTS_DIR, f"y_test{suffix}.npy"),  y_test)

    pd.Series(feature_names).to_csv(
        os.path.join(RESULTS_DIR, "feature_names.csv"), index=False, header=["feature"]
    )
    print(f"[INFO] Données sauvegardées dans outputs/results/ (suffixe='{suffix}')")
