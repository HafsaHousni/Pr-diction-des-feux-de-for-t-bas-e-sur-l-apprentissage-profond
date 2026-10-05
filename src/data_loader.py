"""
data_loader.py — Chargement du dataset et analyse exploratoire (EDA).

Rôle de ce module :
Fournir des fonctions réutilisables pour charger les données brutes
et produire toutes les visualisations d'exploration nécessaires au rapport.
Chaque figure est sauvegardée en PNG dans outputs/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.config import RAW_CSV, FIGURES_DIR, TARGET_COL


def charger_donnees() -> pd.DataFrame:
    """
    Charge le fichier CSV brut et retourne un DataFrame pandas.

    Pourquoi pandas ?
    pandas est la bibliothèque standard pour manipuler des données tabulaires
    en Python : lecture efficace, affichage clair, nombreuses méthodes d'analyse.
    """
    print(f"[INFO] Chargement du fichier : {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"[INFO] Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def afficher_infos_generales(df: pd.DataFrame) -> None:
    """
    Affiche les informations structurelles du dataset :
    shape, types, valeurs manquantes, statistiques descriptives.

    Pourquoi cette étape ?
    Avant tout traitement, il est indispensable de comprendre la nature
    des données : types de colonnes, présence de valeurs nulles,
    plages de valeurs. C'est la base de tout projet de data science.
    """
    print("\n" + "="*60)
    print("INFORMATIONS GÉNÉRALES DU DATASET")
    print("="*60)

    print(f"\n[Dimensions] {df.shape[0]} lignes, {df.shape[1]} colonnes")

    print("\n[Types des colonnes et valeurs non-nulles]")
    print(df.info())

    print("\n[Statistiques descriptives (colonnes numeriques)]")
    print(df.describe().round(3).to_string())

    print("\n[Valeurs manquantes par colonne]")
    missing = df.isnull().sum()
    print(missing[missing >= 0].to_string())

    print("\n[Valeurs uniques par colonne]")
    for col in df.columns:
        n = df[col].nunique()
        print(f"   {col:<15} : {n} valeurs uniques")


def tracer_distribution_cible(df: pd.DataFrame) -> None:
    """
    Trace et sauvegarde la distribution de la variable cible 'type'.

    Pourquoi visualiser la cible ?
    Le déséquilibre des classes est un problème majeur en classification.
    Le visualiser permet de justifier les techniques de rééquilibrage
    qui seront appliquées dans le prétraitement (SMOTE, class_weight).

    Classes :
        0 → Incendie végétation (Vegetation fire)
        2 → Volcan (Volcano)
        3 → Autre feu statique (Other static land source)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    counts = df[TARGET_COL].value_counts().sort_index()
    labels_map = {0: "Végétation\n(0)", 2: "Volcan\n(2)", 3: "Autre\n(3)"}
    labels = [labels_map.get(i, str(i)) for i in counts.index]
    colors = ["#e74c3c", "#f39c12", "#3498db"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution de la variable cible 'type'", fontsize=14, fontweight="bold")

    # Graphique en barres
    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Nombre d'occurrences par classe")
    axes[0].set_ylabel("Nombre d'échantillons")
    axes[0].set_xlabel("Classe")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    # Graphique en camembert
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    axes[1].set_title("Répartition proportionnelle des classes")

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "01_distribution_cible.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


def tracer_distributions_features(df: pd.DataFrame) -> None:
    """
    Trace les histogrammes de toutes les colonnes numériques.

    Pourquoi ?
    Visualiser la distribution de chaque feature permet de détecter
    des asymétries (skewness), des valeurs aberrantes, ou des
    distributions bimodales qui peuvent affecter l'apprentissage.
    """
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_num = [c for c in cols_num if c != TARGET_COL]

    n = len(cols_num)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    fig.suptitle("Distribution des variables numériques", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(cols_num):
        axes[i].hist(df[col].dropna(), bins=40, color="#2980b9", edgecolor="white", linewidth=0.5)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Valeur")
        axes[i].set_ylabel("Fréquence")
        axes[i].xaxis.set_major_formatter(ticker.ScalarFormatter())

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "02_distributions_features.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


def tracer_matrice_correlation(df: pd.DataFrame) -> None:
    """
    Calcule et trace la matrice de corrélation (Pearson) entre variables numériques.

    Pourquoi la corrélation ?
    La corrélation de Pearson mesure la relation linéaire entre deux variables.
    Une corrélation élevée entre deux features (multicolinéarité) peut réduire
    l'apport informatif d'une variable et dégrader certains modèles.
    Elle aide aussi à identifier les features les plus liées à la cible.
    """
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols_num].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # masque triangle supérieur

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Matrice de corrélation (Pearson) — variables numériques",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "03_matrice_correlation.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


def tracer_boxplots_par_classe(df: pd.DataFrame) -> None:
    """
    Trace des boîtes à moustaches (boxplots) pour les principales features
    groupées par classe cible.

    Pourquoi les boxplots par classe ?
    Ils permettent de visualiser si une feature permet de discriminer
    les classes, ce qui est un indicateur préliminaire de son pouvoir
    prédictif. Une feature dont les distributions se chevauchent
    fortement entre classes apporte peu d'information.
    """
    features_clés = ["brightness", "frp", "bright_t31", "scan", "track"]
    features_clés = [f for f in features_clés if f in df.columns]

    n = len(features_clés)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle("Boxplots des features principales par classe cible",
                 fontsize=13, fontweight="bold")

    palette = {0: "#e74c3c", 2: "#f39c12", 3: "#3498db"}

    for i, feat in enumerate(features_clés):
        ax = axes[i] if n > 1 else axes
        data_plot = [df.loc[df[TARGET_COL] == cls, feat].dropna().values
                     for cls in sorted(df[TARGET_COL].unique())]
        bp = ax.boxplot(data_plot, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        colors_list = [palette.get(c, "#95a5a6") for c in sorted(df[TARGET_COL].unique())]
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=11)
        ax.set_xticklabels([f"Cl.{c}" for c in sorted(df[TARGET_COL].unique())])
        ax.set_ylabel("Valeur")

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "04_boxplots_par_classe.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


def tracer_repartition_temporelle(df: pd.DataFrame) -> None:
    """
    Trace la répartition des détections d'incendies par mois.

    Pourquoi l'analyse temporelle ?
    Les feux de forêt ont une forte saisonnalité. Visualiser la distribution
    temporelle permet de vérifier si la variable 'acq_date' porte de
    l'information, et donc si en extraire le mois est pertinent.
    """
    df_temp = df.copy()
    df_temp["acq_date"] = pd.to_datetime(df_temp["acq_date"])
    df_temp["mois"] = df_temp["acq_date"].dt.month

    mois_noms = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                 "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

    fig, ax = plt.subplots(figsize=(10, 5))
    counts_mois = df_temp["mois"].value_counts().sort_index()
    ax.bar(counts_mois.index, counts_mois.values, color="#27ae60", edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(mois_noms)
    ax.set_title("Répartition mensuelle des détections (2020 — Turquie)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de détections")
    for x, y in zip(counts_mois.index, counts_mois.values):
        ax.text(x, y + 10, str(y), ha="center", fontsize=8)

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "05_repartition_mensuelle.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")


def tracer_carte_geographique(df: pd.DataFrame) -> None:
    """
    Trace la carte géographique des points de détection colorés par classe.

    Pourquoi la visualisation spatiale ?
    Le dataset contient des coordonnées GPS (latitude/longitude).
    Visualiser la répartition géographique permet de vérifier la cohérence
    des données (zones attendues en Turquie) et d'identifier si la
    localisation joue un rôle discriminant entre les types de feux.
    """
    palette = {0: "#e74c3c", 2: "#f39c12", 3: "#3498db"}
    labels_map = {0: "Végétation (0)", 2: "Volcan (2)", 3: "Autre (3)"}

    fig, ax = plt.subplots(figsize=(10, 7))

    for cls in sorted(df[TARGET_COL].unique()):
        subset = df[df[TARGET_COL] == cls]
        ax.scatter(
            subset["longitude"], subset["latitude"],
            c=palette.get(cls, "gray"),
            label=labels_map.get(cls, str(cls)),
            s=10, alpha=0.6, linewidths=0
        )

    ax.set_title("Répartition géographique des détections — Turquie 2020",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Type", markerscale=2)

    plt.tight_layout()
    chemin = os.path.join(FIGURES_DIR, "06_carte_geographique.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Sauvegardée : {chemin}")
