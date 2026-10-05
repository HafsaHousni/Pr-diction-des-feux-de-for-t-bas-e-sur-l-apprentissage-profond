"""
01_eda.py — Script principal d'Analyse Exploratoire des Données (EDA).

Objectif :
Comprendre la structure, la qualité et les caractéristiques statistiques
du dataset MODIS avant tout traitement. Cette étape est incontournable
dans tout projet de machine learning : elle guide les choix de prétraitement
et permet de détecter des anomalies tôt.

Exécution :
    python 01_eda.py
"""

import sys
import os

# Permet d'importer les modules du dossier src/ depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    charger_donnees,
    afficher_infos_generales,
    tracer_distribution_cible,
    tracer_distributions_features,
    tracer_matrice_correlation,
    tracer_boxplots_par_classe,
    tracer_repartition_temporelle,
    tracer_carte_geographique,
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ÉTAPE 1 — ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("="*60)

    # ── 1. Chargement ────────────────────────────────────────────
    # On charge le CSV brut en DataFrame pandas.
    df = charger_donnees()

    # ── 2. Informations générales ─────────────────────────────────
    # Affiche shape, types, valeurs manquantes et statistiques descriptives.
    afficher_infos_generales(df)

    # ── 3. Distribution de la variable cible ─────────────────────
    # Visualise le déséquilibre entre les classes (barres + camembert).
    tracer_distribution_cible(df)

    # ── 4. Distributions des features numériques ──────────────────
    # Histogrammes pour détecter asymétries et valeurs aberrantes.
    tracer_distributions_features(df)

    # ── 5. Matrice de corrélation ─────────────────────────────────
    # Heatmap Pearson pour repérer redondances et liens avec la cible.
    tracer_matrice_correlation(df)

    # ── 6. Boxplots par classe ────────────────────────────────────
    # Pouvoir discriminant des features principales selon le type.
    tracer_boxplots_par_classe(df)

    # ── 7. Répartition temporelle ─────────────────────────────────
    # Saisonnalité des détections pour justifier l'extraction du mois.
    tracer_repartition_temporelle(df)

    # ── 8. Carte géographique ─────────────────────────────────────
    # Distribution spatiale des points pour valider la cohérence du dataset.
    tracer_carte_geographique(df)

    print("\n" + "="*60)
    print("  EDA TERMINÉE — Figures sauvegardées dans outputs/figures/")
    print("="*60)
