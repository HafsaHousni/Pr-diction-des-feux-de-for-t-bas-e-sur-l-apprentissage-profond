"""
config.py — Centralisation de tous les chemins et hyperparamètres du projet.

Pourquoi ce fichier ?
Regrouper les constantes en un seul endroit permet de modifier facilement
les chemins ou les paramètres sans toucher au code métier.
"""

import os

# ─── Racine du projet ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Chemins des données ─────────────────────────────────────────────────────
DATA_DIR     = os.path.join(BASE_DIR, "data")
RAW_CSV      = os.path.join(BASE_DIR, "modis_2020_Turkey (1).csv")  # fichier source

# ─── Chemins des sorties ─────────────────────────────────────────────────────
FIGURES_DIR  = os.path.join(BASE_DIR, "outputs", "figures")
MODELS_DIR   = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "outputs", "results")

# ─── Colonne cible ───────────────────────────────────────────────────────────
TARGET_COL   = "type"

# ─── Colonnes à supprimer (constantes, sans valeur discriminante) ─────────────
COLS_TO_DROP = ["instrument", "version"]

# ─── Paramètres du split train/test ──────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ─── Hyperparamètres d'entraînement ──────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 50
LEARNING_RATE = 1e-3
