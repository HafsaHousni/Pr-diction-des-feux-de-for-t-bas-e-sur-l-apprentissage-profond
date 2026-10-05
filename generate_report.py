"""
generate_report.py — Génération du rapport académique final en Word (.docx).

Ce script produit un document Word structuré contenant :
  - Page de titre
  - Résumé (abstract)
  - Introduction et problématique
  - Description du dataset
  - Analyse exploratoire (avec figures)
  - Prétraitement
  - Architectures CNN
  - Résultats et évaluation (avec figures, tableaux)
  - Discussion et conclusion
  - Références bibliographiques

Exécution :
    python generate_report.py
"""

import os
import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "results")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Rapport_PFA_Prediction_Feux_Foret.docx")


# ═══════════════════════════════════════════════════════════════════════
# UTILITAIRES DE MISE EN FORME
# ═══════════════════════════════════════════════════════════════════════

def set_cell_bg(cell, hex_color: str):
    """Colore le fond d'une cellule de tableau."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_cell_border(table):
    """Ajoute des bordures fines à toutes les cellules du tableau."""
    for row in table.rows:
        for cell in row.cells:
            tc   = cell._tc
            tcPr = tc.get_or_add_tcPr()
            for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
                border = OxmlElement(f"w:{side}")
                border.set(qn("w:val"), "single")
                border.set(qn("w:sz"), "4")
                border.set(qn("w:space"), "0")
                border.set(qn("w:color"), "AAAAAA")
                tcBorders = tcPr.find(qn("w:tcBorders"))
                if tcBorders is None:
                    tcBorders = OxmlElement("w:tcBorders")
                    tcPr.append(tcBorders)
                tcBorders.append(border)


def titre_section(doc, texte: str, niveau: int = 1):
    """Ajoute un titre de section stylisé."""
    heading = doc.add_heading(texte, level=niveau)
    run = heading.runs[0] if heading.runs else heading.add_run(texte)
    if niveau == 1:
        run.font.color.rgb = RGBColor(0x1a, 0x53, 0x7a)
        run.font.size = Pt(16)
    elif niveau == 2:
        run.font.color.rgb = RGBColor(0x2e, 0x7d, 0xa8)
        run.font.size = Pt(13)
    elif niveau == 3:
        run.font.color.rgb = RGBColor(0x37, 0x6e, 0x8a)
        run.font.size = Pt(11)


def paragraphe(doc, texte: str, indent: bool = False, gras: bool = False,
               italic: bool = False, taille: int = 11, espace_avant: int = 0):
    """Ajoute un paragraphe de texte formaté."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.space_before = Pt(espace_avant)
    if indent:
        p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(texte)
    run.font.size  = Pt(taille)
    run.font.bold  = gras
    run.font.italic = italic
    run.font.name  = "Calibri"
    return p


def ajouter_figure(doc, nom_fichier: str, legende: str, largeur: float = 14.0):
    """Insère une figure avec sa légende centrée."""
    chemin = os.path.join(FIGURES_DIR, nom_fichier)
    if not os.path.exists(chemin):
        paragraphe(doc, f"[Figure non trouvée : {nom_fichier}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(chemin, width=Cm(largeur))
    p_leg = doc.add_paragraph(legende)
    p_leg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_leg.paragraph_format.space_after = Pt(10)
    run_leg = p_leg.runs[0]
    run_leg.font.size   = Pt(9)
    run_leg.font.italic = True
    run_leg.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def ajouter_tableau(doc, headers: list, rows: list, col_widths: list = None,
                    header_color: str = "1A537A", alt_color: str = "EBF3F8"):
    """Crée un tableau formaté avec en-tête coloré."""
    n_cols = len(headers)
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style     = "Table Grid"

    # En-tête
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_bg(hdr_cells[i], header_color)
        run = hdr_cells[i].paragraphs[0].runs[0]
        run.font.bold  = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size  = Pt(10)
        hdr_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Données
    for ri, row_data in enumerate(rows):
        row_cells = table.rows[ri + 1].cells
        bg = alt_color if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row_data):
            row_cells[ci].text = str(val)
            set_cell_bg(row_cells[ci], bg)
            run = row_cells[ci].paragraphs[0].runs[0]
            run.font.size = Pt(10)
            row_cells[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Largeurs colonnes
    if col_widths:
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                cell.width = Cm(col_widths[ci])

    doc.add_paragraph()
    return table


def separateur(doc):
    """Ajoute une ligne horizontale fine."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "AAAAAA")
    pBdr.append(bottom)
    pPr.append(pBdr)


# ═══════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU DOCUMENT
# ═══════════════════════════════════════════════════════════════════════

def construire_rapport():
    doc = Document()

    # ── Marges du document ─────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    # ── Style de paragraphe par défaut ─────────────────────────────
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ═══════════════════════════════════════════════════════════════
    # PAGE DE TITRE
    # ═══════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_paragraph()

    p_univ = doc.add_paragraph("Université / École d'Ingénieurs")
    p_univ.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_univ.runs[0].font.size = Pt(13)
    p_univ.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    p_dept = doc.add_paragraph("Département Informatique — Master / Licence")
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.runs[0].font.size = Pt(12)
    p_dept.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_paragraph()
    doc.add_paragraph()
    separateur(doc)
    doc.add_paragraph()

    p_titre = doc.add_paragraph()
    p_titre.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = p_titre.add_run("Prédiction des Feux de Forêt")
    run_t.font.size = Pt(24)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1a, 0x53, 0x7a)

    p_sous = doc.add_paragraph()
    p_sous.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_s = p_sous.add_run("basée sur des Modèles d'Apprentissage Profond")
    run_s.font.size = Pt(18)
    run_s.font.bold = True
    run_s.font.color.rgb = RGBColor(0x1a, 0x53, 0x7a)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Comparaison AlexNet-1D vs ResNet-1D sur données MODIS — Turquie 2020")
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph()
    separateur(doc)
    doc.add_paragraph()
    doc.add_paragraph()

    for label, valeur in [("Auteur :", "Nom Prénom"), ("Encadrant :", "Nom Encadrant"),
                           ("Filière :", "Informatique / Data Science"),
                           ("Année universitaire :", "2024 - 2025")]:
        p_info = doc.add_paragraph()
        p_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p_info.add_run(f"{label}  ")
        r1.font.bold = True
        r1.font.size = Pt(12)
        r2 = p_info.add_run(valeur)
        r2.font.size = Pt(12)

    doc.add_paragraph()
    p_date = doc.add_paragraph(f"Date : {datetime.date.today().strftime('%d %B %Y')}")
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.runs[0].font.size = Pt(11)
    p_date.runs[0].font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════
    # RÉSUMÉ (ABSTRACT)
    # ═══════════════════════════════════════════════════════════════
    titre_section(doc, "Résumé", 1)
    paragraphe(doc,
        "Ce projet de fin d'années porte sur la classification automatique des types de "
        "feux de forêt à partir des données satellitaires MODIS (Moderate Resolution Imaging "
        "Spectroradiometer) enregistrées en Turquie durant l'année 2020. Le dataset contient "
        "8 834 observations décrites par 15 variables incluant des mesures thermiques "
        "(brightness, FRP), géographiques (latitude, longitude) et temporelles.")
    paragraphe(doc,
        "Face au fort déséquilibre des classes (92.9% de feux de végétation contre 7% de "
        "volcans et 0.08% d'autres sources), deux stratégies de rééquilibrage sont comparées : "
        "les poids de classes (class_weight) et le sur-échantillonnage synthétique (SMOTE). "
        "Deux architectures CNN 1D adaptées aux données tabulaires sont implémentées : "
        "une inspirée d'AlexNet et une inspirée de ResNet, toutes deux sous Keras/TensorFlow.")
    paragraphe(doc,
        "Les résultats montrent que ResNet-1D avec SMOTE obtient les meilleures performances "
        "avec un F1-score pondéré de 90.67% et un RMSE de 0.6249, surpassant légèrement "
        "AlexNet-1D avec class_weight (F1 = 90.10%, RMSE = 0.6334). Les deux modèles classifient "
        "très bien la classe majoritaire (F1 = 0.95) mais peinent sur les classes minoritaires, "
        "notamment la classe 3 représentée par seulement 7 exemples dans l'ensemble du dataset.")

    p_kw = doc.add_paragraph()
    r_kw1 = p_kw.add_run("Mots-clés : ")
    r_kw1.font.bold = True
    r_kw1.font.size = Pt(10)
    r_kw2 = p_kw.add_run(
        "Feux de forêt, MODIS, CNN 1D, AlexNet, ResNet, SMOTE, "
        "class_weight, déséquilibre des classes, classification, deep learning")
    r_kw2.font.italic = True
    r_kw2.font.size = Pt(10)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════
    # 1. INTRODUCTION
    # ═══════════════════════════════════════════════════════════════
    titre_section(doc, "1. Introduction et Problématique", 1)

    titre_section(doc, "1.1 Contexte", 2)
    paragraphe(doc,
        "Les feux de forêt constituent une menace environnementale, économique et humanitaire "
        "croissante, particulièrement dans les régions méditerranéennes. La Turquie a connu "
        "en 2020 une saison de feux exceptionnellement active, avec des milliers d'hectares "
        "dévastés. La détection rapide et la classification précise des types de feux à partir "
        "de données satellitaires sont essentielles pour les systèmes d'alerte précoce et "
        "la gestion des risques.")
    paragraphe(doc,
        "Le satellite MODIS (NASA) fournit des données de détection active des feux en temps "
        "quasi-réel, incluant la puissance radiative (FRP), la température de brillance, et "
        "la localisation géographique. Ces données tabulaires multi-variables constituent "
        "une base idéale pour les méthodes d'apprentissage automatique.")

    titre_section(doc, "1.2 Problématique", 2)
    paragraphe(doc,
        "La variable cible (type de feu) présente un déséquilibre extrême entre ses trois "
        "classes (0 : végétation, 2 : volcan, 3 : autre). Ce déséquilibre, combiné à la "
        "nature tabulaire des données (ne disposant pas de la structure spatiale des images), "
        "pose deux défis majeurs :")
    for item in [
        "Comment adapter l'architecture CNN — conçue pour les données structurées — à des données tabulaires 1D ?",
        "Quelle stratégie de gestion du déséquilibre (class_weight ou SMOTE) est la plus efficace ?"
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        r = p_b.add_run(item)
        r.font.size = Pt(11)

    titre_section(doc, "1.3 Objectifs", 2)
    for obj in [
        "Réaliser une analyse exploratoire complète du dataset MODIS Turquie 2020",
        "Développer un pipeline de prétraitement adapté (encodage, feature engineering, normalisation)",
        "Implémenter deux architectures CNN 1D (AlexNet-inspired et ResNet-inspired) sous Keras",
        "Comparer deux stratégies anti-déséquilibre : class_weight et SMOTE",
        "Évaluer les modèles avec les métriques académiques standard (Accuracy, F1, RMSE, matrices de confusion)"
    ]:
        p_num = doc.add_paragraph(style="List Bullet")
        p_num.paragraph_format.left_indent = Cm(1.5)
        p_num.add_run(obj).font.size = Pt(11)

    # ═══════════════════════════════════════════════════════════════
    # 2. DATASET
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "2. Présentation du Dataset", 1)

    titre_section(doc, "2.1 Source des données", 2)
    paragraphe(doc,
        "Le dataset provient du système NASA FIRMS (Fire Information for Resource Management "
        "System), qui diffuse en temps réel les détections d'incendies actifs du satellite MODIS. "
        "Il couvre l'ensemble du territoire turc pour l'année 2020 et a été obtenu depuis "
        "earthdata.nasa.gov. Deux capteurs sont représentés : Terra (satellite A) et Aqua (satellite B).")

    titre_section(doc, "2.2 Caractéristiques générales", 2)
    ajouter_tableau(doc,
        ["Propriété", "Valeur"],
        [
            ["Dimensions", "8 834 lignes × 15 colonnes"],
            ["Valeurs manquantes", "Aucune (dataset complet)"],
            ["Variable cible", "type (0 = végétation, 2 = volcan, 3 = autre)"],
            ["Période", "Année 2020 — Turquie"],
            ["Source", "NASA FIRMS — MODIS Terra & Aqua"],
        ],
        col_widths=[5, 11]
    )

    titre_section(doc, "2.3 Description des colonnes", 2)
    ajouter_tableau(doc,
        ["Colonne", "Type", "Description", "Statut"],
        [
            ["latitude", "float", "Latitude GPS du foyer (°N)", "Feature"],
            ["longitude", "float", "Longitude GPS du foyer (°E)", "Feature"],
            ["brightness", "float", "Température brillance canal 21/22 (K)", "Feature"],
            ["scan", "float", "Taille pixel — sens balayage (km)", "Feature"],
            ["track", "float", "Taille pixel — sens déplacement (km)", "Feature"],
            ["acq_date", "string", "Date d'acquisition (YYYY-MM-DD)", "→ mois extrait"],
            ["acq_time", "int", "Heure d'acquisition (HHMM)", "→ heure extraite"],
            ["satellite", "string", "'A' (Terra) ou 'B' (Aqua)", "Encodée 0/1"],
            ["instrument", "string", "Toujours 'MODIS' — constante", "Supprimée"],
            ["confidence", "int", "Niveau de confiance (0–100)", "Feature"],
            ["version", "float", "Toujours 6.03 — constante", "Supprimée"],
            ["bright_t31", "float", "Température brillance canal 31 (K)", "Feature"],
            ["frp", "float", "Fire Radiative Power (MW/sr/m²)", "Feature"],
            ["daynight", "string", "'D' (jour) ou 'N' (nuit)", "Encodée 0/1"],
            ["type", "int", "Type de feu (variable cible)", "CIBLE"],
        ],
        col_widths=[3.2, 2, 7.5, 3]
    )

    titre_section(doc, "2.4 Déséquilibre des classes (enjeu majeur)", 2)
    paragraphe(doc,
        "La variable cible présente un déséquilibre extrême, constitutif du principal "
        "défi de ce projet :")
    ajouter_tableau(doc,
        ["Classe", "Label", "Effectif", "Proportion", "Ratio vs Classe 3"],
        [
            ["0", "Végétation (Vegetation fire)", "8 208", "92.9%", "1172 : 1"],
            ["2", "Volcan (Volcano)", "619", "7.0%", "88 : 1"],
            ["3", "Autre feu statique", "7", "0.08%", "1 : 1"],
        ],
        col_widths=[1.8, 5.5, 2.5, 2.5, 3.5]
    )
    paragraphe(doc,
        "Un modèle naïf prédisant systématiquement la classe 0 atteindrait 92.9% d'accuracy "
        "sans apprendre quoi que ce soit. C'est pourquoi le F1-score pondéré est retenu "
        "comme métrique principale, et deux stratégies de rééquilibrage sont comparées.",
        italic=True)

    titre_section(doc, "2.5 Statistiques descriptives clés", 2)
    ajouter_tableau(doc,
        ["Feature", "Min", "Max", "Moyenne", "Écart-type"],
        [
            ["brightness (K)", "300.0", "500.0", "320.7", "15.4"],
            ["frp (MW/sr/m²)", "2.4", "2787.6", "27.0", "66.2"],
            ["confidence (%)", "0", "100", "64.8", "19.7"],
            ["scan (km)", "1.0", "4.8", "1.48", "0.70"],
            ["track (km)", "1.0", "2.0", "1.17", "0.22"],
            ["bright_t31 (K)", "268.3", "330.9", "301.0", "10.2"],
        ],
        col_widths=[4.5, 2.5, 2.5, 2.5, 2.5]
    )

    # ═══════════════════════════════════════════════════════════════
    # 3. EDA
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "3. Analyse Exploratoire des Données (EDA)", 1)

    titre_section(doc, "3.1 Distribution de la variable cible", 2)
    paragraphe(doc,
        "La Figure 1 illustre le fort déséquilibre entre les trois classes. "
        "La classe 0 (végétation) représente plus de 92% des observations, "
        "rendant indispensable un traitement spécifique avant l'entraînement.")
    ajouter_figure(doc, "01_distribution_cible.png",
                   "Figure 1 — Distribution de la variable cible 'type' (barres et camembert)")

    titre_section(doc, "3.2 Distributions des features numériques", 2)
    paragraphe(doc,
        "Les histogrammes (Figure 2) révèlent des profils très variés : "
        "brightness suit une distribution quasi-normale, tandis que frp (Fire Radiative Power) "
        "est fortement asymétrique à droite avec quelques valeurs extrêmes dépassant 500 MW, "
        "témoignant de feux d'intensité exceptionnelle.")
    ajouter_figure(doc, "02_distributions_features.png",
                   "Figure 2 — Histogrammes des variables numériques")

    titre_section(doc, "3.3 Matrice de corrélation", 2)
    paragraphe(doc,
        "La heatmap de corrélation de Pearson (Figure 3) met en évidence plusieurs relations "
        "importantes :")
    for item in [
        "brightness ↔ bright_t31 : corrélation forte (r > 0.7) — mesures du même phénomène physique sur deux canaux infrarouges différents",
        "scan ↔ track : corrélation forte — ces deux variables encodent la géométrie du satellite (angle d'observation)",
        "frp est la feature la plus corrélée avec type → meilleur prédicteur individuel"
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        p_b.add_run(item).font.size = Pt(11)
    ajouter_figure(doc, "03_matrice_correlation.png",
                   "Figure 3 — Matrice de corrélation de Pearson (triangle inférieur)")

    titre_section(doc, "3.4 Boxplots par classe et analyse temporelle", 2)
    paragraphe(doc,
        "Les boxplots (Figure 4) montrent que frp et brightness permettent une séparation "
        "partielle entre les classes : les volcans (classe 2) présentent des valeurs de "
        "brightness systématiquement plus élevées que les feux de végétation. "
        "L'analyse temporelle (Figure 5) confirme la saisonnalité attendue : "
        "un pic de détections en juillet-août-septembre 2020, saison sèche en Turquie.")
    ajouter_figure(doc, "04_boxplots_par_classe.png",
                   "Figure 4 — Boxplots des features principales par classe cible", largeur=13)
    ajouter_figure(doc, "05_repartition_mensuelle.png",
                   "Figure 5 — Répartition mensuelle des détections (saisonnalité)")

    titre_section(doc, "3.5 Répartition géographique", 2)
    paragraphe(doc,
        "La carte (Figure 6) montre que les feux de végétation sont concentrés sur la "
        "côte méditerranéenne et égéenne, tandis que les volcans sont géographiquement "
        "localisés dans des zones spécifiques de l'Anatolie orientale. Cette information "
        "spatiale est capturée par les features latitude et longitude.")
    ajouter_figure(doc, "06_carte_geographique.png",
                   "Figure 6 — Répartition géographique des détections par type (Turquie 2020)")

    # ═══════════════════════════════════════════════════════════════
    # 4. PRÉTRAITEMENT
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "4. Prétraitement des Données", 1)

    titre_section(doc, "4.1 Suppression des colonnes constantes", 2)
    paragraphe(doc,
        "Les colonnes instrument (toujours 'MODIS') et version (toujours 6.03) ont une "
        "variance nulle — elles n'apportent aucune information discriminante entre les classes "
        "et sont supprimées. Après suppression, le dataset passe à 13 colonnes.")

    titre_section(doc, "4.2 Encodage des variables catégorielles", 2)
    paragraphe(doc,
        "Les variables satellite et daynight sont encodées par Label Encoding binaire, "
        "choix justifié par leur nature strictement binaire (2 valeurs) :")
    ajouter_tableau(doc,
        ["Variable", "Valeurs originales", "Encodage", "Justification"],
        [
            ["satellite", "'A' (Terra) / 'B' (Aqua)", "0 / 1", "Binaire — pas de biais ordinal"],
            ["daynight", "'D' (jour) / 'N' (nuit)", "0 / 1", "Binaire — relation naturelle"],
        ],
        col_widths=[3, 4.5, 2.5, 5.5]
    )

    titre_section(doc, "4.3 Feature Engineering temporel", 2)
    paragraphe(doc,
        "La colonne acq_date (format YYYY-MM-DD) est transformée en mois (1–12) pour "
        "capturer la saisonnalité. La colonne acq_time (format HHMM) est convertie en "
        "heure locale (0–23) par division entière par 100. Les colonnes originales sont "
        "ensuite supprimées. Ces deux nouvelles features portent une information "
        "temporelle directement exploitable par le modèle.")

    titre_section(doc, "4.4 Split stratifié train/test (80/20)", 2)
    paragraphe(doc,
        "Le split stratifié garantit que les proportions de chaque classe sont préservées "
        "dans les ensembles d'entraînement et de test, condition indispensable avec des "
        "données aussi déséquilibrées :")
    ajouter_tableau(doc,
        ["Ensemble", "Taille totale", "Classe 0", "Classe 2", "Classe 3"],
        [
            ["Train (80%)", "7 067", "6 566", "495", "6"],
            ["Test (20%)", "1 767", "1 642", "124", "1"],
        ],
        col_widths=[3.5, 3.5, 3, 3, 3]
    )

    titre_section(doc, "4.5 Gestion du déséquilibre", 2)

    titre_section(doc, "4.5.1 Approche 1 — class_weight (AlexNet-1D)", 3)
    paragraphe(doc,
        "La fonction compute_class_weight de scikit-learn calcule des poids inversement "
        "proportionnels à la fréquence de chaque classe. Ces poids sont passés à "
        "model.fit(class_weight=...) et multiplient la contribution de chaque exemple "
        "à la fonction de perte lors de l'entraînement.")
    paragraphe(doc, "Formule : w_i = N_total / (N_classes × N_i)", indent=True)
    ajouter_tableau(doc,
        ["Classe", "Poids calculé", "Interprétation"],
        [
            ["0 — Végétation", "0.359", "Peu pénalisée (majoritaire)"],
            ["2 — Volcan", "4.759", "13× plus pénalisée que classe 0"],
            ["3 — Autre", "392.6", "1 094× plus pénalisée que classe 0"],
        ],
        col_widths=[4, 3.5, 8]
    )

    titre_section(doc, "4.5.2 Approche 2 — SMOTE (ResNet-1D)", 3)
    paragraphe(doc,
        "SMOTE (Synthetic Minority Over-sampling TEchnique, Chawla et al. 2002) génère "
        "des exemples synthétiques pour les classes minoritaires par interpolation linéaire "
        "entre k plus proches voisins. L'algorithme est appliqué uniquement sur le train "
        "set pour éviter tout data leakage.")
    paragraphe(doc,
        "Le paramètre k_neighbors=2 (au lieu de 5 par défaut) est nécessaire car la "
        "classe 3 ne contient que 6 exemples en train — insuffisant pour 5 voisins.",
        italic=True)
    ajouter_tableau(doc,
        ["Classe", "Avant SMOTE", "Après SMOTE", "Variation"],
        [
            ["0 — Végétation", "6 566", "6 566", "— (référence)"],
            ["2 — Volcan", "495", "6 566", "+1 226% (×13.3)"],
            ["3 — Autre", "6", "6 566", "+109 333% (×1 094)"],
            ["Total train", "7 067", "19 698", "+178%"],
        ],
        col_widths=[4, 3.5, 3.5, 4.5]
    )
    ajouter_figure(doc, "07_smote_comparison.png",
                   "Figure 7 — Comparaison de la distribution des classes avant et après SMOTE")

    titre_section(doc, "4.6 Normalisation (StandardScaler)", 2)
    paragraphe(doc,
        "Les 12 features finales sont normalisées par z-score : x' = (x − μ) / σ, où "
        "μ et σ sont calculés exclusivement sur le train set. Le scaler ajusté est ensuite "
        "appliqué (transform) sur le test set. Cette approche évite tout data leakage "
        "et garantit que les plages d'entrée du réseau sont homogènes.")

    # ═══════════════════════════════════════════════════════════════
    # 5. ARCHITECTURES CNN
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "5. Architectures CNN 1D", 1)

    titre_section(doc, "5.1 Adaptation tabulaire → 1D-CNN", 2)
    paragraphe(doc,
        "Les CNN sont conçus pour exploiter la structure locale des données. Sur données "
        "tabulaires, les features sont ordonnées sémantiquement (géographiques, thermiques, "
        "géométriques, temporelles) et traitées comme une séquence 1D. Un filtre Conv1D "
        "de taille k=3 apprend des interactions entre 3 features consécutives.")
    paragraphe(doc,
        "Reshape appliqué : (n, 12) → (n, 12, 1)  [batch_size, steps, channels]",
        indent=True, gras=True)
    paragraphe(doc,
        "Ordre des 12 features dans la séquence : [0] latitude · [1] longitude · "
        "[2] brightness · [3] scan · [4] track · [5] satellite · [6] confidence · "
        "[7] bright_t31 · [8] frp · [9] daynight · [10] mois · [11] heure",
        indent=True, italic=True)

    titre_section(doc, "5.2 Architecture 1 — AlexNet-1D", 2)
    paragraphe(doc,
        "Inspirée de AlexNet (Krizhevsky et al., 2012), première architecture CNN à gagner "
        "ILSVRC. La philosophie est conservée : plusieurs blocs de convolution séquentiels "
        "suivis d'une tête dense. Les couches de Batch Normalization remplacent la "
        "Local Response Normalization originale, plus stable en pratique.")
    ajouter_tableau(doc,
        ["Couche", "Sortie", "Paramètres", "Description"],
        [
            ["Input", "(12, 1)", "0", "Séquence de 12 features"],
            ["Conv1D(64, k=3) + BN + ReLU", "(12, 64)", "192 + 256", "Bloc 1"],
            ["Conv1D(128, k=3) + BN + ReLU + MaxPool(2)", "(6, 128)", "24 576 + 512", "Bloc 2"],
            ["Conv1D(256, k=3) + BN + ReLU", "(6, 256)", "98 304 + 1 024", "Bloc 3"],
            ["Conv1D(256, k=3) + BN + ReLU", "(6, 256)", "196 608 + 1 024", "Bloc 4"],
            ["Conv1D(128, k=3) + BN + ReLU + MaxPool(2)", "(3, 128)", "98 304 + 512", "Bloc 5"],
            ["GlobalAveragePooling1D", "(128,)", "0", "Agrégation"],
            ["Dense(256) + Dropout(0.5)", "(256,)", "33 024", "FC1"],
            ["Dense(128) + Dropout(0.3)", "(128,)", "32 896", "FC2"],
            ["Dense(3, softmax)", "(3,)", "387", "Sortie"],
        ],
        col_widths=[6, 2.5, 2.5, 4.5]
    )
    paragraphe(doc, "Total paramètres : 487 619  |  Régularisation : L2(1e-4) + Dropout",
               gras=True)

    titre_section(doc, "5.3 Architecture 2 — ResNet-1D", 2)
    paragraphe(doc,
        "Inspirée de ResNet (He et al., 2016), vainqueur ILSVRC 2015. L'innovation clé "
        "est la connexion résiduelle (skip connection) : H(x) = F(x) + x. Le gradient "
        "de la perte vaut ∂L/∂x = ∂L/∂H × (∂F/∂x + 1), le terme '+1' garantissant "
        "qu'un gradient non nul se propage toujours, évitant le problème du gradient "
        "évanescent dans les réseaux profonds.")
    ajouter_tableau(doc,
        ["Stage", "Composants", "Filtres", "Paramètres"],
        [
            ["Entrée", "Conv1D(64, k=3) + BN + ReLU", "64", "192"],
            ["Stage 1", "2 × ResBlock(64)  [shortcut identité]", "64", "~49 920"],
            ["Stage 2", "2 × ResBlock(128) [+ projection 64→128]", "128", "~182 400"],
            ["Stage 3", "1 × ResBlock(256) [+ projection 128→256]", "256", "~330 496"],
            ["Tête", "GlobalAvgPool + Dense(128) + Dropout(0.3) + Dense(3)", "—", "33 283"],
        ],
        col_widths=[2.5, 8, 2.5, 2.5]
    )
    paragraphe(doc, "Total paramètres : 597 443  |  Régularisation : L2(1e-4) + Dropout + Skip connections",
               gras=True)

    # ═══════════════════════════════════════════════════════════════
    # 6. ENTRAÎNEMENT
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "6. Entraînement", 1)

    titre_section(doc, "6.1 Hyperparamètres", 2)
    ajouter_tableau(doc,
        ["Paramètre", "Valeur", "Justification"],
        [
            ["Optimiseur", "Adam (lr = 1e-3)", "Adaptatif, convergence rapide, standard deep learning"],
            ["Fonction de perte", "Sparse Categorical Crossentropy", "Classification multi-classes — labels entiers"],
            ["Batch size", "32", "Compromis vitesse/stabilité du gradient stochastique"],
            ["Époques maximum", "50", "EarlyStopping arrête avant si nécessaire"],
            ["Validation split", "15% du train", "Suivi de la généralisation pendant l'entraînement"],
            ["Régularisation L2", "λ = 1e-4", "Pénalise les grands poids — réduit le sur-apprentissage"],
        ],
        col_widths=[4, 4, 7.5]
    )

    titre_section(doc, "6.2 Callbacks Keras", 2)
    for nom, desc in [
        ("EarlyStopping (patience=10, restore_best_weights=True)",
         "Arrête l'entraînement si val_loss ne s'améliore plus pendant 10 époques et restaure automatiquement les poids de la meilleure époque. Évite le sur-apprentissage et réduit le temps de calcul."),
        ("ReduceLROnPlateau (patience=5, factor=0.5, min_lr=1e-6)",
         "Divise le taux d'apprentissage par 2 si val_loss stagne 5 époques. Permet de sortir des plateaux de convergence en affinant les poids dans un voisinage plus restreint."),
        ("ModelCheckpoint (save_best_only=True)",
         "Sauvegarde automatiquement le modèle dont la val_loss est minimale. Permet la réutilisation du meilleur état sans ré-entraîner."),
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        r1 = p_b.add_run(nom + " : ")
        r1.font.bold = True
        r1.font.size = Pt(11)
        r2 = p_b.add_run(desc)
        r2.font.size = Pt(11)

    titre_section(doc, "6.3 Courbes d'entraînement", 2)
    paragraphe(doc,
        "AlexNet-1D (Figure 8a) a convergé très rapidement : la meilleure époque est "
        "l'époque 2, avec val_loss = 0.543. L'EarlyStopping a été déclenché à l'époque 12. "
        "La loss d'entraînement est artificiellement haute (effet des class_weight élevés "
        "sur les classes rares), mais la val_loss est faible dès le début.")
    ajouter_figure(doc, "08_courbes_alexnet_1d.png",
                   "Figure 8a — Courbes d'entraînement AlexNet-1D (loss et accuracy)", largeur=13)

    paragraphe(doc,
        "ResNet-1D (Figure 8b) a suivi une progression plus régulière sur les 50 époques "
        "complètes. La val_accuracy finale atteint 100% sur le set de validation (issu du "
        "train SMOTE équilibré) et la val_loss descend à 0.0602 — une convergence excellente "
        "facilitée par le dataset synthétiquement équilibré.")
    ajouter_figure(doc, "08_courbes_resnet_1d.png",
                   "Figure 8b — Courbes d'entraînement ResNet-1D (loss et accuracy)", largeur=13)

    # ═══════════════════════════════════════════════════════════════
    # 7. RÉSULTATS
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "7. Résultats et Évaluation", 1)

    titre_section(doc, "7.1 Tableau comparatif global", 2)
    ajouter_tableau(doc,
        ["Modèle", "Stratégie", "Accuracy", "Precision", "Recall", "F1-score", "RMSE"],
        [
            ["AlexNet-1D", "class_weight", "90.04%", "90.17%", "90.04%", "90.10%", "0.6334"],
            ["ResNet-1D",  "SMOTE",        "90.38%", "91.00%", "90.38%", "90.67%", "0.6249"],
        ],
        col_widths=[3, 3, 2.5, 2.5, 2.5, 2.5, 2]
    )
    paragraphe(doc,
        "ResNet-1D avec SMOTE obtient les meilleures performances sur tous les indicateurs. "
        "L'écart de F1-score est de +0.57% en sa faveur, et son RMSE est inférieur (-0.0085), "
        "indiquant des erreurs de classification d'amplitude légèrement plus faible.",
        italic=True)

    titre_section(doc, "7.2 Rapport de classification par classe", 2)
    paragraphe(doc, "AlexNet-1D — Rapport détaillé (meilleure époque : 2)", gras=True)
    ajouter_tableau(doc,
        ["Classe", "Précision", "Rappel", "F1-score", "Support"],
        [
            ["0 — Végétation", "0.95", "0.95", "0.95", "1 642"],
            ["2 — Volcan",     "0.30", "0.31", "0.31", "124"],
            ["3 — Autre",      "0.00", "0.00", "0.00", "1"],
            ["Weighted avg",   "0.90", "0.90", "0.90", "1 767"],
        ],
        col_widths=[4, 2.5, 2.5, 2.5, 2.5]
    )
    doc.add_paragraph()
    paragraphe(doc, "ResNet-1D — Rapport détaillé (50 époques, best epoch = 50)", gras=True)
    ajouter_tableau(doc,
        ["Classe", "Précision", "Rappel", "F1-score", "Support"],
        [
            ["0 — Végétation", "0.95", "0.94", "0.95", "1 642"],
            ["2 — Volcan",     "0.35", "0.40", "0.37", "124"],
            ["3 — Autre",      "0.00", "0.00", "0.00", "1"],
            ["Weighted avg",   "0.91", "0.90", "0.91", "1 767"],
        ],
        col_widths=[4, 2.5, 2.5, 2.5, 2.5]
    )

    titre_section(doc, "7.3 Matrices de confusion", 2)
    paragraphe(doc,
        "Les matrices de confusion (Figures 9a et 9b) permettent d'analyser les erreurs "
        "par classe. La version normalisée (droite) montre les taux de recall par classe, "
        "indépendamment des effectifs.")
    ajouter_figure(doc, "09_confusion_alexnet_1d.png",
                   "Figure 9a — Matrices de confusion AlexNet-1D (brute et normalisée)", largeur=14)
    ajouter_figure(doc, "09_confusion_resnet_1d.png",
                   "Figure 9b — Matrices de confusion ResNet-1D (brute et normalisée)", largeur=14)

    titre_section(doc, "7.4 Tableau comparatif final", 2)
    ajouter_figure(doc, "10_comparaison_finale.png",
                   "Figure 10 — Tableau et graphique comparatifs des deux architectures CNN", largeur=15)

    # ═══════════════════════════════════════════════════════════════
    # 8. DISCUSSION
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "8. Discussion", 1)

    titre_section(doc, "8.1 Analyse des résultats", 2)
    paragraphe(doc,
        "Les deux architectures obtiennent des performances similaires (~90% F1 pondéré), "
        "ce qui s'explique principalement par la domination de la classe 0 dans le calcul "
        "du weighted average. L'écart entre les modèles est plus significatif lorsqu'on "
        "analyse la classe 2 (volcans) :")
    for item in [
        "AlexNet-1D : recall classe 2 = 0.31 → 31% des volcans sont correctement identifiés",
        "ResNet-1D  : recall classe 2 = 0.40 → 40% des volcans détectés — gain de +29% relatif",
        "Les connexions résiduelles de ResNet permettent un apprentissage plus robuste des features discriminantes des volcans (brightness élevée, FRP modéré)"
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        p_b.add_run(item).font.size = Pt(11)

    titre_section(doc, "8.2 Limites identifiées", 2)
    for titre_item, corps in [
        ("Classe 3 indétectable :",
         "Avec seulement 7 exemples dans l'ensemble complet (1 en test), "
         "cette classe est statistiquement impossible à apprendre. F1 = 0.00 pour les deux modèles."),
        ("Limite de SMOTE sur données mixtes :",
         "Les features binaires (satellite, daynight, mois) génèrent des valeurs "
         "interpolées non réalistes lors de SMOTE (ex. satellite = 0.37). SMOTE-NC "
         "serait plus approprié pour les données mixtes continues/catégorielles."),
        ("Séquence courte (12 features) :",
         "La longueur de séquence limite la profondeur effective des CNN 1D. "
         "Les poolings réduisent la dimension à 3 dès le bloc 2 d'AlexNet, "
         "ce qui peut affecter la capacité d'extraction de features."),
        ("Absence de GPU :",
         "L'entraînement sur CPU (TensorFlow ≥ 2.11 ne supporte pas le GPU natif "
         "sur Windows) a limité la possibilité d'explorer des hyperparamètres plus "
         "fins ou des architectures plus profondes."),
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        r1 = p_b.add_run(titre_item + " ")
        r1.font.bold = True
        r1.font.size = Pt(11)
        r2 = p_b.add_run(corps)
        r2.font.size = Pt(11)

    titre_section(doc, "8.3 Perspectives d'amélioration", 2)
    for item in [
        "SMOTE-NC (Nominal+Continuous) : gère correctement les features catégorielles en ne les interpolant pas",
        "TabTransformer ou FT-Transformer : architectures Transformer spécialement conçues pour données tabulaires",
        "Collecte de données supplémentaires de classe 3 : indispensable pour rendre la classe détectable",
        "Validation croisée stratifiée k-fold : évaluation plus robuste sur un dataset de taille modeste",
        "Recherche d'hyperparamètres (Optuna/Keras Tuner) : optimisation du learning rate, batch size, profondeur"
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        p_b.add_run(item).font.size = Pt(11)

    # ═══════════════════════════════════════════════════════════════
    # 9. CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "9. Conclusion", 1)
    paragraphe(doc,
        "Ce projet a permis de concevoir et d'évaluer un pipeline complet de classification "
        "des feux de forêt à partir de données satellitaires MODIS, en s'appuyant sur des "
        "architectures CNN 1D adaptées aux données tabulaires.")
    paragraphe(doc,
        "Les principaux apports sont :")
    for item in [
        "La démonstration de la faisabilité des CNN 1D sur données tabulaires via le reshape (n, 12) → (n, 12, 1)",
        "La comparaison rigoureuse de deux stratégies anti-déséquilibre : class_weight (plus simple) et SMOTE (plus efficace)",
        "ResNet-1D avec SMOTE s'impose comme la meilleure configuration : F1 = 90.67%, RMSE = 0.6249, recall volcans = 40%",
        "AlexNet-1D avec class_weight reste compétitif malgré sa convergence ultra-rapide (2 époques), avec F1 = 90.10%"
    ]:
        p_b = doc.add_paragraph(style="List Bullet")
        p_b.paragraph_format.left_indent = Cm(1.5)
        p_b.add_run(item).font.size = Pt(11)
    paragraphe(doc,
        "Les deux modèles atteignent leurs limites sur les classes très minoritaires "
        "(classe 3 : 7 exemples), ce qui constitue la principale piste d'amélioration "
        "future : enrichir le dataset avec davantage d'exemples de feux non-végétaux "
        "et adopter des techniques de rééquilibrage plus adaptées aux données mixtes "
        "comme SMOTE-NC ou les méthodes d'augmentation basées sur les GANs (CTGAN).",
        espace_avant=6)

    # ═══════════════════════════════════════════════════════════════
    # 10. RÉFÉRENCES
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()
    titre_section(doc, "Références Bibliographiques", 1)
    refs = [
        "[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 1097–1105.",
        "[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE CVPR 2016 (pp. 770–778).",
        "[3] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.",
        "[4] Giglio, L., Schroeder, W., & Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. Remote Sensing of Environment, 178, 31–41.",
        "[5] Kiranyaz, S., Avci, O., Abdeljaber, O., Ince, T., Gabbouj, M., & Inman, D. J. (2021). 1D convolutional neural networks and applications: A survey. Mechanical Systems and Signal Processing, 151, 107398.",
        "[6] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.",
        "[7] Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning Publications.",
        "[8] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    ]
    for ref in refs:
        p_r = doc.add_paragraph(ref)
        p_r.paragraph_format.left_indent   = Cm(1.0)
        p_r.paragraph_format.first_line_indent = Cm(-1.0)
        p_r.paragraph_format.space_after   = Pt(5)
        p_r.runs[0].font.size = Pt(10)

    # ═══════════════════════════════════════════════════════════════
    # SAUVEGARDE
    # ═══════════════════════════════════════════════════════════════
    doc.save(REPORT_PATH)
    print(f"\n[OK] Rapport genere : {REPORT_PATH}")
    print(f"     Taille : {os.path.getsize(REPORT_PATH) / 1024:.0f} Ko")


if __name__ == "__main__":
    print("Generation du rapport Word en cours...")
    construire_rapport()
