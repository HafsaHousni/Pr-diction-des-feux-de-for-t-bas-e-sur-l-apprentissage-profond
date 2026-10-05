"""
app.py — Application Streamlit de prédiction du risque d'incendie de forêt.

Utilise le modèle ResNet-1D entraîné sur les données MODIS Turquie 2020.
Prend les 12 features en entrée et retourne :
  - Le niveau de risque (CRITIQUE / ÉLEVÉ / MODÉRÉ / FAIBLE / AUCUN)
  - Le type de feu prédit (végétation, volcan, autre)
  - Les probabilités par classe
  - Une carte interactive du point analysé

Lancement :
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import folium
import streamlit as st
from streamlit_folium import st_folium

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ─── Chemins ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "outputs", "models", "ResNet-1D_best.keras")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "results", "scaler.pkl")

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="FireRisk AI — Prédiction d'Incendies",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #1a537a 0%, #e74c3c 100%);
    padding: 28px 32px 22px;
    border-radius: 16px;
    color: white;
    margin-bottom: 28px;
  }
  .main-header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 6px; }
  .main-header p  { font-size: 1rem; opacity: .88; margin: 0; }

  .risk-box {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,.12);
  }
  .risk-critique { background: linear-gradient(135deg,#c0392b,#e74c3c); color: white; }
  .risk-eleve    { background: linear-gradient(135deg,#d35400,#e67e22); color: white; }
  .risk-modere   { background: linear-gradient(135deg,#b7950b,#f1c40f); color: #2c3e50; }
  .risk-faible   { background: linear-gradient(135deg,#1e8449,#27ae60); color: white; }
  .risk-aucun    { background: linear-gradient(135deg,#1a5276,#2980b9); color: white; }

  .risk-emoji { font-size: 3.5rem; margin-bottom: 10px; }
  .risk-label { font-size: 2rem; font-weight: 800; letter-spacing: .05em; }
  .risk-sub   { font-size: .9rem; opacity: .85; margin-top: 6px; }

  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
    border-left: 5px solid #1a537a;
    margin-bottom: 12px;
  }
  .metric-card.orange { border-color: #e67e22; }
  .metric-card.green  { border-color: #27ae60; }
  .metric-card.red    { border-color: #e74c3c; }
  .metric-label { font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; color:#888; }
  .metric-value { font-size:1.6rem; font-weight:800; color:#1a537a; margin-top:4px; }
  .metric-card.orange .metric-value { color:#e67e22; }
  .metric-card.green  .metric-value { color:#27ae60; }
  .metric-card.red    .metric-value { color:#e74c3c; }

  .prob-bar-wrap { background:#eee; border-radius:8px; height:18px; overflow:hidden; margin-top:4px; }
  .prob-bar-fill { height:100%; border-radius:8px; transition:width .6s ease; }

  .info-box {
    background: #f0f6fa;
    border-left: 4px solid #1a537a;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: .88rem;
    color: #34495e;
    margin: 10px 0;
  }
  .warn-box {
    background: #fef9e7;
    border-left: 4px solid #f39c12;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: .88rem;
    color: #7d6608;
    margin: 10px 0;
  }

  div[data-testid="stSidebar"] { background: #f8fafc; }
  .stButton > button {
    background: linear-gradient(135deg, #1a537a, #2e7da8) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(26,83,122,.3) !important;
    transition: all .2s !important;
  }
  .stButton > button:hover { opacity: .9 !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE ET DU SCALER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def charger_modele():
    """
    Charge le modèle ResNet-1D et le StandardScaler une seule fois.
    @st.cache_resource évite de recharger à chaque interaction.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : {MODEL_PATH}\n"
                 "Veuillez d'abord exécuter : python 03_train_evaluate.py")
        st.stop()
    model  = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = charger_modele()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIQUE DE PRÉDICTION ET CALCUL DU RISQUE
# ═══════════════════════════════════════════════════════════════════════════════

# Remap inverse : sortie modèle {0,1,2} → labels originaux {0,2,3}
INV_MAP      = {0: 0, 1: 2, 2: 3}
NOMS_CLASSES = {0: "Végétation (feu de forêt)", 2: "Volcan / activité géothermique", 3: "Autre source statique"}
EMOJIS_CLS   = {0: "🌿", 2: "🌋", 3: "🔥"}


def calculer_risque(type_predit: int, proba_veg: float,
                    frp: float, confidence: int, brightness: float) -> dict:
    """
    Calcule le niveau de risque d'incendie à partir de la prédiction
    et des features physiques clés.

    Logique de décision :
      Le type prédit par le modèle indique la nature du feu détecté.
      On combine cela avec les indicateurs physiques :
        - FRP (Fire Radiative Power) : intensité du rayonnement thermique
        - brightness : température de brillance (K)
        - confidence : niveau de confiance de la détection MODIS (0-100)
        - proba_veg  : probabilité d'appartenir à la classe "végétation"

    Niveaux :
      CRITIQUE → feu de végétation probable + FRP > 100 + confiance élevée
      ÉLEVÉ    → feu de végétation probable + FRP > 30
      MODÉRÉ   → signal fire detectable mais intensité modérée
      FAIBLE   → probabilités basses ou FRP très faible
    """
    # Pré-filtre : données trop faibles pour caractériser un incendie réel
    if brightness < 305 and frp < 3 and confidence < 25:
        return {
            "niveau": "AUCUN RISQUE DÉTECTÉ",
            "classe": "AUCUN",
            "emoji": "🟢",
            "css": "risk-aucun",
            "explication": (
                "Les valeurs mesurées (brightness, FRP, confiance) sont trop faibles "
                "pour caractériser une activité de feu active. Aucun risque immédiat détecté."
            ),
            "conseil": "Continuer la surveillance. Vérifier à nouveau si les conditions évoluent."
        }

    if type_predit == 0:  # Feu de végétation
        if proba_veg > 0.75 and frp > 100 and confidence > 70:
            return {
                "niveau": "CRITIQUE",
                "classe": "Feu de végétation actif",
                "emoji": "🔴",
                "css": "risk-critique",
                "explication": (
                    f"Feu de végétation actif confirmé avec une probabilité de {proba_veg*100:.1f}%. "
                    f"FRP = {frp:.1f} MW/sr/m² (très élevé) — incendie d'intensité extrême. "
                    f"Confiance MODIS = {confidence}%."
                ),
                "conseil": "⚠️ Intervention immédiate requise. Alerte aux services de pompiers. Évacuation des zones proches."
            }
        elif proba_veg > 0.55 and frp > 30:
            return {
                "niveau": "ÉLEVÉ",
                "classe": "Feu de végétation probable",
                "emoji": "🟠",
                "css": "risk-eleve",
                "explication": (
                    f"Forte probabilité de feu de végétation ({proba_veg*100:.1f}%). "
                    f"FRP = {frp:.1f} MW/sr/m² — incendie en cours avec propagation possible."
                ),
                "conseil": "⚠️ Mobiliser les équipes d'intervention. Surveiller la progression du feu."
            }
        else:
            return {
                "niveau": "MODÉRÉ",
                "classe": "Foyer de végétation (faible intensité)",
                "emoji": "🟡",
                "css": "risk-modere",
                "explication": (
                    f"Signal de feu de végétation détecté ({proba_veg*100:.1f}%) "
                    f"mais d'intensité modérée (FRP = {frp:.1f} MW/sr/m²). "
                    "Peut correspondre à un brûlage agricole contrôlé."
                ),
                "conseil": "Surveiller l'évolution. Vérifier s'il s'agit d'un brûlage contrôlé ou d'un départ de feu."
            }

    elif type_predit == 2:  # Volcan
        return {
            "niveau": "MODÉRÉ",
            "classe": "Activité volcanique / géothermique",
            "emoji": "🌋",
            "css": "risk-modere",
            "explication": (
                f"Signal classifié comme activité volcanique ou géothermique. "
                f"FRP = {frp:.1f} MW/sr/m². "
                "Le risque d'incendie de forêt associé dépend de la zone environnante."
            ),
            "conseil": "Surveiller la zone de cendres et de coulées de lave. Risque de feux secondaires."
        }

    else:  # Classe 3 — Autre
        return {
            "niveau": "FAIBLE",
            "classe": "Source thermique non végétale",
            "emoji": "🟡",
            "css": "risk-faible",
            "explication": (
                f"Source thermique détectée mais classifiée comme non végétale (type 3). "
                f"FRP = {frp:.1f} MW/sr/m². Peut être une installation industrielle ou un feu de décharge."
            ),
            "conseil": "Surveillance recommandée. Identifier la source pour confirmer l'absence de risque forestier."
        }


def predire(latitude, longitude, brightness, scan, track,
            satellite, confidence, bright_t31, frp, daynight, mois, heure):
    """
    Pipeline complet : features → normalisation → reshape → prédiction → risque.
    """
    sat_enc = 0 if satellite == "Terra (A)" else 1
    dn_enc  = 0 if daynight == "Jour (D)"   else 1

    X = np.array([[latitude, longitude, brightness, scan, track,
                   sat_enc, confidence, bright_t31, frp,
                   dn_enc, mois, heure]], dtype=float)

    X_scaled = scaler.transform(X)
    X_3d     = X_scaled.reshape(1, 12, 1)

    probas        = model.predict(X_3d, verbose=0)[0]
    idx_predit    = int(np.argmax(probas))
    type_reel     = INV_MAP[idx_predit]
    confiance_mod = float(probas[idx_predit])
    proba_veg     = float(probas[0])

    risque = calculer_risque(type_reel, proba_veg, frp, confidence, brightness)

    return {
        "type_reel"    : type_reel,
        "nom_classe"   : NOMS_CLASSES[type_reel],
        "emoji_classe" : EMOJIS_CLS[type_reel],
        "confiance_mod": confiance_mod,
        "probas"       : probas,
        "risque"       : risque,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALISATION DU SESSION STATE
# Pourquoi ? Streamlit réexécute le script entier à chaque interaction
# (clic, slider, rendu de carte...). Sans session_state, les résultats
# de prédiction disparaissent dès que la carte Folium se charge.
# session_state persiste entre les reruns dans la même session.
# ═══════════════════════════════════════════════════════════════════════════════
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None
if "historique"  not in st.session_state:
    st.session_state["historique"]  = []
# Coordonnées — partagées entre la carte cliquable et les inputs manuels
if "lat" not in st.session_state:
    st.session_state["lat"] = 37.50
if "lon" not in st.session_state:
    st.session_state["lon"] = 36.00
# Flag pour détecter un nouveau clic carte (évite les reruns infinis)
if "last_map_click" not in st.session_state:
    st.session_state["last_map_click"] = None


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE UTILISATEUR
# ═══════════════════════════════════════════════════════════════════════════════

# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔥 FireRisk AI — Prédiction du Risque d'Incendie</h1>
  <p>Système de classification basé sur le modèle ResNet-1D entraîné sur les données MODIS NASA · Turquie 2020</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Paramètres MODIS")
    st.markdown("---")

    st.markdown("**📍 Localisation active**")
    st.markdown(f"""
    <div style="background:#e8f4f8;border-radius:8px;padding:10px 14px;font-size:.88rem;margin-bottom:6px">
      🎯 <b>{st.session_state['lat']:.4f}°N</b> &nbsp;,&nbsp; <b>{st.session_state['lon']:.4f}°E</b><br/>
      <span style="color:#666;font-size:.78rem">Modifiez via la carte ou les champs ci-dessous</span>
    </div>
    """, unsafe_allow_html=True)
    # Inputs manuels synchronisés avec session_state
    # Bornes élargies au monde entier — la carte est mondiale, pas limitée à la Turquie
    lat_manual = st.number_input("Latitude (°N)",  min_value=-90.0, max_value=90.0,
                                  value=float(st.session_state["lat"]), step=0.001, format="%.4f")
    lon_manual = st.number_input("Longitude (°E)", min_value=-180.0, max_value=180.0,
                                  value=float(st.session_state["lon"]), step=0.001, format="%.4f")
    # Mise à jour si l'utilisateur a changé un champ manuellement
    if round(lat_manual, 4) != round(st.session_state["lat"], 4) or \
       round(lon_manual, 4) != round(st.session_state["lon"], 4):
        st.session_state["lat"] = round(lat_manual, 4)
        st.session_state["lon"] = round(lon_manual, 4)
        st.rerun()
    latitude  = st.session_state["lat"]
    longitude = st.session_state["lon"]

    st.markdown("---")
    st.markdown("**🌡️ Mesures thermiques**")
    brightness  = st.slider("Brightness (K)",       300.0, 500.0, 322.0, 0.1)
    bright_t31  = st.slider("Bright T31 (K)",       268.0, 331.0, 300.0, 0.1)
    frp         = st.slider("FRP (MW/sr/m²)",       0.0,   800.0,  15.0, 0.5,
                             help="Fire Radiative Power — intensité du rayonnement thermique")

    st.markdown("---")
    st.markdown("**🛰️ Géométrie satellite**")
    scan        = st.slider("Scan (km)",  1.0, 5.0, 1.3, 0.1)
    track       = st.slider("Track (km)", 1.0, 2.0, 1.1, 0.1)
    satellite   = st.selectbox("Satellite", ["Terra (A)", "Aqua (B)"])

    st.markdown("---")
    st.markdown("**📊 Qualité de détection**")
    confidence  = st.slider("Confiance MODIS (%)", 0, 100, 65)

    st.markdown("---")
    st.markdown("**🕒 Conditions temporelles**")
    daynight    = st.selectbox("Période", ["Jour (D)", "Nuit (N)"])
    mois        = st.select_slider("Mois", options=list(range(1, 13)),
                                   value=7,
                                   format_func=lambda m: ["Jan","Fév","Mar","Avr","Mai","Jun",
                                                           "Jul","Aoû","Sep","Oct","Nov","Déc"][m-1])
    heure       = st.slider("Heure locale (0-23)", 0, 23, 10)

    st.markdown("---")
    analyser = st.button("🔍 Analyser le risque")

# ═══════════════════════════════════════════════════════════════════════════════
# SÉLECTION DE LA LOCALISATION — CARTE INTERACTIVE + COORDONNÉES
# L'utilisateur peut :
#   1. Cliquer directement sur la carte → met à jour lat/lon dans session_state
#   2. Modifier les champs numériques dans le sidebar → la carte se déplace
# Les deux modes sont synchronisés via st.session_state["lat"] / ["lon"]
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📍 Sélection de la localisation")
st.caption("Cliquez sur la carte pour définir le point d'analyse, ou modifiez les coordonnées dans le panneau gauche.")

col_carte_sel, col_info_loc = st.columns([3, 1])

with col_carte_sel:
    # Carte de sélection avec marker sur la position courante
    m_sel = folium.Map(
        location=[st.session_state["lat"], st.session_state["lon"]],
        zoom_start=6,
        tiles="CartoDB positron"
    )
    # Marker de position courante (bleu = non encore analysé)
    folium.Marker(
        location=[st.session_state["lat"], st.session_state["lon"]],
        tooltip="📍 Position sélectionnée — cliquez ailleurs pour changer",
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa")
    ).add_to(m_sel)
    # Cercle léger pour visualiser la zone
    folium.Circle(
        location=[st.session_state["lat"], st.session_state["lon"]],
        radius=15000,
        color="#1a537a", fill=True, fill_opacity=0.08,
        weight=1.5
    ).add_to(m_sel)

    map_sel_data = st_folium(
        m_sel, width="100%", height=340,
        key="carte_selection",
        returned_objects=["last_clicked"]
    )

    # ── Traitement du clic ─────────────────────────────────────────────────
    if map_sel_data and map_sel_data.get("last_clicked"):
        clk = map_sel_data["last_clicked"]
        new_lat = round(clk["lat"], 4)
        new_lon = round(clk["lng"], 4)
        prev    = st.session_state["last_map_click"]
        if prev != (new_lat, new_lon):
            st.session_state["lat"]            = new_lat
            st.session_state["lon"]            = new_lon
            st.session_state["last_map_click"] = (new_lat, new_lon)
            st.rerun()

with col_info_loc:
    lat_c = st.session_state["lat"]
    lon_c = st.session_state["lon"]
    # Zone d'entraînement MODIS Turquie : lat 35-42, lon 25-45
    hors_zone = not (35.0 <= lat_c <= 42.5 and 25.0 <= lon_c <= 45.0)
    badge_zone = (
        '<div style="margin-top:12px;padding:8px 10px;background:#fff3cd;'
        'border-radius:8px;font-size:.78rem;color:#856404">'
        '⚠️ Hors zone d\'entraînement<br/>(modèle entraîné sur Turquie)<br/>'
        'Prédiction par extrapolation</div>'
    ) if hors_zone else (
        '<div style="margin-top:12px;padding:8px 10px;background:#d4edda;'
        'border-radius:8px;font-size:.78rem;color:#155724">'
        '✅ Dans la zone d\'entraînement<br/>MODIS Turquie 2020</div>'
    )
    st.markdown(f"""
    <div style="background:#f0f6fa;border-radius:12px;padding:20px 16px;min-height:200px">
      <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:12px">
        Position sélectionnée
      </div>
      <div style="font-size:1.5rem;font-weight:800;color:#1a537a;margin-bottom:4px">
        {lat_c:.4f}°N
      </div>
      <div style="font-size:1.5rem;font-weight:800;color:#1a537a;margin-bottom:8px">
        {lon_c:.4f}°E
      </div>
      <div style="font-size:.82rem;color:#666;line-height:1.8">
        🖱️ Cliquez sur la carte<br/>
        ✏️ Ou saisie manuelle<br/>&nbsp;&nbsp;&nbsp;&nbsp;dans le panneau gauche
      </div>
      {badge_zone}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SCENARIOS PREDEFINIS ──────────────────────────────────────────────────────
st.markdown("#### Scénarios de démonstration")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

scenarios = {
    "🔴 Feu Critique": dict(brightness=385.0, frp=250.0, confidence=90, mois=8, heure=14, bright_t31=315.0, scan=1.2, track=1.1, latitude=37.1, longitude=28.5, satellite="Terra (A)", daynight="Jour (D)"),
    "🟠 Feu Modéré":   dict(brightness=330.0, frp=45.0,  confidence=72, mois=7, heure=11, bright_t31=304.0, scan=1.4, track=1.2, latitude=38.2, longitude=32.0, satellite="Aqua (B)",  daynight="Jour (D)"),
    "🌋 Volcan":        dict(brightness=420.0, frp=90.0,  confidence=85, mois=3, heure=15, bright_t31=319.0, scan=1.0, track=1.0, latitude=38.9, longitude=27.8, satellite="Terra (A)", daynight="Jour (D)"),
    "🟢 Pas de risque": dict(brightness=302.0, frp=2.0,   confidence=18, mois=1, heure=22, bright_t31=270.0, scan=1.8, track=1.3, latitude=40.5, longitude=35.0, satellite="Aqua (B)",  daynight="Nuit (N)"),
}

sc_actif = None
with col_s1:
    if st.button("🔴 Feu Critique"):  sc_actif = scenarios["🔴 Feu Critique"]
with col_s2:
    if st.button("🟠 Feu Modéré"):   sc_actif = scenarios["🟠 Feu Modéré"]
with col_s3:
    if st.button("🌋 Volcan"):         sc_actif = scenarios["🌋 Volcan"]
with col_s4:
    if st.button("🟢 Pas de risque"): sc_actif = scenarios["🟢 Pas de risque"]

# Injecter les valeurs du scénario dans session_state
if sc_actif:
    # Mise à jour des coordonnées dans session_state (synchronise la carte)
    st.session_state["lat"] = sc_actif["latitude"]
    st.session_state["lon"] = sc_actif["longitude"]
    analyser    = True
    latitude    = sc_actif["latitude"];   longitude  = sc_actif["longitude"]
    brightness  = sc_actif["brightness"]; bright_t31 = sc_actif["bright_t31"]
    frp         = sc_actif["frp"];        confidence = sc_actif["confidence"]
    scan        = sc_actif["scan"];       track      = sc_actif["track"]
    satellite   = sc_actif["satellite"];  daynight   = sc_actif["daynight"]
    mois        = sc_actif["mois"];       heure      = sc_actif["heure"]
else:
    # S'assurer que latitude/longitude suivent toujours session_state
    latitude  = st.session_state["lat"]
    longitude = st.session_state["lon"]

st.markdown("---")

# ── DÉCLENCHEMENT DE LA PRÉDICTION ───────────────────────────────────────────
# On stocke immédiatement dans session_state dès qu'un bouton est cliqué.
# Ainsi, même si Streamlit réexécute le script (rendu carte, slider...),
# le résultat reste disponible via st.session_state["last_result"].
import datetime

if analyser or sc_actif:
    with st.spinner("Analyse en cours..."):
        res = predire(latitude, longitude, brightness, scan, track,
                      satellite, confidence, bright_t31, frp, daynight, mois, heure)

    # ── Stocker le résultat et les inputs dans session_state ─────────────────
    st.session_state["last_result"] = res
    st.session_state["last_inputs"] = {
        "latitude": latitude, "longitude": longitude,
        "brightness": brightness, "bright_t31": bright_t31,
        "frp": frp, "scan": scan, "track": track,
        "satellite": satellite, "confidence": confidence,
        "daynight": daynight, "mois": mois, "heure": heure,
    }

    # ── Ajouter à l'historique ───────────────────────────────────────────────
    mois_noms = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
    st.session_state["historique"].insert(0, {
        "horodatage" : datetime.datetime.now().strftime("%H:%M:%S"),
        "risque"     : res["risque"]["niveau"],
        "emoji"      : res["risque"]["emoji"],
        "type"       : res["nom_classe"],
        "lat"        : f"{latitude:.3f}°N",
        "lon"        : f"{longitude:.3f}°E",
        "frp"        : f"{frp:.1f}",
        "brightness" : f"{brightness:.1f}",
        "confiance"  : f"{confidence}%",
        "mois"       : mois_noms[mois - 1],
        "prob_veg"   : f"{float(res['probas'][0])*100:.1f}%",
        "prob_vol"   : f"{float(res['probas'][1])*100:.1f}%",
        "prob_aut"   : f"{float(res['probas'][2])*100:.1f}%",
    })
    # Garder les 20 dernières analyses
    st.session_state["historique"] = st.session_state["historique"][:20]

# ── AFFICHAGE DES RÉSULTATS (depuis session_state — persistant) ───────────────
if st.session_state["last_result"] is not None:
    res    = st.session_state["last_result"]
    inp    = st.session_state["last_inputs"]
    risque = res["risque"]
    probas = res["probas"]

    # Récupérer les inputs pour la carte
    lat_r  = inp["latitude"];     lon_r  = inp["longitude"]
    frp_r  = inp["frp"];          br_r   = inp["brightness"]
    conf_r = inp["confidence"];   mois_r = inp["mois"]
    heure_r= inp["heure"];        sat_r  = inp["satellite"]
    dn_r   = inp["daynight"]

    col_main, col_side = st.columns([1.2, 1])

    # ── Colonne gauche : résultat risque ─────────────────────────────────────
    with col_main:
        st.markdown(f"""
        <div class="risk-box {risque['css']}">
          <div class="risk-emoji">{risque['emoji']}</div>
          <div class="risk-label">{risque['niveau']}</div>
          <div class="risk-sub">{risque['classe']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
          <b>🔬 Analyse :</b> {risque['explication']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warn-box">
          <b>📋 Recommandation :</b> {risque['conseil']}
        </div>
        """, unsafe_allow_html=True)

        # Métriques clés
        st.markdown("#### Indicateurs physiques")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card {'red' if frp_r > 100 else 'orange' if frp_r > 30 else 'green'}">
              <div class="metric-label">FRP (MW/sr/m²)</div>
              <div class="metric-value">{frp_r:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card {'red' if br_r > 380 else 'orange' if br_r > 330 else ''}">
              <div class="metric-label">Brightness (K)</div>
              <div class="metric-value">{br_r:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card {'green' if conf_r > 70 else 'orange'}">
              <div class="metric-label">Confiance MODIS</div>
              <div class="metric-value">{conf_r}%</div>
            </div>""", unsafe_allow_html=True)

        # Probabilités par classe
        st.markdown("#### Probabilités du modèle ResNet-1D")
        noms_affich = ["🌿 Végétation (0)", "🌋 Volcan (2)", "🔥 Autre (3)"]
        couleurs    = ["#e74c3c", "#f39c12", "#3498db"]
        for i, (nom, c) in enumerate(zip(noms_affich, couleurs)):
            p = float(probas[i])
            st.markdown(f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;font-size:.85rem;font-weight:600;margin-bottom:4px">
                <span>{nom}</span><span>{p*100:.1f}%</span>
              </div>
              <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{c}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:.8rem;color:#888;margin-top:8px">
          Type prédit : <b>{res['emoji_classe']} {res['nom_classe']}</b> &nbsp;·&nbsp;
          Confiance modèle : <b>{res['confiance_mod']*100:.1f}%</b>
        </div>""", unsafe_allow_html=True)

    # ── Colonne droite : carte ────────────────────────────────────────────────
    with col_side:
        st.markdown("#### 🗺️ Localisation du point analysé")

        couleur_marker = {
            "CRITIQUE": "red", "ÉLEVÉ": "orange",
            "MODÉRÉ": "beige", "FAIBLE": "green",
            "AUCUN RISQUE DÉTECTÉ": "blue",
        }.get(risque["niveau"], "red")

        m_carte = folium.Map(location=[lat_r, lon_r], zoom_start=7,
                             tiles="CartoDB positron")

        rayon = min(max(frp_r * 300, 5000), 80000)
        folium.Circle(
            location=[lat_r, lon_r], radius=rayon,
            color=couleur_marker, fill=True, fill_opacity=0.2,
            popup=f"Rayon estimé : {rayon/1000:.0f} km"
        ).add_to(m_carte)

        folium.Marker(
            location=[lat_r, lon_r],
            popup=folium.Popup(
                f"<b>{risque['emoji']} {risque['niveau']}</b><br/>"
                f"Type : {res['nom_classe']}<br/>"
                f"FRP : {frp_r:.1f} MW/sr/m²<br/>"
                f"Brightness : {br_r:.1f} K<br/>"
                f"Confiance : {conf_r}%<br/>"
                f"({lat_r:.3f}°N, {lon_r:.3f}°E)",
                max_width=250
            ),
            tooltip=f"{risque['emoji']} {risque['niveau']} — cliquer pour détails",
            icon=folium.Icon(color=couleur_marker, icon="fire", prefix="fa")
        ).add_to(m_carte)

        st_folium(m_carte, width=520, height=420, key="carte_principale")

        st.markdown(f"""
        <div style="background:#f8fafc;border-radius:8px;padding:12px 16px;font-size:.85rem;margin-top:8px">
          📍 <b>Coordonnées :</b> {lat_r:.4f}°N, {lon_r:.4f}°E<br/>
          📅 <b>Période :</b> Mois {mois_r} — {heure_r:02d}h00 — {dn_r}<br/>
          🛰️ <b>Satellite :</b> {sat_r}
        </div>""", unsafe_allow_html=True)

else:
    # ── État initial : aucune analyse encore effectuée ────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:50px 20px;color:#999">
      <div style="font-size:4rem;margin-bottom:16px">🛰️</div>
      <div style="font-size:1.2rem;font-weight:600;margin-bottom:8px">Configurez les paramètres</div>
      <div style="font-size:.9rem">
        Ajustez les valeurs dans le panneau de gauche ou choisissez<br/>
        un scénario de démonstration, puis cliquez sur <b>Analyser le risque</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)
    m_default = folium.Map(location=[39.0, 35.0], zoom_start=5, tiles="CartoDB positron")
    st_folium(m_default, width=None, height=320, key="carte_defaut")


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORIQUE DES ANALYSES
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["historique"]:
    st.markdown("---")
    st.markdown("## 📋 Historique des analyses")
    st.caption(f"{len(st.session_state['historique'])} analyse(s) effectuée(s) — 20 dernières conservées")

    # Bouton de réinitialisation
    col_btn, col_empty = st.columns([1, 5])
    with col_btn:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state["historique"] = []
            st.rerun()

    hist = st.session_state["historique"]

    # ── Cartes compactes pour chaque analyse ─────────────────────────────────
    COULEURS_NIV = {
        "CRITIQUE":             "#e74c3c",
        "ÉLEVÉ":               "#e67e22",
        "MODÉRÉ":              "#f1c40f",
        "FAIBLE":              "#27ae60",
        "AUCUN RISQUE DÉTECTÉ": "#2980b9",
    }

    for i, h in enumerate(hist):
        couleur = COULEURS_NIV.get(h["risque"], "#888")
        bg      = "fff8f8" if h["risque"] == "CRITIQUE" else \
                  "fff5ee" if h["risque"] == "ÉLEVÉ"   else \
                  "fffff0" if h["risque"] == "MODÉRÉ"  else \
                  "f0fff0" if h["risque"] == "FAIBLE"  else "f0f8ff"

        st.markdown(f"""
        <div style="background:#{bg};border-left:5px solid {couleur};border-radius:10px;
                    padding:14px 18px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,.07)">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
            <div>
              <span style="font-size:1.3rem">{h['emoji']}</span>
              <span style="font-weight:800;font-size:1rem;color:{couleur};margin-left:8px">{h['risque']}</span>
              <span style="color:#666;font-size:.85rem;margin-left:10px">— {h['type']}</span>
            </div>
            <span style="font-size:.78rem;color:#aaa;font-family:monospace">#{len(hist)-i} &nbsp; {h['horodatage']}</span>
          </div>
          <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:10px;font-size:.83rem;color:#555">
            <span>📍 {h['lat']}, {h['lon']}</span>
            <span>🔥 FRP : <b>{h['frp']}</b> MW</span>
            <span>🌡️ Bright : <b>{h['brightness']}</b> K</span>
            <span>📊 Confiance : <b>{h['confiance']}</b></span>
            <span>📅 {h['mois']}</span>
          </div>
          <div style="display:flex;gap:16px;margin-top:8px;font-size:.8rem;color:#777">
            <span>🌿 Vég : <b>{h['prob_veg']}</b></span>
            <span>🌋 Vol : <b>{h['prob_vol']}</b></span>
            <span>🔥 Aut : <b>{h['prob_aut']}</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tableau récapitulatif exportable ─────────────────────────────────────
    with st.expander("📊 Tableau récapitulatif de l'historique"):
        df_hist = pd.DataFrame(hist)[
            ["horodatage", "emoji", "risque", "type", "lat", "lon",
             "frp", "brightness", "confiance", "mois", "prob_veg", "prob_vol"]
        ]
        df_hist.columns = ["Heure", "", "Risque", "Type détecté", "Latitude", "Longitude",
                           "FRP (MW)", "Brightness (K)", "Confiance", "Mois",
                           "P(Végét.)", "P(Volcan)"]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # Export CSV
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger l'historique (CSV)",
            data=csv,
            file_name=f"historique_firerisk_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#aaa;font-size:.8rem;padding:8px">
  🔥 FireRisk AI &nbsp;·&nbsp; Modèle ResNet-1D entraîné sur MODIS NASA FIRMS &nbsp;·&nbsp;
  Turquie 2020 &nbsp;·&nbsp; Projet de Fin d'Année — Deep Learning
</div>
""", unsafe_allow_html=True)
