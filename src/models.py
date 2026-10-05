"""
models.py — Adaptation tabulaire → 1D-CNN et définition des deux architectures.

═══════════════════════════════════════════════════════════════════════════════
POURQUOI UN CNN SUR DES DONNÉES TABULAIRES ?
═══════════════════════════════════════════════════════════════════════════════

Un CNN (Convolutional Neural Network) est conçu pour exploiter la structure
locale des données : en vision, un pixel est lié à ses voisins ; en texte,
un mot est lié aux mots adjacents.

Sur des données tabulaires, l'idée est la suivante :
→ On ordonne les features de façon sémantiquement cohérente (ex. features
  physiques ensemble, features temporelles ensemble).
→ Un filtre Conv1D de taille k apprend des INTERACTIONS entre k features
  consécutives, ce que fait un MLP uniquement avec des couches plus profondes.
→ La convolution partage ses poids sur toute la séquence, réduisant le
  nombre de paramètres et améliorant la généralisation.

═══════════════════════════════════════════════════════════════════════════════
LE RESHAPE : (n, 12) → (n, 12, 1)
═══════════════════════════════════════════════════════════════════════════════

Un Conv1D Keras attend des entrées de forme (batch, steps, channels) :
  - steps   = longueur de la séquence = nombre de features = 12
  - channels = nombre de "canaux" = 1 (chaque feature est un scalaire)

On ajoute donc une dimension de canal :
    X.reshape(-1, 12, 1)   ou   X[..., np.newaxis]

C'est analogue à transformer une image en niveaux de gris (1 canal) en RGB
(3 canaux) : on passe de (H, W) à (H, W, 1).

Ordre des features dans la séquence (important pour la convolution) :
    [0]  latitude      ← position géographique
    [1]  longitude     ← position géographique
    [2]  brightness    ← intensité thermique (feature principale)
    [3]  scan          ← géométrie satellite
    [4]  track         ← géométrie satellite
    [5]  satellite     ← source instrumentale
    [6]  confidence    ← fiabilité détection
    [7]  bright_t31    ← température 11µm (corrélée à brightness)
    [8]  frp           ← puissance radiative (feature principale)
    [9]  daynight      ← condition lumineuse
    [10] mois          ← saisonnalité
    [11] heure         ← moment de la journée

Les features physiques proches (brightness/bright_t31, scan/track, lat/lon)
sont regroupées pour maximiser l'information locale capturée par les filtres.

═══════════════════════════════════════════════════════════════════════════════
DEUX ARCHITECTURES COMPARÉES
═══════════════════════════════════════════════════════════════════════════════

1. AlexNet-1D : architecture profonde séquentielle
   Inspirée de AlexNet (Krizhevsky et al., 2012), pionnière de l'ère
   deep learning. Architecture linéaire avec plusieurs blocs Conv→Pool→FC.

2. ResNet-1D : architecture à connexions résiduelles
   Inspirée de ResNet (He et al., 2016). Les connexions skip (raccourcis)
   permettent au gradient de circuler directement vers les premières couches,
   résolvant le problème du gradient évanescent dans les réseaux profonds.
   Formule d'un bloc résiduel : H(x) = F(x) + x
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ═══════════════════════════════════════════════════════════════════════
# RESHAPE : DONNÉES TABULAIRES → FORMAT 1D-CNN
# ═══════════════════════════════════════════════════════════════════════

def reshape_pour_cnn(X: np.ndarray) -> np.ndarray:
    """
    Transforme des données tabulaires (n, n_features) en (n, n_features, 1).

    Pourquoi ?
    Keras Conv1D attend (batch_size, steps, channels).
    Chaque feature devient un "pas de temps" avec 1 canal.
    """
    return X.reshape(X.shape[0], X.shape[1], 1)


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 1 : ALEXNET-INSPIRED 1D-CNN
# ═══════════════════════════════════════════════════════════════════════

def construire_alexnet_1d(n_features: int = 12,
                          n_classes: int = 3,
                          learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    CNN séquentiel inspiré d'AlexNet, adapté aux données 1D tabulaires.

    ─── INSPIRATION ALEXNET ────────────────────────────────────────────
    AlexNet original (2012) : traitement d'images 227×227×3
        Conv(96, 11×11, stride=4) → LRN → Pool(3×3, stride=2)
        Conv(256, 5×5) → LRN → Pool
        Conv(384, 3×3) → Conv(384, 3×3) → Conv(256, 3×3) → Pool
        FC(4096) → FC(4096) → FC(1000, softmax)

    Notre adaptation 1D (entrée 12×1) :
        → Même philosophie : plusieurs blocs conv successifs + FC finale
        → Filtres plus petits (taille 3) car séquence courte (12 pas)
        → Batch Normalization à la place de LRN (plus stable)
        → GlobalAveragePooling remplace le Flatten (moins de paramètres)
        → Dropout pour régulariser (éviter le sur-apprentissage)

    Architecture :
        Entrée (12, 1)
        ├─ Bloc 1 : Conv1D(64, 3) → BN → ReLU
        ├─ Bloc 2 : Conv1D(128, 3) → BN → ReLU → MaxPool(2)
        ├─ Bloc 3 : Conv1D(256, 3) → BN → ReLU
        ├─ Bloc 4 : Conv1D(256, 3) → BN → ReLU
        ├─ Bloc 5 : Conv1D(128, 3) → BN → ReLU → MaxPool(2)
        ├─ GlobalAveragePooling1D
        ├─ Dense(256) → Dropout(0.5)
        ├─ Dense(128) → Dropout(0.3)
        └─ Dense(3, softmax)
    ─────────────────────────────────────────────────────────────────────
    """
    inp = tf.keras.Input(shape=(n_features, 1), name="input")

    # ── Bloc 1 ─────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)

    # ── Bloc 2 (+ pooling) ─────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── Bloc 3 ─────────────────────────────────────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    # ── Bloc 4 ─────────────────────────────────────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)

    # ── Bloc 5 (+ pooling) ─────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu", name="relu5")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── Tête de classification (FC) ────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.5, name="drop1")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="AlexNet_1D")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 2 : RESNET-INSPIRED 1D-CNN
# ═══════════════════════════════════════════════════════════════════════

def _bloc_residuel(x, n_filtres: int, nom: str):
    """
    Bloc résiduel : F(x) + x  (identité shortcut).

    ─── PRINCIPE DU BLOC RÉSIDUEL ──────────────────────────────────────
    Dans un réseau classique : la sortie est H(x) = F(x)
    Dans un réseau résiduel  : la sortie est H(x) = F(x) + x

    Avantage mathématique :
    Le gradient de la perte par rapport à x vaut :
        ∂L/∂x = ∂L/∂H · (∂F/∂x + 1)

    Le "+1" garantit qu'un gradient non nul se propage toujours,
    même si ∂F/∂x ≈ 0 (gradient évanescent évité).

    Structure du bloc :
        x ─→ Conv1D → BN → ReLU → Conv1D → BN ─→ + ─→ ReLU
        └──────────────────────────────────────────┘
    ─────────────────────────────────────────────────────────────────────

    Si le nombre de filtres change, on utilise une projection linéaire
    (Conv1D 1×1) pour adapter la dimension du shortcut.
    """
    shortcut = x

    # Première convolution du bloc
    x = layers.Conv1D(n_filtres, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"{nom}_conv_a")(x)
    x = layers.BatchNormalization(name=f"{nom}_bn_a")(x)
    x = layers.Activation("relu", name=f"{nom}_relu_a")(x)

    # Deuxième convolution du bloc
    x = layers.Conv1D(n_filtres, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"{nom}_conv_b")(x)
    x = layers.BatchNormalization(name=f"{nom}_bn_b")(x)

    # Projection du shortcut si les dimensions ne correspondent pas
    if shortcut.shape[-1] != n_filtres:
        shortcut = layers.Conv1D(n_filtres, kernel_size=1, padding="same",
                                 use_bias=False, name=f"{nom}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{nom}_proj_bn")(shortcut)

    # Connexion résiduelle : addition de la branche principale et du shortcut
    x = layers.Add(name=f"{nom}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{nom}_relu_out")(x)
    return x


def construire_resnet_1d(n_features: int = 12,
                         n_classes: int = 3,
                         learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    CNN résiduel inspiré de ResNet, adapté aux données 1D tabulaires.

    ─── INSPIRATION RESNET ─────────────────────────────────────────────
    ResNet original (He et al., 2016) : traitement d'images 224×224×3
        Conv(64, 7×7, stride=2) → BN → ReLU → MaxPool
        2× ResBlock(64) → 2× ResBlock(128) → 2× ResBlock(256)
        → GlobalAvgPool → FC(1000, softmax)

    Notre adaptation 1D (entrée 12×1) :
        → Même philosophie : blocs résiduels empilés
        → Moins de blocs (séquence courte) mais même structure
        → Pas de stride > 1 (séquence trop courte pour sous-échantillonner)

    Architecture :
        Entrée (12, 1)
        ├─ Conv1D(64, 3) → BN → ReLU          [couche d'entrée]
        ├─ ResBlock(64) × 2                   [stage 1]
        ├─ ResBlock(128) × 2                  [stage 2, +projection]
        ├─ ResBlock(256) × 1                  [stage 3, +projection]
        ├─ GlobalAveragePooling1D
        ├─ Dense(128) → Dropout(0.3)
        └─ Dense(3, softmax)
    ─────────────────────────────────────────────────────────────────────
    """
    inp = tf.keras.Input(shape=(n_features, 1), name="input")

    # ── Couche d'entrée ────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # ── Stage 1 : 2 blocs résiduels (64 filtres) ──────────────────
    x = _bloc_residuel(x, 64,  nom="res1a")
    x = _bloc_residuel(x, 64,  nom="res1b")

    # ── Stage 2 : 2 blocs résiduels (128 filtres) ─────────────────
    # La projection dans _bloc_residuel gère l'augmentation 64→128
    x = _bloc_residuel(x, 128, nom="res2a")
    x = _bloc_residuel(x, 128, nom="res2b")

    # ── Stage 3 : 1 bloc résiduel (256 filtres) ───────────────────
    x = _bloc_residuel(x, 256, nom="res3a")

    # ── Tête de classification ─────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.3, name="drop1")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="ResNet_1D")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# UTILITAIRE : AFFICHAGE DU RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════

def afficher_resume_modeles(n_features: int = 12, n_classes: int = 3) -> None:
    """
    Affiche le résumé des deux architectures (nb paramètres, couches).
    Permet de comparer la complexité des deux modèles avant entraînement.
    """
    print("\n" + "="*60)
    print("  ARCHITECTURE 1 : AlexNet-1D")
    print("="*60)
    m1 = construire_alexnet_1d(n_features, n_classes)
    m1.summary()

    print("\n" + "="*60)
    print("  ARCHITECTURE 2 : ResNet-1D")
    print("="*60)
    m2 = construire_resnet_1d(n_features, n_classes)
    m2.summary()

    p1 = m1.count_params()
    p2 = m2.count_params()
    print(f"\n[COMPARAISON] AlexNet-1D : {p1:,} paramètres")
    print(f"[COMPARAISON] ResNet-1D  : {p2:,} paramètres")
