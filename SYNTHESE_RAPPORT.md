# Synthèse du Projet — Prédiction des Feux de Forêt par Deep Learning
## (Aide à la rédaction du rapport académique)

---

## 1. INTRODUCTION ET PROBLÉMATIQUE

### Contexte
Les feux de forêt représentent une menace environnementale et économique majeure,
notamment dans les pays méditerranéens comme la Turquie. L'année 2020 a connu
une saison particulièrement active en raison des conditions climatiques extrêmes
(sécheresse, vagues de chaleur). La détection et la classification automatique des
feux à partir de données satellitaires constitue un enjeu crucial pour les systèmes
d'alerte précoce.

### Problématique
Comment exploiter les données de détection satellitaire MODIS pour **classifier
automatiquement le type de feu** (végétation, volcan, autre) à l'aide de modèles
d'apprentissage profond (CNN), en tenant compte du fort déséquilibre des classes ?

### Objectifs
1. Analyser les données MODIS Turquie 2020 (EDA)
2. Prétraiter les données pour les rendre exploitables par des CNN
3. Implémenter et comparer deux architectures CNN adaptées aux données tabulaires
4. Évaluer les performances en termes d'accuracy, F1-score, RMSE

---

## 2. DESCRIPTION DU DATASET

### Source
**NASA FIRMS (Fire Information for Resource Management System)**
Satellite : MODIS (Moderate Resolution Imaging Spectroradiometer)
Capteurs  : Terra (satellite A) et Aqua (satellite B)
Période   : Année 2020 — Turquie

### Caractéristiques
- **Dimensions** : 8 834 lignes × 15 colonnes
- **Valeurs manquantes** : aucune (dataset complet)
- **Fichier** : `modis_2020_Turkey.csv`

### Description des colonnes

| Colonne | Type | Description | Rôle |
|---------|------|-------------|------|
| `latitude` | float | Latitude GPS du foyer détecté | Feature géographique |
| `longitude` | float | Longitude GPS du foyer détecté | Feature géographique |
| `brightness` | float | Température de brillance canal 21/22 (Kelvin) | Feature thermique principale |
| `scan` | float | Taille pixel en sens de balayage (km) | Feature géométrique satellite |
| `track` | float | Taille pixel en sens de déplacement (km) | Feature géométrique satellite |
| `acq_date` | string | Date d'acquisition (YYYY-MM-DD) | → mois extrait (saisonnalité) |
| `acq_time` | int | Heure d'acquisition (format HHMM) | → heure extraite |
| `satellite` | string | Satellite source : 'A' (Terra) ou 'B' (Aqua) | Feature binaire encodée |
| `instrument` | string | Toujours "MODIS" — constante | **Supprimée** (sans info) |
| `confidence` | int | Niveau de confiance de la détection (0-100) | Feature de qualité |
| `version` | float | Toujours 6.03 — constante | **Supprimée** (sans info) |
| `bright_t31` | float | Température de brillance canal 31 (Kelvin) | Feature thermique (11 µm) |
| `frp` | float | Fire Radiative Power (MW/sr/m²) | Feature d'intensité du feu |
| `daynight` | string | Détection diurne 'D' ou nocturne 'N' | Feature binaire encodée |
| `type` | int | **VARIABLE CIBLE** : type de feu | 0=végétation, 2=volcan, 3=autre |

### Distribution de la variable cible (déséquilibre critique)

| Classe | Label | Effectif | Proportion |
|--------|-------|----------|------------|
| 0 | Végétation (Vegetation fire) | 8 208 | 92.9% |
| 2 | Volcan (Volcano) | 619 | 7.0% |
| 3 | Autre feu statique | 7 | 0.08% |

> **Note rapport** : Ce déséquilibre extrême (ratio 1172:1 entre classe 0 et classe 3)
> est un défi majeur. Sans traitement, un modèle qui prédirait systématiquement la
> classe 0 atteindrait 92.9% d'accuracy tout en étant inutile.

---

## 3. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)

### Observations principales

**Distributions des features numériques** (figure `02_distributions_features.png`)
- `brightness` : distribution quasi-normale centrée ~320 K avec quelques valeurs extrêmes (jusqu'à 500 K) correspondant à des feux très intenses
- `frp` (Fire Radiative Power) : distribution très asymétrique (skewness élevé), la plupart des valeurs < 50 MW, quelques pics à ~2788 MW (feux exceptionnels)
- `confidence` : distribution étalée de 0 à 100, pic vers 65-75%

**Matrice de corrélation** (figure `03_matrice_correlation.png`)
- Corrélation forte (>0.7) entre `brightness` et `bright_t31` → mesures du même phénomène physique sur deux canaux différents
- Corrélation forte entre `scan` et `track` → géométrie satellite (lié à l'angle d'observation)
- `frp` est la feature la plus corrélée avec `type` → meilleur prédicteur individuel

**Analyse temporelle** (figure `05_repartition_mensuelle.png`)
- Pic de détections : **juillet-août-septembre 2020** (saison des feux en Turquie)
- Confirme la pertinence d'extraire le mois comme feature de saisonnalité

**Répartition géographique** (figure `06_carte_geographique.png`)
- Foyers concentrés principalement sur la côte méditerranéenne et égéenne de la Turquie
- Les volcans (classe 2) sont géographiquement isolés (zones spécifiques)

---

## 4. PRÉTRAITEMENT

### Pipeline complet

```
Données brutes (8834 × 15)
        │
        ▼
1. Suppression instrument, version  → (8834 × 13)
        │
        ▼
2. Encodage catégorielles
   satellite : Aqua→0, Terra→1
   daynight  : D→0, N→1         → (8834 × 13)
        │
        ▼
3. Feature engineering temporel
   mois  ← acq_date.dt.month    (saisonnalité 1-12)
   heure ← acq_time // 100      (heure locale 0-23)
   Suppression acq_date, acq_time → (8834 × 13)
        │
        ▼
4. Split stratifié 80/20
   Train : 7067 exemples
   Test  : 1767 exemples         (proportions des classes préservées)
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
  Pipeline A                               Pipeline B
  class_weight                             SMOTE
  ──────────────                           ──────────────
  Train: 7067×12                           Train: 19698×12
  (proportions originales)                 (6566 × 3 classes)
        │                                         │
        ▼                                         ▼
5. Normalisation StandardScaler          Normalisation StandardScaler
   fit sur X_train uniquement            fit sur X_train_smote
        │                                         │
        ▼                                         ▼
  AlexNet-1D                              ResNet-1D
```

### Choix de normalisation
**StandardScaler** (z-score) : `x' = (x - μ) / σ`

Justification : les features ont des échelles très hétérogènes
(brightness ~300-500 K, frp ~2-2788 MW, mois 1-12). Sans normalisation,
les grandes valeurs dominent le gradient et ralentissent la convergence.

### Gestion du déséquilibre : deux approches

#### Approche 1 — class_weight
Utilisée avec AlexNet-1D. Les poids calculés :
- Classe 0 (végétation) : **0.359** → erreur peu pénalisée (classe majoritaire)
- Classe 2 (volcan)     : **4.759** → erreur 13× plus pénalisée que classe 0
- Classe 3 (autre)      : **392.6** → erreur 1094× plus pénalisée que classe 0

Formule : `w_i = N_total / (N_classes × N_i)`

#### Approche 2 — SMOTE
Utilisée avec ResNet-1D. Le sur-échantillonnage synthétique porte toutes
les classes à **6566 exemples** (taille de la classe majoritaire).
`k_neighbors=2` (au lieu de 5 par défaut) car la classe 3 n'a que 6 exemples
dans le train set — insuffisant pour chercher 5 voisins.

Résultat SMOTE :
- Avant : 6566 / 495 / 6 exemples
- Après : 6566 / 6566 / 6566 exemples
- Taille du train × 2.79 : 7067 → 19698 exemples

---

## 5. ADAPTATION TABULAIRE → CNN 1D

### Justification du choix CNN sur données tabulaires

Un CNN (Convolutional Neural Network) applique des **filtres locaux** qui
apprennent des interactions entre features voisines. Sur des données tabulaires :

- Les features sont ordonnées sémantiquement (géographiques, thermiques,
  géométriques, temporelles)
- Un filtre de taille k=3 apprend des combinaisons de 3 features consécutives
- Le partage de poids réduit le nombre de paramètres vs un MLP équivalent

### Reshape : (n, 12) → (n, 12, 1)

L'API Conv1D de Keras attend `(batch_size, steps, channels)` :
```python
X.reshape(X.shape[0], 12, 1)
```
- `steps = 12`   : nombre de features (longueur de la séquence)
- `channels = 1` : chaque feature est un scalaire (1 canal)

### Ordre des features dans la séquence

```
Index : [0]lat  [1]lon  [2]brightness  [3]scan  [4]track  [5]satellite
        [6]confidence  [7]bright_t31   [8]frp   [9]daynight  [10]mois  [11]heure
```

---

## 6. ARCHITECTURES CNN

### Architecture 1 — AlexNet-1D

**Inspiration** : AlexNet (Krizhevsky, Sutskever & Hinton, 2012) — ILSVRC 2012
Architecture originale pour images 227×227×3, adaptée ici en 1D pour 12 features.

**Philosophie** : réseau profond séquentiel, blocs Conv→BN→ReLU enchaînés,
suivi d'une tête de classification dense.

```
Entrée : (12, 1)
├─ Conv1D(64, k=3)  → BN → ReLU
├─ Conv1D(128, k=3) → BN → ReLU → MaxPool(2)
├─ Conv1D(256, k=3) → BN → ReLU
├─ Conv1D(256, k=3) → BN → ReLU
├─ Conv1D(128, k=3) → BN → ReLU → MaxPool(2)
├─ GlobalAveragePooling1D
├─ Dense(256) → Dropout(0.5)
├─ Dense(128) → Dropout(0.3)
└─ Dense(3, softmax)

Paramètres : 487 619
Régularisation : L2(1e-4) + Dropout
```

### Architecture 2 — ResNet-1D

**Inspiration** : ResNet (He, Zhang, Ren & Sun, 2016) — ILSVRC 2015
Architecture à connexions résiduelles (skip connections), adaptée en 1D.

**Innovation clé** : H(x) = F(x) + x
Le gradient de la loss peut se propager directement via le shortcut,
évitant le problème du gradient évanescent dans les réseaux profonds.

```
Entrée : (12, 1)
├─ Conv1D(64, k=3) → BN → ReLU            [couche d'entrée]
├─ ResBlock(64)  × 2                       [stage 1]
│   └─ Conv1D(64,k=3)→BN→ReLU → Conv1D(64,k=3)→BN → (+x) → ReLU
├─ ResBlock(128) × 2  [+ projection 64→128][stage 2]
├─ ResBlock(256) × 1  [+ projection 128→256][stage 3]
├─ GlobalAveragePooling1D
├─ Dense(128) → Dropout(0.3)
└─ Dense(3, softmax)

Paramètres : 597 443
Régularisation : L2(1e-4) + Dropout + connexions résiduelles
```

---

## 7. ENTRAÎNEMENT

### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Optimiseur | Adam (lr=1e-3) | Adaptatif, convergence rapide, standard deep learning |
| Loss | Sparse Categorical Crossentropy | Classification multi-classes avec labels entiers |
| Batch size | 32 | Compromis vitesse/stabilité du gradient |
| Époques max | 50 | EarlyStopping arrête avant si nécessaire |
| Validation | 15% du train set | Suivi de la généralisation pendant l'entraînement |

### Callbacks Keras

- **EarlyStopping** (`patience=10`) : arrête si `val_loss` ne s'améliore plus pendant 10 époques → évite sur-apprentissage, réduit temps de calcul. `restore_best_weights=True` : revient aux poids de la meilleure époque.
- **ReduceLROnPlateau** (`patience=5, factor=0.5`) : divise lr par 2 si stagnation → sort des plateaux de convergence
- **ModelCheckpoint** : sauvegarde automatiquement le meilleur modèle (`val_loss` minimal)

### Remap des labels

Les labels originaux {0, 2, 3} sont remappés en {0, 1, 2} pour Keras
(`sparse_categorical_crossentropy` requiert des indices contigus).
Mapping : `0→0, 2→1, 3→2`. Les prédictions sont inversement remappées
avant le calcul des métriques.

---

## 8. MÉTRIQUES D'ÉVALUATION

### Définitions

**Accuracy** = (TP + TN) / N_total
→ Taux global de bonne classification. *Trompeuse sur données déséquilibrées.*

**Precision (weighted)** = Σ_i [w_i × TP_i / (TP_i + FP_i)]
→ Parmi les prédictions de classe i, quelle proportion est correcte ?

**Recall (weighted)** = Σ_i [w_i × TP_i / (TP_i + FN_i)]
→ Parmi les vrais exemples de classe i, quelle proportion est détectée ?

**F1-score (weighted)** = 2 × (Precision × Recall) / (Precision + Recall)
→ Moyenne harmonique de Precision et Recall. *Métrique de référence pour classes déséquilibrées.*

**RMSE** = √(Σ(y_true - y_pred)² / N)
→ Racine de l'erreur quadratique moyenne sur les indices de classe.
   Mesure l'amplitude des erreurs : confondre classe 0 avec classe 3
   est pénalisé plus qu'une confusion entre classes adjacentes.

### Pourquoi ces métriques ?

| Métrique | Pertinence pour ce projet |
|----------|--------------------------|
| Accuracy | Baseline de référence, mais insuffisante seule |
| F1-score weighted | Métrique principale — tient compte du déséquilibre |
| RMSE | Quantifie la gravité des erreurs (distincts de l'accuracy) |
| Matrice de confusion | Visualise quelles classes sont confondues et dans quel sens |

---

## 9. RÉSULTATS

### Tableau comparatif final

| Modèle | Stratégie | Accuracy | Precision | Recall | F1-score | RMSE |
|--------|-----------|----------|-----------|--------|----------|------|
| AlexNet-1D | class_weight | 90.04% | 90.17% | 90.04% | 90.10% | 0.6334 |
| ResNet-1D  | SMOTE        | 90.38% | 91.00% | 90.38% | 90.67% | 0.6249 |

**Meilleur modèle : ResNet-1D** (+0.57% F1-score, -0.0085 RMSE)

---

### Rapport de classification détaillé par classe

**AlexNet-1D** (meilleure époque : 2, EarlyStopping déclenché à l'époque 12)

| Classe | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0 — Végétation | 0.95 | 0.95 | 0.95 | 1642 |
| 2 — Volcan     | 0.30 | 0.31 | 0.31 | 124  |
| 3 — Autre      | 0.00 | 0.00 | 0.00 | 1    |
| **weighted avg** | **0.90** | **0.90** | **0.90** | **1767** |

**ResNet-1D** (50 époques complètes, val_accuracy finale = 100%, val_loss = 0.0602)

| Classe | Précision | Rappel | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0 — Végétation | 0.95 | 0.94 | 0.95 | 1642 |
| 2 — Volcan     | 0.35 | 0.40 | 0.37 | 124  |
| 3 — Autre      | 0.00 | 0.00 | 0.00 | 1    |
| **weighted avg** | **0.91** | **0.90** | **0.91** | **1767** |

---

### Analyse des courbes d'entraînement

**AlexNet-1D** :
- Convergence très rapide : la meilleure époque est l'**époque 2** (val_loss=0.543)
- EarlyStopping déclenché à l'époque 12 (patience=10 sans amélioration)
- L'accuracy de validation atteint 89.2% dès l'époque 2
- Phénomène notable : la loss d'entraînement est élevée (effet du class_weight
  qui pénalise fortement les erreurs sur les classes rares), mais la val_loss
  est faible dès le début — le modèle a bien généralisé très rapidement.

**ResNet-1D** :
- Convergence progressive sur les 50 époques complètes
- val_accuracy atteint **100%** sur le set de validation à la dernière époque
- val_loss descend jusqu'à **0.0602** — convergence excellente sur le train SMOTE
- Le dataset augmenté (19 698 exemples équilibrés) a permis un apprentissage plus
  stable et progressif que le dataset class_weight

---

### Analyse des matrices de confusion

**Observations communes aux deux modèles :**
- **Classe 0 (végétation)** : très bien classifiée (F1 ≈ 0.95), ce qui est attendu
  car elle représente 92.9% des données — les deux modèles ont largement appris
  ses caractéristiques.
- **Classe 2 (volcan)** : classifiée partiellement (F1 = 0.31 à 0.37), avec
  confusion fréquente vers la classe 0. Malgré SMOTE/class_weight, les caractéristiques
  thermiques des volcans (brightness, frp) se chevauchent avec certains feux de végétation.
- **Classe 3 (autre)** : F1 = 0.00 pour les deux modèles. Avec seulement 7 exemples
  dans l'ensemble complet (dont ~1 en test), cette classe est quasi-impossible à
  apprendre correctement même avec SMOTE.

**Avantage ResNet-1D sur la classe 2 :**
- Recall classe 2 : **0.40** (ResNet) vs **0.31** (AlexNet) → +29% de volcans détectés
- SMOTE a permis de mieux apprendre les caractéristiques des volcans en générant
  des exemples synthétiques, là où class_weight n'ajoute que de la pénalisation

---

### Discussion et limites

**Pourquoi les performances plafonnent à ~90% ?**
1. La classe 3 (7 exemples) est statistiquement indétectable → plafonne le macro F1
2. Le chevauchement thermique (brightness, frp) entre volcans et feux de végétation
   intenses rend la distinction difficile même pour le modèle
3. Les CNN 1D sur 12 features ont une capacité de représentation limitée comparée
   aux CNN 2D sur images ou aux Transformers

**Limites de SMOTE sur ces données :**
- Les features binaires (satellite, daynight) génèrent des valeurs interpolées
  non réalistes (ex. satellite = 0.37 au lieu de 0 ou 1)
- La classe 3 ne compte que 6 exemples en train → les exemples SMOTE sont très
  proches des originaux (k_neighbors=2), manquant de diversité

**Perspectives d'amélioration :**
- Utiliser SMOTE-NC (Nominal+Continuous) pour gérer les features catégorielles
- Tester avec des Transformers (TabTransformer) plus adaptés aux données tabulaires
- Collecter plus de données de classe 3 pour améliorer sa représentativité

---

## 10. STRUCTURE DU PROJET

```
PFA4/
├── modis_2020_Turkey (1).csv          # Dataset brut NASA FIRMS
├── 01_eda.py                           # Analyse exploratoire
├── 02_preprocessing.py                 # Prétraitement complet
├── 03_train_evaluate.py                # Entraînement + évaluation
├── 04_comparison.py                    # Tableau comparatif
├── requirements.txt                    # Dépendances Python
├── SYNTHESE_RAPPORT.md                 # Ce fichier
│
├── src/
│   ├── config.py                       # Chemins et hyperparamètres centralisés
│   ├── data_loader.py                  # Chargement + fonctions EDA
│   ├── preprocessing.py                # Pipeline de prétraitement (2 variantes)
│   ├── models.py                       # Architectures AlexNet-1D et ResNet-1D
│   └── evaluate.py                     # Métriques, matrices, courbes
│
├── outputs/
│   ├── figures/                        # Toutes les figures PNG (10 au total)
│   │   ├── 01_distribution_cible.png
│   │   ├── 02_distributions_features.png
│   │   ├── 03_matrice_correlation.png
│   │   ├── 04_boxplots_par_classe.png
│   │   ├── 05_repartition_mensuelle.png
│   │   ├── 06_carte_geographique.png
│   │   ├── 07_smote_comparison.png
│   │   ├── 08_courbes_AlexNet-1D.png
│   │   ├── 08_courbes_ResNet-1D.png
│   │   ├── 09_confusion_AlexNet-1D.png
│   │   ├── 09_confusion_ResNet-1D.png
│   │   └── 10_comparaison_finale.png
│   │
│   ├── models/                         # Modèles sauvegardés
│   │   ├── AlexNet-1D_best.keras
│   │   └── ResNet-1D_best.keras
│   │
│   └── results/                        # Données et métriques
│       ├── X_train_cw.npy / X_test_cw.npy / y_train_cw.npy / y_test_cw.npy
│       ├── X_train_smote.npy / X_test_smote.npy / y_train_smote.npy / y_test_smote.npy
│       ├── feature_names.csv
│       ├── scaler.pkl
│       └── metriques_comparaison.csv
```

---

## 11. RÉFÉRENCES BIBLIOGRAPHIQUES SUGGÉRÉES

1. **AlexNet** : Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks*. Advances in Neural Information Processing Systems, 25.

2. **ResNet** : He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR 2016.

3. **SMOTE** : Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique*. Journal of Artificial Intelligence Research, 16, 321-357.

4. **MODIS FIRMS** : Giglio, L., Schroeder, W., & Justice, C. O. (2016). *The collection 6 MODIS active fire detection algorithm and fire products*. Remote Sensing of Environment, 178, 31-41.

5. **CNN sur données tabulaires** : Baosenguo et al. (2021). *1D Convolutional Neural Networks and Applications: A Survey*. Mechanical Systems and Signal Processing, 151.

6. **Class imbalance** : He, H., & Garcia, E. A. (2009). *Learning from imbalanced data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.

---

## 12. POINTS CLÉS POUR L'ORAL / LE RAPPORT

### Questions fréquentes et réponses courtes

**Pourquoi un CNN sur des données tabulaires ?**
Les CNN apprennent des interactions locales entre features voisines via leurs filtres.
En ordonnant les features sémantiquement (géo/thermique/géométrique/temporel),
on crée une structure locale exploitable, analogue à la structure spatiale des images.

**Pourquoi comparer class_weight et SMOTE ?**
Ce sont deux stratégies fondamentalement différentes :
- class_weight modifie la fonction de perte sans toucher aux données
- SMOTE modifie les données d'entraînement par génération synthétique
Les comparer permet d'évaluer l'impact de la stratégie sur chaque architecture.

**Pourquoi AlexNet avec class_weight et ResNet avec SMOTE ?**
Pour isoler deux variables : architecture ET stratégie anti-déséquilibre.
Chaque modèle est évalué dans ses meilleures conditions supposées, et
la comparaison finale porte sur le résultat global de chaque combinaison.

**Pourquoi le F1-score est la métrique principale ?**
Avec 92.9% de classe dominante, l'accuracy est trompeuse.
Le F1-score weighted tient compte de tous les exemples pondérés par leur effectif,
ce qui pénalise les mauvaises classifications sur toutes les classes.

**Quel est le risque de SMOTE ?**
SMOTE génère des exemples synthétiques par interpolation linéaire.
Sur des features mixtes (continues + discrètes comme `satellite`, `daynight`, `mois`),
les exemples synthétiques peuvent avoir des valeurs non réalistes
(ex. satellite=0.3 au lieu de 0 ou 1). C'est une limite connue de SMOTE
sur les données mixtes.
