# Projet NLP - Système Intelligent de Recommandations Nutritionnelles

## Description

Ce projet consiste à développer un système intelligent capable de générer des solutions nutritionnelles et des régimes alimentaires personnalisés grâce aux techniques avancées du Traitement Automatique du Langage Naturel (NLP).

Le système analyse différentes sources d'information — articles scientifiques, recommandations nutritionnelles et données fournies par l'utilisateur — afin de proposer des conseils adaptés et scientifiquement fiables.

## Structure du Projet

```
projet_nlp_nutrition/
├── scripts/                    # Scripts d'extraction
│   ├── extract_entities.py     # Extraction d'entités (aliments, nutriments, etc.)
│   └── extract_relations.py    # Extraction de relations entre entités
├── data/                       # Données extraites
│   ├── entities_extracted.csv  # Entités extraites (CSV)
│   ├── entities_extracted.json # Entités extraites (JSON)
│   ├── relations_extracted.csv # Relations extraites (CSV)
│   ├── relations_extracted.json # Relations extraites (JSON)
│   └── entities_schema_sample.csv # Schéma d'exemple
├── docs/                       # Documentation
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)
- (Optionnel) GPU NVIDIA avec 16GB+ VRAM pour le fine-tuning LLM

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Vérification de l'installation

```bash
python scripts/check_setup.py
```

Ce script vérifie que tous les fichiers et dépendances sont en place.

## Utilisation

### 1. Extraction d'entités

Pour extraire les entités nutritionnelles depuis vos PDFs :

```bash
python scripts/extract_entities.py [chemin_vers_dossier_pdfs]
```

**Exemple :**
```bash
python scripts/extract_entities.py "/home/mehrez/study/nlp/article scientifique"
```

**Résultat :** Génère `data/entities_extracted.csv` et `data/entities_extracted.json`

### 2. Extraction de relations

Pour extraire les relations entre entités (triplets sujet-relation-objet) :

```bash
python scripts/extract_relations.py [chemin_vers_dossier_pdfs]
```

**Exemple :**
```bash
python scripts/extract_relations.py "/home/mehrez/study/nlp/article scientifique"
```

**Résultat :** Génère `data/relations_extracted.csv` et `data/relations_extracted.json`

### 3. Moteur de recommandation

Pour générer des recommandations personnalisées :

```bash
python scripts/recommender.py
```

**Utilisation en Python :**
```python
from scripts.recommender import NutritionRecommender

# Initialiser
recommender = NutritionRecommender()

# Profil utilisateur
profile = {
    "condition": "anémie",
    "objectif": "santé"
}

# Obtenir les recommandations
results = recommender.get_recommendations(profile)

# Afficher les résultats
for rec in results['recommendations']:
    print(f"{rec['recommandation']}: {rec['explication']}")
```

**Voir aussi :**
- `scripts/example_usage.py` : Exemples d'utilisation détaillés
- `docs/RECOMMENDER.md` : Documentation complète du recommandeur

### 4. Fine-tuning LLM (Nouveau ⭐)

Créer un chatbot nutritionnel spécialisé avec fine-tuning :

#### Étape 1 : Préparer les données
```bash
python scripts/prepare_training_data.py
```

#### Étape 2 : Fine-tuning avec QLoRA
```bash
python scripts/train_llm.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --use_qlora \
    --num_epochs 3
```

#### Étape 3 : Utiliser le chatbot
```bash
python scripts/chat.py --model_path models/mistral-7b-nutrition
```

**Voir aussi :**
- `docs/LLM_TRAINING.md` : Guide complet du fine-tuning

## Types d'Entités Extraites

- **ALIMENT** : Nom de l'aliment, ingrédient ou plat
- **NUTRIMENT** : Molécule spécifique (macro ou micro-nutriment)
- **MESURE/VALEUR** : Quantités, dosages, pourcentages, valeurs énergétiques
- **OBJECTIF_CORPS** : État corporel souhaité ou maintenance
- **CONDITION_SANTE** : Maladie, allergie, intolérance, ou état physiologique
- **RECOMMANDATION** : Verbes ou expressions indiquant relation positive/négative

## Types de Relations Extraites

- **PRÉVIENT** : Relation de prévention (ex: "Les antioxydants préviennent les déficits")
- **TRAITE** : Relation de traitement (ex: "Le calcium traite l'hypertension")
- **RÉDUIT** : Relation de réduction (ex: "Les lipides réduisent le déficit")
- **AUGMENTE** : Relation d'augmentation
- **RECOMMANDÉ_POUR** : Relation de recommandation
- **CONTRE_INDICATIF** : Relation contre-indiquée
- **CONTIENT** : Relation de composition (ex: "Le brocoli contient de la vitamine C")
- **AIDE_À** : Relation d'aide (ex: "La vitamine E aide à la santé")

## Format des Données

### Entités (CSV)
```csv
Type d'Entité,Description,Exemple d'Annotation dans un texte
ALIMENT,"Nom de l'aliment, ingrédient ou plat.","La consommation de brocolis..."
```

### Relations (CSV)
```csv
Sujet,Type_Sujet,Relation,Objet,Type_Objet,Contexte
calcium,NUTRIMENT,TRAITE,hypertension,CONDITION_SANTE,"KEY WORDS: calcium, hypertension..."
```

## Fonctionnalités

### ✅ Moteur de recommandation

Le système peut maintenant générer des recommandations personnalisées basées sur :
- **Conditions de santé** : Recherche de nutriments qui TRAITENT ou PRÉVIENNENT
- **Objectifs corporels** : Recherche de nutriments qui AIDENT À atteindre l'objectif
- **Contexte scientifique** : Chaque recommandation inclut le contexte extrait des articles

### 🤖 Fine-tuning LLM (Nouveau)

Système hybride combinant :
- **LLM fine-tuné** : Modèle spécialisé (Mistral-7B ou Llama-3-8B) entraîné sur vos données
- **LoRA/QLoRA** : Entraînement efficace sur machine grand public
- **Garde-fou scientifique** : Validation des réponses par le moteur de recommandation
- **Réponses enrichies** : Combinaison de génération LLM + faits scientifiques validés

### 🔄 Intégration RAG

Le recommandeur est préparé pour l'intégration avec des LLMs :
- Génération de contexte formaté pour prompts LLM
- Combinaison de faits scientifiques avec génération de texte
- Recommandations plus détaillées et personnalisées

## Démarrage Rapide

### 1. Vérification

```bash
python scripts/check_setup.py
```

### 2. Démonstration complète

```bash
python scripts/demo_complete.py
```

### 3. Utilisation du recommandeur

```bash
python scripts/recommender.py
```

Voir `QUICKSTART.md` pour un guide détaillé.

## Prochaines Étapes

1. **Enrichir les données** :
   - Ajouter plus de PDFs scientifiques
   - Améliorer les patterns d'extraction
   - Utiliser des modèles NLP plus avancés (spaCy, transformers)

2. **Fine-tuning LLM** :
   - Entraîner sur plus de données
   - Optimiser les hyperparamètres
   - Évaluer la qualité des réponses

3. **Interface utilisateur** :
   - Créer une API REST pour exposer le système
   - Développer une interface web
   - Application mobile

4. **Génération de plans nutritionnels** :
   - Calcul des apports recommandés en nutriments
   - Suggestions de repas personnalisés
   - Recettes adaptées
   - Conseils quotidiens

## Auteur

Projet développé dans le cadre d'un système intelligent de recommandations nutritionnelles.

## Licence

À définir selon vos besoins.

