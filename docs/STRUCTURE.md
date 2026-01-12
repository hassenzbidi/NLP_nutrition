# Structure du Projet NLP Nutrition

## Organisation des Fichiers

### 📁 scripts/
Contient les scripts Python d'extraction de données :

- **extract_entities.py** : Extraction d'entités nutritionnelles
  - Détecte : ALIMENT, NUTRIMENT, MESURE/VALEUR, OBJECTIF_CORPS, CONDITION_SANTE, RECOMMANDATION
  - Sortie : `data/entities_extracted.csv` et `.json`

- **extract_relations.py** : Extraction de relations entre entités
  - Détecte : PRÉVIENT, TRAITE, RÉDUIT, AUGMENTE, RECOMMANDÉ_POUR, CONTRE_INDICATIF, CONTIENT, AIDE_À
  - Sortie : `data/relations_extracted.csv` et `.json`

### 📁 data/
Contient les données extraites :

- **entities_extracted.csv/json** : 173 entités extraites
- **relations_extracted.csv/json** : 9 relations extraites (triplets sujet-relation-objet)
- **entities_schema_sample.csv** : Schéma d'exemple pour les entités

### 📁 docs/
Documentation du projet

## Utilisation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Extraire les entités
python scripts/extract_entities.py "/chemin/vers/dossier/pdfs"

# 3. Extraire les relations
python scripts/extract_relations.py "/chemin/vers/dossier/pdfs"
```

## Format des Relations

Chaque relation est un triplet :
- **Sujet** : L'entité source (ex: "calcium")
- **Type_Sujet** : Type de l'entité (ex: "NUTRIMENT")
- **Relation** : Type de relation (ex: "TRAITE")
- **Objet** : L'entité cible (ex: "hypertension")
- **Type_Objet** : Type de l'objet (ex: "CONDITION_SANTE")
- **Contexte** : Phrase d'origine (200 premiers caractères)

## Exemples de Relations Extraites

1. `calcium` (NUTRIMENT) → **TRAITE** → `hypertension` (CONDITION_SANTE)
2. `fer` (NUTRIMENT) → **TRAITE** → `anémie` (CONDITION_SANTE)
3. `antioxydants` (NUTRIMENT) → **PRÉVIENT** → `déficits` (CONDITION_SANTE)
4. `vitamine E` (NUTRIMENT) → **AIDE_À** → `santé` (OBJECTIF_CORPS)


