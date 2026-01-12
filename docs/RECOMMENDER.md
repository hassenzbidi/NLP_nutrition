# Documentation du Moteur de Recommandation

## Vue d'ensemble

Le script `recommender.py` est un moteur de recommandation nutritionnelle intelligent qui utilise les relations scientifiques extraites pour générer des recommandations personnalisées.

## Installation

```bash
# Depuis le dossier racine du projet
cd projet_nlp_nutrition
python scripts/recommender.py
```

## Utilisation de base

### 1. Initialisation

```python
from recommender import NutritionRecommender

# Initialiser le recommandeur
recommender = NutritionRecommender()
```

### 2. Profil utilisateur

Le profil utilisateur est un dictionnaire avec les clés suivantes :

```python
user_profile = {
    "condition": "anémie",      # Condition de santé (optionnel)
    "objectif": "santé",        # Objectif corporel (optionnel)
    "restrictions": "sans gluten"  # Restrictions alimentaires (optionnel)
}
```

### 3. Obtenir des recommandations

```python
results = recommender.get_recommendations(user_profile)

# Structure de la réponse
{
    "recommendations": [
        {
            "recommandation": "fer (NUTRIMENT)",
            "relation": "TRAITE",
            "cible": "anémie",
            "explication": "fer est utilisé pour traiter anémie...",
            "contexte_scientifique": "L'anémie par carence en fer...",
            "source": "Article scientifique extrait",
            "confiance": "Élevée"
        }
    ],
    "summary": "Résumé des recommandations",
    "sources": 1,
    "profile": {...},
    "stats": {...}
}
```

## Logique de filtrage

### Condition de santé

Si l'utilisateur a une `CONDITION_SANTE`, le système cherche :
- Relations de type **TRAITE** ou **PRÉVIENT**
- Où l'objet correspond à la condition

**Exemple :**
```python
profile = {"condition": "hypertension"}
# → Cherche: NUTRIMENT → TRAITE/PRÉVIENT → hypertension
```

### Objectif corporel

Si l'utilisateur a un `OBJECTIF_CORPS`, le système cherche :
- Relations de type **AIDE_À**
- Où l'objet correspond à l'objectif

**Exemple :**
```python
profile = {"objectif": "santé"}
# → Cherche: NUTRIMENT → AIDE_À → santé
```

## Intégration RAG (Retrieval-Augmented Generation)

Le script est préparé pour l'intégration avec des LLMs (ChatGPT, Claude, etc.) :

```python
# Préparer le contexte pour un LLM
rag_context = recommender.prepare_rag_context(
    user_profile,
    max_contexts=5
)

# Utiliser ce contexte comme prompt pour un LLM
prompt = f"""
{rag_context}

Génère un plan nutritionnel personnalisé basé sur ces faits scientifiques.
"""
```

### Format du contexte RAG

Le contexte généré contient :
- Profil utilisateur
- Nombre de sources scientifiques
- Faits scientifiques extraits avec :
  - Recommandation
  - Relation
  - Cible
  - Explication
  - Contexte scientifique
  - Niveau de confiance

## Types de relations supportées

| Relation | Description | Confiance |
|----------|-------------|-----------|
| TRAITE | Traite une condition | Élevée |
| PRÉVIENT | Préviens une condition | Élevée |
| AIDE_À | Aide à atteindre un objectif | Élevée |
| RÉDUIT | Réduit les effets | Moyenne |
| AUGMENTE | Augmente un paramètre | Moyenne |
| RECOMMANDÉ_POUR | Recommandé pour | Moyenne |
| CONTIENT | Contient un nutriment | Modérée |

## Exemples d'utilisation

### Exemple 1 : Condition de santé

```python
from recommender import NutritionRecommender

recommender = NutritionRecommender()
profile = {"condition": "anémie"}

results = recommender.get_recommendations(profile)

for rec in results['recommendations']:
    print(f"{rec['recommandation']}: {rec['explication']}")
```

**Sortie :**
```
fer (NUTRIMENT): fer est utilisé pour traiter anémie selon la littérature scientifique.
```

### Exemple 2 : Objectif corporel

```python
profile = {"objectif": "santé"}
results = recommender.get_recommendations(profile)
```

### Exemple 3 : Intégration API REST

```python
from flask import Flask, jsonify, request
from recommender import NutritionRecommender

app = Flask(__name__)
recommender = NutritionRecommender()

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_profile = request.json
    results = recommender.get_recommendations(user_profile)
    return jsonify(results)

if __name__ == '__main__':
    app.run()
```

## Statistiques

Afficher les statistiques des relations chargées :

```python
recommender.print_stats()
```

**Sortie :**
```
=== STATISTIQUES DES RELATIONS ===
Total de relations: 9

Par type de relation:
  - AIDE_À: 2
  - PRÉVIENT: 3
  - RÉDUIT: 1
  - TRAITE: 3
```

## Prochaines étapes

1. **Enrichir les données** : Ajouter plus de relations en exécutant `extract_relations.py` sur plus de PDFs
2. **Intégrer un LLM** : Utiliser `prepare_rag_context()` avec ChatGPT/Claude pour des recommandations plus détaillées
3. **Créer une API** : Exposer le recommandeur via une API REST
4. **Interface utilisateur** : Créer une interface web pour saisir le profil utilisateur

## Dépannage

### Erreur : "Fichier de relations non trouvé"

**Solution :** Exécutez d'abord l'extraction de relations :
```bash
python scripts/extract_relations.py "/chemin/vers/dossier/pdfs"
```

### Aucune recommandation trouvée

**Causes possibles :**
- La condition/objectif ne correspond à aucune relation dans la base
- Les termes utilisés ne correspondent pas exactement (essayez des variantes)

**Solution :** Vérifiez les relations disponibles avec `recommender.print_stats()`

## Support

Pour toute question ou problème, consultez :
- `README.md` : Documentation principale
- `STRUCTURE.md` : Structure du projet
- `example_usage.py` : Exemples d'utilisation


