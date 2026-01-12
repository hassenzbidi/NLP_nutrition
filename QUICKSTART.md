# Guide de Démarrage Rapide

## 🚀 Démarrage en 5 minutes

### 1. Installation

```bash
cd projet_nlp_nutrition
pip install -r requirements.txt
```

### 2. Vérification

```bash
python scripts/demo_complete.py
```

Ce script va :
- ✅ Vérifier que les données sont présentes
- ✅ Préparer les données d'entraînement
- ✅ Tester le moteur de recommandation
- ✅ Afficher des exemples

### 3. Utilisation du Recommandeur

```bash
# Mode interactif avec exemples
python scripts/recommender.py

# Exemples détaillés
python scripts/example_usage.py
```

### 4. Fine-tuning LLM (Optionnel - Nécessite GPU)

```bash
# Étape 1: Préparer les données (déjà fait)
python scripts/prepare_training_data.py

# Étape 2: Fine-tuning
python scripts/train_llm.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --use_qlora \
    --num_epochs 3

# Étape 3: Utiliser le chatbot
python scripts/chat.py --model_path models/mistral-7b-nutrition
```

## 📋 Checklist de Vérification

### Fichiers requis

- [x] `data/entities_extracted.json` - Entités extraites
- [x] `data/relations_extracted.json` - Relations extraites
- [x] `data/training_data_alpaca.jsonl` - Données d'entraînement

### Scripts disponibles

- [x] `scripts/extract_entities.py` - Extraction d'entités
- [x] `scripts/extract_relations.py` - Extraction de relations
- [x] `scripts/prepare_training_data.py` - Préparation données
- [x] `scripts/recommender.py` - Moteur de recommandation
- [x] `scripts/train_llm.py` - Fine-tuning LLM
- [x] `scripts/chat.py` - Chatbot
- [x] `scripts/demo_complete.py` - Démonstration complète

## 🎯 Cas d'usage rapides

### Cas 1: Recommandation simple

```python
from scripts.recommender import NutritionRecommender

recommender = NutritionRecommender()
profile = {"condition": "anémie"}
results = recommender.get_recommendations(profile)
print(results['summary'])
```

### Cas 2: Contexte RAG pour LLM externe

```python
from scripts.recommender import NutritionRecommender

recommender = NutritionRecommender()
profile = {"condition": "hypertension", "objectif": "santé"}
rag_context = recommender.prepare_rag_context(profile)

# Utiliser rag_context comme prompt pour ChatGPT/Claude
```

### Cas 3: Extraction depuis nouveaux PDFs

```bash
# Extraire entités
python scripts/extract_entities.py "/chemin/vers/nouveaux/pdfs"

# Extraire relations
python scripts/extract_relations.py "/chemin/vers/nouveaux/pdfs"

# Re-préparer les données
python scripts/prepare_training_data.py
```

## 🔧 Dépannage

### Erreur: "Fichier non trouvé"

**Solution:** Exécutez d'abord les scripts d'extraction :
```bash
python scripts/extract_entities.py [dossier_pdfs]
python scripts/extract_relations.py [dossier_pdfs]
```

### Erreur: "Module not found"

**Solution:** Installez les dépendances :
```bash
pip install -r requirements.txt
```

### Erreur: "CUDA out of memory" (fine-tuning)

**Solutions:**
1. Utilisez `--use_qlora` (quantification 4-bit)
2. Réduisez `--batch_size` (ex: 2 ou 1)
3. Réduisez `--max_length` (ex: 256)

## 📚 Documentation complète

- `README.md` - Vue d'ensemble
- `docs/RECOMMENDER.md` - Moteur de recommandation
- `docs/LLM_TRAINING.md` - Fine-tuning LLM
- `docs/STRUCTURE.md` - Structure du projet

## 🎓 Prochaines étapes

1. **Enrichir les données** : Ajouter plus de PDFs scientifiques
2. **Améliorer l'extraction** : Utiliser des modèles NLP plus avancés
3. **Fine-tuner le LLM** : Créer votre chatbot spécialisé
4. **Créer une API** : Exposer le système via REST API
5. **Interface web** : Créer une interface utilisateur

## 💡 Astuces

- Utilisez `demo_complete.py` pour voir toutes les fonctionnalités
- Le recommandeur peut être utilisé sans fine-tuning LLM
- Le contexte RAG peut être utilisé avec n'importe quel LLM externe
- Les données peuvent être enrichies progressivement


