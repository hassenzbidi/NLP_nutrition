# Guide de Fine-Tuning LLM pour la Nutrition

## Vue d'ensemble

Ce guide explique comment fine-tuner un modèle LLM (Mistral-7B ou Llama-3-8B) pour créer un chatbot nutritionnel spécialisé utilisant LoRA/QLoRA.

## Architecture

Le système combine deux approches :

1. **Fine-tuning LLM** : Entraînement d'un modèle de langage sur vos données nutritionnelles
2. **RAG hybride** : Utilisation du moteur de recommandation (`recommender.py`) comme garde-fou pour valider les réponses

## Prérequis

### Matériel recommandé

- **GPU** : NVIDIA avec au moins 16GB VRAM (pour QLoRA 4-bit)
- **RAM** : 16GB minimum
- **Stockage** : 50GB libres (pour les modèles)

### Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Pour CUDA (si GPU disponible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Workflow complet

### Étape 1 : Préparer les données d'entraînement

Convertir vos entités et relations en format Instruction-Response :

```bash
python scripts/prepare_training_data.py
```

**Résultat :** Génère `data/training_data_alpaca.jsonl` avec des paires instruction-réponse.

**Format généré :**
```json
{
  "instruction": "Quel nutriment traite l'anémie ?",
  "input": "",
  "output": "Le fer traite l'anémie. L'anémie par carence en fer..."
}
```

### Étape 2 : Fine-tuning du modèle

#### Option A : Mistral-7B avec QLoRA (recommandé)

```bash
python scripts/train_llm.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --use_qlora \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

#### Option B : Llama-3-8B avec QLoRA

```bash
python scripts/train_llm.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --use_qlora \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4
```

#### Option C : Sans QLoRA (nécessite plus de VRAM)

```bash
python scripts/train_llm.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --no_qlora \
    --batch_size 1 \
    --num_epochs 2
```

**Paramètres importants :**
- `--use_qlora` : Active la quantification 4-bit (économise la VRAM)
- `--num_epochs` : Nombre d'époques (3-5 recommandé)
- `--batch_size` : Taille du batch (ajuster selon VRAM)
- `--learning_rate` : Taux d'apprentissage (2e-4 recommandé)

**Temps d'entraînement estimé :**
- QLoRA 4-bit : ~2-4 heures (GPU 16GB)
- LoRA 8-bit : ~4-6 heures (GPU 16GB)
- Sans LoRA : ~8-12 heures (GPU 24GB+)

### Étape 3 : Utiliser le modèle entraîné

#### Mode interactif

```bash
python scripts/chat.py --model_path models/mistral-7b-nutrition
```

#### Mode non-interactif (question unique)

```bash
python scripts/chat.py \
    --model_path models/mistral-7b-nutrition \
    --question "Quel nutriment traite l'hypertension ?"
```

#### Sans validation recommandeur

```bash
python scripts/chat.py \
    --model_path models/mistral-7b-nutrition \
    --no_recommender
```

## Architecture du système hybride

### Fonctionnement

1. **Question utilisateur** → LLM fine-tuné
2. **Réponse LLM** → Validation par le recommandeur
3. **Recommandations scientifiques** → Enrichissement de la réponse
4. **Réponse finale** = Réponse LLM + Faits scientifiques validés

### Exemple de flux

```
Utilisateur: "Que recommandez-vous pour l'anémie ?"

1. LLM génère: "Je recommande de consommer des aliments riches en fer..."

2. Recommandeur valide:
   - Extrait: condition="anémie"
   - Trouve: fer (NUTRIMENT) → TRAITE → anémie
   - Contexte scientifique: "L'anémie par carence en fer..."

3. Réponse finale enrichie:
   "Je recommande de consommer des aliments riches en fer...
   
   **Recommandations basées sur la littérature scientifique:**
   - fer (NUTRIMENT): fer est utilisé pour traiter anémie..."
```

## Configuration LoRA/QLoRA

### Paramètres LoRA

Les paramètres par défaut dans `train_llm.py` :

```python
LoraConfig(
    r=16,              # Rank (plus bas = moins de paramètres)
    lora_alpha=32,     # Scaling factor
    target_modules=[   # Modules à adapter
        "q_proj", "v_proj", "k_proj", "o_proj"
    ],
    lora_dropout=0.05,
)
```

### Ajustements possibles

- **Augmenter `r`** (ex: 32, 64) : Plus de capacité, plus de paramètres
- **Réduire `r`** (ex: 8) : Moins de paramètres, entraînement plus rapide
- **Modifier `target_modules`** : Adapter d'autres couches

## Dépannage

### Erreur : "Out of memory"

**Solutions :**
1. Réduire `--batch_size` (ex: 2 ou 1)
2. Utiliser `--use_qlora` (quantification 4-bit)
3. Réduire `--max_length` (ex: 256 au lieu de 512)

### Erreur : "Model not found"

**Solution :** Télécharger le modèle manuellement :
```bash
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```

### Erreur : "CUDA out of memory"

**Solutions :**
1. Utiliser QLoRA (4-bit) au lieu de LoRA (8-bit)
2. Réduire la taille du batch
3. Utiliser `gradient_accumulation_steps` plus élevé

## Améliorations possibles

### 1. Augmenter les données d'entraînement

- Générer plus d'exemples depuis vos relations
- Ajouter des questions-réponses manuelles
- Utiliser des techniques d'augmentation de données

### 2. Hyperparamètres

- Tester différents `learning_rate` (1e-4 à 5e-4)
- Ajuster `num_epochs` selon la convergence
- Utiliser un scheduler de learning rate

### 3. Évaluation

- Créer un dataset de validation
- Mesurer la précision des réponses
- Comparer avec le modèle de base

## Structure des fichiers

```
projet_nlp_nutrition/
├── scripts/
│   ├── prepare_training_data.py  # Préparation des données
│   ├── train_llm.py              # Fine-tuning
│   └── chat.py                   # Interface d'inférence
├── data/
│   └── training_data_alpaca.jsonl  # Données d'entraînement
├── models/
│   └── mistral-7b-nutrition/     # Modèle entraîné
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── training_config.json
└── docs/
    └── LLM_TRAINING.md           # Ce fichier
```

## Ressources

- [Hugging Face PEFT](https://github.com/huggingface/peft) : Documentation LoRA
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) : Article scientifique
- [Mistral AI](https://mistral.ai/) : Modèles Mistral
- [Llama 3](https://ai.meta.com/llama/) : Modèles Llama

## Support

Pour toute question ou problème :
1. Vérifier les logs d'erreur
2. Consulter la documentation Hugging Face
3. Vérifier que toutes les dépendances sont installées


