#!/usr/bin/env python3
"""
Script de fine-tuning d'un modèle LLM pour la nutrition.

Utilise LoRA (Low-Rank Adaptation) ou QLoRA pour un entraînement
efficace sur machine grand public.

Modèles supportés:
- Mistral-7B-Instruct
- Llama-3-8B-Instruct
- Autres modèles Hugging Face compatibles

Usage:
    python scripts/train_llm.py --model_name mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Chemins
DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_DATA_FILE = DATA_DIR / "training_data_alpaca.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_training_data(data_file: Path):
    """Charge les données d'entraînement depuis JSONL."""
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    return dataset


def format_prompt(example):
    """Formate les exemples au format Alpaca."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenise les exemples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def setup_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = True,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
):
    """
    Configure le modèle et le tokenizer avec LoRA/QLoRA.
    
    Args:
        model_name: Nom du modèle Hugging Face
        use_qlora: Utiliser QLoRA (quantification 4-bit)
        load_in_4bit: Charger en 4-bit (QLoRA)
        load_in_8bit: Charger en 8-bit (LoRA)
    """
    print(f"Chargement du modèle: {model_name}")
    
    # Configuration de quantification pour QLoRA
    if use_qlora and load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    # Préparer le modèle pour l'entraînement k-bit
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # Configuration LoRA
    lora_config = LoraConfig(
        r=16,  # Rank (plus bas = moins de paramètres)
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Modules à adapter
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Appliquer LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ajouter des tokens spéciaux si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def train(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    use_qlora: bool = True,
    output_dir: Optional[str] = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
):
    """
    Entraîne le modèle avec LoRA/QLoRA.
    
    Args:
        model_name: Nom du modèle Hugging Face
        use_qlora: Utiliser QLoRA
        output_dir: Dossier de sortie
        num_epochs: Nombre d'époques
        batch_size: Taille du batch
        learning_rate: Taux d'apprentissage
        max_length: Longueur maximale des séquences
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / model_name.split("/")[-1]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FINE-TUNING LLM POUR LA NUTRITION")
    print("="*60)
    print(f"Modèle: {model_name}")
    print(f"Méthode: {'QLoRA' if use_qlora else 'LoRA'}")
    print(f"Sortie: {output_dir}")
    print("="*60)
    
    # Charger les données
    print("\n1. Chargement des données...")
    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier de données non trouvé: {TRAINING_DATA_FILE}\n"
            f"Exécutez d'abord: python scripts/prepare_training_data.py"
        )
    
    dataset = load_training_data(TRAINING_DATA_FILE)
    print(f"   ✓ {len(dataset)} exemples chargés")
    
    # Formater les données
    print("\n2. Formatage des données...")
    dataset = dataset.map(format_prompt)
    
    # Configurer le modèle
    print("\n3. Configuration du modèle...")
    model, tokenizer = setup_model_and_tokenizer(model_name, use_qlora=use_qlora)
    
    # Tokeniser
    print("\n4. Tokenisation...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Diviser en train/validation (90/10)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"   ✓ Train: {len(train_dataset)} exemples")
    print(f"   ✓ Validation: {len(eval_dataset)} exemples")
    
    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",  # Désactiver wandb/tensorboard par défaut
        warmup_steps=50,
        max_steps=-1,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, pas de masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Entraînement
    print("\n5. Démarrage de l'entraînement...")
    print("   (Cela peut prendre du temps selon votre matériel)")
    trainer.train()
    
    # Sauvegarder
    print("\n6. Sauvegarde du modèle...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Sauvegarder la configuration
    config = {
        "model_name": model_name,
        "use_qlora": use_qlora,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Entraînement terminé!")
    print(f"   Modèle sauvegardé dans: {output_dir}")
    print(f"\nPour utiliser le modèle:")
    print(f"   python scripts/chat.py --model_path {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM pour la nutrition")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Nom du modèle Hugging Face",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        default=True,
        help="Utiliser QLoRA (quantification 4-bit)",
    )
    parser.add_argument(
        "--no_qlora",
        action="store_false",
        dest="use_qlora",
        help="Ne pas utiliser QLoRA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Dossier de sortie pour le modèle",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Nombre d'époques",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Taille du batch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Taux d'apprentissage",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Longueur maximale des séquences",
    )
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        use_qlora=args.use_qlora,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()


