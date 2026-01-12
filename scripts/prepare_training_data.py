#!/usr/bin/env python3
"""
Préparation des données pour le fine-tuning LLM.

Convertit les entités et relations JSON en format Instruction-Response
pour l'entraînement d'un modèle de langage.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).parent.parent / "data"
ENTITIES_FILE = DATA_DIR / "entities_extracted.json"
RELATIONS_FILE = DATA_DIR / "relations_extracted.json"
OUTPUT_FILE = DATA_DIR / "training_data.jsonl"


def load_json(file_path: Path) -> List[Dict]:
    """Charge un fichier JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_instruction_response_pairs(
    entities: List[Dict],
    relations: List[Dict]
) -> List[Dict]:
    """
    Crée des paires Instruction-Response à partir des entités et relations.
    """
    training_examples = []
    
    # 1. Questions sur les relations (TRAITE, PRÉVIENT, AIDE_À)
    for relation in relations:
        sujet = relation.get("Sujet", "")
        relation_type = relation.get("Relation", "")
        objet = relation.get("Objet", "")
        contexte = relation.get("Contexte", "")
        type_sujet = relation.get("Type_Sujet", "")
        type_objet = relation.get("Type_Objet", "")
        
        if not sujet or not objet:
            continue
        
        # Question 1: Quel nutriment traite/prévient X ?
        if relation_type in ["TRAITE", "PRÉVIENT"]:
            if type_sujet == "NUTRIMENT" and type_objet == "CONDITION_SANTE":
                instruction = f"Quel nutriment {relation_type.lower()} {objet} ?"
                response = f"Le {sujet} {relation_type.lower()} {objet}. {contexte[:200]}"
                training_examples.append({
                    "instruction": instruction,
                    "response": response,
                    "source": "relation",
                    "metadata": {
                        "relation_type": relation_type,
                        "sujet": sujet,
                        "objet": objet
                    }
                })
        
        # Question 2: Quel nutriment aide à X ?
        if relation_type == "AIDE_À":
            if type_sujet == "NUTRIMENT" and type_objet == "OBJECTIF_CORPS":
                instruction = f"Quel nutriment aide à {objet} ?"
                response = f"Le {sujet} aide à {objet}. {contexte[:200]}"
                training_examples.append({
                    "instruction": instruction,
                    "response": response,
                    "source": "relation",
                    "metadata": {
                        "relation_type": relation_type,
                        "sujet": sujet,
                        "objet": objet
                    }
                })
        
        # Question 3: Que recommandez-vous pour X ?
        if type_objet == "CONDITION_SANTE":
            instruction = f"Que recommandez-vous pour {objet} ?"
            response = f"Je recommande {sujet} car il {relation_type.lower()} {objet}. {contexte[:200]}"
            training_examples.append({
                "instruction": instruction,
                "response": response,
                "source": "relation",
                "metadata": {
                    "relation_type": relation_type,
                    "sujet": sujet,
                    "objet": objet
                }
            })
    
    # 2. Questions sur les entités (aliments, nutriments)
    for entity in entities[:100]:  # Limiter pour éviter trop de données
        entity_type = entity.get("Type d'Entité", "")
        exemple = entity.get("Exemple d'Annotation dans un texte", "")
        
        if entity_type == "ALIMENT":
            # Question: Quels sont les bienfaits de X ?
            instruction = f"Quels sont les bienfaits nutritionnels de {exemple[:50]} ?"
            response = f"{exemple[:300]}"
            training_examples.append({
                "instruction": instruction,
                "response": response,
                "source": "entity",
                "metadata": {"entity_type": entity_type}
            })
    
    # 3. Questions génériques sur la nutrition
    generic_questions = [
        {
            "instruction": "Qu'est-ce qu'une alimentation équilibrée ?",
            "response": "Une alimentation équilibrée comprend des protéines, des glucides, des lipides, des vitamines et des minéraux en quantités appropriées selon les besoins individuels.",
            "source": "generic"
        },
        {
            "instruction": "Comment puis-je améliorer ma santé nutritionnelle ?",
            "response": "Pour améliorer votre santé nutritionnelle, consommez une variété d'aliments riches en nutriments, maintenez un équilibre entre les macronutriments, et suivez les recommandations basées sur votre condition de santé et vos objectifs.",
            "source": "generic"
        }
    ]
    training_examples.extend(generic_questions)
    
    return training_examples


def format_for_training(examples: List[Dict], format_type: str = "alpaca") -> List[Dict]:
    """
    Formate les exemples selon différents formats d'entraînement.
    
    Formats supportés:
    - alpaca: Format Alpaca (instruction, input, output)
    - chatml: Format ChatML pour les modèles de chat
    - simple: Format simple (instruction, response)
    """
    formatted = []
    
    for example in examples:
        if format_type == "alpaca":
            formatted.append({
                "instruction": example["instruction"],
                "input": "",
                "output": example["response"]
            })
        elif format_type == "chatml":
            formatted.append({
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            })
        else:  # simple
            formatted.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })
    
    return formatted


def save_training_data(examples: List[Dict], output_file: Path, format_type: str = "alpaca"):
    """Sauvegarde les données d'entraînement au format JSONL."""
    formatted = format_for_training(examples, format_type)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in formatted:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ {len(formatted)} exemples sauvegardés dans {output_file}")
    print(f"  Format: {format_type}")
    
    # Statistiques
    sources = {}
    for ex in examples:
        source = ex.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\nStatistiques:")
    for source, count in sources.items():
        print(f"  - {source}: {count} exemples")


def main():
    """Fonction principale."""
    print("="*60)
    print("PRÉPARATION DES DONNÉES POUR LE FINE-TUNING")
    print("="*60)
    
    # Charger les données
    print("\n1. Chargement des données...")
    entities = load_json(ENTITIES_FILE)
    relations = load_json(RELATIONS_FILE)
    print(f"   ✓ {len(entities)} entités chargées")
    print(f"   ✓ {len(relations)} relations chargées")
    
    # Créer les paires instruction-response
    print("\n2. Création des paires Instruction-Response...")
    examples = create_instruction_response_pairs(entities, relations)
    print(f"   ✓ {len(examples)} exemples créés")
    
    # Sauvegarder en différents formats
    print("\n3. Sauvegarde des données d'entraînement...")
    
    # Format Alpaca (recommandé pour la plupart des modèles)
    output_alpaca = DATA_DIR / "training_data_alpaca.jsonl"
    save_training_data(examples, output_alpaca, format_type="alpaca")
    
    # Format ChatML (pour les modèles de chat)
    output_chatml = DATA_DIR / "training_data_chatml.jsonl"
    save_training_data(examples, output_chatml, format_type="chatml")
    
    # Format simple
    output_simple = DATA_DIR / "training_data_simple.jsonl"
    save_training_data(examples, output_simple, format_type="simple")
    
    print("\n" + "="*60)
    print("✅ Préparation terminée!")
    print("="*60)
    print(f"\nFichiers générés:")
    print(f"  - {output_alpaca.name} (Format Alpaca)")
    print(f"  - {output_chatml.name} (Format ChatML)")
    print(f"  - {output_simple.name} (Format simple)")


if __name__ == "__main__":
    main()


