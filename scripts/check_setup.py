#!/usr/bin/env python3
"""
Script de vérification de l'installation et de la configuration.

Vérifie que tous les fichiers nécessaires sont présents et
que les dépendances sont installées.
"""

import sys
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

def check_files():
    """Vérifie la présence des fichiers essentiels."""
    print("="*60)
    print("VÉRIFICATION DES FICHIERS")
    print("="*60)
    
    required_files = {
        "Scripts": [
            "extract_entities.py",
            "extract_relations.py",
            "prepare_training_data.py",
            "recommender.py",
            "train_llm.py",
            "chat.py",
        ],
        "Données": [
            "entities_extracted.json",
            "relations_extracted.json",
        ],
        "Documentation": [
            "README.md",
            "requirements.txt",
        ],
    }
    
    all_ok = True
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            if category == "Scripts":
                path = SCRIPTS_DIR / file
            elif category == "Données":
                path = DATA_DIR / file
            else:
                path = PROJECT_ROOT / file
            
            if path.exists():
                size = path.stat().st_size
                print(f"  ✓ {file} ({size:,} bytes)")
            else:
                print(f"  ✗ {file} (MANQUANT)")
                all_ok = False
    
    return all_ok


def check_dependencies():
    """Vérifie les dépendances Python."""
    print("\n" + "="*60)
    print("VÉRIFICATION DES DÉPENDANCES")
    print("="*60)
    
    dependencies = {
        "pymupdf": "PyMuPDF (lecture PDF)",
        "PyPDF2": "PyPDF2 (lecture PDF alternative)",
        "torch": "PyTorch (fine-tuning LLM)",
        "transformers": "Transformers (modèles Hugging Face)",
        "datasets": "Datasets (gestion données)",
        "peft": "PEFT (LoRA/QLoRA)",
        "bitsandbytes": "BitsAndBytes (quantification)",
    }
    
    all_ok = True
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {module} - {description}")
        except ImportError:
            print(f"  ✗ {module} - {description} (NON INSTALLÉ)")
            all_ok = False
    
    return all_ok


def check_data():
    """Vérifie la qualité des données."""
    print("\n" + "="*60)
    print("VÉRIFICATION DES DONNÉES")
    print("="*60)
    
    try:
        import json
        
        # Vérifier entities_extracted.json
        entities_file = DATA_DIR / "entities_extracted.json"
        if entities_file.exists():
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)
            print(f"  ✓ {len(entities)} entités dans entities_extracted.json")
        else:
            print(f"  ✗ entities_extracted.json manquant")
            return False
        
        # Vérifier relations_extracted.json
        relations_file = DATA_DIR / "relations_extracted.json"
        if relations_file.exists():
            with open(relations_file, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            print(f"  ✓ {len(relations)} relations dans relations_extracted.json")
        else:
            print(f"  ✗ relations_extracted.json manquant")
            return False
        
        # Vérifier training_data
        training_file = DATA_DIR / "training_data_alpaca.jsonl"
        if training_file.exists():
            count = sum(1 for _ in open(training_file, 'r', encoding='utf-8'))
            print(f"  ✓ {count} exemples dans training_data_alpaca.jsonl")
        else:
            print(f"  ⚠ training_data_alpaca.jsonl manquant (exécutez prepare_training_data.py)")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Erreur lors de la vérification: {e}")
        return False


def check_gpu():
    """Vérifie la disponibilité GPU (pour fine-tuning)."""
    print("\n" + "="*60)
    print("VÉRIFICATION GPU (pour fine-tuning)")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU disponible: {gpu_name}")
            print(f"  ✓ Mémoire VRAM: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 16:
                print(f"  ✓ VRAM suffisante pour QLoRA (4-bit)")
            elif gpu_memory >= 8:
                print(f"  ⚠ VRAM limitée, utilisez QLoRA avec batch_size=1")
            else:
                print(f"  ✗ VRAM insuffisante pour fine-tuning")
        else:
            print(f"  ⚠ GPU non disponible (fine-tuning impossible)")
            print(f"    Le recommandeur fonctionne toujours sans GPU")
    
    except ImportError:
        print(f"  ⚠ PyTorch non installé (impossible de vérifier GPU)")


def main():
    """Fonction principale."""
    print("\n" + "="*60)
    print("VÉRIFICATION DE L'INSTALLATION")
    print("="*60)
    print(f"\nDossier du projet: {PROJECT_ROOT}\n")
    
    results = {
        "Fichiers": check_files(),
        "Dépendances": check_dependencies(),
        "Données": check_data(),
    }
    
    check_gpu()
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    
    all_ok = all(results.values())
    
    for check, result in results.items():
        status = "✓ OK" if result else "✗ PROBLÈME"
        print(f"  {check}: {status}")
    
    if all_ok:
        print("\n✅ Tous les composants sont prêts!")
        print("\nProchaines étapes:")
        print("  1. python scripts/demo_complete.py")
        print("  2. python scripts/recommender.py")
        print("  3. (Optionnel) python scripts/train_llm.py --use_qlora")
    else:
        print("\n⚠️  Certains composants manquent.")
        print("\nActions recommandées:")
        if not results["Fichiers"]:
            print("  - Vérifiez que tous les scripts sont présents")
        if not results["Dépendances"]:
            print("  - Exécutez: pip install -r requirements.txt")
        if not results["Données"]:
            print("  - Exécutez: python scripts/extract_entities.py [dossier_pdfs]")
            print("  - Exécutez: python scripts/extract_relations.py [dossier_pdfs]")
    
    print("="*60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())


