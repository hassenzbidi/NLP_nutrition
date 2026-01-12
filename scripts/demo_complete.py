#!/usr/bin/env python3
"""
Démonstration complète du système NLP de nutrition.

Ce script montre le workflow complet :
1. Extraction des données
2. Préparation pour le fine-tuning
3. Utilisation du recommandeur
4. (Optionnel) Fine-tuning et chat
"""

import sys
from pathlib import Path

# Ajouter le dossier scripts au path
sys.path.insert(0, str(Path(__file__).parent))

def demo_extraction():
    """Démonstration de l'extraction de données."""
    print("="*60)
    print("ÉTAPE 1: EXTRACTION DES DONNÉES")
    print("="*60)
    print("\nCette étape extrait les entités et relations depuis les PDFs.")
    print("Pour l'exécuter:")
    print("  python scripts/extract_entities.py [dossier_pdfs]")
    print("  python scripts/extract_relations.py [dossier_pdfs]")
    print("\n✓ Fichiers générés:")
    print("  - data/entities_extracted.json")
    print("  - data/relations_extracted.json")


def demo_preparation():
    """Démonstration de la préparation des données."""
    print("\n" + "="*60)
    print("ÉTAPE 2: PRÉPARATION DES DONNÉES POUR FINE-TUNING")
    print("="*60)
    
    try:
        from prepare_training_data import main as prepare_main
        print("\nExécution de la préparation...")
        prepare_main()
    except Exception as e:
        print(f"\n⚠️  Erreur: {e}")
        print("Assurez-vous que les fichiers entities_extracted.json et")
        print("relations_extracted.json existent dans le dossier data/")


def demo_recommender():
    """Démonstration du moteur de recommandation."""
    print("\n" + "="*60)
    print("ÉTAPE 3: MOTEUR DE RECOMMANDATION")
    print("="*60)
    
    try:
        from recommender import NutritionRecommender
        
        recommender = NutritionRecommender()
        
        # Exemple 1
        print("\n--- Exemple 1: Condition de santé ---")
        profile1 = {"condition": "anémie", "objectif": ""}
        results1 = recommender.get_recommendations(profile1)
        print(f"Question: Que recommandez-vous pour l'anémie ?")
        print(f"Réponse: {results1['summary']}")
        if results1['recommendations']:
            print(f"\nRecommandation principale:")
            rec = results1['recommendations'][0]
            print(f"  • {rec['recommandation']}")
            print(f"    {rec['explication']}")
        
        # Exemple 2
        print("\n--- Exemple 2: Objectif corporel ---")
        profile2 = {"condition": "", "objectif": "santé"}
        results2 = recommender.get_recommendations(profile2)
        print(f"Question: Quels nutriments aident à la santé ?")
        print(f"Réponse: {results2['summary']}")
        if results2['recommendations']:
            for rec in results2['recommendations'][:2]:
                print(f"  • {rec['recommandation']}: {rec['explication']}")
        
        # Exemple 3: Contexte RAG
        print("\n--- Exemple 3: Contexte RAG pour LLM ---")
        rag_context = recommender.prepare_rag_context(profile1, max_contexts=2)
        print("Contexte formaté pour LLM (extrait):")
        print(rag_context[:300] + "...")
        
    except Exception as e:
        print(f"\n⚠️  Erreur: {e}")
        import traceback
        traceback.print_exc()


def demo_llm_training():
    """Démonstration du fine-tuning LLM."""
    print("\n" + "="*60)
    print("ÉTAPE 4: FINE-TUNING LLM (Optionnel)")
    print("="*60)
    print("\nPour fine-tuner un modèle LLM:")
    print("\n1. Préparer les données (déjà fait à l'étape 2)")
    print("2. Entraîner le modèle:")
    print("   python scripts/train_llm.py \\")
    print("       --model_name mistralai/Mistral-7B-Instruct-v0.2 \\")
    print("       --use_qlora \\")
    print("       --num_epochs 3")
    print("\n3. Utiliser le chatbot:")
    print("   python scripts/chat.py --model_path models/mistral-7b-nutrition")
    print("\n⚠️  Note: Le fine-tuning nécessite un GPU avec au moins 16GB VRAM")
    print("   Voir docs/LLM_TRAINING.md pour plus de détails")


def demo_chat():
    """Démonstration du chatbot."""
    print("\n" + "="*60)
    print("ÉTAPE 5: CHATBOT (Si modèle entraîné)")
    print("="*60)
    print("\nPour utiliser le chatbot:")
    print("  python scripts/chat.py --model_path models/[votre_modele]")
    print("\nOu en mode non-interactif:")
    print("  python scripts/chat.py \\")
    print("      --model_path models/[votre_modele] \\")
    print("      --question 'Quel nutriment traite l\'hypertension ?'")
    print("\n⚠️  Note: Le modèle doit être entraîné d'abord (étape 4)")


def main():
    """Fonction principale de démonstration."""
    print("\n" + "="*60)
    print("DÉMONSTRATION COMPLÈTE - SYSTÈME NLP NUTRITION")
    print("="*60)
    print("\nCe script démontre toutes les fonctionnalités du système.")
    print("Suivez les étapes pour comprendre le workflow complet.\n")
    
    # Étape 1: Extraction
    demo_extraction()
    
    # Étape 2: Préparation
    demo_preparation()
    
    # Étape 3: Recommandeur
    demo_recommender()
    
    # Étape 4: Fine-tuning (info seulement)
    demo_llm_training()
    
    # Étape 5: Chat (info seulement)
    demo_chat()
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DU WORKFLOW")
    print("="*60)
    print("""
1. Extraction des données:
   python scripts/extract_entities.py [dossier_pdfs]
   python scripts/extract_relations.py [dossier_pdfs]

2. Préparation pour fine-tuning:
   python scripts/prepare_training_data.py

3. Utilisation du recommandeur:
   python scripts/recommender.py
   python scripts/example_usage.py

4. Fine-tuning LLM (optionnel, nécessite GPU):
   python scripts/train_llm.py --use_qlora

5. Chatbot (si modèle entraîné):
   python scripts/chat.py --model_path models/[modele]

Documentation:
   - docs/RECOMMENDER.md : Moteur de recommandation
   - docs/LLM_TRAINING.md : Fine-tuning LLM
   - docs/STRUCTURE.md : Structure du projet
    """)
    
    print("="*60)
    print("✅ Démonstration terminée!")
    print("="*60)


if __name__ == "__main__":
    main()


