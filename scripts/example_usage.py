#!/usr/bin/env python3
"""
Exemple d'utilisation du moteur de recommandation nutritionnelle.

Ce script montre comment utiliser le recommandeur dans votre application.
"""

from recommender import NutritionRecommender
import json

def example_basic_usage():
    """Exemple d'utilisation basique."""
    print("="*60)
    print("EXEMPLE 1: Utilisation basique")
    print("="*60)
    
    # Initialiser le recommandeur
    recommender = NutritionRecommender()
    
    # Profil utilisateur avec condition
    user_profile = {
        "condition": "hypertension",
        "objectif": ""
    }
    
    # Obtenir les recommandations
    results = recommender.get_recommendations(user_profile)
    
    # Afficher les résultats
    print(f"\n📊 Résumé: {results['summary']}")
    print(f"📚 Sources scientifiques: {results['sources']}")
    print(f"\n💡 Recommandations trouvées: {len(results['recommendations'])}")
    
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"\n  {i}. {rec['recommandation']}")
        print(f"     🔗 Relation: {rec['relation']}")
        print(f"     🎯 Cible: {rec['cible']}")
        print(f"     📝 {rec['explication']}")
        print(f"     ⚡ Confiance: {rec['confiance']}")


def example_with_objective():
    """Exemple avec objectif corporel."""
    print("\n" + "="*60)
    print("EXEMPLE 2: Avec objectif corporel")
    print("="*60)
    
    recommender = NutritionRecommender()
    
    user_profile = {
        "condition": "",
        "objectif": "santé"
    }
    
    results = recommender.get_recommendations(user_profile, max_results=5)
    
    print(f"\n📊 Résumé: {results['summary']}")
    print(f"📚 Sources: {results['sources']}")
    
    for rec in results['recommendations']:
        print(f"\n  ✓ {rec['recommandation']}")
        print(f"    {rec['explication']}")


def example_rag_integration():
    """Exemple d'intégration RAG pour LLM."""
    print("\n" + "="*60)
    print("EXEMPLE 3: Préparation du contexte RAG pour LLM")
    print("="*60)
    
    recommender = NutritionRecommender()
    
    user_profile = {
        "condition": "anémie",
        "objectif": "santé"
    }
    
    # Préparer le contexte pour un LLM
    rag_context = recommender.prepare_rag_context(user_profile, max_contexts=3)
    
    print("\n📄 Contexte formaté pour LLM:")
    print("-" * 60)
    print(rag_context)
    print("-" * 60)
    
    print("\n💡 Ce contexte peut être utilisé comme prompt pour un LLM:")
    print("   - ChatGPT, Claude, ou tout autre modèle de langage")
    print("   - Pour générer des recommandations plus détaillées")
    print("   - En combinant les faits scientifiques avec la génération de texte")


def example_json_output():
    """Exemple de sortie JSON pour API."""
    print("\n" + "="*60)
    print("EXEMPLE 4: Sortie JSON (pour API)")
    print("="*60)
    
    recommender = NutritionRecommender()
    
    user_profile = {
        "condition": "hypertension",
        "objectif": ""
    }
    
    results = recommender.get_recommendations(user_profile)
    
    # Convertir en JSON (pour API REST par exemple)
    json_output = json.dumps(results, ensure_ascii=False, indent=2)
    
    print("\n📦 Format JSON (prêt pour API):")
    print(json_output[:500] + "...")


def example_custom_integration():
    """Exemple d'intégration personnalisée."""
    print("\n" + "="*60)
    print("EXEMPLE 5: Intégration dans votre application")
    print("="*60)
    
    recommender = NutritionRecommender()
    
    # Simuler des données utilisateur depuis un formulaire
    user_input = {
        "age": 35,
        "poids": 70,
        "condition": "diabète",
        "objectif": "perte de poids",
        "restrictions": "sans gluten"
    }
    
    # Extraire les informations pertinentes
    profile = {
        "condition": user_input.get("condition", ""),
        "objectif": user_input.get("objectif", "")
    }
    
    # Obtenir les recommandations
    results = recommender.get_recommendations(profile)
    
    # Générer une réponse personnalisée
    print(f"\n👤 Profil utilisateur:")
    print(f"   - Âge: {user_input['age']} ans")
    print(f"   - Poids: {user_input['poids']} kg")
    print(f"   - Condition: {user_input['condition']}")
    print(f"   - Objectif: {user_input['objectif']}")
    print(f"   - Restrictions: {user_input['restrictions']}")
    
    print(f"\n💊 Recommandations nutritionnelles:")
    if results['recommendations']:
        for rec in results['recommendations']:
            print(f"\n   • {rec['recommandation']}")
            print(f"     {rec['explication']}")
            print(f"     📖 Source: {rec['contexte_scientifique'][:100]}...")
    else:
        print("   Aucune recommandation spécifique trouvée.")
        print("   Consultez un professionnel de la santé.")


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_with_objective()
        example_rag_integration()
        example_json_output()
        example_custom_integration()
        
        print("\n" + "="*60)
        print("✅ Tous les exemples ont été exécutés avec succès!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


