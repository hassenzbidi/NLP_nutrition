import sys
from pathlib import Path

# Ajouter le dossier scripts pour importer le moteur
sys.path.append('scripts')
from recommender import NutritionRecommender

def start_chat():
    recommender = NutritionRecommender()
    print("\n" + "="*50)
    print("      BIENVENUE SUR VOTRE CHATBOT NUTRITION")
    print("="*50)
    print("(Tapez 'quitter' pour arrêter)")

    while True:
        condition = input("\nAvez-vous une condition médicale ? (ex: anémie, hypertension) : ").lower()
        if condition == 'quitter': break
        
        objectif = input("Quel est votre objectif santé ? (ex: santé, énergie) : ").lower()
        if objectif == 'quitter': break

        profile = {"condition": condition, "objectif": objectif}
        results = recommender.get_recommendations(profile)

        print("\n--- RÉPONSE DU CHATBOT ---")
        print(results['summary'])
        
        for rec in results['recommendations']:
            print(f"\n💡 Recommandation : {rec['recommandation']}")
            print(f"📝 Explication : {rec['explication']}")
            print(f"📚 Source : {rec['contexte_scientifique'][:100]}...")

start_chat()