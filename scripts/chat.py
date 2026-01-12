#!/usr/bin/env python3
"""
Interface d'inférence pour le modèle LLM fine-tuné.

Utilise le moteur de recommandation (recommender.py) comme garde-fou
pour valider et enrichir les réponses du LLM.

Usage:
    python scripts/chat.py --model_path models/mistral-7b-nutrition
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importer le recommandeur
sys.path.insert(0, str(Path(__file__).parent))
from recommender import NutritionRecommender

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "mistral-7b-nutrition"


class NutritionChatBot:
    """
    Chatbot nutritionnel hybride combinant LLM et moteur de recommandation.
    """
    
    def __init__(
        self,
        model_path: Path,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_recommender: bool = True,
    ):
        """
        Initialise le chatbot.
        
        Args:
            model_path: Chemin vers le modèle fine-tuné
            base_model_name: Nom du modèle de base (si LoRA)
            use_recommender: Utiliser le recommandeur comme garde-fou
        """
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.use_recommender = use_recommender
        
        print("="*60)
        print("CHARGEMENT DU CHATBOT NUTRITIONNEL")
        print("="*60)
        
        # Charger le modèle LLM
        print("\n1. Chargement du modèle LLM...")
        self.model, self.tokenizer = self._load_model()
        print("   ✓ Modèle LLM chargé")
        
        # Charger le recommandeur
        if self.use_recommender:
            print("\n2. Chargement du moteur de recommandation...")
            self.recommender = NutritionRecommender()
            print("   ✓ Moteur de recommandation chargé")
        else:
            self.recommender = None
    
    def _load_model(self):
        """Charge le modèle fine-tuné."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé: {self.model_path}\n"
                f"Entraînez d'abord le modèle avec: python scripts/train_llm.py"
            )
        
        # Vérifier si c'est un modèle LoRA
        if (self.model_path / "adapter_config.json").exists():
            # Charger le modèle de base
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            # Charger les adaptateurs LoRA
            model = PeftModel.from_pretrained(base_model, str(self.model_path))
            print(f"   Modèle LoRA chargé depuis {self.model_path}")
        else:
            # Modèle complet
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def generate_response(
        self,
        user_input: str,
        use_recommender: Optional[bool] = None,
        max_length: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        """
        Génère une réponse à la question de l'utilisateur.
        
        Args:
            user_input: Question de l'utilisateur
            use_recommender: Utiliser le recommandeur (override)
            max_length: Longueur maximale de la réponse
            temperature: Température pour la génération
        
        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        use_recommender = use_recommender if use_recommender is not None else self.use_recommender
        
        # 1. Générer la réponse avec le LLM
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        llm_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraire seulement la partie réponse
        if "### Response:" in llm_response:
            llm_response = llm_response.split("### Response:")[-1].strip()
        
        result = {
            "user_input": user_input,
            "llm_response": llm_response,
            "recommender_validation": None,
            "final_response": llm_response,
            "sources": [],
        }
        
        # 2. Valider avec le recommandeur (garde-fou)
        if use_recommender and self.recommender:
            try:
                # Extraire les informations du profil utilisateur depuis la question
                profile = self._extract_profile_from_question(user_input)
                
                if profile.get("condition") or profile.get("objectif"):
                    # Obtenir les recommandations scientifiques
                    recommendations = self.recommender.get_recommendations(profile)
                    
                    if recommendations["recommendations"]:
                        # Enrichir la réponse avec les faits scientifiques
                        scientific_facts = []
                        for rec in recommendations["recommendations"][:3]:
                            scientific_facts.append(
                                f"- {rec['recommandation']}: {rec['explication']}"
                            )
                        
                        result["recommender_validation"] = {
                            "validated": True,
                            "recommendations": recommendations["recommendations"],
                            "summary": recommendations["summary"],
                        }
                        
                        # Combiner la réponse LLM avec les faits scientifiques
                        result["final_response"] = (
                            f"{llm_response}\n\n"
                            f"**Recommandations basées sur la littérature scientifique:**\n"
                            + "\n".join(scientific_facts)
                        )
                        result["sources"] = [
                            rec["contexte_scientifique"][:100] + "..."
                            for rec in recommendations["recommendations"][:2]
                        ]
            except Exception as e:
                print(f"⚠️  Erreur lors de la validation: {e}", file=sys.stderr)
                result["recommender_validation"] = {"validated": False, "error": str(e)}
        
        return result
    
    def _extract_profile_from_question(self, question: str) -> dict:
        """
        Extrait le profil utilisateur depuis la question (simple heuristique).
        """
        question_lower = question.lower()
        profile = {}
        
        # Conditions de santé courantes
        conditions = [
            "anémie", "hypertension", "diabète", "diabetes", "obésité", "obesite",
            "cholestérol", "cholesterol", "grossesse", "pregnancy",
        ]
        for condition in conditions:
            if condition in question_lower:
                profile["condition"] = condition
                break
        
        # Objectifs courants
        objectives = [
            "perte de poids", "weight loss", "maigrir", "santé", "health",
            "prise de masse", "muscle gain", "maintien", "maintenance",
        ]
        for objective in objectives:
            if objective in question_lower:
                profile["objectif"] = objective
                break
        
        return profile
    
    def chat_interactive(self):
        """Mode interactif pour discuter avec le chatbot."""
        print("\n" + "="*60)
        print("CHATBOT NUTRITIONNEL - MODE INTERACTIF")
        print("="*60)
        print("Tapez 'quit' ou 'exit' pour quitter\n")
        
        while True:
            try:
                user_input = input("Vous: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nAu revoir!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🤖 Bot: ", end="", flush=True)
                result = self.generate_response(user_input)
                print(result["final_response"])
                
                if result.get("recommender_validation") and result["recommender_validation"].get("validated"):
                    print("\n📚 Sources scientifiques validées ✓")
                
                print()
            
            except KeyboardInterrupt:
                print("\n\nAu revoir!")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Chatbot nutritionnel avec LLM")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Chemin vers le modèle fine-tuné",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Nom du modèle de base (pour LoRA)",
    )
    parser.add_argument(
        "--no_recommender",
        action="store_true",
        help="Désactiver le moteur de recommandation",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question unique (mode non-interactif)",
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = NutritionChatBot(
            model_path=args.model_path,
            base_model_name=args.base_model,
            use_recommender=not args.no_recommender,
        )
        
        if args.question:
            # Mode non-interactif
            result = chatbot.generate_response(args.question)
            print("\n" + "="*60)
            print("RÉPONSE")
            print("="*60)
            print(f"\nQuestion: {result['user_input']}")
            print(f"\nRéponse:\n{result['final_response']}")
            if result.get("sources"):
                print(f"\n📚 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"   {i}. {source}")
        else:
            # Mode interactif
            chatbot.chat_interactive()
    
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


