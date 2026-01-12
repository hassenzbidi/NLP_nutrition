#!/usr/bin/env python3
"""
Moteur de recommandation nutritionnelle basé sur les relations extraites.

Ce script charge les relations scientifiques extraites et génère des
recommandations personnalisées selon le profil utilisateur.

Usage:
    python scripts/recommender.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Chemin vers les données
DATA_DIR = Path(__file__).parent.parent / "data"
RELATIONS_FILE = DATA_DIR / "relations_extracted.json"


class NutritionRecommender:
    """
    Moteur de recommandation nutritionnelle basé sur les relations scientifiques.
    """
    
    def __init__(self, relations_file: Path = RELATIONS_FILE):
        """
        Initialise le recommandeur en chargeant les relations.
        
        Args:
            relations_file: Chemin vers le fichier JSON des relations
        """
        self.relations = self._load_relations(relations_file)
        self.stats = self._compute_stats()
    
    def _load_relations(self, relations_file: Path) -> List[Dict]:
        """Charge les relations depuis le fichier JSON."""
        if not relations_file.exists():
            raise FileNotFoundError(
                f"Fichier de relations non trouvé : {relations_file}\n"
                f"Exécutez d'abord : python scripts/extract_relations.py"
            )
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        print(f"✓ {len(relations)} relations chargées depuis {relations_file.name}")
        return relations
    
    def _compute_stats(self) -> Dict:
        """Calcule des statistiques sur les relations chargées."""
        stats = {
            "total": len(self.relations),
            "by_relation": {},
            "by_subject_type": {},
            "by_object_type": {},
        }
        
        for rel in self.relations:
            # Par type de relation
            rel_type = rel.get("Relation", "UNKNOWN")
            stats["by_relation"][rel_type] = stats["by_relation"].get(rel_type, 0) + 1
            
            # Par type de sujet
            subj_type = rel.get("Type_Sujet", "UNKNOWN")
            stats["by_subject_type"][subj_type] = stats["by_subject_type"].get(subj_type, 0) + 1
            
            # Par type d'objet
            obj_type = rel.get("Type_Objet", "UNKNOWN")
            stats["by_object_type"][obj_type] = stats["by_object_type"].get(obj_type, 0) + 1
        
        return stats
    
    def get_recommendations(
        self,
        user_profile: Dict[str, str],
        max_results: int = 10
    ) -> Dict[str, any]:
        """
        Génère des recommandations basées sur le profil utilisateur.
        
        Args:
            user_profile: Dictionnaire avec les clés possibles :
                - "condition": Condition de santé (ex: "anémie", "hypertension")
                - "objectif": Objectif corporel (ex: "santé", "perte de poids")
                - "restrictions": Restrictions alimentaires (optionnel)
            max_results: Nombre maximum de recommandations à retourner
        
        Returns:
            Dictionnaire avec :
                - "recommendations": Liste de recommandations structurées
                - "summary": Résumé des recommandations
                - "sources": Nombre de sources scientifiques utilisées
        """
        recommendations = []
        matched_relations = []
        
        # Normaliser le profil utilisateur
        condition = user_profile.get("condition", "").lower().strip()
        objectif = user_profile.get("objectif", "").lower().strip()
        restrictions = user_profile.get("restrictions", []).lower().strip() if isinstance(user_profile.get("restrictions"), str) else []
        
        if not condition and not objectif:
            return {
                "recommendations": [],
                "summary": "Aucun critère fourni. Veuillez spécifier une condition ou un objectif.",
                "sources": 0,
                "error": "Profil utilisateur incomplet"
            }
        
        # Filtrer les relations selon la logique
        for relation in self.relations:
            matched = False
            reason = ""
            
            # Cas 1: CONDITION_SANTE → cherche TRAITE ou PRÉVIENT
            if condition:
                obj = relation.get("Objet", "").lower()
                obj_type = relation.get("Type_Objet", "")
                rel_type = relation.get("Relation", "")
                
                # Vérifier si l'objet correspond à la condition
                if obj_type == "CONDITION_SANTE" and condition in obj:
                    # Chercher les relations TRAITE ou PRÉVIENT
                    if rel_type in ["TRAITE", "PRÉVIENT"]:
                        matched = True
                        reason = f"Relation {rel_type} trouvée pour la condition '{condition}'"
            
            # Cas 2: OBJECTIF_CORPS → cherche AIDE_À
            if objectif and not matched:
                obj = relation.get("Objet", "").lower()
                obj_type = relation.get("Type_Objet", "")
                rel_type = relation.get("Relation", "")
                
                # Vérifier si l'objet correspond à l'objectif
                if obj_type == "OBJECTIF_CORPS" and objectif in obj:
                    # Chercher la relation AIDE_À
                    if rel_type == "AIDE_À":
                        matched = True
                        reason = f"Relation {rel_type} trouvée pour l'objectif '{objectif}'"
            
            if matched:
                matched_relations.append((relation, reason))
        
        # Générer les recommandations structurées
        for relation, reason in matched_relations[:max_results]:
            sujet = relation.get("Sujet", "N/A")
            sujet_type = relation.get("Type_Sujet", "N/A")
            rel_type = relation.get("Relation", "N/A")
            objet = relation.get("Objet", "N/A")
            contexte = relation.get("Contexte", "N/A")
            
            # Construire la recommandation
            recommendation = {
                "recommandation": f"{sujet} ({sujet_type})",
                "relation": rel_type,
                "cible": f"{objet}",
                "explication": self._generate_explanation(sujet, rel_type, objet),
                "contexte_scientifique": contexte[:300] + "..." if len(contexte) > 300 else contexte,
                "source": "Article scientifique extrait",
                "confiance": self._compute_confidence(rel_type)
            }
            
            recommendations.append(recommendation)
        
        # Générer un résumé
        summary = self._generate_summary(recommendations, condition, objectif)
        
        return {
            "recommendations": recommendations,
            "summary": summary,
            "sources": len(matched_relations),
            "profile": user_profile,
            "stats": {
                "total_relations_checked": len(self.relations),
                "matched_relations": len(matched_relations),
                "recommendations_returned": len(recommendations)
            }
        }
    
    def _generate_explanation(self, sujet: str, relation: str, objet: str) -> str:
        """Génère une explication lisible de la recommandation."""
        explanations = {
            "TRAITE": f"{sujet} est utilisé pour traiter {objet} selon la littérature scientifique.",
            "PRÉVIENT": f"{sujet} aide à prévenir {objet} selon les études scientifiques.",
            "AIDE_À": f"{sujet} contribue à améliorer {objet} selon les recherches.",
            "RÉDUIT": f"{sujet} réduit les effets de {objet} selon les données scientifiques.",
            "AUGMENTE": f"{sujet} augmente {objet} selon les études.",
            "RECOMMANDÉ_POUR": f"{sujet} est recommandé pour {objet} selon la littérature.",
            "CONTIENT": f"{sujet} contient {objet}, ce qui peut être bénéfique.",
        }
        
        return explanations.get(relation, f"{sujet} a une relation {relation} avec {objet}.")
    
    def _compute_confidence(self, relation_type: str) -> str:
        """Calcule un niveau de confiance basé sur le type de relation."""
        high_confidence = ["TRAITE", "PRÉVIENT", "AIDE_À"]
        medium_confidence = ["RÉDUIT", "AUGMENTE", "RECOMMANDÉ_POUR"]
        
        if relation_type in high_confidence:
            return "Élevée"
        elif relation_type in medium_confidence:
            return "Moyenne"
        else:
            return "Modérée"
    
    def _generate_summary(
        self,
        recommendations: List[Dict],
        condition: str,
        objectif: str
    ) -> str:
        """Génère un résumé des recommandations."""
        if not recommendations:
            return "Aucune recommandation trouvée pour votre profil."
        
        summary_parts = []
        
        if condition:
            summary_parts.append(f"Pour votre condition ({condition}),")
        if objectif:
            summary_parts.append(f"pour votre objectif ({objectif}),")
        
        summary_parts.append(f"nous avons trouvé {len(recommendations)} recommandation(s) basée(s) sur la littérature scientifique.")
        
        # Lister les sujets recommandés
        sujets = [rec["recommandation"] for rec in recommendations[:5]]
        if sujets:
            summary_parts.append(f"\nNutriments/Aliments recommandés : {', '.join(sujets)}")
        
        return " ".join(summary_parts)
    
    def prepare_rag_context(
        self,
        user_profile: Dict[str, str],
        max_contexts: int = 5
    ) -> str:
        """
        Prépare le contexte pour un système RAG (Retrieval-Augmented Generation).
        
        Cette fonction peut être utilisée pour fournir du contexte à un LLM.
        
        Args:
            user_profile: Profil utilisateur
            max_contexts: Nombre maximum de contextes à inclure
        
        Returns:
            Chaîne de caractères formatée pour être utilisée comme contexte LLM
        """
        recommendations = self.get_recommendations(user_profile, max_results=max_contexts)
        
        if not recommendations["recommendations"]:
            return "Aucune information scientifique pertinente trouvée."
        
        context_parts = [
            "=== CONTEXTE SCIENTIFIQUE POUR RECOMMANDATIONS NUTRITIONNELLES ===\n",
            f"Profil utilisateur: {recommendations['profile']}\n",
            f"Nombre de sources: {recommendations['sources']}\n\n",
            "FAITS SCIENTIFIQUES EXTRAITS:\n"
        ]
        
        for i, rec in enumerate(recommendations["recommendations"], 1):
            context_parts.append(f"\n--- Fait {i} ---")
            context_parts.append(f"Recommandation: {rec['recommandation']}")
            context_parts.append(f"Relation: {rec['relation']}")
            context_parts.append(f"Cible: {rec['cible']}")
            context_parts.append(f"Explication: {rec['explication']}")
            context_parts.append(f"Contexte scientifique: {rec['contexte_scientifique']}")
            context_parts.append(f"Confiance: {rec['confiance']}")
        
        context_parts.append("\n=== FIN DU CONTEXTE ===\n")
        context_parts.append("\nInstructions pour le LLM:")
        context_parts.append("Utilisez ces faits scientifiques pour générer des recommandations")
        context_parts.append("nutritionnelles personnalisées et scientifiquement fondées.")
        
        return "\n".join(context_parts)
    
    def print_stats(self):
        """Affiche les statistiques des relations chargées."""
        print("\n=== STATISTIQUES DES RELATIONS ===")
        print(f"Total de relations: {self.stats['total']}")
        print("\nPar type de relation:")
        for rel_type, count in sorted(self.stats['by_relation'].items()):
            print(f"  - {rel_type}: {count}")
        print("\nPar type de sujet:")
        for subj_type, count in sorted(self.stats['by_subject_type'].items()):
            print(f"  - {subj_type}: {count}")
        print("\nPar type d'objet:")
        for obj_type, count in sorted(self.stats['by_object_type'].items()):
            print(f"  - {obj_type}: {count}")


def main():
    """Fonction principale pour tester le recommandeur."""
    try:
        # Initialiser le recommandeur
        recommender = NutritionRecommender()
        recommender.print_stats()
        
        # Exemples de profils utilisateur
        print("\n" + "="*60)
        print("EXEMPLES DE RECOMMANDATIONS")
        print("="*60)
        
        # Exemple 1: Condition de santé
        print("\n--- Exemple 1: Condition de santé (anémie) ---")
        profile1 = {
            "condition": "anémie",
            "objectif": ""
        }
        results1 = recommender.get_recommendations(profile1)
        print(f"\nRésumé: {results1['summary']}")
        print(f"Sources: {results1['sources']}")
        for i, rec in enumerate(results1['recommendations'][:3], 1):
            print(f"\n  {i}. {rec['recommandation']}")
            print(f"     Relation: {rec['relation']} → {rec['cible']}")
            print(f"     {rec['explication']}")
        
        # Exemple 2: Objectif corporel
        print("\n--- Exemple 2: Objectif (santé) ---")
        profile2 = {
            "condition": "",
            "objectif": "santé"
        }
        results2 = recommender.get_recommendations(profile2)
        print(f"\nRésumé: {results2['summary']}")
        print(f"Sources: {results2['sources']}")
        for i, rec in enumerate(results2['recommendations'][:3], 1):
            print(f"\n  {i}. {rec['recommandation']}")
            print(f"     Relation: {rec['relation']} → {rec['cible']}")
            print(f"     {rec['explication']}")
        
        # Exemple 3: Contexte RAG
        print("\n--- Exemple 3: Contexte RAG pour LLM ---")
        rag_context = recommender.prepare_rag_context(profile1, max_contexts=3)
        print(rag_context[:500] + "...")
        
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


