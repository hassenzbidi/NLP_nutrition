#!/usr/bin/env python3
"""
Extraction de relations entre entités nutritionnelles depuis les PDFs.
Produit un CSV avec les triplets (Sujet, Relation, Objet) pour permettre
au système de comprendre "Les œufs sont bons pour le diabète" et non juste
détecter "œufs" et "diabète" séparément.

Usage:
    python scripts/extract_relations.py [chemin_vers_dossier_pdfs]
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Chemin par défaut : dossier parent (où se trouvent les PDFs)
DEFAULT_PDF_DIR = Path(__file__).parent.parent.parent
OUT_CSV = Path(__file__).parent.parent / "data" / "relations_extracted.csv"
OUT_JSON = Path(__file__).parent.parent / "data" / "relations_extracted.json"

# Listes d'entités (réutilisées depuis extract_entities.py)
ALIMENTS = [
    "poulet", "poisson", "saumon", "thon", "brocoli", "tomate", "lait",
    "fromage", "yaourt", "riz", "pomme", "banane", "noix", "amande",
    "huile d'olive", "avocat", "légumes", "fruits", "œuf", "œufs", "oeuf", "oeufs",
    "épinards", "epinards", "carotte", "patate", "pomme de terre",
    "viande", "bœuf", "boeuf", "porc", "dinde", "canard",
    "céréales", "cereales", "blé", "ble", "avoine", "quinoa",
    "haricot", "lentille", "pois chiche", "tofu", "soja",
    "champignon", "oignon", "ail", "citron", "orange", "raisin",
    "huile", "beurre", "margarine", "sucre", "miel",
]

NUTRIMENTS = [
    "protéine", "protéines", "glucide", "glucides", "lipide", "lipides",
    "omega-3", "oméga-3", "omega 3", "oméga 3", "vitamine", "vitamine c",
    "vitamine d", "vitamine a", "vitamine e", "vitamine k", "vitamine b",
    "fer", "calcium", "magnésium", "magnesium", "zinc", "fibres", "fibre",
    "sodium", "potassium", "phosphore", "iode", "sélénium", "selenium",
    "acide folique", "folate", "biotine", "niacine", "riboflavine",
    "thiamine", "antioxydant", "antioxydants", "polyphénol", "polyphenol",
    "caroténoïde", "carotenoid", "flavonoïde", "flavonoid",
]

CONDITIONS = [
    "diabète", "diabetes", "diabète de type 2", "diabetes type 2",
    "hypertension", "obésité", "obesite", "obese", "grossesse", "pregnancy",
    "allergie", "allergie alimentaire", "intolérance", "intolerance",
    "intolérance au lactose", "lactose intolerance", "cholestérol", "cholesterol",
    "maladie cardiaque", "heart disease", "maladie cardiovasculaire",
    "cardiovascular disease", "cancer", "ostéoporose", "osteoporose",
    "anémie", "anemie", "anemia", "déficit", "deficit", "carence",
    "déficience", "deficience", "syndrome métabolique", "metabolic syndrome",
]

OBJECTIFS = [
    "perte de poids", "weight loss", "maigrir", "lose weight",
    "maintenir le poids", "maintain weight", "prise de masse", "muscle gain",
    "masse musculaire", "muscle mass", "musculaire", "maintien",
    "maintenance", "gain de poids", "weight gain", "prise de poids",
    "amélioration", "improvement", "performance", "endurance",
    "récupération", "recovery", "santé", "health",
]

# Patterns de relations (verbes, prépositions, expressions)
# Relations plus spécifiques pour une meilleure précision
RELATION_PATTERNS = {
    "PRÉVIENT": [
        r"prévient",
        r"prévention\s+(?:de|du|des)",
        r"prévenir",
        r"protège\s+(?:contre|de|du|des)",
        r"protection\s+(?:contre|de|du|des)",
        r"réduit\s+(?:le|la|les)\s+risque",
        r"diminue\s+(?:le|la|les)\s+risque",
        r"évite",
        r"éviter",
    ],
    "TRAITE": [
        r"trait[éeé]?\s+(?:de|du|des)",
        r"traitement\s+(?:de|du|des)",
        r"soigne",
        r"soigner",
        r"guérit",
        r"guérir",
        r"combat",
        r"combattre",
        r"lutte\s+(?:contre|contre le|contre la)",
    ],
    "RÉDUIT": [
        r"réduit\s+(?:le|la|les)",
        r"réduction\s+(?:de|du|des)",
        r"diminue\s+(?:le|la|les)",
        r"diminution\s+(?:de|du|des)",
        r"baisse\s+(?:le|la|les)",
        r"abaisse\s+(?:le|la|les)",
        r"baisse\s+(?:de|du|des)",
    ],
    "AUGMENTE": [
        r"augmente\s+(?:le|la|les)",
        r"augmentation\s+(?:de|du|des)",
        r"élève\s+(?:le|la|les)",
        r"hausse\s+(?:de|du|des)",
        r"améliore\s+(?:le|la|les)",
        r"amélioration\s+(?:de|du|des)",
    ],
    "RECOMMANDÉ_POUR": [
        r"recommand[éeé]?\s+(?:pour|en cas de|dans)",
        r"conseill[éeé]?\s+(?:pour|en cas de|dans)",
        r"bénéfique\s+(?:pour|à|en cas de)",
        r"bon\s+(?:pour|à)",
        r"utile\s+(?:pour|à|en cas de)",
        r"indiqu[éeé]?\s+(?:pour|en cas de)",
        r"appropri[éeé]?\s+(?:pour|en cas de)",
        r"adapt[éeé]?\s+(?:pour|en cas de)",
    ],
    "CONTRE_INDICATIF": [
        r"éviter\s+(?:pour|en cas de|dans)",
        r"déconseill[éeé]?\s+(?:pour|en cas de)",
        r"contre-indiqu[éeé]?\s+(?:pour|en cas de)",
        r"mauvais\s+(?:pour|à)",
        r"néfaste\s+(?:pour|à)",
        r"dangereux\s+(?:pour|en cas de)",
        r"aggrave",
        r"aggravation",
        r"provoque",
        r"provoquer",
    ],
    "CONTIENT": [
        r"contient",
        r"riche\s+(?:en|de)",
        r"source\s+(?:de|d')",
        r"apport\s+(?:de|en)",
        r"fournit",
        r"apporte",
        r"composé\s+(?:de|d')",
        r"constitué\s+(?:de|d')",
        r"enrichi\s+(?:en|de)",
    ],
    "AIDE_À": [
        r"aide\s+(?:à|pour)",
        r"favorise",
        r"contribue\s+(?:à|pour)",
        r"permet\s+(?:de|d')",
        r"facilite",
        r"soutient",
        r"optimise",
        r"renforce",
        r"stimule",
    ],
    "QUANTITÉ": [
        r"\d+\s*(?:g|mg|µg|mcg|kg|kcal|cal|kj)\s+(?:de|d')",
        r"quantité\s+(?:de|d')",
        r"dose\s+(?:de|d')",
        r"apport\s+(?:de|en)",
        r"dosage\s+(?:de|d')",
    ],
}

# Compilation des regex
RELATION_REGEX = {
    rel_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for rel_type, patterns in RELATION_PATTERNS.items()
}


def load_text(pdf_path: Path) -> str:
    """Retourne le texte du PDF via PyMuPDF si dispo, sinon PyPDF2."""
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

    if fitz:
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages)

    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Installez pymupdf ou PyPDF2 (pip install pymupdf PyPDF2)"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def sentences(text: str) -> List[str]:
    """Découpe en phrases."""
    # Nettoyage basique
    text = re.sub(r"\s+", " ", text)
    # Découpage par ponctuation
    sents = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 30]


def find_entities_in_sentence(sent: str) -> Dict[str, List[str]]:
    """Trouve toutes les entités dans une phrase avec validation."""
    lower = sent.lower()
    entities: Dict[str, List[str]] = {
        "ALIMENT": [],
        "NUTRIMENT": [],
        "CONDITION_SANTE": [],
        "OBJECTIF_CORPS": [],
    }
    
    # Mots à exclure (faux positifs courants)
    EXCLUDE_WORDS = {
        "ailleurs", "par ailleurs", "d'ailleurs",
    }
    
    # Chercher les aliments (avec validation plus stricte)
    for aliment in ALIMENTS:
        # Éviter les mots trop courts qui peuvent être des faux positifs
        if len(aliment) < 3:
            continue
        pattern = r"\b" + re.escape(aliment) + r"(?:s|es)?\b"
        matches = re.finditer(pattern, lower)
        for match in matches:
            start, end = match.span()
            original = sent[start:end].strip()
            # Vérifier que ce n'est pas un faux positif
            if original.lower() in EXCLUDE_WORDS:
                continue
            # Vérifier que le mot n'est pas dans un autre mot (ex: "ailleurs" dans "ailleurs")
            if len(original) < 3:
                continue
            if original not in entities["ALIMENT"]:
                entities["ALIMENT"].append(original)
    
    # Chercher les nutriments
    for nutriment in NUTRIMENTS:
        if len(nutriment) < 3:
            continue
        # Patterns plus flexibles pour les nutriments composés
        pattern = r"\b" + re.escape(nutriment) + r"(?:s|es)?\b"
        matches = re.finditer(pattern, lower)
        for match in matches:
            start, end = match.span()
            original = sent[start:end].strip()
            if len(original) < 3:
                continue
            if original not in entities["NUTRIMENT"]:
                entities["NUTRIMENT"].append(original)
    
    # Chercher les conditions (avec patterns plus flexibles)
    for condition in CONDITIONS:
        if len(condition) < 3:
            continue
        pattern = r"\b" + re.escape(condition) + r"(?:s|es)?\b"
        matches = re.finditer(pattern, lower)
        for match in matches:
            start, end = match.span()
            original = sent[start:end].strip()
            if len(original) < 3:
                continue
            if original not in entities["CONDITION_SANTE"]:
                entities["CONDITION_SANTE"].append(original)
    
    # Chercher les objectifs
    for objectif in OBJECTIFS:
        if len(objectif) < 3:
            continue
        # Pour les objectifs composés, chercher la phrase complète
        pattern = r"\b" + re.escape(objectif) + r"(?:s|es)?\b"
        matches = re.finditer(pattern, lower)
        for match in matches:
            start, end = match.span()
            original = sent[start:end].strip()
            if len(original) < 3:
                continue
            if original not in entities["OBJECTIF_CORPS"]:
                entities["OBJECTIF_CORPS"].append(original)
    
    return entities


def find_relations_in_sentence(sent: str) -> List[str]:
    """Trouve les types de relations présents dans la phrase."""
    lower = sent.lower()
    found_relations = []
    
    for rel_type, regexes in RELATION_REGEX.items():
        for regex in regexes:
            if regex.search(lower):
                if rel_type not in found_relations:
                    found_relations.append(rel_type)
                break
    
    return found_relations


def extract_triplets(sent: str) -> List[Dict[str, str]]:
    """Extrait les triplets (Sujet, Relation, Objet) d'une phrase avec logique améliorée."""
    triplets = []
    
    entities = find_entities_in_sentence(sent)
    relations = find_relations_in_sentence(sent)
    
    if not relations or not any(entities.values()):
        return triplets
    
    # Générer des triplets selon les relations trouvées
    for rel_type in relations:
        # Relations spécifiques pour les conditions de santé
        if rel_type in ["PRÉVIENT", "TRAITE", "RÉDUIT", "AUGMENTE"]:
            # ALIMENT/NUTRIMENT -> PRÉVIENT/TRAITE/RÉDUIT/AUGMENTE -> CONDITION_SANTE
            if entities["CONDITION_SANTE"]:
                # Avec ALIMENT
                if entities["ALIMENT"]:
                    for aliment in entities["ALIMENT"]:
                        for condition in entities["CONDITION_SANTE"]:
                            triplets.append({
                                "Sujet": aliment,
                                "Type_Sujet": "ALIMENT",
                                "Relation": rel_type,
                                "Objet": condition,
                                "Type_Objet": "CONDITION_SANTE",
                                "Contexte": sent[:200],
                            })
                # Avec NUTRIMENT
                if entities["NUTRIMENT"]:
                    for nutriment in entities["NUTRIMENT"]:
                        for condition in entities["CONDITION_SANTE"]:
                            triplets.append({
                                "Sujet": nutriment,
                                "Type_Sujet": "NUTRIMENT",
                                "Relation": rel_type,
                                "Objet": condition,
                                "Type_Objet": "CONDITION_SANTE",
                                "Contexte": sent[:200],
                            })
        
        # Relation CONTIENT : uniquement ALIMENT -> NUTRIMENT
        if rel_type == "CONTIENT" and entities["ALIMENT"] and entities["NUTRIMENT"]:
            for aliment in entities["ALIMENT"]:
                for nutriment in entities["NUTRIMENT"]:
                    triplets.append({
                        "Sujet": aliment,
                        "Type_Sujet": "ALIMENT",
                        "Relation": rel_type,
                        "Objet": nutriment,
                        "Type_Objet": "NUTRIMENT",
                        "Contexte": sent[:200],
                    })
        
        # Relation RECOMMANDÉ_POUR : ALIMENT/NUTRIMENT -> CONDITION_SANTE/OBJECTIF_CORPS
        if rel_type == "RECOMMANDÉ_POUR":
            # ALIMENT -> CONDITION_SANTE
            if entities["ALIMENT"] and entities["CONDITION_SANTE"]:
                for aliment in entities["ALIMENT"]:
                    for condition in entities["CONDITION_SANTE"]:
                        triplets.append({
                            "Sujet": aliment,
                            "Type_Sujet": "ALIMENT",
                            "Relation": rel_type,
                            "Objet": condition,
                            "Type_Objet": "CONDITION_SANTE",
                            "Contexte": sent[:200],
                        })
            # ALIMENT -> OBJECTIF_CORPS
            if entities["ALIMENT"] and entities["OBJECTIF_CORPS"]:
                for aliment in entities["ALIMENT"]:
                    for objectif in entities["OBJECTIF_CORPS"]:
                        triplets.append({
                            "Sujet": aliment,
                            "Type_Sujet": "ALIMENT",
                            "Relation": rel_type,
                            "Objet": objectif,
                            "Type_Objet": "OBJECTIF_CORPS",
                            "Contexte": sent[:200],
                        })
            # NUTRIMENT -> CONDITION_SANTE
            if entities["NUTRIMENT"] and entities["CONDITION_SANTE"]:
                for nutriment in entities["NUTRIMENT"]:
                    for condition in entities["CONDITION_SANTE"]:
                        triplets.append({
                            "Sujet": nutriment,
                            "Type_Sujet": "NUTRIMENT",
                            "Relation": rel_type,
                            "Objet": condition,
                            "Type_Objet": "CONDITION_SANTE",
                            "Contexte": sent[:200],
                        })
            # NUTRIMENT -> OBJECTIF_CORPS
            if entities["NUTRIMENT"] and entities["OBJECTIF_CORPS"]:
                for nutriment in entities["NUTRIMENT"]:
                    for objectif in entities["OBJECTIF_CORPS"]:
                        triplets.append({
                            "Sujet": nutriment,
                            "Type_Sujet": "NUTRIMENT",
                            "Relation": rel_type,
                            "Objet": objectif,
                            "Type_Objet": "OBJECTIF_CORPS",
                            "Contexte": sent[:200],
                        })
        
        # Relation CONTRE_INDICATIF : ALIMENT/NUTRIMENT -> CONDITION_SANTE
        if rel_type == "CONTRE_INDICATIF" and entities["CONDITION_SANTE"]:
            if entities["ALIMENT"]:
                for aliment in entities["ALIMENT"]:
                    for condition in entities["CONDITION_SANTE"]:
                        triplets.append({
                            "Sujet": aliment,
                            "Type_Sujet": "ALIMENT",
                            "Relation": rel_type,
                            "Objet": condition,
                            "Type_Objet": "CONDITION_SANTE",
                            "Contexte": sent[:200],
                        })
            if entities["NUTRIMENT"]:
                for nutriment in entities["NUTRIMENT"]:
                    for condition in entities["CONDITION_SANTE"]:
                        triplets.append({
                            "Sujet": nutriment,
                            "Type_Sujet": "NUTRIMENT",
                            "Relation": rel_type,
                            "Objet": condition,
                            "Type_Objet": "CONDITION_SANTE",
                            "Contexte": sent[:200],
                        })
        
        # Relation AIDE_À : ALIMENT/NUTRIMENT -> OBJECTIF_CORPS
        if rel_type == "AIDE_À" and entities["OBJECTIF_CORPS"]:
            if entities["ALIMENT"]:
                for aliment in entities["ALIMENT"]:
                    for objectif in entities["OBJECTIF_CORPS"]:
                        triplets.append({
                            "Sujet": aliment,
                            "Type_Sujet": "ALIMENT",
                            "Relation": rel_type,
                            "Objet": objectif,
                            "Type_Objet": "OBJECTIF_CORPS",
                            "Contexte": sent[:200],
                        })
            if entities["NUTRIMENT"]:
                for nutriment in entities["NUTRIMENT"]:
                    for objectif in entities["OBJECTIF_CORPS"]:
                        triplets.append({
                            "Sujet": nutriment,
                            "Type_Sujet": "NUTRIMENT",
                            "Relation": rel_type,
                            "Objet": objectif,
                            "Type_Objet": "OBJECTIF_CORPS",
                            "Contexte": sent[:200],
                        })
    
    return triplets


def process_pdf(pdf_path: Path) -> List[Dict[str, str]]:
    """Traite un PDF et retourne la liste des triplets."""
    text = load_text(pdf_path)
    all_triplets = []
    
    for sent in sentences(text):
        triplets = extract_triplets(sent)
        all_triplets.extend(triplets)
    
    return all_triplets


def main() -> int:
    # Utiliser le chemin fourni en argument ou le chemin par défaut
    if len(sys.argv) > 1:
        PDF_DIR = Path(sys.argv[1])
    else:
        PDF_DIR = DEFAULT_PDF_DIR
    
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Aucun PDF trouvé dans {PDF_DIR}")
        print(f"Usage: {sys.argv[0]} [chemin_vers_dossier_pdfs]")
        return 1
    
    # Créer le dossier data s'il n'existe pas
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    all_relations = []
    
    for pdf in pdfs:
        try:
            triplets = process_pdf(pdf)
            all_relations.extend(triplets)
            print(f"{pdf.name}: {len(triplets)} relations")
        except Exception as exc:
            print(f"Erreur sur {pdf.name}: {exc}", file=sys.stderr)
    
    # Écrire CSV
    if all_relations:
        fieldnames = ["Sujet", "Type_Sujet", "Relation", "Objet", "Type_Objet", "Contexte"]
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_relations)
        print(f"Total écrit: {len(all_relations)} relations dans {OUT_CSV}")
        
        # Écrire JSON
        with OUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(all_relations, f, ensure_ascii=False, indent=2)
        print(f"Total écrit: {len(all_relations)} relations dans {OUT_JSON}")
    else:
        print("Aucune relation extraite.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


