#!/usr/bin/env python3
"""
Extraction simple d'entités nutritionnelles depuis les PDFs du dossier.
Produit un CSV avec la même structure que entities_schema_sample.csv.

Heuristique légère (mots-clés + regex). Pour de meilleurs résultats, il
faudrait un modèle NER dédié, mais ce script donne une base rapide.

Usage:
    python scripts/extract_entities.py [chemin_vers_dossier_pdfs]
"""

import csv
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Chemin par défaut : dossier parent (où se trouvent les PDFs)
DEFAULT_PDF_DIR = Path(__file__).parent.parent.parent
OUT_CSV = Path(__file__).parent.parent / "data" / "entities_extracted.csv"

# Listes simples pour détecter quelques catégories
ALIMENTS = [
    "poulet",
    "poisson",
    "saumon",
    "thon",
    "brocoli",
    "tomate",
    "lait",
    "fromage",
    "yaourt",
    "riz",
    "pomme",
    "banane",
    "noix",
    "amande",
    "huile d'olive",
    "avocat",
    "légumes",
    "fruits",
]

NUTRIMENTS = [
    "protéine",
    "protéines",
    "glucide",
    "glucides",
    "lipide",
    "lipides",
    "omega-3",
    "oméga-3",
    "vitamine",
    "vitamine c",
    "vitamine d",
    "fer",
    "calcium",
    "magnésium",
    "zinc",
    "fibres",
]

CONDITIONS = [
    "diabète",
    "diabetes",
    "hypertension",
    "obésité",
    "obesite",
    "grossesse",
    "allergie",
    "intolérance",
    "intolerance",
    "cholestérol",
    "cholesterol",
]

OBJECTIFS = [
    "perte de poids",
    "maigrir",
    "maintenir le poids",
    "prise de masse",
    "mass musculaire",
    "musculaire",
    "maintien",
]

RECO_PATTERNS = [
    "doit",
    "doivent",
    "devrait",
    "limiter",
    "éviter",
    "eviter",
    "recommand",
    "conseille",
    "essentiel",
    "favorise",
]

MEASURE_RE = re.compile(
    r"\b\d+[\s\u00A0]?(?:g|mg|µg|mcg|kg|kcal|cal|kj|g/j|g jour|kcal/j)\b",
    flags=re.IGNORECASE,
)


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


def sentences(text: str) -> Iterable[str]:
    """Découpe grossièrement en phrases."""
    for chunk in re.split(r"[.!?]\s+", text):
        s = chunk.strip()
        if len(s) > 20:  # éviter les fragments trop courts
            yield s


def find_matches(sent: str) -> List[Tuple[str, str, str]]:
    """Retourne des lignes (type, description, exemple) pour une phrase."""
    lower = sent.lower()
    rows: List[Tuple[str, str, str]] = []

    if any(a in lower for a in ALIMENTS):
        rows.append(
            (
                "ALIMENT",
                "Nom de l'aliment, ingrédient ou plat.",
                sent,
            )
        )
    if any(n in lower for n in NUTRIMENTS):
        rows.append(
            (
                "NUTRIMENT",
                "Molécule spécifique (macro ou micro-nutriment).",
                sent,
            )
        )
    if MEASURE_RE.search(sent):
        rows.append(
            (
                "MESURE/VALEUR",
                "Quantités, dosages, pourcentages, valeurs énergétiques.",
                sent,
            )
        )
    if any(o in lower for o in OBJECTIFS):
        rows.append(
            (
                "OBJECTIF_CORPS",
                "État corporel souhaité ou maintenance.",
                sent,
            )
        )
    if any(c in lower for c in CONDITIONS):
        rows.append(
            (
                "CONDITION_SANTE",
                "Maladie, allergie, intolérance, ou état physiologique.",
                sent,
            )
        )
    if any(r in lower for r in RECO_PATTERNS):
        rows.append(
            (
                "RECOMMANDATION",
                "Verbes ou expressions indiquant relation positive/négative.",
                sent,
            )
        )
    return rows


def process_pdf(pdf_path: Path, writer: csv.writer) -> int:
    text = load_text(pdf_path)
    count = 0
    for sent in sentences(text):
        for row in find_matches(sent):
            writer.writerow(row)
            count += 1
    return count


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

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Type d'Entité", "Description", "Exemple d'Annotation dans un texte"]
        )
        total = 0
        for pdf in pdfs:
            try:
                n = process_pdf(pdf, writer)
                print(f"{pdf.name}: {n} lignes")
                total += n
            except Exception as exc:
                print(f"Erreur sur {pdf.name}: {exc}", file=sys.stderr)
        print(f"Total écrit: {total} lignes dans {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


