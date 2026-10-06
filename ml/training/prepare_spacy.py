import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "spacy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONVERT JSON → SPACY .SPACY
# ============================================================

def convert_to_spacy(input_file, output_file):

    print(f"\nProcessing: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Blank English pipeline
    nlp = spacy.blank("en")

    # DocBin stores training documents efficiently
    doc_bin = DocBin(store_user_data=True)

    skipped = 0
    total_entities = 0

    for item in data:

        text = item["text"]

        # Create spaCy Doc
        doc = nlp.make_doc(text)

        spans = []

        for entity in item["entities"]:

            start = entity["start"]
            end = entity["end"]
            label = entity["label"]

            # Convert character positions → token positions
            span = doc.char_span(
                start,
                end,
                label=label,
                alignment_mode="contract"
            )

            if span is None:
                skipped += 1
                continue

            spans.append(span)
            total_entities += 1

        # Remove overlapping entities
        spans = filter_spans(spans)

        doc.ents = spans

        doc_bin.add(doc)

    doc_bin.to_disk(output_file)

    print(f"Saved: {output_file}")
    print(f"Documents: {len(data)}")
    print(f"Entities: {total_entities}")
    print(f"Skipped entities: {skipped}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PREPARING SPACY DATASET")
    print("=" * 60)

    convert_to_spacy(
        PROCESSED_DIR / "train.json",
        OUTPUT_DIR / "train.spacy"
    )

    convert_to_spacy(
        PROCESSED_DIR / "validation.json",
        OUTPUT_DIR / "validation.spacy"
    )

    print("\n" + "=" * 60)
    print("SPACY DATA PREPARATION COMPLETE")
    print("=" * 60)