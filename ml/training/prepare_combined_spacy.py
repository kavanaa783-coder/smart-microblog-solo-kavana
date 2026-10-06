import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "ml" / "data" / "combined"
OUTPUT_DIR = BASE_DIR / "ml" / "data" / "spacy_combined"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REMOVE OVERLAPPING ENTITIES
# ============================================================

def remove_overlaps(spans):

    # Longest entities first
    spans = sorted(
        spans,
        key=lambda x: (x[1] - x[0]),
        reverse=True
    )

    selected = []

    for span in spans:

        start, end, label = span

        overlap = False

        for existing_start, existing_end, existing_label in selected:

            # Check overlap
            if not (
                end <= existing_start
                or start >= existing_end
            ):
                overlap = True
                break

        if not overlap:
            selected.append(span)

    # Sort back according to text position
    selected.sort(key=lambda x: (x[0], x[1]))

    return selected


# ============================================================
# PREPARE SPACY DATA
# ============================================================

def prepare_spacy(input_file, output_file):

    print("\n" + "=" * 70)
    print("PREPARING COMBINED DATASET FOR SPACY")
    print("=" * 70)

    print("\nInput:")
    print(input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Documents:", len(data))

    nlp = spacy.blank("en")

    doc_bin = DocBin(store_user_data=True)

    entity_count = 0
    skipped_entities = 0
    overlapping_entities = 0

    for item in data:

        text = item["text"]

        doc = nlp.make_doc(text)

        raw_entities = []

        # ----------------------------------------------------
        # READ ENTITIES
        # ----------------------------------------------------

        for entity in item["entities"]:

            start = entity["start"]
            end = entity["end"]
            label = entity["label"]

            # Basic boundary validation
            if start < 0 or end > len(text) or start >= end:
                skipped_entities += 1
                continue

            raw_entities.append(
                (start, end, label)
            )

        # ----------------------------------------------------
        # REMOVE OVERLAPPING ENTITIES
        # ----------------------------------------------------

        before = len(raw_entities)

        clean_entities = remove_overlaps(
            raw_entities
        )

        after = len(clean_entities)

        overlapping_entities += (
            before - after
        )

        # ----------------------------------------------------
        # CREATE SPACY SPANS
        # ----------------------------------------------------

        spans = []

        for start, end, label in clean_entities:

            span = doc.char_span(
                start,
                end,
                label=label,
                alignment_mode="contract"
            )

            if span is None:

                skipped_entities += 1
                continue

            spans.append(span)

        # ----------------------------------------------------
        # SET ENTITIES
        # ----------------------------------------------------

        doc.ents = spans

        entity_count += len(spans)

        doc_bin.add(doc)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    doc_bin.to_disk(output_file)

    print("\nSaved:")
    print(output_file)

    print("\nDocuments:", len(data))
    print("Valid entities:", entity_count)
    print("Skipped entities:", skipped_entities)
    print("Overlapping entities removed:", overlapping_entities)


# ============================================================
# TRAINING DATA
# ============================================================

prepare_spacy(
    INPUT_DIR / "train.json",
    OUTPUT_DIR / "train.spacy"
)


# ============================================================
# VALIDATION DATA
# ============================================================

prepare_spacy(
    INPUT_DIR / "validation.json",
    OUTPUT_DIR / "validation.spacy"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("COMBINED SPACY DATA PREPARATION COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)