import json
from pathlib import Path
import spacy
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models" / "spacy_pii"
VALIDATION_DATA = BASE_DIR / "data" / "processed" / "validation.json"

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "spacy_error_analysis.txt"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SPACY ERROR ANALYSIS")
print("=" * 70)

nlp = spacy.load(MODEL_DIR)

with open(VALIDATION_DATA, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# ANALYZE
# ============================================================

errors = defaultdict(list)

total_errors = 0

for item in data:

    text = item["text"]

    # Gold entities
    gold = {
        (
            entity["start"],
            entity["end"],
            entity["label"]
        )
        for entity in item["entities"]
    }

    # Model prediction
    doc = nlp(text)

    predicted = {
        (
            ent.start_char,
            ent.end_char,
            ent.label_
        )
        for ent in doc.ents
    }

    # False negatives
    missing = gold - predicted

    # False positives
    extra = predicted - gold

    # Record missing entities
    for start, end, label in missing:

        if label in ["DOB", "PIN"]:

            errors[label].append({
                "type": "MISSED",
                "text": text,
                "entity": text[start:end]
            })

            total_errors += 1

    # Record incorrect predictions
    for start, end, label in extra:

        if label in ["DOB", "PIN"]:

            errors[label].append({
                "type": "FALSE_POSITIVE",
                "text": text,
                "entity": text[start:end]
            })

            total_errors += 1


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

for label in ["DOB", "PIN"]:

    print(f"\n{label}")
    print("-" * 50)

    print(
        "Total errors:",
        len(errors[label])
    )


# ============================================================
# SAVE EXAMPLES
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("SPACY MODEL ERROR ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    for label in ["DOB", "PIN"]:

        f.write(f"\n{label}\n")
        f.write("-" * 70 + "\n")

        examples = errors[label][:50]

        for i, error in enumerate(examples, 1):

            f.write(f"\nExample {i}\n")
            f.write(f"Type: {error['type']}\n")
            f.write(f"Entity: {error['entity']}\n")
            f.write(f"Text: {error['text']}\n")

    f.write("\n\nTotal analyzed errors: ")
    f.write(str(total_errors))


print("\nDetailed examples saved to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)