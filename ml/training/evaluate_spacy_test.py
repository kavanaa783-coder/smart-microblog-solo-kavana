import json
import spacy
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "spacy_pii_combined"
)

TEST_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "combined"
    / "test.json"
)

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "spacy_test_results.txt"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("spaCy PII MODEL TEST EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_DIR)

print("\nTest data:")
print(TEST_PATH)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading spaCy model...")

nlp = spacy.load(MODEL_DIR)

print("spaCy model loaded successfully.")

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    test_data = json.load(f)

print(
    "Test examples:",
    len(test_data)
)

# ============================================================
# HELPER — EXTRACT GOLD ENTITIES
# ============================================================

def extract_gold_entities(item):

    entities = []

    # --------------------------------------------------------
    # Format 1:
    # {
    #   "text": "...",
    #   "entities": [
    #       {"start": 0, "end": 5, "label": "PERSON"}
    #   ]
    # }
    # --------------------------------------------------------

    raw_entities = item.get("entities", [])

    for entity in raw_entities:

        if isinstance(entity, dict):

            start = entity.get("start")
            end = entity.get("end")

            label = (
                entity.get("label")
                or entity.get("type")
                or entity.get("entity")
            )

            if (
                start is not None
                and end is not None
                and label
            ):

                entities.append(
                    (
                        int(start),
                        int(end),
                        str(label)
                    )
                )

        # ----------------------------------------------------
        # Format 2:
        # [start, end, label]
        # ----------------------------------------------------

        elif isinstance(entity, list):

            if len(entity) >= 3:

                start = entity[0]
                end = entity[1]
                label = entity[2]

                entities.append(
                    (
                        int(start),
                        int(end),
                        str(label)
                    )
                )

    return entities


# ============================================================
# EVALUATION STORAGE
# ============================================================

total_gold = 0
total_predicted = 0
total_correct = 0

per_entity_gold = defaultdict(int)
per_entity_pred = defaultdict(int)
per_entity_correct = defaultdict(int)

# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning evaluation...")

for index, item in enumerate(test_data):

    text = item.get("text", "")

    if not text:
        continue

    # --------------------------------------------------------
    # GOLD ENTITIES
    # --------------------------------------------------------

    gold_entities = extract_gold_entities(item)

    gold_set = set(gold_entities)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    doc = nlp(text)

    predicted_entities = []

    for ent in doc.ents:

        predicted_entities.append(
            (
                ent.start_char,
                ent.end_char,
                ent.label_
            )
        )

    pred_set = set(predicted_entities)

    # --------------------------------------------------------
    # CORRECT ENTITIES
    # Exact span + exact label match
    # --------------------------------------------------------

    correct_set = (
        gold_set.intersection(pred_set)
    )

    # --------------------------------------------------------
    # OVERALL COUNTS
    # --------------------------------------------------------

    total_gold += len(gold_set)
    total_predicted += len(pred_set)
    total_correct += len(correct_set)

    # --------------------------------------------------------
    # PER ENTITY COUNTS
    # --------------------------------------------------------

    for start, end, label in gold_set:

        per_entity_gold[label] += 1

    for start, end, label in pred_set:

        per_entity_pred[label] += 1

    for start, end, label in correct_set:

        per_entity_correct[label] += 1

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (index + 1) % 500 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(test_data)}"
        )

# ============================================================
# OVERALL METRICS
# ============================================================

precision = (
    total_correct / total_predicted
    if total_predicted > 0
    else 0.0
)

recall = (
    total_correct / total_gold
    if total_gold > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)

# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("spaCy TEST EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nGold entities      : {total_gold}"
)

print(
    f"Predicted entities : {total_predicted}"
)

print(
    f"Correct entities   : {total_correct}"
)

print(
    f"\nPrecision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

# ============================================================
# PER-ENTITY RESULTS
# ============================================================

all_labels = sorted(
    set(per_entity_gold.keys())
    | set(per_entity_pred.keys())
)

print("\n" + "=" * 70)
print("PER-ENTITY RESULTS")
print("=" * 70)

print(
    f"{'ENTITY':<25}"
    f"{'PRECISION':<15}"
    f"{'RECALL':<15}"
    f"{'F1':<15}"
    f"{'SUPPORT':<10}"
)

print("-" * 80)

results = []

for label in all_labels:

    gold = per_entity_gold[label]
    predicted = per_entity_pred[label]
    correct = per_entity_correct[label]

    entity_precision = (
        correct / predicted
        if predicted > 0
        else 0.0
    )

    entity_recall = (
        correct / gold
        if gold > 0
        else 0.0
    )

    entity_f1 = (
        2
        * entity_precision
        * entity_recall
        / (entity_precision + entity_recall)
        if (
            entity_precision
            + entity_recall
        ) > 0
        else 0.0
    )

    print(
        f"{label:<25}"
        f"{entity_precision:<15.4f}"
        f"{entity_recall:<15.4f}"
        f"{entity_f1:<15.4f}"
        f"{gold:<10}"
    )

    results.append(
        (
            label,
            entity_precision,
            entity_recall,
            entity_f1,
            gold
        )
    )

# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "spaCy PII MODEL TEST EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Model: {MODEL_DIR}\n"
    )

    f.write(
        f"Test data: {TEST_PATH}\n\n"
    )

    f.write(
        f"Test examples: {len(test_data)}\n\n"
    )

    f.write(
        f"Gold entities      : {total_gold}\n"
    )

    f.write(
        f"Predicted entities : {total_predicted}\n"
    )

    f.write(
        f"Correct entities   : {total_correct}\n\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1-score  : {f1:.4f}\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("PER-ENTITY RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"{'ENTITY':<25}"
        f"{'PRECISION':<15}"
        f"{'RECALL':<15}"
        f"{'F1':<15}"
        f"{'SUPPORT':<10}\n"
    )

    f.write("-" * 80 + "\n")

    for (
        label,
        entity_precision,
        entity_recall,
        entity_f1,
        support
    ) in results:

        f.write(
            f"{label:<25}"
            f"{entity_precision:<15.4f}"
            f"{entity_recall:<15.4f}"
            f"{entity_f1:<15.4f}"
            f"{support:<10}\n"
        )

# ============================================================
# COMPLETE
# ============================================================

print("\nResults saved to:")

print(RESULT_PATH)

print("\n" + "=" * 70)
print("spaCy TEST EVALUATION COMPLETE")
print("=" * 70)