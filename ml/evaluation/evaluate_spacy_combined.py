import spacy
from spacy.tokens import DocBin
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "ml" / "models" / "spacy_pii_combined"

VAL_PATH = BASE_DIR / "ml" / "data" / "spacy_combined" / "validation.spacy"

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "spacy_combined_validation_results.txt"
)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("IMPROVED SPACY MODEL EVALUATION")
print("=" * 70)

print("\nLoading improved model...")

nlp = spacy.load(MODEL_DIR)

print("Model loaded successfully.")

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

docbin = DocBin().from_disk(VAL_PATH)

gold_docs = list(
    docbin.get_docs(nlp.vocab)
)

print(
    "Validation documents:",
    len(gold_docs)
)

# ============================================================
# ENTITY MATCHING
# ============================================================

gold_entities = []
pred_entities = []

per_entity_gold = defaultdict(list)
per_entity_pred = defaultdict(list)

print("\nRunning predictions...")

for doc in gold_docs:

    predicted_doc = nlp(doc.text)

    gold_set = set()

    for ent in doc.ents:

        item = (
            ent.start_char,
            ent.end_char,
            ent.label_
        )

        gold_entities.append(item)
        gold_set.add(item)

        per_entity_gold[ent.label_].append(
            (ent.start_char, ent.end_char)
        )

    predicted_set = set()

    for ent in predicted_doc.ents:

        item = (
            ent.start_char,
            ent.end_char,
            ent.label_
        )

        pred_entities.append(item)
        predicted_set.add(item)

        per_entity_pred[ent.label_].append(
            (ent.start_char, ent.end_char)
        )

# ============================================================
# OVERALL RESULTS
# ============================================================

gold_set = set(gold_entities)
pred_set = set(pred_entities)

correct = len(
    gold_set.intersection(pred_set)
)

precision = (
    correct / len(pred_set)
    if pred_set else 0
)

recall = (
    correct / len(gold_set)
    if gold_set else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Gold entities      : {len(gold_set)}"
)

print(
    f"Predicted entities : {len(pred_set)}"
)

print(
    f"Correct entities   : {correct}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

# ============================================================
# PER ENTITY RESULTS
# ============================================================

all_labels = sorted(
    set(per_entity_gold.keys()) |
    set(per_entity_pred.keys())
)

print("\n" + "=" * 70)
print("PER-ENTITY RESULTS")
print("=" * 70)

print(
    f"{'ENTITY':<25}"
    f"{'PRECISION':<15}"
    f"{'RECALL':<15}"
    f"{'F1':<15}"
)

print("-" * 70)

results = []

for label in all_labels:

    gold = set(
        per_entity_gold[label]
    )

    pred = set(
        per_entity_pred[label]
    )

    tp = len(
        gold.intersection(pred)
    )

    p = (
        tp / len(pred)
        if pred else 0
    )

    r = (
        tp / len(gold)
        if gold else 0
    )

    entity_f1 = (
        2 * p * r / (p + r)
        if (p + r) > 0
        else 0
    )

    print(
        f"{label:<25}"
        f"{p:<15.4f}"
        f"{r:<15.4f}"
        f"{entity_f1:<15.4f}"
    )

    results.append(
        (
            label,
            p,
            r,
            entity_f1
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
        "IMPROVED SPACY MODEL VALIDATION RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Gold entities      : {len(gold_set)}\n"
    )

    f.write(
        f"Predicted entities : {len(pred_set)}\n"
    )

    f.write(
        f"Correct entities   : {correct}\n\n"
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

    for label, p, r, entity_f1 in results:

        f.write(
            f"{label:<25}"
            f"{p:<15.4f}"
            f"{r:<15.4f}"
            f"{entity_f1:<15.4f}\n"
        )

print("\nResults saved to:")
print(RESULT_PATH)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)