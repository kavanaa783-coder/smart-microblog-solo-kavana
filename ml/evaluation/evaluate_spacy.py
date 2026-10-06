import spacy
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models" / "spacy_pii"
VALIDATION_DATA = BASE_DIR / "data" / "spacy" / "validation.spacy"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SPACY MODEL EVALUATION")
print("=" * 70)

print("\nLoading trained model...")

nlp = spacy.load(MODEL_DIR)

print("Model loaded successfully.")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

doc_bin = spacy.tokens.DocBin().from_disk(VALIDATION_DATA)

gold_docs = list(doc_bin.get_docs(nlp.vocab))

print("Validation documents:", len(gold_docs))


# ============================================================
# ENTITY-LEVEL EVALUATION
# ============================================================

print("\nRunning predictions...")

gold_entities = []
pred_entities = []

per_label_gold = defaultdict(int)
per_label_pred = defaultdict(int)
per_label_correct = defaultdict(int)

total_gold = 0
total_pred = 0
total_correct = 0


for gold_doc in gold_docs:

    # Run model
    pred_doc = nlp(gold_doc.text)

    # Convert entities into sets
    gold_set = {
        (ent.start_char, ent.end_char, ent.label_)
        for ent in gold_doc.ents
    }

    pred_set = {
        (ent.start_char, ent.end_char, ent.label_)
        for ent in pred_doc.ents
    }

    # Counts
    total_gold += len(gold_set)
    total_pred += len(pred_set)

    correct = gold_set.intersection(pred_set)

    total_correct += len(correct)

    # Per-label counts
    for start, end, label in gold_set:
        per_label_gold[label] += 1

    for start, end, label in pred_set:
        per_label_pred[label] += 1

    for start, end, label in correct:
        per_label_correct[label] += 1


# ============================================================
# OVERALL METRICS
# ============================================================

if total_pred > 0:

    precision = total_correct / total_pred

else:

    precision = 0


if total_gold > 0:

    recall = total_correct / total_gold

else:

    recall = 0


if precision + recall > 0:

    f1 = 2 * precision * recall / (precision + recall)

else:

    f1 = 0


# ============================================================
# DISPLAY OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"\nGold entities      : {total_gold}")
print(f"Predicted entities : {total_pred}")
print(f"Correct entities   : {total_correct}")

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# PER-ENTITY METRICS
# ============================================================

print("\n" + "=" * 70)
print("PER-ENTITY RESULTS")
print("=" * 70)

all_labels = sorted(
    set(per_label_gold.keys()) |
    set(per_label_pred.keys())
)

print(
    f"\n{'ENTITY':<20}"
    f"{'PRECISION':<15}"
    f"{'RECALL':<15}"
    f"{'F1':<15}"
)

print("-" * 65)


for label in all_labels:

    gold = per_label_gold[label]
    pred = per_label_pred[label]
    correct = per_label_correct[label]

    label_precision = correct / pred if pred > 0 else 0
    label_recall = correct / gold if gold > 0 else 0

    if label_precision + label_recall > 0:
        label_f1 = (
            2 * label_precision * label_recall
            / (label_precision + label_recall)
        )
    else:
        label_f1 = 0

    print(
        f"{label:<20}"
        f"{label_precision:<15.4f}"
        f"{label_recall:<15.4f}"
        f"{label_f1:<15.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

results_file = RESULTS_DIR / "spacy_validation_results.txt"

with open(results_file, "w", encoding="utf-8") as f:

    f.write("CUSTOM SPACY NER - VALIDATION RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Validation documents: {len(gold_docs)}\n")
    f.write(f"Gold entities: {total_gold}\n")
    f.write(f"Predicted entities: {total_pred}\n")
    f.write(f"Correct entities: {total_correct}\n\n")

    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n\n")

    f.write("PER-ENTITY RESULTS\n")
    f.write("-" * 60 + "\n")

    for label in all_labels:

        gold = per_label_gold[label]
        pred = per_label_pred[label]
        correct = per_label_correct[label]

        p = correct / pred if pred > 0 else 0
        r = correct / gold if gold > 0 else 0

        f1_label = (
            2 * p * r / (p + r)
            if p + r > 0
            else 0
        )

        f.write(
            f"{label}: "
            f"Precision={p:.4f}, "
            f"Recall={r:.4f}, "
            f"F1={f1_label:.4f}\n"
        )


print("\nResults saved to:")
print(results_file)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)