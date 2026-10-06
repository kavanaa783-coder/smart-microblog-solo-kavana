import json
from collections import defaultdict

from .config import (
    VALIDATION_DATA,
    HYBRID_RESULTS_FILE,
)

from .hybrid_ensemble import (
    HybridPIIEnsemble,
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("HYBRID PII MODEL EVALUATION")
print("=" * 70)

print("\nValidation data:")
print(VALIDATION_DATA)

with open(
    VALIDATION_DATA,
    "r",
    encoding="utf-8"
) as f:

    validation_data = json.load(f)

print(
    "\nValidation documents:",
    len(validation_data)
)


# ============================================================
# LOAD HYBRID
# ============================================================

print("\nLoading hybrid ensemble...")

hybrid = HybridPIIEnsemble()


# ============================================================
# STORAGE
# ============================================================

total_gold = 0
total_predicted = 0
total_correct = 0

per_label_gold = defaultdict(int)
per_label_predicted = defaultdict(int)
per_label_correct = defaultdict(int)


# ============================================================
# EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING HYBRID PREDICTIONS")
print("=" * 70)

for index, item in enumerate(
    validation_data,
    start=1
):

    text = item["text"]

    # --------------------------------------------------------
    # Gold entities
    # --------------------------------------------------------

    gold_set = {
        (
            entity["start"],
            entity["end"],
            entity["label"]
        )
        for entity in item["entities"]
    }

    # --------------------------------------------------------
    # Hybrid prediction
    # --------------------------------------------------------

    predicted_entities = (
        hybrid.predict(text)
    )

    predicted_set = {
        (
            entity["start"],
            entity["end"],
            entity["label"]
        )
        for entity in predicted_entities
    }

    # --------------------------------------------------------
    # Correct entities
    # --------------------------------------------------------

    correct_set = (
        gold_set.intersection(
            predicted_set
        )
    )

    # --------------------------------------------------------
    # Overall counts
    # --------------------------------------------------------

    total_gold += len(gold_set)

    total_predicted += len(
        predicted_set
    )

    total_correct += len(
        correct_set
    )

    # --------------------------------------------------------
    # Per-label gold
    # --------------------------------------------------------

    for (
        start,
        end,
        label
    ) in gold_set:

        per_label_gold[label] += 1

    # --------------------------------------------------------
    # Per-label predictions
    # --------------------------------------------------------

    for (
        start,
        end,
        label
    ) in predicted_set:

        per_label_predicted[label] += 1

    # --------------------------------------------------------
    # Per-label correct
    # --------------------------------------------------------

    for (
        start,
        end,
        label
    ) in correct_set:

        per_label_correct[label] += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index % 100 == 0
        or index == len(validation_data)
    ):

        print(
            f"Processed "
            f"{index}/"
            f"{len(validation_data)}"
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
    if precision + recall > 0
    else 0.0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("HYBRID OVERALL RESULTS")
print("=" * 70)

print(
    f"\nGold entities      : "
    f"{total_gold}"
)

print(
    f"Predicted entities : "
    f"{total_predicted}"
)

print(
    f"Correct entities   : "
    f"{total_correct}"
)

print(
    f"\nPrecision : "
    f"{precision:.4f}"
)

print(
    f"Recall    : "
    f"{recall:.4f}"
)

print(
    f"F1-score  : "
    f"{f1:.4f}"
)


# ============================================================
# PER ENTITY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-ENTITY RESULTS")
print("=" * 70)

print(
    f"\n"
    f"{'ENTITY':<25}"
    f"{'PRECISION':<15}"
    f"{'RECALL':<15}"
    f"{'F1':<15}"
)

print("-" * 70)

labels = sorted(
    set(per_label_gold.keys())
    |
    set(per_label_predicted.keys())
)

per_entity_results = []

for label in labels:

    gold = (
        per_label_gold[label]
    )

    predicted = (
        per_label_predicted[label]
    )

    correct = (
        per_label_correct[label]
    )

    label_precision = (
        correct / predicted
        if predicted > 0
        else 0.0
    )

    label_recall = (
        correct / gold
        if gold > 0
        else 0.0
    )

    label_f1 = (
        2
        * label_precision
        * label_recall
        / (
            label_precision
            + label_recall
        )
        if (
            label_precision
            + label_recall
        ) > 0
        else 0.0
    )

    print(
        f"{label:<25}"
        f"{label_precision:<15.4f}"
        f"{label_recall:<15.4f}"
        f"{label_f1:<15.4f}"
    )

    per_entity_results.append(
        (
            label,
            label_precision,
            label_recall,
            label_f1
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving results...")

with open(
    HYBRID_RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "HYBRID PII MODEL VALIDATION RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Validation documents : "
        f"{len(validation_data)}\n"
    )

    f.write(
        f"Gold entities        : "
        f"{total_gold}\n"
    )

    f.write(
        f"Predicted entities   : "
        f"{total_predicted}\n"
    )

    f.write(
        f"Correct entities     : "
        f"{total_correct}\n\n"
    )

    f.write(
        f"Precision : "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Recall    : "
        f"{recall:.4f}\n"
    )

    f.write(
        f"F1-score  : "
        f"{f1:.4f}\n\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        "PER-ENTITY RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"{'ENTITY':<25}"
        f"{'PRECISION':<15}"
        f"{'RECALL':<15}"
        f"{'F1':<15}\n"
    )

    f.write("-" * 70 + "\n")

    for (
        label,
        p,
        r,
        entity_f1
    ) in per_entity_results:

        f.write(
            f"{label:<25}"
            f"{p:<15.4f}"
            f"{r:<15.4f}"
            f"{entity_f1:<15.4f}\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nResults saved to:")

print(
    HYBRID_RESULTS_FILE
)

print("\n")
print("=" * 70)
print("HYBRID EVALUATION COMPLETE")
print("=" * 70)