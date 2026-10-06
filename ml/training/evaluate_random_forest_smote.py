import json
import joblib
import numpy as np

from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "ml" / "models" / "random_forest_smote"

MODEL_PATH = MODEL_DIR / "random_forest_smote.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_PATH = MODEL_DIR / "label_names.json"

TEST_PATH = BASE_DIR / "ml" / "data" / "combined" / "test.json"

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "random_forest_smote_test_results.txt"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RANDOM FOREST + SMOTE MODEL EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nTest data:")
print(TEST_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Random Forest + SMOTE model...")

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")


# ============================================================
# LOAD TF-IDF
# ============================================================

print("\nLoading TF-IDF vectorizer...")

vectorizer = joblib.load(VECTORIZER_PATH)

print("TF-IDF vectorizer loaded successfully.")


# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading label names...")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_data = json.load(f)


# Support either:
# {"0": "B-AADHAAR", ...}
# or ["B-AADHAAR", ...]
if isinstance(label_data, dict):
    label_names = {}

    for key, value in label_data.items():
        try:
            label_names[int(key)] = value
        except:
            pass

else:
    label_names = {
        i: value
        for i, value in enumerate(label_data)
    }


print("Labels loaded.")

print("\nLabel mapping:")

for index in sorted(label_names):
    print(f"{index} -> {label_names[index]}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("Test examples:", len(test_data))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_text(item):
    """
    Extract text from different possible JSON formats.
    """

    if isinstance(item, dict):

        if "text" in item:
            return item["text"]

        if "sentence" in item:
            return item["sentence"]

        if "content" in item:
            return item["content"]

    return ""


def get_entities(item):
    """
    Extract entities from the dataset.

    Expected common format:

    entities = [
        {
            "start": 10,
            "end": 20,
            "label": "PERSON"
        }
    ]

    Also supports:
    [start, end, label]
    """

    entities = []

    if not isinstance(item, dict):
        return entities

    raw_entities = item.get("entities", [])

    for entity in raw_entities:

        start = None
        end = None
        label = None

        if isinstance(entity, dict):

            start = entity.get("start")
            end = entity.get("end")

            label = (
                entity.get("label")
                or entity.get("entity")
                or entity.get("type")
            )

        elif isinstance(entity, (list, tuple)):

            if len(entity) >= 3:
                start = entity[0]
                end = entity[1]
                label = entity[2]

        if (
            start is not None
            and end is not None
            and label is not None
        ):

            entities.append(
                (
                    int(start),
                    int(end),
                    str(label)
                )
            )

    return entities


def make_token_texts(text):
    """
    Simple whitespace tokenization.

    Returns:
        tokens
        start_positions
        end_positions
    """

    tokens = []
    starts = []
    ends = []

    position = 0

    for part in text.split():

        start = text.find(part, position)

        if start == -1:
            continue

        end = start + len(part)

        tokens.append(part)
        starts.append(start)
        ends.append(end)

        position = end

    return tokens, starts, ends


def normalize_label(label):
    """
    Converts BIO labels into entity labels.

    B-PERSON -> PERSON
    I-PERSON -> PERSON
    PERSON   -> PERSON
    """

    label = str(label)

    if label.startswith("B-"):
        return label[2:]

    if label.startswith("I-"):
        return label[2:]

    return label


# ============================================================
# PREPARE TOKEN-LEVEL TEST DATA
# ============================================================

print("\nPreparing evaluation data...")

X_text = []
y_gold = []

token_metadata = []

for item_index, item in enumerate(test_data):

    text = get_text(item)

    entities = get_entities(item)

    tokens, starts, ends = make_token_texts(text)

    # --------------------------------------------------------
    # Create gold BIO labels for tokens
    # --------------------------------------------------------

    token_labels = ["O"] * len(tokens)

    for entity_start, entity_end, entity_label in entities:

        entity_tokens = []

        for i, (start, end) in enumerate(
            zip(starts, ends)
        ):

            # Token overlaps entity span
            if end > entity_start and start < entity_end:

                entity_tokens.append(i)

        if not entity_tokens:
            continue

        for j, token_index in enumerate(entity_tokens):

            if j == 0:
                token_labels[token_index] = (
                    "B-" + normalize_label(entity_label)
                )
            else:
                token_labels[token_index] = (
                    "I-" + normalize_label(entity_label)
                )

    # --------------------------------------------------------
    # Add tokens
    # --------------------------------------------------------

    for i, token in enumerate(tokens):

        X_text.append(token)
        y_gold.append(token_labels[i])

        token_metadata.append(
            (
                item_index,
                starts[i],
                ends[i]
            )
        )

    if (item_index + 1) % 500 == 0:

        print(
            f"Prepared {item_index + 1}/{len(test_data)}"
        )


print("\nTotal token samples:", len(X_text))


# ============================================================
# TF-IDF TRANSFORMATION
# ============================================================

print("\nTransforming test data with TF-IDF...")

X = vectorizer.transform(X_text)

print("TF-IDF matrix:", X.shape)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning Random Forest predictions...")

raw_predictions = model.predict(X)

y_pred = []

for value in raw_predictions:

    # The trained Random Forest already returns string labels
    # such as B-PERSON, I-PERSON, O, etc.
    if isinstance(value, str):
        y_pred.append(value)

    else:
        # Safety handling if a numeric prediction is returned
        try:
            value = int(value)

            if value in label_names:
                y_pred.append(label_names[value])
            else:
                y_pred.append("O")

        except (ValueError, TypeError):
            y_pred.append("O")


print("Prediction complete.")

print("\nSample predictions:")

for i in range(min(10, len(y_pred))):
    print(
        f"{i + 1}. "
        f"Token: {X_text[i][:50]!r} "
        f"Gold: {y_gold[i]} "
        f"Pred: {y_pred[i]}"
    )

# ============================================================
# TOKEN-LEVEL RESULTS
# ============================================================

print("\nCalculating token-level metrics...")

labels_for_report = sorted(
    set(y_gold) | set(y_pred)
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_gold,
        y_pred,
        labels=labels_for_report,
        average=None,
        zero_division=0
    )
)


macro_p, macro_r, macro_f1, _ = (
    precision_recall_fscore_support(
        y_gold,
        y_pred,
        labels=labels_for_report,
        average="macro",
        zero_division=0
    )
)


weighted_p, weighted_r, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_gold,
        y_pred,
        labels=labels_for_report,
        average="weighted",
        zero_division=0
    )
)


micro_p, micro_r, micro_f1, _ = (
    precision_recall_fscore_support(
        y_gold,
        y_pred,
        labels=labels_for_report,
        average="micro",
        zero_division=0
    )
)


# ============================================================
# ENTITY-LEVEL RECONSTRUCTION
# ============================================================

print("\nReconstructing predicted entities...")

gold_entities = []
pred_entities = []


def extract_entities_from_bio(
    labels,
    metadata,
    dataset_items
):

    entities = []

    current_label = None
    current_start = None
    current_end = None
    current_doc = None

    for label, meta in zip(labels, metadata):

        doc_index, start, end = meta

        if label == "O":

            if current_label is not None:

                entities.append(
                    (
                        current_doc,
                        current_start,
                        current_end,
                        current_label
                    )
                )

                current_label = None
                current_start = None
                current_end = None
                current_doc = None

            continue

        if label.startswith("B-"):

            if current_label is not None:

                entities.append(
                    (
                        current_doc,
                        current_start,
                        current_end,
                        current_label
                    )
                )

            current_label = normalize_label(label)
            current_start = start
            current_end = end
            current_doc = doc_index

        elif label.startswith("I-"):

            entity_label = normalize_label(label)

            if (
                current_label is not None
                and current_doc == doc_index
                and current_label == entity_label
            ):

                current_end = end

            else:

                if current_label is not None:

                    entities.append(
                        (
                            current_doc,
                            current_start,
                            current_end,
                            current_label
                        )
                    )

                current_label = entity_label
                current_start = start
                current_end = end
                current_doc = doc_index

    if current_label is not None:

        entities.append(
            (
                current_doc,
                current_start,
                current_end,
                current_label
            )
        )

    return entities


# Gold BIO -> entities
gold_entities = extract_entities_from_bio(
    y_gold,
    token_metadata,
    test_data
)


# Predicted BIO -> entities
pred_entities = extract_entities_from_bio(
    y_pred,
    token_metadata,
    test_data
)


gold_set = set(gold_entities)
pred_set = set(pred_entities)

correct_entities = len(
    gold_set.intersection(pred_set)
)


entity_precision = (
    correct_entities / len(pred_set)
    if pred_set
    else 0
)

entity_recall = (
    correct_entities / len(gold_set)
    if gold_set
    else 0
)

entity_f1 = (
    2 * entity_precision * entity_recall
    / (entity_precision + entity_recall)
    if entity_precision + entity_recall > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST + SMOTE TEST RESULTS")
print("=" * 70)

print(
    f"\nToken-level Precision : {micro_p:.4f}"
)

print(
    f"Token-level Recall    : {micro_r:.4f}"
)

print(
    f"Token-level F1-score  : {micro_f1:.4f}"
)

print(
    f"\nMacro Precision : {macro_p:.4f}"
)

print(
    f"Macro Recall    : {macro_r:.4f}"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print(
    f"\nWeighted Precision : {weighted_p:.4f}"
)

print(
    f"Weighted Recall    : {weighted_r:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)


print("\n" + "=" * 70)
print("ENTITY-LEVEL RESULTS")
print("=" * 70)

print(
    f"\nGold entities      : {len(gold_set)}"
)

print(
    f"Predicted entities : {len(pred_set)}"
)

print(
    f"Correct entities   : {correct_entities}"
)

print(
    f"Entity Precision   : {entity_precision:.4f}"
)

print(
    f"Entity Recall      : {entity_recall:.4f}"
)

print(
    f"Entity F1-score    : {entity_f1:.4f}"
)


# ============================================================
# PER ENTITY RESULTS
# ============================================================

gold_by_label = defaultdict(set)
pred_by_label = defaultdict(set)

for doc_index, start, end, label in gold_set:

    gold_by_label[label].add(
        (
            doc_index,
            start,
            end
        )
    )

for doc_index, start, end, label in pred_set:

    pred_by_label[label].add(
        (
            doc_index,
            start,
            end
        )
    )


all_entity_labels = sorted(
    set(gold_by_label.keys())
    |
    set(pred_by_label.keys())
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

entity_results = []


for label in all_entity_labels:

    gold = gold_by_label[label]
    pred = pred_by_label[label]

    tp = len(gold.intersection(pred))

    p = (
        tp / len(pred)
        if pred
        else 0
    )

    r = (
        tp / len(gold)
        if gold
        else 0
    )

    f = (
        2 * p * r / (p + r)
        if p + r > 0
        else 0
    )

    support_count = len(gold)

    print(
        f"{label:<25}"
        f"{p:<15.4f}"
        f"{r:<15.4f}"
        f"{f:<15.4f}"
        f"{support_count:<10}"
    )

    entity_results.append(
        (
            label,
            p,
            r,
            f,
            support_count
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
        "RANDOM FOREST + SMOTE TEST EVALUATION RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test examples: {len(test_data)}\n"
    )

    f.write(
        f"Token samples: {len(X_text)}\n\n"
    )

    f.write(
        "TOKEN-LEVEL RESULTS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Precision : {micro_p:.4f}\n"
    )

    f.write(
        f"Recall    : {micro_r:.4f}\n"
    )

    f.write(
        f"F1-score  : {micro_f1:.4f}\n\n"
    )

    f.write(
        f"Macro Precision : {macro_p:.4f}\n"
    )

    f.write(
        f"Macro Recall    : {macro_r:.4f}\n"
    )

    f.write(
        f"Macro F1        : {macro_f1:.4f}\n\n"
    )

    f.write(
        f"Weighted Precision : {weighted_p:.4f}\n"
    )

    f.write(
        f"Weighted Recall    : {weighted_r:.4f}\n"
    )

    f.write(
        f"Weighted F1        : {weighted_f1:.4f}\n\n"
    )

    f.write(
        "ENTITY-LEVEL RESULTS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Gold entities      : {len(gold_set)}\n"
    )

    f.write(
        f"Predicted entities : {len(pred_set)}\n"
    )

    f.write(
        f"Correct entities   : {correct_entities}\n\n"
    )

    f.write(
        f"Precision : {entity_precision:.4f}\n"
    )

    f.write(
        f"Recall    : {entity_recall:.4f}\n"
    )

    f.write(
        f"F1-score  : {entity_f1:.4f}\n\n"
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
        f"{'F1':<15}"
        f"{'SUPPORT':<10}\n"
    )

    f.write("-" * 80 + "\n")

    for label, p, r, f1_value, support_count in entity_results:

        f.write(
            f"{label:<25}"
            f"{p:<15.4f}"
            f"{r:<15.4f}"
            f"{f1_value:<15.4f}"
            f"{support_count:<10}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\nResults saved to:")
print(RESULT_PATH)

print("\n" + "=" * 70)
print("RANDOM FOREST + SMOTE EVALUATION COMPLETE")
print("=" * 70)