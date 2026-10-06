import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

BERT_DIR = BASE_DIR / "ml" / "models" / "bert_pii"
ROBERTA_DIR = BASE_DIR / "ml" / "models" / "roberta_pii"

TEST_PATH = BASE_DIR / "ml" / "data" / "combined" / "test.json"

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "hybrid_bert_roberta_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID MODEL 1 — BERT + RoBERTa")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading test dataset...")

with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("Test examples:", len(test_data))


# ============================================================
# LABEL MAPPING
# ============================================================

LABELS = [
    "O",
    "B-AADHAAR", "I-AADHAAR",
    "B-ACCOUNT", "I-ACCOUNT",
    "B-ADDRESS", "I-ADDRESS",
    "B-AGE", "I-AGE",
    "B-API_KEY", "I-API_KEY",
    "B-BANK_ACCOUNT", "I-BANK_ACCOUNT",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-CRYPTO_ADDRESS", "I-CRYPTO_ADDRESS",
    "B-DATE", "I-DATE",
    "B-DEVICE_ID", "I-DEVICE_ID",
    "B-DOB", "I-DOB",
    "B-DRIVER_LICENSE", "I-DRIVER_LICENSE",
    "B-EMAIL", "I-EMAIL",
    "B-IP_ADDRESS", "I-IP_ADDRESS",
    "B-LOCATION", "I-LOCATION",
    "B-MAC_ADDRESS", "I-MAC_ADDRESS",
    "B-PAN", "I-PAN",
    "B-PASSPORT", "I-PASSPORT",
    "B-PASSWORD", "I-PASSWORD",
    "B-PERSON", "I-PERSON",
    "B-PHONE", "I-PHONE",
    "B-PIN", "I-PIN",
    "B-SSN", "I-SSN",
    "B-UPI_ID", "I-UPI_ID",
    "B-URL", "I-URL",
    "B-USERNAME", "I-USERNAME",
    "B-USER_AGENT", "I-USER_AGENT",
    "B-VEHICLE_REG", "I-VEHICLE_REG"
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

print("\nNumber of labels:", len(LABELS))


# ============================================================
# LOAD BERT
# ============================================================

print("\nLoading BERT model...")

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
bert_model = AutoModelForTokenClassification.from_pretrained(BERT_DIR)

bert_model.to(DEVICE)
bert_model.eval()

print("BERT loaded successfully.")


# ============================================================
# LOAD ROBERTA
# ============================================================

print("\nLoading RoBERTa model...")

roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
roberta_model = AutoModelForTokenClassification.from_pretrained(ROBERTA_DIR)

roberta_model.to(DEVICE)
roberta_model.eval()

print("RoBERTa loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_model(text, tokenizer, model):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    model_inputs = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )[0].cpu().tolist()

    token_predictions = []

    for pred_id, offset in zip(predictions, offsets):

        start, end = offset

        if start == end:
            continue

        label = ID_TO_LABEL.get(
            pred_id,
            "O"
        )

        token_predictions.append(
            (start, end, label)
        )

    return token_predictions


# ============================================================
# CONVERT BIO TOKENS TO ENTITIES
# ============================================================

def build_entities(text, predictions):

    entities = []

    current_start = None
    current_end = None
    current_label = None

    for start, end, label in predictions:

        if label == "O":

            if current_label is not None:

                entities.append(
                    (
                        current_start,
                        current_end,
                        current_label
                    )
                )

            current_start = None
            current_end = None
            current_label = None

            continue

        if label.startswith("B-"):

            if current_label is not None:

                entities.append(
                    (
                        current_start,
                        current_end,
                        current_label
                    )
                )

            current_start = start
            current_end = end
            current_label = label[2:]

        elif label.startswith("I-"):

            entity_type = label[2:]

            if (
                current_label == entity_type
                and current_end is not None
                and start >= current_end
            ):

                current_end = end

            else:

                if current_label is not None:

                    entities.append(
                        (
                            current_start,
                            current_end,
                            current_label
                        )
                    )

                current_start = start
                current_end = end
                current_label = entity_type

    if current_label is not None:

        entities.append(
            (
                current_start,
                current_end,
                current_label
            )
        )

    return entities


# ============================================================
# HYBRID COMBINATION
# ============================================================

def hybrid_entities(bert_entities, roberta_entities):

    combined = {}

    # Add BERT predictions
    for start, end, label in bert_entities:

        key = (start, end, label)

        combined[key] = combined.get(
            key,
            0
        ) + 1

    # Add RoBERTa predictions
    for start, end, label in roberta_entities:

        key = (start, end, label)

        combined[key] = combined.get(
            key,
            0
        ) + 1

    # Keep entities supported by both models.
    #
    # If an entity is predicted by only one model,
    # keep it only when there is no conflicting
    # overlapping entity.

    result = []

    all_items = list(combined.items())

    # First: consensus predictions
    for key, votes in all_items:

        if votes >= 2:

            result.append(key)

    # If no consensus exists for a region,
    # use the higher-confidence model behavior
    # indirectly through BERT/RoBERTa agreement.
    #
    # For fair evaluation, consensus is preferred.

    return result


# ============================================================
# GOLD ENTITY EXTRACTION
# ============================================================

def get_gold_entities(example):

    text = example.get("text", "")

    entities = example.get(
        "entities",
        []
    )

    result = []

    for entity in entities:

        if isinstance(entity, dict):

            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label")

            if (
                start is not None
                and end is not None
                and label is not None
            ):

                result.append(
                    (
                        int(start),
                        int(end),
                        str(label)
                    )
                )

        elif isinstance(entity, list):

            if len(entity) >= 3:

                result.append(
                    (
                        int(entity[0]),
                        int(entity[1]),
                        str(entity[2])
                    )
                )

    return result


# ============================================================
# EVALUATION
# ============================================================

gold_all = []
bert_all = []
roberta_all = []
hybrid_all = []

print("\nRunning hybrid evaluation...")

for index, example in enumerate(test_data, start=1):

    text = example.get(
        "text",
        ""
    )

    gold = get_gold_entities(example)

    bert_tokens = predict_model(
        text,
        bert_tokenizer,
        bert_model
    )

    roberta_tokens = predict_model(
        text,
        roberta_tokenizer,
        roberta_model
    )

    bert_entities = build_entities(
        text,
        bert_tokens
    )

    roberta_entities = build_entities(
        text,
        roberta_tokens
    )

    hybrid = hybrid_entities(
        bert_entities,
        roberta_entities
    )

    gold_all.extend(gold)
    bert_all.extend(bert_entities)
    roberta_all.extend(roberta_entities)
    hybrid_all.extend(hybrid)

    if index % 500 == 0:

        print(
            f"Processed {index}/{len(test_data)}"
        )


# ============================================================
# ENTITY METRICS
# ============================================================

def calculate_metrics(
    gold,
    predicted
):

    gold_set = set(gold)
    pred_set = set(predicted)

    correct = len(
        gold_set.intersection(pred_set)
    )

    precision = (
        correct / len(pred_set)
        if pred_set
        else 0
    )

    recall = (
        correct / len(gold_set)
        if gold_set
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return (
        precision,
        recall,
        f1,
        len(gold_set),
        len(pred_set),
        correct
    )


# ============================================================
# RESULTS
# ============================================================

bert_results = calculate_metrics(
    gold_all,
    bert_all
)

roberta_results = calculate_metrics(
    gold_all,
    roberta_all
)

hybrid_results = calculate_metrics(
    gold_all,
    hybrid_all
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("HYBRID BERT + RoBERTa RESULTS")
print("=" * 70)

print("\nBERT")
print("-" * 70)

print(
    f"Precision : {bert_results[0]:.4f}"
)

print(
    f"Recall    : {bert_results[1]:.4f}"
)

print(
    f"F1-score  : {bert_results[2]:.4f}"
)


print("\nRoBERTa")
print("-" * 70)

print(
    f"Precision : {roberta_results[0]:.4f}"
)

print(
    f"Recall    : {roberta_results[1]:.4f}"
)

print(
    f"F1-score  : {roberta_results[2]:.4f}"
)


print("\nHYBRID — BERT + RoBERTa")
print("-" * 70)

print(
    f"Precision : {hybrid_results[0]:.4f}"
)

print(
    f"Recall    : {hybrid_results[1]:.4f}"
)

print(
    f"F1-score  : {hybrid_results[2]:.4f}"
)

print(
    f"Gold entities      : {hybrid_results[3]}"
)

print(
    f"Predicted entities : {hybrid_results[4]}"
)

print(
    f"Correct entities   : {hybrid_results[5]}"
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
        "HYBRID BERT + RoBERTa EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("BERT\n")
    f.write(
        f"Precision : {bert_results[0]:.4f}\n"
    )
    f.write(
        f"Recall    : {bert_results[1]:.4f}\n"
    )
    f.write(
        f"F1-score  : {bert_results[2]:.4f}\n\n"
    )

    f.write("RoBERTa\n")
    f.write(
        f"Precision : {roberta_results[0]:.4f}\n"
    )
    f.write(
        f"Recall    : {roberta_results[1]:.4f}\n"
    )
    f.write(
        f"F1-score  : {roberta_results[2]:.4f}\n\n"
    )

    f.write(
        "HYBRID — BERT + RoBERTa\n"
    )

    f.write(
        f"Precision : {hybrid_results[0]:.4f}\n"
    )

    f.write(
        f"Recall    : {hybrid_results[1]:.4f}\n"
    )

    f.write(
        f"F1-score  : {hybrid_results[2]:.4f}\n"
    )

    f.write(
        f"Gold entities      : {hybrid_results[3]}\n"
    )

    f.write(
        f"Predicted entities : {hybrid_results[4]}\n"
    )

    f.write(
        f"Correct entities   : {hybrid_results[5]}\n"
    )


print("\nResults saved to:")
print(RESULT_PATH)

print("\n" + "=" * 70)
print("HYBRID EVALUATION COMPLETE")
print("=" * 70)