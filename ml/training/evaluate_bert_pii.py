import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "ml" / "models" / "bert_pii"
TEST_FILE = BASE_DIR / "ml" / "data" / "combined" / "test.json"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BERT PII MODEL EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_DIR)

print("\nTest data:")
print(TEST_FILE)

print("\nDevice:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("BERT model loaded.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Test examples:", len(data))


# ============================================================
# LABEL MAPPING
# ============================================================

id2label = model.config.id2label

print("\nLabels:")
for idx, label in id2label.items():
    print(idx, "->", label)


# ============================================================
# CONVERT CHARACTER ENTITIES TO TOKEN LABELS
# ============================================================

def create_bio_labels(text, entities, offsets):

    labels = ["O"] * len(offsets)

    for entity in entities:

        start = entity["start"]
        end = entity["end"]
        entity_type = entity["label"]

        token_indexes = []

        for i, (token_start, token_end) in enumerate(offsets):

            if token_start == token_end:
                continue

            # Token overlaps entity
            if token_start < end and token_end > start:
                token_indexes.append(i)

        if not token_indexes:
            continue

        first = True

        for token_idx in token_indexes:

            if first:
                labels[token_idx] = "B-" + entity_type
                first = False
            else:
                labels[token_idx] = "I-" + entity_type

    return labels


# ============================================================
# EVALUATION
# ============================================================

true_sequences = []
pred_sequences = []

print("\nRunning evaluation...")

with torch.no_grad():

    for count, item in enumerate(data, start=1):

        text = item["text"]
        entities = item.get("entities", [])

        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )[0].cpu().tolist()

        predicted_labels = [
            id2label[prediction]
            for prediction in predictions
        ]

        true_labels = create_bio_labels(
            text,
            entities,
            offsets
        )

        # Remove special tokens
        valid_true = []
        valid_pred = []

        for (start, end), true_label, pred_label in zip(
            offsets,
            true_labels,
            predicted_labels
        ):

            if start == end:
                continue

            valid_true.append(true_label)
            valid_pred.append(pred_label)

        true_sequences.append(valid_true)
        pred_sequences.append(valid_pred)

        if count % 500 == 0:
            print(
                f"Processed {count}/{len(data)}"
            )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BERT EVALUATION RESULTS")
print("=" * 70)

precision = precision_score(
    true_sequences,
    pred_sequences
)

recall = recall_score(
    true_sequences,
    pred_sequences
)

f1 = f1_score(
    true_sequences,
    pred_sequences
)

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# DETAILED REPORT
# ============================================================

print("\n" + "=" * 70)
print("PER-ENTITY RESULTS")
print("=" * 70)

print(
    classification_report(
        true_sequences,
        pred_sequences,
        digits=4
    )
)

print("\n" + "=" * 70)
print("BERT EVALUATION COMPLETE")
print("=" * 70)