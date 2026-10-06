import json
from pathlib import Path

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

BERT_DIR = BASE_DIR / "ml" / "models" / "bert_pii"

SPACY_DIR = BASE_DIR / "ml" / "models" / "spacy_pii_combined"

TEST_PATH = BASE_DIR / "ml" / "data" / "combined" / "test.json"

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "hybrid_bert_spacy_results.txt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LABELS
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

ID_TO_LABEL = {
    i: label
    for i, label in enumerate(LABELS)
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("HYBRID MODEL 2 — BERT + spaCy")
print("=" * 70)

print("\nDevice:", DEVICE)


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
# LOAD BERT
# ============================================================

print("\nLoading BERT model...")

bert_tokenizer = AutoTokenizer.from_pretrained(
    BERT_DIR
)

bert_model = AutoModelForTokenClassification.from_pretrained(
    BERT_DIR
)

bert_model.to(DEVICE)
bert_model.eval()

print("BERT loaded successfully.")


# ============================================================
# LOAD SPACY
# ============================================================

print("\nLoading spaCy model...")

nlp = spacy.load(
    SPACY_DIR
)

print("spaCy loaded successfully.")


# ============================================================
# GOLD EXTRACTION
# ============================================================

def get_gold_entities(example):

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
# BERT PREDICTION
# ============================================================

def predict_bert(text):

    encoded = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )

    offsets = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    inputs = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = bert_model(
            **inputs
        )

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )[0].cpu().tolist()

    token_predictions = []

    for pred_id, offset in zip(
        predictions,
        offsets
    ):

        start, end = offset

        if start == end:
            continue

        label = ID_TO_LABEL.get(
            pred_id,
            "O"
        )

        token_predictions.append(
            (
                start,
                end,
                label
            )
        )

    return token_predictions


# ============================================================
# BERT BIO → ENTITIES
# ============================================================

def bert_to_entities(
    predictions
):

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
# SPACY PREDICTION
# ============================================================

def predict_spacy(text):

    doc = nlp(text)

    return [
        (
            ent.start_char,
            ent.end_char,
            ent.label_
        )
        for ent in doc.ents
    ]


# ============================================================
# HYBRID COMBINATION
# ============================================================

def overlap(
    a_start,
    a_end,
    b_start,
    b_end
):

    return (
        max(a_start, b_start)
        < min(a_end, b_end)
    )


def hybrid_entities(
    bert_entities,
    spacy_entities
):

    result = []

    used_spacy = set()

    # --------------------------------------------------------
    # 1. Exact agreement
    # --------------------------------------------------------

    for bert in bert_entities:

        b_start, b_end, b_label = bert

        for i, spacy_ent in enumerate(
            spacy_entities
        ):

            s_start, s_end, s_label = spacy_ent

            if (
                b_start == s_start
                and b_end == s_end
                and b_label == s_label
            ):

                result.append(
                    bert
                )

                used_spacy.add(i)

                break

    # --------------------------------------------------------
    # 2. BERT entities not exactly matched
    #    Check overlapping spaCy entity
    # --------------------------------------------------------

    for bert in bert_entities:

        if bert in result:
            continue

        b_start, b_end, b_label = bert

        candidates = []

        for i, spacy_ent in enumerate(
            spacy_entities
        ):

            if i in used_spacy:
                continue

            s_start, s_end, s_label = spacy_ent

            if (
                b_label == s_label
                and overlap(
                    b_start,
                    b_end,
                    s_start,
                    s_end
                )
            ):

                candidates.append(
                    (
                        i,
                        spacy_ent
                    )
                )

        if candidates:

            # Choose the entity with the
            # largest overlap.

            best_i = None
            best_entity = None
            best_overlap = 0

            for i, ent in candidates:

                s_start, s_end, _ = ent

                overlap_size = (
                    min(
                        b_end,
                        s_end
                    )
                    -
                    max(
                        b_start,
                        s_start
                    )
                )

                if overlap_size > best_overlap:

                    best_overlap = overlap_size
                    best_i = i
                    best_entity = ent

            if best_entity is not None:

                # Prefer the longer span because
                # spaCy may recover complete words
                # while BERT may use subword offsets.

                s_start, s_end, s_label = best_entity

                if (
                    s_end - s_start
                    >
                    b_end - b_start
                ):

                    result.append(
                        best_entity
                    )

                else:

                    result.append(
                        bert
                    )

                used_spacy.add(
                    best_i
                )

    # --------------------------------------------------------
    # 3. Add remaining spaCy entities only when
    #    they do not conflict with BERT
    # --------------------------------------------------------

    for i, spacy_ent in enumerate(
        spacy_entities
    ):

        if i in used_spacy:
            continue

        s_start, s_end, s_label = spacy_ent

        conflict = False

        for bert in bert_entities:

            b_start, b_end, b_label = bert

            if overlap(
                s_start,
                s_end,
                b_start,
                b_end
            ):

                conflict = True
                break

        if not conflict:

            result.append(
                spacy_ent
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    return list(
        set(result)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    gold,
    predicted
):

    gold_set = set(gold)
    pred_set = set(predicted)

    correct = len(
        gold_set.intersection(
            pred_set
        )
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
        2 * precision * recall
        /
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
# EVALUATION
# ============================================================

gold_all = []
bert_all = []
spacy_all = []
hybrid_all = []

print("\nRunning hybrid evaluation...")

for index, example in enumerate(
    test_data,
    start=1
):

    text = example.get(
        "text",
        ""
    )

    gold = get_gold_entities(
        example
    )

    bert_tokens = predict_bert(
        text
    )

    bert_entities = bert_to_entities(
        bert_tokens
    )

    spacy_entities = predict_spacy(
        text
    )

    hybrid = hybrid_entities(
        bert_entities,
        spacy_entities
    )

    gold_all.extend(
        gold
    )

    bert_all.extend(
        bert_entities
    )

    spacy_all.extend(
        spacy_entities
    )

    hybrid_all.extend(
        hybrid
    )

    if index % 500 == 0:

        print(
            f"Processed {index}/{len(test_data)}"
        )


# ============================================================
# CALCULATE
# ============================================================

bert_results = calculate_metrics(
    gold_all,
    bert_all
)

spacy_results = calculate_metrics(
    gold_all,
    spacy_all
)

hybrid_results = calculate_metrics(
    gold_all,
    hybrid_all
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("HYBRID BERT + spaCy RESULTS")
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


print("\nspaCy")
print("-" * 70)

print(
    f"Precision : {spacy_results[0]:.4f}"
)

print(
    f"Recall    : {spacy_results[1]:.4f}"
)

print(
    f"F1-score  : {spacy_results[2]:.4f}"
)


print("\nHYBRID — BERT + spaCy")
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
# SAVE
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
        "HYBRID BERT + spaCy EVALUATION\n"
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

    f.write("spaCy\n")

    f.write(
        f"Precision : {spacy_results[0]:.4f}\n"
    )

    f.write(
        f"Recall    : {spacy_results[1]:.4f}\n"
    )

    f.write(
        f"F1-score  : {spacy_results[2]:.4f}\n\n"
    )

    f.write(
        "HYBRID — BERT + spaCy\n"
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
print("HYBRID BERT + spaCy EVALUATION COMPLETE")
print("=" * 70)