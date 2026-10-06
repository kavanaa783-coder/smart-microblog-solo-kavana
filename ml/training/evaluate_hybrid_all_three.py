import json
import torch
import spacy
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

BERT_DIR = BASE_DIR / "ml" / "models" / "bert_pii"
ROBERTA_DIR = BASE_DIR / "ml" / "models" / "roberta_pii"
SPACY_DIR = BASE_DIR / "ml" / "models" / "spacy_pii_combined"

TEST_PATH = BASE_DIR / "ml" / "data" / "combined" / "test.json"

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "hybrid_all_three_results.txt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL HYBRID MODEL — BERT + RoBERTa + spaCy")
print("=" * 70)

print("\nDevice:", DEVICE)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

with open(TEST_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Test examples:", len(data))


# ============================================================
# LOAD BERT
# ============================================================

print("\nLoading BERT model...")

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)

bert_model = AutoModelForTokenClassification.from_pretrained(
    BERT_DIR
)

bert_model.to(DEVICE)
bert_model.eval()

bert_id2label = {
    int(k): v
    for k, v in bert_model.config.id2label.items()
}

print("BERT loaded successfully.")


# ============================================================
# LOAD RoBERTa
# ============================================================

print("\nLoading RoBERTa model...")

roberta_tokenizer = AutoTokenizer.from_pretrained(
    ROBERTA_DIR
)

roberta_model = AutoModelForTokenClassification.from_pretrained(
    ROBERTA_DIR
)

roberta_model.to(DEVICE)
roberta_model.eval()

roberta_id2label = {
    int(k): v
    for k, v in roberta_model.config.id2label.items()
}

print("RoBERTa loaded successfully.")


# ============================================================
# LOAD spaCy
# ============================================================

print("\nLoading spaCy model...")

nlp = spacy.load(SPACY_DIR)

print("spaCy loaded successfully.")


# ============================================================
# TRANSFORMER ENTITY FUNCTION
# ============================================================

def get_transformer_entities(
    text,
    tokenizer,
    model,
    id2label
):

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    offsets = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = model(**encoded)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )[0].cpu().tolist()

    entities = []

    current_start = None
    current_end = None
    current_label = None

    for pred_id, (start, end) in zip(
        predictions,
        offsets
    ):

        # Skip special tokens
        if start == end:
            continue

        label = id2label.get(
            pred_id,
            "O"
        )

        # ----------------------------------------------------
        # Outside
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Beginning
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Inside
        # ----------------------------------------------------

        elif label.startswith("I-"):

            entity_label = label[2:]

            if (
                current_label == entity_label
                and current_start is not None
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
                current_label = entity_label

    # Final entity

    if current_label is not None:

        entities.append(
            (
                current_start,
                current_end,
                current_label
            )
        )

    return set(entities)


# ============================================================
# spaCy ENTITY FUNCTION
# ============================================================

def get_spacy_entities(text):

    doc = nlp(text)

    return set(
        (
            ent.start_char,
            ent.end_char,
            ent.label_
        )
        for ent in doc.ents
    )


# ============================================================
# FINAL 3-MODEL ENSEMBLE
# ============================================================

def get_hybrid_entities(
    bert_entities,
    roberta_entities,
    spacy_entities
):

    # --------------------------------------------------------
    # Count how many models agree on each entity
    # --------------------------------------------------------

    all_entities = (
        bert_entities
        | roberta_entities
        | spacy_entities
    )

    hybrid = set()

    for entity in all_entities:

        votes = 0

        if entity in bert_entities:
            votes += 1

        if entity in roberta_entities:
            votes += 1

        if entity in spacy_entities:
            votes += 1

        # ----------------------------------------------------
        # Majority voting
        # ----------------------------------------------------
        #
        # An entity must be predicted by at least
        # 2 out of 3 models.
        #
        # ----------------------------------------------------

        if votes >= 2:

            hybrid.add(entity)

    return hybrid


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    gold,
    predicted
):

    gold = set(gold)
    predicted = set(predicted)

    correct = len(
        gold.intersection(predicted)
    )

    precision = (
        correct / len(predicted)
        if predicted
        else 0
    )

    recall = (
        correct / len(gold)
        if gold
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return (
        precision,
        recall,
        f1,
        correct
    )


# ============================================================
# EVALUATION
# ============================================================

gold_all = []

bert_all = []

roberta_all = []

spacy_all = []

hybrid_all = []


print("\nRunning final 3-model hybrid evaluation...")

for index, item in enumerate(
    data,
    start=1
):

    text = item.get(
        "text",
        ""
    )

    # --------------------------------------------------------
    # GOLD ENTITIES
    # --------------------------------------------------------

    gold_entities = set()

    for entity in item.get(
        "entities",
        []
    ):

        if isinstance(
            entity,
            dict
        ):

            start = entity.get(
                "start"
            )

            end = entity.get(
                "end"
            )

            label = entity.get(
                "label"
            )

        else:

            start = entity[0]
            end = entity[1]
            label = entity[2]

        if (
            start is not None
            and end is not None
            and label is not None
        ):

            gold_entities.add(
                (
                    int(start),
                    int(end),
                    str(label)
                )
            )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    bert_entities = get_transformer_entities(
        text,
        bert_tokenizer,
        bert_model,
        bert_id2label
    )

    # --------------------------------------------------------
    # RoBERTa
    # --------------------------------------------------------

    roberta_entities = get_transformer_entities(
        text,
        roberta_tokenizer,
        roberta_model,
        roberta_id2label
    )

    # --------------------------------------------------------
    # spaCy
    # --------------------------------------------------------

    spacy_entities = get_spacy_entities(
        text
    )

    # --------------------------------------------------------
    # FINAL ENSEMBLE
    # --------------------------------------------------------

    hybrid_entities = get_hybrid_entities(
        bert_entities,
        roberta_entities,
        spacy_entities
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    gold_all.extend(
        gold_entities
    )

    bert_all.extend(
        bert_entities
    )

    roberta_all.extend(
        roberta_entities
    )

    spacy_all.extend(
        spacy_entities
    )

    hybrid_all.extend(
        hybrid_entities
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 500 == 0:

        print(
            f"Processed {index}/{len(data)}"
        )


# ============================================================
# CALCULATE RESULTS
# ============================================================

bert_p, bert_r, bert_f1, bert_correct = calculate_metrics(
    gold_all,
    bert_all
)

roberta_p, roberta_r, roberta_f1, roberta_correct = calculate_metrics(
    gold_all,
    roberta_all
)

spacy_p, spacy_r, spacy_f1, spacy_correct = calculate_metrics(
    gold_all,
    spacy_all
)

hybrid_p, hybrid_r, hybrid_f1, hybrid_correct = calculate_metrics(
    gold_all,
    hybrid_all
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL 3-MODEL HYBRID RESULTS")
print("=" * 70)


print("\nBERT")
print("-" * 70)

print(
    f"Precision : {bert_p:.4f}"
)

print(
    f"Recall    : {bert_r:.4f}"
)

print(
    f"F1-score  : {bert_f1:.4f}"
)


print("\nRoBERTa")
print("-" * 70)

print(
    f"Precision : {roberta_p:.4f}"
)

print(
    f"Recall    : {roberta_r:.4f}"
)

print(
    f"F1-score  : {roberta_f1:.4f}"
)


print("\nspaCy")
print("-" * 70)

print(
    f"Precision : {spacy_p:.4f}"
)

print(
    f"Recall    : {spacy_r:.4f}"
)

print(
    f"F1-score  : {spacy_f1:.4f}"
)


print("\nHYBRID — BERT + RoBERTa + spaCy")
print("-" * 70)

print(
    f"Precision : {hybrid_p:.4f}"
)

print(
    f"Recall    : {hybrid_r:.4f}"
)

print(
    f"F1-score  : {hybrid_f1:.4f}"
)

print(
    f"Gold entities      : {len(set(gold_all))}"
)

print(
    f"Predicted entities : {len(set(hybrid_all))}"
)

print(
    f"Correct entities   : {hybrid_correct}"
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
        "FINAL HYBRID — BERT + RoBERTa + spaCy\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write("BERT\n")
    f.write(
        f"Precision : {bert_p:.4f}\n"
    )
    f.write(
        f"Recall    : {bert_r:.4f}\n"
    )
    f.write(
        f"F1-score  : {bert_f1:.4f}\n\n"
    )

    f.write("RoBERTa\n")
    f.write(
        f"Precision : {roberta_p:.4f}\n"
    )
    f.write(
        f"Recall    : {roberta_r:.4f}\n"
    )
    f.write(
        f"F1-score  : {roberta_f1:.4f}\n\n"
    )

    f.write("spaCy\n")
    f.write(
        f"Precision : {spacy_p:.4f}\n"
    )
    f.write(
        f"Recall    : {spacy_r:.4f}\n"
    )
    f.write(
        f"F1-score  : {spacy_f1:.4f}\n\n"
    )

    f.write(
        "HYBRID — BERT + RoBERTa + spaCy\n"
    )

    f.write(
        f"Precision : {hybrid_p:.4f}\n"
    )

    f.write(
        f"Recall    : {hybrid_r:.4f}\n"
    )

    f.write(
        f"F1-score  : {hybrid_f1:.4f}\n"
    )

    f.write(
        f"Gold entities      : {len(set(gold_all))}\n"
    )

    f.write(
        f"Predicted entities : {len(set(hybrid_all))}\n"
    )

    f.write(
        f"Correct entities   : {hybrid_correct}\n"
    )


print("\nResults saved to:")
print(RESULT_PATH)

print("\n" + "=" * 70)
print("FINAL HYBRID EVALUATION COMPLETE")
print("=" * 70)