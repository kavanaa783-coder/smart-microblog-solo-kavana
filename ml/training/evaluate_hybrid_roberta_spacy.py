import json
import torch
import spacy
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ROBERTA_DIR = BASE_DIR / "ml" / "models" / "roberta_pii"
SPACY_DIR = BASE_DIR / "ml" / "models" / "spacy_pii_combined"
TEST_PATH = BASE_DIR / "ml" / "data" / "combined" / "test.json"

RESULT_PATH = (
    BASE_DIR
    / "ml"
    / "results"
    / "hybrid_roberta_spacy_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("HYBRID MODEL 3 — RoBERTa + spaCy")
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
# LOAD ROBERTA
# ============================================================

print("\nLoading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)

roberta = AutoModelForTokenClassification.from_pretrained(
    ROBERTA_DIR
)

roberta.to(DEVICE)
roberta.eval()

print("RoBERTa loaded successfully.")


# ============================================================
# LOAD SPACY
# ============================================================

print("\nLoading spaCy model...")

nlp = spacy.load(SPACY_DIR)

print("spaCy loaded successfully.")


# ============================================================
# LABEL MAPPING
# ============================================================

id2label = roberta.config.id2label

# Make sure keys are integers
id2label = {
    int(k): v
    for k, v in id2label.items()
}


# ============================================================
# HELPER — CONVERT ROBERTA TOKEN PREDICTIONS TO ENTITIES
# ============================================================

def get_roberta_entities(text):
    """
    Returns entities as:
        (start_char, end_char, label)
    """

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    encoded = {
        k: v.to(DEVICE)
        for k, v in encoded.items()
    }

    with torch.no_grad():
        outputs = roberta(**encoded)

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

        # Special tokens
        if start == end:
            continue

        label = id2label.get(
            pred_id,
            "O"
        )

        # O
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

        # B-label
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

        # I-label
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

    return entities


# ============================================================
# HELPER — SPACY ENTITIES
# ============================================================

def get_spacy_entities(text):

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
# HYBRID DECISION
# ============================================================

def hybrid_entities(text):

    roberta_entities = get_roberta_entities(text)

    spacy_entities = get_spacy_entities(text)

    roberta_set = set(roberta_entities)
    spacy_set = set(spacy_entities)

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    agreed = roberta_set.intersection(
        spacy_set
    )

    # --------------------------------------------------------
    # Include RoBERTa predictions
    # --------------------------------------------------------

    hybrid = set()

    for entity in roberta_set:

        start, end, label = entity

        # If spaCy agrees exactly, keep it
        if entity in agreed:
            hybrid.add(entity)

        else:
            # Keep RoBERTa prediction
            hybrid.add(entity)

    # --------------------------------------------------------
    # Add spaCy entities only when they don't conflict
    # --------------------------------------------------------

    for entity in spacy_set:

        start, end, label = entity

        conflict = False

        for existing in hybrid:

            e_start, e_end, e_label = existing

            # overlapping spans
            if (
                start < e_end
                and end > e_start
            ):
                conflict = True
                break

        if not conflict:
            hybrid.add(entity)

    return hybrid


# ============================================================
# EVALUATION
# ============================================================

gold_all = []
roberta_all = []
spacy_all = []
hybrid_all = []

print("\nRunning hybrid evaluation...")

for index, item in enumerate(data, start=1):

    text = item.get("text", "")

    # --------------------------------------------------------
    # GOLD ENTITIES
    # --------------------------------------------------------

    gold_entities = set()

    # Expected format:
    # entities: [[start, end, label], ...]

    for entity in item.get("entities", []):

        if isinstance(entity, dict):

            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label")

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
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    roberta_entities = set(
        get_roberta_entities(text)
    )

    spacy_entities = set(
        get_spacy_entities(text)
    )

    hybrid = hybrid_entities(text)

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    gold_all.extend(gold_entities)
    roberta_all.extend(roberta_entities)
    spacy_all.extend(spacy_entities)
    hybrid_all.extend(hybrid)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 500 == 0:
        print(
            f"Processed {index}/{len(data)}"
        )


# ============================================================
# METRIC FUNCTION
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
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1


# ============================================================
# RESULTS
# ============================================================

roberta_p, roberta_r, roberta_f1 = calculate_metrics(
    gold_all,
    roberta_all
)

spacy_p, spacy_r, spacy_f1 = calculate_metrics(
    gold_all,
    spacy_all
)

hybrid_p, hybrid_r, hybrid_f1 = calculate_metrics(
    gold_all,
    hybrid_all
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HYBRID RoBERTa + spaCy RESULTS")
print("=" * 70)


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


print("\nHYBRID — RoBERTa + spaCy")
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
    f"Correct entities   : "
    f"{len(set(gold_all).intersection(set(hybrid_all)))}"
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
        "HYBRID RoBERTa + spaCy RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")

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

    f.write("HYBRID — RoBERTa + spaCy\n")
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
        f"Correct entities   : "
        f"{len(set(gold_all).intersection(set(hybrid_all)))}\n"
    )


print("\nResults saved to:")
print(RESULT_PATH)

print("\n" + "=" * 70)
print("HYBRID EVALUATION COMPLETE")
print("=" * 70)