import spacy
from pathlib import Path
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
import random


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_DATA = BASE_DIR / "data" / "spacy" / "train.spacy"
VALIDATION_DATA = BASE_DIR / "data" / "spacy" / "validation.spacy"

MODEL_DIR = BASE_DIR / "models" / "spacy_pii"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

N_ITER = 15
DROPOUT = 0.3
SEED = 42

random.seed(SEED)


# ============================================================
# CREATE SPACY PIPELINE
# ============================================================

print("=" * 70)
print("CUSTOM SPACY PII NER TRAINING")
print("=" * 70)

nlp = spacy.blank("en")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_docbin = DocBin().from_disk(TRAIN_DATA)

train_docs = list(
    train_docbin.get_docs(nlp.vocab)
)

print("Training documents:", len(train_docs))


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

val_docbin = DocBin().from_disk(VALIDATION_DATA)

val_docs = list(
    val_docbin.get_docs(nlp.vocab)
)

print("Validation documents:", len(val_docs))


# ============================================================
# CREATE NER PIPELINE
# ============================================================

print("\nCreating NER pipeline...")

ner = nlp.add_pipe("ner")


# ============================================================
# ADD ENTITY LABELS
# ============================================================

labels = set()

for doc in train_docs:

    for ent in doc.ents:

        labels.add(ent.label_)


print("\nEntity labels:")

for label in sorted(labels):

    print(" -", label)

    ner.add_label(label)


print("\nTotal labels:", len(labels))


# ============================================================
# CREATE TRAINING EXAMPLES
# ============================================================

print("\nPreparing training examples...")

train_examples = []

for doc in train_docs:

    entities = [
        (ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
    ]

    example = Example.from_dict(
        doc,
        {
            "entities": entities
        }
    )

    train_examples.append(example)


print("Training examples:", len(train_examples))


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\nInitializing model...")

nlp.initialize(
    get_examples=lambda: train_examples[:100]
)

print("Initialization complete.")


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for iteration in range(N_ITER):

    random.shuffle(train_examples)

    losses = {}

    batches = minibatch(
        train_examples,
        size=compounding(
            4.0,
            32.0,
            1.5
        )
    )

    for batch in batches:

        nlp.update(
            batch,
            drop=DROPOUT,
            losses=losses
        )

    print(
        f"Epoch {iteration + 1}/{N_ITER} "
        f"Loss: {losses.get('ner', 0):.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

nlp.to_disk(MODEL_DIR)

print("Model saved to:")
print(MODEL_DIR)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)