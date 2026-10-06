import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from pathlib import Path
import random
import re

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "spacy_combined"
    / "train.spacy"
)

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "spacy_pii_combined"
)

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

TOTAL_EPOCHS = 15
DROPOUT = 0.25
SEED = 42

random.seed(SEED)

# ============================================================
# FIND LATEST CHECKPOINT
# ============================================================

def get_latest_checkpoint():

    checkpoints = list(
        CHECKPOINT_DIR.glob("epoch_*")
    )

    if not checkpoints:
        return None

    valid = []

    for path in checkpoints:

        match = re.match(
            r"epoch_(\d+)$",
            path.name
        )

        if match:

            epoch_number = int(
                match.group(1)
            )

            valid.append(
                (epoch_number, path)
            )

    if not valid:
        return None

    valid.sort(
        key=lambda x: x[0]
    )

    return valid[-1]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("IMPROVED SPACY PII NER TRAINING")
print("=" * 70)

print("\nLoading training data...")

base_nlp = spacy.blank("en")

train_docbin = DocBin().from_disk(
    TRAIN_PATH
)

train_docs = list(
    train_docbin.get_docs(
        base_nlp.vocab
    )
)

print(
    "Training documents:",
    len(train_docs)
)

# ============================================================
# CHECK FOR PREVIOUS CHECKPOINT
# ============================================================

latest_checkpoint = get_latest_checkpoint()

if latest_checkpoint:

    # IMPORTANT:
    # get_latest_checkpoint() returns:
    # (epoch_number, checkpoint_path)

    checkpoint_epoch, checkpoint_path = (
        latest_checkpoint
    )

    print("\nPrevious checkpoint found:")
    print(checkpoint_path)

    print(
        f"Resuming from epoch "
        f"{checkpoint_epoch}"
    )

    # Load checkpoint model
    nlp = spacy.load(
        checkpoint_path
    )

    # Resume training
    optimizer = nlp.resume_training()

    start_epoch = checkpoint_epoch

else:

    print("\nNo checkpoint found.")
    print("Initializing a new model...")

    nlp = spacy.blank("en")

    # ========================================================
    # CREATE NER PIPELINE
    # ========================================================

    ner = nlp.add_pipe(
        "ner",
        last=True
    )

    # ========================================================
    # ADD ENTITY LABELS
    # ========================================================

    labels = set()

    for doc in train_docs:

        for ent in doc.ents:

            labels.add(
                ent.label_
            )

    print(
        "\nEntity labels:",
        len(labels)
    )

    for label in sorted(labels):

        ner.add_label(
            label
        )

    print("\nLabels:")

    for label in sorted(labels):

        print(
            " -",
            label
        )

    # ========================================================
    # INITIALIZE MODEL
    # ========================================================

    print(
        "\nInitializing model..."
    )

    optimizer = nlp.initialize(
        get_examples=lambda: [
            spacy.training.Example.from_dict(
                doc,
                {
                    "entities": [
                        [
                            ent.start_char,
                            ent.end_char,
                            ent.label_
                        ]
                        for ent in doc.ents
                    ]
                }
            )
            for doc in train_docs[:100]
        ]
    )

    start_epoch = 0


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    f"Training epochs: {TOTAL_EPOCHS}"
)

print(
    f"Starting from epoch: {start_epoch}"
)

# ============================================================
# TRAIN EACH EPOCH
# ============================================================

for epoch in range(
    start_epoch,
    TOTAL_EPOCHS
):

    current_epoch = epoch + 1

    print(
        f"\nStarting Epoch "
        f"{current_epoch}/{TOTAL_EPOCHS}..."
    )

    # Shuffle training documents
    random.shuffle(
        train_docs
    )

    losses = {}

    # ========================================================
    # CREATE BATCHES
    # ========================================================

    batches = minibatch(
        train_docs,
        size=compounding(
            4.0,
            32.0,
            1.001
        )
    )

    # ========================================================
    # TRAIN BATCHES
    # ========================================================

    for batch in batches:

        examples = []

        for doc in batch:

            entities = [
                [
                    ent.start_char,
                    ent.end_char,
                    ent.label_
                ]
                for ent in doc.ents
            ]

            example = (
                spacy.training.Example.from_dict(
                    doc,
                    {
                        "entities": entities
                    }
                )
            )

            examples.append(
                example
            )

        nlp.update(
            examples,
            drop=DROPOUT,
            losses=losses
        )

    # ========================================================
    # PRINT LOSS
    # ========================================================

    loss = losses.get(
        "ner",
        0
    )

    print(
        f"Epoch {current_epoch}/{TOTAL_EPOCHS} "
        f"Loss: {loss:.4f}"
    )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"epoch_{current_epoch}"
    )

    print(
        "\nSaving checkpoint..."
    )

    nlp.to_disk(
        checkpoint_path
    )

    print(
        "Checkpoint saved:"
    )

    print(
        checkpoint_path
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL IMPROVED MODEL")
print("=" * 70)

nlp.to_disk(
    MODEL_DIR
)

print("\nModel saved to:")

print(
    MODEL_DIR
)

print("\n" + "=" * 70)
print("IMPROVED MODEL 1 TRAINING COMPLETE")
print("=" * 70)