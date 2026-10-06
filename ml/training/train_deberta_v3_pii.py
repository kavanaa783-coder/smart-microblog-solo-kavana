# ============================================================
# MODEL 5 — DeBERTa-v3 PII NER
# ============================================================
#
# Fine-tuning microsoft/deberta-v3-base for token classification
# using the combined AI4Privacy + Maskara Indian PII dataset.
#
# Compatible with Transformers 5.x
# Designed for NVIDIA RTX 3050 4GB
#
# ============================================================

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-base"

MAX_LENGTH = 128

NUM_EPOCHS = 5

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.01

SEED = 42

# IMPORTANT:
# Disable FP16 because your previous run produced:
# "Attempting to unscale FP16 gradients."
FP16 = False

# BF16 is not recommended for this RTX 3050 setup.
BF16 = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    BASE_DIR
    / "ml"
    / "data"
    / "combined"
    / "train.json"
)

VALIDATION_FILE = (
    BASE_DIR
    / "ml"
    / "data"
    / "combined"
    / "validation.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "deberta_v3_pii"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HARDWARE CHECK
# ============================================================

print("=" * 70)
print("MODEL 5 — DeBERTa-v3 PII NER")
print("=" * 70)

print("\nHardware check:")

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

else:

    device = torch.device("cpu")

    print("WARNING: CUDA is NOT available.")


# ============================================================
# LOAD JSON DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

print("\nTraining file:")
print(TRAIN_FILE)

print("\nValidation file:")
print(VALIDATION_FILE)


with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    train_data = json.load(f)


with open(
    VALIDATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    validation_data = json.load(f)


print(
    "\nTraining documents:",
    len(train_data)
)

print(
    "Validation documents:",
    len(validation_data)
)


# ============================================================
# BUILD ENTITY LABEL MAP
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LABEL MAP")
print("=" * 70)


entity_types = sorted(
    {
        entity["label"]
        for item in train_data
        for entity in item["entities"]
    }
)


print(
    "\nEntity types:",
    len(entity_types)
)


# ------------------------------------------------------------
# BIO LABELS
# ------------------------------------------------------------

labels = ["O"]

for entity_type in entity_types:

    labels.append(
        f"B-{entity_type}"
    )

    labels.append(
        f"I-{entity_type}"
    )


label2id = {
    label: index
    for index, label in enumerate(labels)
}


id2label = {
    index: label
    for label, index in label2id.items()
}


print(
    "BIO labels:",
    len(labels)
)


print("\nEntities:")

for entity_type in entity_types:

    print(
        " -",
        entity_type
    )


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

with open(
    OUTPUT_DIR / "label_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label2id": label2id,
            "id2label": {
                str(k): v
                for k, v in id2label.items()
            }
        },
        f,
        indent=2
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING DeBERTa-v3 TOKENIZER")
print("=" * 70)

print(
    "\nModel:",
    MODEL_NAME
)

print(
    "\nUsing slow tokenizer intentionally."
)

print(
    "This avoids the SentencePiece/TikToken conversion problem."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


print(
    "\nTokenizer loaded successfully."
)


# ============================================================
# CONVERT DATASET
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING DATASET")
print("=" * 70)


train_dataset = Dataset.from_list(
    train_data
)

validation_dataset = Dataset.from_list(
    validation_data
)


# ============================================================
# TOKENIZE + ALIGN BIO LABELS
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING AND ALIGNING LABELS")
print("=" * 70)


def tokenize_and_align_labels(examples):

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_offsets_mapping=True
    )

    all_labels = []

    for i, offsets in enumerate(
        tokenized["offset_mapping"]
    ):

        entities = examples["entities"][i]

        labels_for_tokens = []

        for token_start, token_end in offsets:

            # Special token
            if token_start == token_end:

                labels_for_tokens.append(
                    -100
                )

                continue

            token_label = "O"

            # Find entity covering this token
            for entity in entities:

                entity_start = entity["start"]
                entity_end = entity["end"]
                entity_type = entity["label"]

                # No overlap
                if (
                    token_end <= entity_start
                    or token_start >= entity_end
                ):

                    continue

                # Token overlaps entity
                if token_start <= entity_start:

                    token_label = (
                        f"B-{entity_type}"
                    )

                else:

                    token_label = (
                        f"I-{entity_type}"
                    )

                break

            labels_for_tokens.append(
                label2id[token_label]
            )

        all_labels.append(
            labels_for_tokens
        )

    tokenized["labels"] = all_labels

    # offset_mapping is only required during alignment.
    # Remove it before training.
    tokenized.pop(
        "offset_mapping"
    )

    return tokenized


# ============================================================
# TOKENIZE TRAINING DATA
# ============================================================

print("\nTokenizing training data...")

tokenized_train = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    desc="Tokenizing training dataset"
)


# ============================================================
# TOKENIZE VALIDATION DATA
# ============================================================

print("\nTokenizing validation data...")

tokenized_validation = validation_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    desc="Tokenizing validation dataset"
)


print(
    "\nTraining tokenized examples:",
    len(tokenized_train)
)

print(
    "Validation tokenized examples:",
    len(tokenized_validation)
)


# ============================================================
# LOAD DeBERTa MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING DeBERTa-v3 MODEL")
print("=" * 70)


model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)


print(
    "\nModel loaded successfully."
)


# ============================================================
# MOVE MODEL TO GPU
# ============================================================

if torch.cuda.is_available():

    print(
        "\nCUDA detected."
    )

    print(
        "Trainer will use NVIDIA GPU."
    )

else:

    print(
        "\nWARNING: Training on CPU."
    )


# ============================================================
# DATA COLLATOR
# ============================================================

print("\n" + "=" * 70)
print("BUILDING DATA COLLATOR")
print("=" * 70)


# IMPORTANT:
# Transformers 5.x expects tokenizer here.
# NOT processing_class.
#
# Correct:
# DataCollatorForTokenClassification(tokenizer=tokenizer)
#
# Incorrect:
# DataCollatorForTokenClassification(processing_class=tokenizer)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt"
)


print(
    "\nData collator ready."
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels_batch = eval_prediction

    # predictions shape:
    # [batch, sequence_length, num_labels]

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels_batch
    ):

        current_predictions = []
        current_labels = []

        for pred_id, label_id in zip(
            prediction,
            label
        ):

            # Ignore special/padded tokens
            if label_id == -100:
                continue

            current_predictions.append(
                int(pred_id)
            )

            current_labels.append(
                int(label_id)
            )

        true_predictions.extend(
            current_predictions
        )

        true_labels.extend(
            current_labels
        )

    if not true_labels:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # Exclude O for PII-focused metrics
    non_o_indices = [
        i
        for i, label_id in enumerate(true_labels)
        if label_id != label2id["O"]
    ]

    if not non_o_indices:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    filtered_predictions = [
        true_predictions[i]
        for i in non_o_indices
    ]

    filtered_labels = [
        true_labels[i]
        for i in non_o_indices
    ]

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            filtered_labels,
            filtered_predictions,
            average="micro",
            zero_division=0
        )
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    "\nEpochs:",
    NUM_EPOCHS
)

print(
    "Train batch size:",
    TRAIN_BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Effective batch size:",
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

print(
    "Evaluation batch size:",
    EVAL_BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Maximum sequence length:",
    MAX_LENGTH
)

print(
    "FP16:",
    FP16
)

print(
    "BF16:",
    BF16
)

print(
    "CUDA:",
    torch.cuda.is_available()
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(
        CHECKPOINT_DIR
    ),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    # Transformers 5.x
    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=FP16,

    bf16=BF16,

    report_to="none",

    seed=SEED,

    data_seed=SEED,

    dataloader_num_workers=0,

    remove_unused_columns=False,

    optim="adamw_torch"
)


# ============================================================
# CREATE TRAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAINER")
print("=" * 70)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    data_collator=data_collator,

    # processing_class belongs HERE,
    # not inside DataCollatorForTokenClassification.
    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


print(
    "\nTrainer created successfully."
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING DeBERTa-v3 TRAINING")
print("=" * 70)

print(
    "\nThis is the point where actual GPU training starts."
)

print(
    "Checkpoint will be saved after every epoch."
)


start_time = time.time()


train_result = trainer.train()


training_time = (
    time.time()
    - start_time
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


evaluation_start = time.time()


metrics = trainer.evaluate()


evaluation_time = (
    time.time()
    - evaluation_start
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)


trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": MODEL_NAME,

    "training_documents": len(train_data),

    "validation_documents": len(validation_data),

    "entity_types": len(entity_types),

    "bio_labels": len(labels),

    "max_length": MAX_LENGTH,

    "epochs": NUM_EPOCHS,

    "train_batch_size": TRAIN_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS,

    "learning_rate": LEARNING_RATE,

    "fp16": FP16,

    "bf16": BF16,

    "training_time_seconds":
        training_time,

    "evaluation_time_seconds":
        evaluation_time,

    "evaluation_metrics": metrics
}


with open(
    OUTPUT_DIR / "validation_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL 5 — DeBERTa-v3 RESULTS")
print("=" * 70)

print(
    "\nTraining time:",
    round(training_time, 2),
    "seconds"
)

print(
    "Evaluation time:",
    round(evaluation_time, 2),
    "seconds"
)

print(
    "\nPrecision:",
    metrics.get(
        "eval_precision",
        "N/A"
    )
)

print(
    "Recall:",
    metrics.get(
        "eval_recall",
        "N/A"
    )
)

print(
    "F1:",
    metrics.get(
        "eval_f1",
        "N/A"
    )
)

print(
    "\nFinal model saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCheckpoint directory:"
)

print(
    CHECKPOINT_DIR
)

print("\n" + "=" * 70)
print("MODEL 5 TRAINING COMPLETE")
print("=" * 70)