import json
import time
import random
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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

VAL_FILE = (
    BASE_DIR
    / "ml"
    / "data"
    / "combined"
    / "validation.json"
)

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "roberta_pii"
)

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

RESULT_DIR = (
    BASE_DIR
    / "ml"
    / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128

# RTX 3050 4 GB
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 3e-5

NUM_EPOCHS = 5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

SEED = 42

SAVE_TOTAL_LIMIT = 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("MODEL 4 — RoBERTa PII NER")
print("=" * 70)

print("\nHardware check:")

print("PyTorch:", torch.__version__)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

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

    DEVICE = "cuda"

else:

    print(
        "WARNING: CUDA unavailable."
    )

    DEVICE = "cpu"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

print("\nTraining file:")
print(TRAIN_FILE)

print("\nValidation file:")
print(VAL_FILE)


with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    train_data = json.load(f)


with open(
    VAL_FILE,
    "r",
    encoding="utf-8"
) as f:

    val_data = json.load(f)


print(
    "\nTraining documents:",
    len(train_data)
)

print(
    "Validation documents:",
    len(val_data)
)


# ============================================================
# COLLECT ENTITY LABELS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LABEL MAP")
print("=" * 70)

entity_labels = set()

for item in train_data:

    for entity in item["entities"]:

        entity_labels.add(
            entity["label"]
        )


entity_labels = sorted(
    entity_labels
)


# ============================================================
# BIO LABELS
# ============================================================

label_names = ["O"]

for label in entity_labels:

    label_names.append(
        "B-" + label
    )

    label_names.append(
        "I-" + label
    )


label2id = {
    label: index
    for index, label in enumerate(
        label_names
    )
}

id2label = {
    index: label
    for label, index in label2id.items()
}


print(
    "\nEntity types:",
    len(entity_labels)
)

print(
    "BIO labels:",
    len(label_names)
)

print("\nEntities:")

for label in entity_labels:

    print(
        " -",
        label
    )


# ============================================================
# TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING RoBERTa TOKENIZER")
print("=" * 70)

print(
    "\nModel:",
    MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# CONVERT DATASET
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING DATASET")
print("=" * 70)


def convert_to_hf_format(data):

    texts = []

    entities_list = []

    for item in data:

        texts.append(
            item["text"]
        )

        entities_list.append(
            item["entities"]
        )

    return Dataset.from_dict(
        {
            "text": texts,
            "entities": entities_list
        }
    )


train_dataset = convert_to_hf_format(
    train_data
)

val_dataset = convert_to_hf_format(
    val_data
)


# ============================================================
# TOKENIZE + ALIGN BIO LABELS
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING AND ALIGNING LABELS")
print("=" * 70)


def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        padding=False
    )

    all_labels = []

    for batch_index in range(
        len(examples["text"])
    ):

        text = examples["text"][batch_index]

        entities = examples["entities"][batch_index]

        offsets = tokenized_inputs[
            "offset_mapping"
        ][batch_index]

        labels = [
            "O"
            for _ in offsets
        ]

        # ----------------------------------------------------
        # ASSIGN BIO LABELS
        # ----------------------------------------------------

        for entity in entities:

            entity_start = entity["start"]

            entity_end = entity["end"]

            entity_label = entity["label"]

            overlapping_tokens = []

            for token_index, (
                token_start,
                token_end
            ) in enumerate(offsets):

                # Special tokens
                if token_start == token_end:
                    continue

                # Token overlaps entity
                if (
                    token_start < entity_end
                    and token_end > entity_start
                ):

                    overlapping_tokens.append(
                        token_index
                    )

            if not overlapping_tokens:
                continue

            for position, token_index in enumerate(
                overlapping_tokens
            ):

                if position == 0:

                    labels[token_index] = (
                        "B-" + entity_label
                    )

                else:

                    labels[token_index] = (
                        "I-" + entity_label
                    )

        all_labels.append(
            [
                label2id[label]
                for label in labels
            ]
        )

    tokenized_inputs["labels"] = all_labels

    # Offset mappings are not passed to the model
    tokenized_inputs.pop(
        "offset_mapping"
    )

    return tokenized_inputs


print(
    "\nTokenizing training data..."
)

train_tokenized = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=[
        "text",
        "entities"
    ],
    desc="Tokenizing training dataset"
)


print(
    "\nTokenizing validation data..."
)

val_tokenized = val_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=[
        "text",
        "entities"
    ],
    desc="Tokenizing validation dataset"
)


print(
    "\nTraining tokenized examples:",
    len(train_tokenized)
)

print(
    "Validation tokenized examples:",
    len(val_tokenized)
)


# ============================================================
# LOAD RoBERTa MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING RoBERTa MODEL")
print("=" * 70)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []

    true_labels = []

    for prediction_row, label_row in zip(
        predictions,
        labels
    ):

        current_predictions = []

        current_labels = []

        for prediction, label in zip(
            prediction_row,
            label_row
        ):

            # Ignore padding / special tokens
            if label == -100:
                continue

            current_predictions.append(
                id2label[int(prediction)]
            )

            current_labels.append(
                id2label[int(label)]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    precision = precision_score(
        true_labels,
        true_predictions
    )

    recall = recall_score(
        true_labels,
        true_predictions
    )

    f1 = f1_score(
        true_labels,
        true_predictions
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING TIMER
# ============================================================

class TrainingTimerCallback(
    TrainerCallback
):

    def __init__(self):

        self.start_time = None

    def on_train_begin(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        self.start_time = time.time()

        print(
            "\nTraining timer started."
        )

    def on_train_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        if self.start_time:

            elapsed = (
                time.time()
                - self.start_time
            )

            print(
                "\nTotal training time:",
                round(
                    elapsed / 60,
                    2
                ),
                "minutes"
            )


# ============================================================
# TRAINING ARGUMENTS
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
    PER_DEVICE_TRAIN_BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Effective batch size:",
    PER_DEVICE_TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
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
    "Mixed precision:",
    DEVICE == "cuda"
)


training_args = TrainingArguments(

    output_dir=str(
        CHECKPOINT_DIR
    ),

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=(
        PER_DEVICE_TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        PER_DEVICE_EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    eval_accumulation_steps=1,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=SAVE_TOTAL_LIMIT,

    report_to="none",

    fp16=(
        DEVICE == "cuda"
    ),

    dataloader_num_workers=0,

    seed=SEED,

    optim="adamw_torch",

    gradient_checkpointing=True
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_tokenized,

    eval_dataset=val_tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        TrainingTimerCallback()
    ]
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING RoBERTa TRAINING")
print("=" * 70)

print(
    "\nGPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

print(
    "\nIMPORTANT:"
)

print(
    "Do not close this terminal while training."
)

print(
    "Checkpoints are saved after every epoch."
)


start_time = time.time()

train_result = trainer.train()

training_time = (
    time.time()
    - start_time
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

if torch.cuda.is_available():

    torch.cuda.empty_cache()

evaluation_start = time.time()

metrics = trainer.evaluate()

evaluation_time = (
    time.time()
    - evaluation_start
)


print(
    "\nValidation metrics:"
)

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key:<25}: {value:.6f}"
        )

    else:

        print(
            f"{key:<25}: {value}"
        )


# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING BEST RoBERTa MODEL")
print("=" * 70)

trainer.save_model(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)


# ============================================================
# SAVE LABEL MAP
# ============================================================

label_map_file = (
    MODEL_DIR
    / "label_mapping.json"
)

with open(
    label_map_file,
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
# SAVE RESULTS
# ============================================================

result_file = (
    RESULT_DIR
    / "roberta_validation_results.txt"
)

with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 70
        + "\n"
    )

    f.write(
        "MODEL 4 — RoBERTa PII NER\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Model: {MODEL_NAME}\n"
    )

    f.write(
        f"Training documents: {len(train_data)}\n"
    )

    f.write(
        f"Validation documents: {len(val_data)}\n"
    )

    f.write(
        f"Entity types: {len(entity_labels)}\n"
    )

    f.write(
        f"BIO labels: {len(label_names)}\n"
    )

    f.write(
        f"Max length: {MAX_LENGTH}\n"
    )

    f.write(
        f"Epochs: {NUM_EPOCHS}\n"
    )

    f.write(
        f"Training time seconds: {training_time:.2f}\n"
    )

    f.write(
        f"Evaluation time seconds: {evaluation_time:.2f}\n\n"
    )

    f.write(
        "OVERALL METRICS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    for key, value in metrics.items():

        f.write(
            f"{key}: {value}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL 4 — RoBERTa TRAINING COMPLETE")
print("=" * 70)

print(
    "\nBest model saved to:"
)

print(
    MODEL_DIR
)

print(
    "\nCheckpoints saved to:"
)

print(
    CHECKPOINT_DIR
)

print(
    "\nResults saved to:"
)

print(
    result_file
)

print(
    "\nTraining time:",
    round(
        training_time / 60,
        2
    ),
    "minutes"
)

print(
    "Evaluation time:",
    round(
        evaluation_time,
        2
    ),
    "seconds"
)

print(
    "\nFinal F1:",
    metrics.get(
        "eval_f1",
        "N/A"
    )
)

print("\n" + "=" * 70)