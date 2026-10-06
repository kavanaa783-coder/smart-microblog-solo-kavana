import json
import re
import time
import joblib

from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "ml" / "data" / "combined"

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "random_forest_smote"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train.json"
VAL_FILE = DATA_DIR / "validation.json"

VECTORIZER_FILE = (
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

MODEL_FILE = (
    MODEL_DIR / "random_forest_smote.joblib"
)

LABEL_FILE = (
    MODEL_DIR / "label_names.json"
)

RESULT_FILE = (
    BASE_DIR
    / "ml"
    / "results"
    / "rf_smote_validation_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Much smaller than previous experiment
MAX_FEATURES = 5000

# Keep training manageable
MAX_O_SAMPLES = 60000

# Maximum examples per PII BIO class
MAX_CLASS_SAMPLES = 3000

# Target number after SMOTE
SMOTE_TARGET = 3000

SMOTE_K = 2

N_ESTIMATORS = 75

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 2


# ============================================================
# TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    re.UNICODE
)


def tokenize_with_offsets(text):

    tokens = []

    for match in TOKEN_PATTERN.finditer(text):

        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return tokens


# ============================================================
# BIO LABEL CREATION
# ============================================================

def create_bio_labels(text, entities):

    tokens = tokenize_with_offsets(text)

    labels = ["O"] * len(tokens)

    for entity in entities:

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        matched = []

        for i, token in enumerate(tokens):

            if (
                token["start"] < end
                and token["end"] > start
            ):
                matched.append(i)

        if not matched:
            continue

        for position, token_index in enumerate(matched):

            if position == 0:
                labels[token_index] = (
                    "B-" + label
                )
            else:
                labels[token_index] = (
                    "I-" + label
                )

    return tokens, labels


# ============================================================
# CREATE TOKEN FEATURES
# ============================================================

def create_token_documents(data):

    token_texts = []
    token_labels = []

    for item in data:

        text = item["text"]
        entities = item["entities"]

        tokens, labels = create_bio_labels(
            text,
            entities
        )

        for i, token in enumerate(tokens):

            left = []

            for j in range(
                max(0, i - 2),
                i
            ):
                left.append(
                    tokens[j]["text"]
                )

            right = []

            for j in range(
                i + 1,
                min(len(tokens), i + 3)
            ):
                right.append(
                    tokens[j]["text"]
                )

            feature_text = (
                " ".join(left)
                + " [TOKEN] "
                + token["text"]
                + " [/TOKEN] "
                + " ".join(right)
            )

            token_texts.append(feature_text)
            token_labels.append(labels[i])

    return token_texts, token_labels


# ============================================================
# CONTROLLED SAMPLING
# ============================================================

def controlled_sampling(texts, labels):

    print("\n" + "=" * 70)
    print("CONTROLLED TRAINING SAMPLING")
    print("=" * 70)

    grouped = {}

    for text, label in zip(texts, labels):

        if label not in grouped:
            grouped[label] = []

        grouped[label].append(text)

    selected_texts = []
    selected_labels = []

    for label, examples in grouped.items():

        if label == "O":

            limit = MAX_O_SAMPLES

        else:

            limit = MAX_CLASS_SAMPLES

        # Deterministic sampling
        examples = examples[:limit]

        selected_texts.extend(examples)

        selected_labels.extend(
            [label] * len(examples)
        )

    counts = Counter(selected_labels)

    print("\nSamples before SMOTE:")

    for label, count in counts.most_common():

        print(
            f"{label:<30}{count}"
        )

    print(
        "\nTotal selected:",
        len(selected_labels)
    )

    return selected_texts, selected_labels


# ============================================================
# BUILD SMOTE TARGET
# ============================================================

def build_smote_strategy(labels):

    counts = Counter(labels)

    strategy = {}

    for label, count in counts.items():

        # Never oversample O
        if label == "O":
            continue

        # Only oversample classes below target
        if count < SMOTE_TARGET:

            # SMOTE cannot create more than target
            strategy[label] = SMOTE_TARGET

    return strategy


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("MODEL 2B — MEMORY-SAFE TF-IDF + SMOTE + RANDOM FOREST")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training dataset...")

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    train_data = json.load(f)

print(
    "Training documents:",
    len(train_data)
)


print("\nLoading validation dataset...")

with open(
    VAL_FILE,
    "r",
    encoding="utf-8"
) as f:

    val_data = json.load(f)

print(
    "Validation documents:",
    len(val_data)
)


# ============================================================
# CREATE TOKEN DATA
# ============================================================

print(
    "\nCreating token-level examples..."
)

start_time = time.time()

X_train_text, y_train = create_token_documents(
    train_data
)

X_val_text, y_val = create_token_documents(
    val_data
)

print(
    "Original training tokens:",
    len(X_train_text)
)

print(
    "Validation tokens:",
    len(X_val_text)
)

print(
    "Preparation time:",
    round(
        time.time() - start_time,
        2
    ),
    "seconds"
)


# ============================================================
# CONTROLLED SAMPLING
# ============================================================

(
    X_train_text,
    y_train
) = controlled_sampling(
    X_train_text,
    y_train
)


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("BUILDING TF-IDF FEATURES")
print("=" * 70)

vectorizer = TfidfVectorizer(

    analyzer="char_wb",

    ngram_range=(2, 5),

    max_features=MAX_FEATURES,

    lowercase=True,

    sublinear_tf=True,

    dtype="float32"
)

print("\nFitting TF-IDF...")

start_time = time.time()

X_train = vectorizer.fit_transform(
    X_train_text
)

print(
    "Training matrix:",
    X_train.shape
)

print(
    "TF-IDF training time:",
    round(
        time.time() - start_time,
        2
    ),
    "seconds"
)


print("\nTransforming validation data...")

X_val = vectorizer.transform(
    X_val_text
)

print(
    "Validation matrix:",
    X_val.shape
)


# ============================================================
# DELETE TEXT DATA FROM MEMORY
# ============================================================

del X_train_text
del X_val_text
del train_data
del val_data


# ============================================================
# SMOTE
# ============================================================

print("\n" + "=" * 70)
print("APPLYING CONTROLLED SMOTE")
print("=" * 70)

before_counts = Counter(y_train)

print("\nBefore SMOTE:")

for label, count in before_counts.most_common():

    print(
        f"{label:<30}{count}"
    )


strategy = build_smote_strategy(
    y_train
)

print("\nSMOTE strategy:")

for label, target in strategy.items():

    print(
        f"{label:<30}{target}"
    )


if strategy:

    print(
        "\nRunning SMOTE only on minority classes..."
    )

    start_time = time.time()

    smote = SMOTE(
        sampling_strategy=strategy,
        random_state=RANDOM_STATE,
        k_neighbors=SMOTE_K
    )

    X_train_resampled, y_train_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    smote_time = time.time() - start_time

else:

    print(
        "\nNo classes require SMOTE."
    )

    X_train_resampled = X_train
    y_train_resampled = y_train

    smote_time = 0


print(
    "\nSMOTE time:",
    round(smote_time, 2),
    "seconds"
)


after_counts = Counter(
    y_train_resampled
)

print("\nAfter SMOTE:")

for label, count in after_counts.most_common():

    print(
        f"{label:<30}{count}"
    )

print(
    "\nTotal training samples after SMOTE:",
    len(y_train_resampled)
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print(
    "\nTrees:",
    N_ESTIMATORS
)

print(
    "Maximum depth:",
    MAX_DEPTH
)

print(
    "Minimum samples per leaf:",
    MIN_SAMPLES_LEAF
)

print("\nTraining...")

start_time = time.time()

rf = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1
)

rf.fit(
    X_train_resampled,
    y_train_resampled
)

training_time = (
    time.time() - start_time
)

print(
    "\nRandom Forest training time:",
    round(training_time, 2),
    "seconds"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "\nGenerating predictions..."
)

start_time = time.time()

y_pred = rf.predict(
    X_val
)

inference_time = (
    time.time() - start_time
)

print(
    "Inference time:",
    round(inference_time, 2),
    "seconds"
)


# ============================================================
# METRICS
# ============================================================

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

report = classification_report(
    y_val,
    y_pred,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print("\n" + "=" * 70)
print("PER-LABEL RESULTS")
print("=" * 70)

print(report)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MODEL 2B — MEMORY-SAFE TF-IDF + SMOTE + RANDOM FOREST\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Max TF-IDF features: {MAX_FEATURES}\n"
    )

    f.write(
        f"Maximum O samples: {MAX_O_SAMPLES}\n"
    )

    f.write(
        f"Maximum samples per class: {MAX_CLASS_SAMPLES}\n"
    )

    f.write(
        f"SMOTE target: {SMOTE_TARGET}\n"
    )

    f.write(
        f"SMOTE k-neighbors: {SMOTE_K}\n\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1-score  : {f1:.4f}\n"
    )

    f.write(
        f"SMOTE time : {smote_time:.2f} seconds\n"
    )

    f.write(
        f"Training time : {training_time:.2f} seconds\n"
    )

    f.write(
        f"Inference time: {inference_time:.2f} seconds\n\n"
    )

    f.write(report)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)

joblib.dump(
    rf,
    MODEL_FILE
)

labels = sorted(
    set(y_train_resampled)
)

with open(
    LABEL_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        labels,
        f,
        indent=2
    )


print(
    "\nVectorizer saved to:"
)

print(
    VECTORIZER_FILE
)

print(
    "\nSMOTE Random Forest saved to:"
)

print(
    MODEL_FILE
)

print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2B COMPLETE")
print("=" * 70)

print(
    f"\nPrecision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"SMOTE time : {smote_time:.2f} seconds"
)

print(
    f"Training time : {training_time:.2f} seconds"
)

print(
    f"Inference time: {inference_time:.2f} seconds"
)

print("\nDone.")