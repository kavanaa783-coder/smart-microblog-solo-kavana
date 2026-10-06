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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "ml" / "data" / "combined"

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "random_forest"
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
    MODEL_DIR / "random_forest.joblib"
)

LABEL_FILE = (
    MODEL_DIR / "label_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Smaller than the previous version so training is practical
MAX_FEATURES = 15000

N_ESTIMATORS = 50

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 2

RANDOM_STATE = 42


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

        tokens.append(
            {
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            }
        )

    return tokens


# ============================================================
# CREATE BIO LABELS
# ============================================================

def create_bio_labels(text, entities):

    tokens = tokenize_with_offsets(text)

    labels = ["O"] * len(tokens)

    for entity in entities:

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        matched_indices = []

        for i, token in enumerate(tokens):

            if (
                token["start"] < end
                and token["end"] > start
            ):

                matched_indices.append(i)

        if not matched_indices:
            continue

        for position, token_index in enumerate(
            matched_indices
        ):

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

            # Previous two tokens
            left = [
                tokens[j]["text"]
                for j in range(
                    max(0, i - 2),
                    i
                )
            ]

            # Next two tokens
            right = [
                tokens[j]["text"]
                for j in range(
                    i + 1,
                    min(
                        len(tokens),
                        i + 3
                    )
                )
            ]

            # Token + local context
            feature_text = (
                " ".join(left)
                + " [TOKEN] "
                + token["text"]
                + " [/TOKEN] "
                + " ".join(right)
            )

            token_texts.append(
                feature_text
            )

            token_labels.append(
                labels[i]
            )

    return token_texts, token_labels


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL 2 — TF-IDF + RANDOM FOREST")
print("=" * 70)

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

print("\nCreating token-level examples...")

start_time = time.time()

X_train_text, y_train = (
    create_token_documents(train_data)
)

X_val_text, y_val = (
    create_token_documents(val_data)
)

print(
    "Training tokens:",
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
# LABEL DISTRIBUTION
# ============================================================

print("\nTraining label distribution:")

label_counts = Counter(y_train)

for label, count in label_counts.most_common():

    print(
        f"{label:<30}{count}"
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

    sublinear_tf=True
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
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print(
    "\nNumber of trees:",
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

    n_jobs=-1,

    class_weight="balanced_subsample"
)

rf.fit(
    X_train,
    y_train
)

training_time = (
    time.time() - start_time
)

print(
    "\nRandom Forest training time:",
    round(
        training_time,
        2
    ),
    "seconds"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print("\nGenerating predictions...")

start_time = time.time()

y_pred = rf.predict(
    X_val
)

inference_time = (
    time.time() - start_time
)

print(
    "Inference time:",
    round(
        inference_time,
        2
    ),
    "seconds"
)


# ============================================================
# OVERALL METRICS
# ============================================================

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

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


# ============================================================
# PER-LABEL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PER-LABEL RESULTS")
print("=" * 70)

report = classification_report(
    y_val,
    y_pred,
    zero_division=0
)

print(report)


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
    set(y_train)
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
    "\nTF-IDF vectorizer saved to:"
)

print(
    VECTORIZER_FILE
)

print(
    "\nRandom Forest model saved to:"
)

print(
    MODEL_FILE
)

print(
    "\nLabels saved to:"
)

print(
    LABEL_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 TRAINING COMPLETE")
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
    f"Training time : {training_time:.2f} seconds"
)

print(
    f"Inference time: {inference_time:.2f} seconds"
)