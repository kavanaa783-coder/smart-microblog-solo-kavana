from pathlib import Path


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# MODEL PATHS
# ============================================================

ROBERTA_MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "roberta_pii"
)

BERT_MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "bert_pii"
)

SPACY_MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "spacy_pii_combined"
)


# ============================================================
# VALIDATION DATA
# ============================================================

# Same validation JSON used by your PII dataset
VALIDATION_DATA = (
    BASE_DIR
    / "ml"
    / "data"
    / "processed"
    / "validation.json"
)


# ============================================================
# OUTPUT
# ============================================================

RESULTS_DIR = (
    BASE_DIR
    / "ml"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HYBRID_RESULTS_FILE = (
    RESULTS_DIR
    / "hybrid_validation_results.txt"
)


# ============================================================
# HYBRID SETTINGS
# ============================================================

# V1:
# RoBERTa + spaCy
#
# BERT is kept available for V2.
USE_ROBERTA = True
USE_BERT = False
USE_SPACY = True


# ============================================================
# TRANSFORMER SETTINGS
# ============================================================

MAX_LENGTH = 128

# Overlap between transformer chunks.
# Useful when an entity is close to a chunk boundary.
STRIDE = 32


# ============================================================
# CONFIDENCE
# ============================================================

# Minimum confidence for a transformer entity.
#
# We start at 0.50 so that the evaluation is not
# unnecessarily aggressive.
TRANSFORMER_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# HYBRID DECISION
# ============================================================

# spaCy is used as a supplementary detector.
#
# A spaCy entity is added only when there is no
# conflicting RoBERTa/BERT entity covering the same text.
#
# This protects the stronger transformer model from
# being overwritten by the weaker model.
SPACY_SUPPLEMENT_ONLY = True


# ============================================================
# ENTITY LABELS
# ============================================================

EXPECTED_ENTITY_TYPES = {
    "AADHAAR",
    "ACCOUNT",
    "ADDRESS",
    "AGE",
    "API_KEY",
    "BANK_ACCOUNT",
    "CREDIT_CARD",
    "CRYPTO_ADDRESS",
    "DATE",
    "DEVICE_ID",
    "DOB",
    "DRIVER_LICENSE",
    "EMAIL",
    "IP_ADDRESS",
    "LOCATION",
    "MAC_ADDRESS",
    "PAN",
    "PASSPORT",
    "PASSWORD",
    "PERSON",
    "PHONE",
    "PIN",
    "SSN",
    "UPI_ID",
    "URL",
    "USERNAME",
    "USER_AGENT",
    "VEHICLE_REG",
}


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_config():
    print("=" * 70)
    print("HYBRID PII CONFIGURATION")
    print("=" * 70)

    print("\nModels:")
    print("RoBERTa :", USE_ROBERTA)
    print("BERT    :", USE_BERT)
    print("spaCy   :", USE_SPACY)

    print("\nPaths:")
    print("RoBERTa:", ROBERTA_MODEL_DIR)
    print("BERT   :", BERT_MODEL_DIR)
    print("spaCy  :", SPACY_MODEL_DIR)

    print("\nValidation:")
    print(VALIDATION_DATA)

    print("\nTransformer max length:", MAX_LENGTH)
    print("Transformer stride    :", STRIDE)
    print(
        "Confidence threshold  :",
        TRANSFORMER_CONFIDENCE_THRESHOLD
    )

    print("=" * 70)