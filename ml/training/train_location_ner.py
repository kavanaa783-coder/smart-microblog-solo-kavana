# ============================================================
# train_location_ner.py
# Smart Microblog Privacy Guard
#
# Purpose:
#   Train a dedicated LOCATION NER model using GeoNames-derived
#   location data.
#
# Architecture:
#   Location examples -> spaCy NER -> LOCATION model
#   GeoNames remains the knowledge/gazetteer layer.
# ============================================================

import json
import random
import shutil
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "locations" / "processed"
MODEL_DIR = BASE_DIR / "models" / "location_ner"

INDIAN_FILE = DATA_DIR / "indian_locations.json"
WORLDWIDE_FILE = DATA_DIR / "worldwide_locations.json"

SEED = 42

# Number of examples generated for training
INDIAN_EXAMPLES = 30000
WORLDWIDE_EXAMPLES = 30000

EPOCHS = 15


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)


# ============================================================
# LOAD LOCATION DATA
# ============================================================

def load_json(path):

    print(f"\nLoading: {path}")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        f"Loaded {len(data):,} records"
    )

    return data


# ============================================================
# NORMALIZE RECORD STRUCTURE
# ============================================================

def extract_location_names(data):

    names = []

    if isinstance(data, dict):

        iterator = data.items()

        for name, value in iterator:

            if not isinstance(name, str):
                continue

            name = name.strip()

            if len(name) < 3:
                continue

            names.append(name)

    elif isinstance(data, list):

        for item in data:

            if isinstance(item, str):

                name = item.strip()

                if len(name) >= 3:
                    names.append(name)

            elif isinstance(item, dict):

                for key in (
                    "name",
                    "location",
                    "asciiname"
                ):

                    value = item.get(key)

                    if isinstance(value, str):

                        value = value.strip()

                        if len(value) >= 3:
                            names.append(value)

                        break

    return list(set(names))


# ============================================================
# FILTER BAD / AMBIGUOUS NAMES
# ============================================================

STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "this",
    "that",
    "live",
    "last",
    "trip",
    "company",
    "friend",
    "she",
    "he",
    "are",
    "was",
    "you",
    "our",
    "your",
    "their",
    "home",
    "work"
}


def valid_location_name(name):

    normalized = name.lower().strip()

    if normalized in STOPWORDS:
        return False

    if len(normalized) < 3:
        return False

    # Ignore names consisting only of numbers
    if normalized.isdigit():
        return False

    return True


# ============================================================
# CREATE SYNTHETIC NER EXAMPLES
# ============================================================

TEMPLATES = [

    "I live in {}.",

    "I am from {}.",

    "I currently stay in {}.",

    "I moved to {}.",

    "I moved from {}.",

    "I am travelling to {}.",

    "I visited {} last year.",

    "My friend lives in {}.",

    "She travelled to {}.",

    "He works in {}.",

    "I am studying in {}.",

    "We are planning a trip to {}.",

    "The event is happening in {}.",

    "The office is located in {}.",

    "They moved from {} to another city.",

    "I recently visited {}.",

    "I travelled through {}.",

    "My hometown is {}.",

    "I was born in {}.",

    "The meeting will be held in {}."
]


def create_examples(names, count):

    examples = []

    if not names:
        return examples

    for _ in range(count):

        location = random.choice(names)

        template = random.choice(TEMPLATES)

        text = template.format(location)

        start = text.find(location)

        if start == -1:
            continue

        end = start + len(location)

        examples.append({
            "text": text,
            "entities": [
                {
                    "start": start,
                    "end": end,
                    "label": "LOCATION"
                }
            ]
        })

    return examples


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMART MICROBLOG PRIVACY GUARD")
print("DEDICATED LOCATION NER TRAINING")
print("=" * 70)

indian_data = load_json(
    INDIAN_FILE
)

worldwide_data = load_json(
    WORLDWIDE_FILE
)


# ============================================================
# EXTRACT NAMES
# ============================================================

indian_names = extract_location_names(
    indian_data
)

worldwide_names = extract_location_names(
    worldwide_data
)


indian_names = [
    name
    for name in indian_names
    if valid_location_name(name)
]

worldwide_names = [
    name
    for name in worldwide_names
    if valid_location_name(name)
]


print(
    f"\nValid Indian location names: "
    f"{len(indian_names):,}"
)

print(
    f"Valid worldwide location names: "
    f"{len(worldwide_names):,}"
)


# ============================================================
# REDUCE HUGE GAZETTEER
# ============================================================

# We do NOT train on millions of records.
# A representative subset is sufficient for the NER model.

if len(indian_names) > INDIAN_EXAMPLES:

    indian_train_names = random.sample(
        indian_names,
        INDIAN_EXAMPLES
    )

else:

    indian_train_names = indian_names


if len(worldwide_names) > WORLDWIDE_EXAMPLES:

    worldwide_train_names = random.sample(
        worldwide_names,
        WORLDWIDE_EXAMPLES
    )

else:

    worldwide_train_names = worldwide_names


print(
    f"\nIndian names selected: "
    f"{len(indian_train_names):,}"
)

print(
    f"Worldwide names selected: "
    f"{len(worldwide_train_names):,}"
)


# ============================================================
# BUILD TRAINING EXAMPLES
# ============================================================

print("\nGenerating training examples...")

indian_examples = create_examples(
    indian_train_names,
    len(indian_train_names)
)

worldwide_examples = create_examples(
    worldwide_train_names,
    len(worldwide_train_names)
)


examples = (
    indian_examples +
    worldwide_examples
)

random.shuffle(examples)


print(
    f"Total generated examples: "
    f"{len(examples):,}"
)


# ============================================================
# SPLIT DATA
# ============================================================

random.shuffle(examples)

total = len(examples)

train_end = int(total * 0.8)

validation_end = int(total * 0.9)

train_data = examples[:train_end]

validation_data = examples[
    train_end:validation_end
]

test_data = examples[
    validation_end:
]


print("\nDataset split:")

print(
    f"Training:   {len(train_data):,}"
)

print(
    f"Validation: {len(validation_data):,}"
)

print(
    f"Testing:    {len(test_data):,}"
)


# ============================================================
# SAVE DATASET
# ============================================================

DATASET_DIR = BASE_DIR / "data" / "locations" / "ner"

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_json(data, filename):

    path = DATASET_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved: {path}"
    )


save_json(
    train_data,
    "train.json"
)

save_json(
    validation_data,
    "validation.json"
)

save_json(
    test_data,
    "test.json"
)


# ============================================================
# CREATE SPACY MODEL
# ============================================================

print("\nCreating spaCy NER pipeline...")

nlp = spacy.blank("en")


# ============================================================
# ADD NER COMPONENT
# ============================================================

ner = nlp.add_pipe(
    "ner"
)

ner.add_label(
    "LOCATION"
)


# ============================================================
# CREATE TRAINING EXAMPLES
# ============================================================

training_examples = []

for item in train_data:

    doc = nlp.make_doc(
        item["text"]
    )

    example = Example.from_dict(
        doc,
        {
            "entities": [
                (
                    entity["start"],
                    entity["end"],
                    entity["label"]
                )
                for entity in item["entities"]
            ]
        }
    )

    training_examples.append(
        example
    )


# ============================================================
# INITIALIZE
# ============================================================

print("\nInitializing model...")

optimizer = nlp.initialize(
    get_examples=lambda: training_examples
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

for epoch in range(EPOCHS):

    random.shuffle(
        training_examples
    )

    losses = {}

    batches = minibatch(
        training_examples,
        size=compounding(
            8.0,
            32.0,
            1.001
        )
    )

    for batch in batches:

        nlp.update(
            batch,
            drop=0.2,
            sgd=optimizer,
            losses=losses
        )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS}"
        f" | Loss: {losses.get('ner', 0):.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

if MODEL_DIR.exists():

    shutil.rmtree(
        MODEL_DIR
    )


nlp.to_disk(
    MODEL_DIR
)


print(
    f"\nModel saved to:\n{MODEL_DIR}"
)


# ============================================================
# QUICK TEST
# ============================================================

print("\n" + "=" * 70)
print("QUICK MODEL TEST")
print("=" * 70)


test_sentences = [

    "I live in Mangalore, Karnataka.",

    "I am studying in Chennai.",

    "My friend moved from Bangalore to Mumbai.",

    "She travelled from New York to London.",

    "We are planning a trip to Dubai and Singapore.",

    "I work in California.",

    "I visited Hyderabad last month."
]


for text in test_sentences:

    doc = nlp(text)

    locations = [
        ent.text
        for ent in doc.ents
        if ent.label_ == "LOCATION"
    ]

    print(
        f"\nTEXT: {text}"
    )

    print(
        f"LOCATION: {locations}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("LOCATION NER TRAINING COMPLETE")
print("=" * 70)

print(
    "\nModel:"
)

print(
    MODEL_DIR
)

print(
    "\nDataset:"
)

print(
    DATASET_DIR
)

print(
    "\nNext:"
)

print(
    "Evaluate the model on unseen test data."
)