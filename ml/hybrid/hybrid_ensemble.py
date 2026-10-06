import json
from pathlib import Path

import spacy
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from .config import (
    ROBERTA_MODEL_DIR,
    BERT_MODEL_DIR,
    SPACY_MODEL_DIR,
    USE_ROBERTA,
    USE_BERT,
    USE_SPACY,
    MAX_LENGTH,
    STRIDE,
    TRANSFORMER_CONFIDENCE_THRESHOLD,
    SPACY_SUPPLEMENT_ONLY,
)


# ============================================================
# HYBRID PII ENSEMBLE
# ============================================================

class HybridPIIEnsemble:

    def __init__(self):

        print("\n")
        print("=" * 70)
        print("INITIALIZING HYBRID PII ENSEMBLE")
        print("=" * 70)

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        print("\nDevice:", self.device)

        self.roberta_tokenizer = None
        self.roberta_model = None

        self.bert_tokenizer = None
        self.bert_model = None

        self.spacy_model = None

        # --------------------------------------------------------
        # Load RoBERTa
        # --------------------------------------------------------

        if USE_ROBERTA:

            print("\nLoading RoBERTa model...")

            self.roberta_tokenizer = (
                AutoTokenizer.from_pretrained(
                    ROBERTA_MODEL_DIR,
                    local_files_only=True
                )
            )

            self.roberta_model = (
                AutoModelForTokenClassification.from_pretrained(
                    ROBERTA_MODEL_DIR,
                    local_files_only=True
                )
            )

            self.roberta_model.to(self.device)
            self.roberta_model.eval()

            print("RoBERTa loaded successfully.")
            print(
                "RoBERTa device:",
                self.device
            )

            print(
                "RoBERTa labels:",
                self.roberta_model.config.num_labels
            )

        # --------------------------------------------------------
        # Load BERT
        # --------------------------------------------------------

        if USE_BERT:

            print("\nLoading BERT model...")

            self.bert_tokenizer = (
                AutoTokenizer.from_pretrained(
                    BERT_MODEL_DIR,
                    local_files_only=True
                )
            )

            self.bert_model = (
                AutoModelForTokenClassification.from_pretrained(
                    BERT_MODEL_DIR,
                    local_files_only=True
                )
            )

            self.bert_model.to(self.device)
            self.bert_model.eval()

            print("BERT loaded successfully.")

            print(
                "BERT labels:",
                self.bert_model.config.num_labels
            )

        # --------------------------------------------------------
        # Load spaCy
        # --------------------------------------------------------

        if USE_SPACY:

            print("\nLoading spaCy Combined model...")

            self.spacy_model = spacy.load(
                SPACY_MODEL_DIR
            )

            print(
                "spaCy Combined loaded successfully."
            )

        print("\nHybrid initialization complete.")
        print("=" * 70)

    # ============================================================
    # LABEL NORMALIZATION
    # ============================================================

    @staticmethod
    def normalize_label(label):

        if not label:
            return None

        if label.startswith("B-"):
            return label[2:]

        if label.startswith("I-"):
            return label[2:]

        return label

    # ============================================================
    # VALID ENTITY LABEL
    # ============================================================

    @staticmethod
    def valid_entity(label):

        return (
            label is not None
            and label not in {
                "O",
                "",
                None,
            }
        )

    # ============================================================
    # CHECK ENTITY OVERLAP
    # ============================================================

    @staticmethod
    def overlaps(
        start1,
        end1,
        start2,
        end2
    ):

        return (
            start1 < end2
            and start2 < end1
        )

    # ============================================================
    # TRANSFORMER PREDICTION
    # ============================================================

    def predict_transformer(
        self,
        text,
        tokenizer,
        model
    ):

        if not text.strip():
            return []

        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding=False,
        )

        input_ids = encoded["input_ids"].to(
            self.device
        )

        attention_mask = encoded["attention_mask"].to(
            self.device
        )

        offset_mapping = encoded[
            "offset_mapping"
        ]

        predictions = []

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            predicted_ids = torch.argmax(
                probabilities,
                dim=-1
            )

        id2label = model.config.id2label

        # --------------------------------------------------------
        # Process each chunk
        # --------------------------------------------------------

        for batch_index in range(
            input_ids.shape[0]
        ):

            current_entities = []

            current_label = None
            current_start = None
            current_end = None
            current_scores = []

            sequence_length = (
                predicted_ids.shape[1]
            )

            for token_index in range(
                sequence_length
            ):

                token_id = int(
                    predicted_ids[
                        batch_index,
                        token_index
                    ].item()
                )

                label = id2label.get(
                    token_id,
                    "O"
                )

                score = float(
                    probabilities[
                        batch_index,
                        token_index,
                        token_id
                    ].item()
                )

                start, end = offset_mapping[
                    batch_index,
                    token_index
                ].tolist()

                # Special token
                if start == end:
                    continue

                normalized = (
                    self.normalize_label(label)
                )

                if not self.valid_entity(
                    normalized
                ):

                    # Close previous entity
                    if current_label is not None:

                        average_score = (
                            sum(current_scores)
                            / len(current_scores)
                        )

                        if (
                            average_score
                            >= TRANSFORMER_CONFIDENCE_THRESHOLD
                        ):

                            current_entities.append(
                                {
                                    "start": current_start,
                                    "end": current_end,
                                    "label": current_label,
                                    "score": average_score,
                                    "source": "transformer",
                                }
                            )

                        current_label = None
                        current_start = None
                        current_end = None
                        current_scores = []

                    continue

                # ------------------------------------------------
                # B-label
                # ------------------------------------------------

                if label.startswith("B-"):

                    # Save previous entity
                    if current_label is not None:

                        average_score = (
                            sum(current_scores)
                            / len(current_scores)
                        )

                        if (
                            average_score
                            >= TRANSFORMER_CONFIDENCE_THRESHOLD
                        ):

                            current_entities.append(
                                {
                                    "start": current_start,
                                    "end": current_end,
                                    "label": current_label,
                                    "score": average_score,
                                    "source": "transformer",
                                }
                            )

                    current_label = normalized
                    current_start = start
                    current_end = end
                    current_scores = [score]

                # ------------------------------------------------
                # I-label
                # ------------------------------------------------

                elif label.startswith("I-"):

                    if (
                        current_label == normalized
                        and current_start is not None
                    ):

                        current_end = end
                        current_scores.append(score)

                    else:

                        # Broken BIO sequence.
                        # Start a new entity rather than
                        # attaching it incorrectly.

                        if current_label is not None:

                            average_score = (
                                sum(current_scores)
                                / len(current_scores)
                            )

                            if (
                                average_score
                                >= TRANSFORMER_CONFIDENCE_THRESHOLD
                            ):

                                current_entities.append(
                                    {
                                        "start": current_start,
                                        "end": current_end,
                                        "label": current_label,
                                        "score": average_score,
                                        "source": "transformer",
                                    }
                                )

                        current_label = normalized
                        current_start = start
                        current_end = end
                        current_scores = [score]

            # ----------------------------------------------------
            # Save final entity
            # ----------------------------------------------------

            if current_label is not None:

                average_score = (
                    sum(current_scores)
                    / len(current_scores)
                )

                if (
                    average_score
                    >= TRANSFORMER_CONFIDENCE_THRESHOLD
                ):

                    current_entities.append(
                        {
                            "start": current_start,
                            "end": current_end,
                            "label": current_label,
                            "score": average_score,
                            "source": "transformer",
                        }
                    )

            predictions.extend(
                current_entities
            )

        # --------------------------------------------------------
        # Remove duplicate entities caused by chunk overlap
        # --------------------------------------------------------

        unique = {}

        for entity in predictions:

            key = (
                entity["start"],
                entity["end"],
                entity["label"]
            )

            if key not in unique:

                unique[key] = entity

            else:

                # Keep higher-confidence duplicate
                if (
                    entity["score"]
                    > unique[key]["score"]
                ):

                    unique[key] = entity

        return list(
            unique.values()
        )

    # ============================================================
    # SPACY PREDICTION
    # ============================================================

    def predict_spacy(self, text):

        if self.spacy_model is None:
            return []

        doc = self.spacy_model(
            text
        )

        entities = []

        for ent in doc.ents:

            entities.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                    "score": 1.0,
                    "source": "spacy",
                }
            )

        return entities

    # ============================================================
    # REMOVE OVERLAPPING TRANSFORMER ENTITIES
    # ============================================================

    def clean_transformer_entities(
        self,
        entities
    ):

        # Sort:
        # 1. higher confidence
        # 2. longer entity
        # 3. earlier position

        entities = sorted(
            entities,
            key=lambda x: (
                -x["score"],
                -(x["end"] - x["start"]),
                x["start"]
            )
        )

        selected = []

        for candidate in entities:

            conflict = False

            for existing in selected:

                if self.overlaps(
                    candidate["start"],
                    candidate["end"],
                    existing["start"],
                    existing["end"]
                ):

                    conflict = True
                    break

            if not conflict:

                selected.append(
                    candidate
                )

        return sorted(
            selected,
            key=lambda x: x["start"]
        )

    # ============================================================
    # ADD SPACY SUPPLEMENT
    # ============================================================

    def add_spacy_supplements(
        self,
        transformer_entities,
        spacy_entities
    ):

        final_entities = list(
            transformer_entities
        )

        for spacy_entity in spacy_entities:

            conflict = False

            for transformer_entity in (
                transformer_entities
            ):

                if self.overlaps(
                    spacy_entity["start"],
                    spacy_entity["end"],
                    transformer_entity["start"],
                    transformer_entity["end"]
                ):

                    conflict = True
                    break

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # spaCy NEVER replaces a transformer prediction.
            #
            # It is allowed only to add a completely separate
            # entity that RoBERTa did not detect.
            # ----------------------------------------------------

            if not conflict:

                final_entities.append(
                    spacy_entity
                )

        return final_entities

    # ============================================================
    # MAIN PREDICTION
    # ============================================================

    def predict(self, text):

        all_transformer_entities = []

        # --------------------------------------------------------
        # RoBERTa
        # --------------------------------------------------------

        if (
            USE_ROBERTA
            and self.roberta_model is not None
        ):

            roberta_entities = (
                self.predict_transformer(
                    text,
                    self.roberta_tokenizer,
                    self.roberta_model
                )
            )

            all_transformer_entities.extend(
                roberta_entities
            )

        # --------------------------------------------------------
        # BERT
        #
        # Currently disabled in config.py.
        # If enabled later, its predictions are added to the
        # transformer pool and cleaned using confidence.
        # --------------------------------------------------------

        if (
            USE_BERT
            and self.bert_model is not None
        ):

            bert_entities = (
                self.predict_transformer(
                    text,
                    self.bert_tokenizer,
                    self.bert_model
                )
            )

            all_transformer_entities.extend(
                bert_entities
            )

        # --------------------------------------------------------
        # Clean transformer predictions
        # --------------------------------------------------------

        transformer_entities = (
            self.clean_transformer_entities(
                all_transformer_entities
            )
        )

        # --------------------------------------------------------
        # If transformer exists, it remains primary.
        # --------------------------------------------------------

        if not USE_SPACY:

            final_entities = (
                transformer_entities
            )

        else:

            spacy_entities = (
                self.predict_spacy(
                    text
                )
            )

            if SPACY_SUPPLEMENT_ONLY:

                final_entities = (
                    self.add_spacy_supplements(
                        transformer_entities,
                        spacy_entities
                    )
                )

            else:

                final_entities = (
                    transformer_entities
                    + spacy_entities
                )

        # --------------------------------------------------------
        # Final cleanup
        # --------------------------------------------------------

        final_entities = sorted(
            final_entities,
            key=lambda x: (
                x["start"],
                x["end"]
            )
        )

        # --------------------------------------------------------
        # Remove internal score/source fields.
        #
        # evaluate_hybrid.py expects:
        # start, end, label
        # --------------------------------------------------------

        output = []

        for entity in final_entities:

            output.append(
                {
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": entity["label"],
                }
            )

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\nTesting HybridPIIEnsemble...")

    hybrid = HybridPIIEnsemble()

    test_text = (
        "My email is test@example.com "
        "and my phone number is 9876543210."
    )

    print("\nInput:")
    print(test_text)

    results = hybrid.predict(
        test_text
    )

    print("\nPredictions:")

    for entity in results:

        print(
            entity,
            "->",
            test_text[
                entity["start"]:
                entity["end"]
            ]
        )

    print("\nTest complete.")