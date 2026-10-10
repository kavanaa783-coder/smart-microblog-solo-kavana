import re

from pathlib import Path

import spacy

# ============================================================

# MODELS

# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SPACY_MODEL_PATH = (

    BASE_DIR

    / "ml"

    / "models"

    / "spacy_pii_combined_case_augmented"

    / "final"

)

try:

    pii_nlp = spacy.load(SPACY_MODEL_PATH)

    general_nlp = spacy.load("en_core_web_sm")

except (OSError, IOError) as exc:

    raise RuntimeError(

        "Could not load the custom PII model or en_core_web_sm."

    ) from exc

# ============================================================

# STRUCTURED PII REGEX

# ============================================================

STRUCTURED_PATTERNS = {

    "PHONE": re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b"),

    "EMAIL": re.compile(

        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    ),

    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),

    "PAN": re.compile(

        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

        re.IGNORECASE,

    ),

    "DOB": re.compile(

        r"\b(?:0?[1-9]|[12][0-9]|3[01])[\\/.-]"

        r"(?:0?[1-9]|1[0-2])[\\/.-](?:19|20)\d{2}\b"

    ),

}

PLACEHOLDER_PATTERN = re.compile(

    r"\b(?:xyz|abc|unknown|anonymous|redacted|placeholder)\b",

    re.IGNORECASE,

)

# ============================================================

# HELPERS

# ============================================================

def _normalize(value: str) -> str:

    return value.strip().casefold()

def _overlaps(start: int, end: int, spans) -> bool:

    return any(

        start < other_end and other_start < end

        for other_start, other_end in spans

    )

def _entity(text, label, start, end, source):

    return {

        "text": text[start:end].strip(),

        "label": label,

        "start": start,

        "end": end,

        "source": source,

    }

def _is_placeholder(text: str, start: int, end: int) -> bool:

    value = text[start:end].strip()

    return bool(value and PLACEHOLDER_PATTERN.fullmatch(value))

# ============================================================

# REGEX DETECTION

# ============================================================

def _detect_structured_pii(text: str):

    entities = []

    protected_spans = []

    for label, pattern in STRUCTURED_PATTERNS.items():

        for match in pattern.finditer(text):

            start, end = match.span()

            if _overlaps(start, end, protected_spans):

                continue

            entities.append(

                _entity(text, label, start, end, "regex")

            )

            protected_spans.append((start, end))

    return entities, protected_spans

# ============================================================

# CONTEXT-BASED LOCATION DETECTION

# ============================================================

LOCATION_CONTEXT_PATTERN = re.compile(

    r"\b(?:from|native\s+is|native\s+of|lives\s+in|located\s+in|"

    r"went\s+to|run\s+to|college\s+is\s+in)\s+"

    r"([a-z][a-z'-]*)",

    re.IGNORECASE,

)

LOCATION_LIST_PATTERN = re.compile(

    r"\band\s+([a-z][a-z'-]*)\b",

    re.IGNORECASE,

)

def _detect_context_locations(text: str, protected_spans):

    entities = []

    patterns = (

        (LOCATION_CONTEXT_PATTERN, "context"),

        (LOCATION_LIST_PATTERN, "context_list"),

    )

    for pattern, source in patterns:

        for match in pattern.finditer(text):

            start, end = match.span(1)

            value = match.group(1)

            if value.casefold() in {"my", "me", "i", "we", "us", "you", "he", "she", "they", "it"}:

                continue

            if _overlaps(start, end, protected_spans):

                continue

            if PLACEHOLDER_PATTERN.fullmatch(value):

                continue

            # Avoid treating names from common personal-name phrases

            # as locations merely because they follow "and".

            if source == "context_list":

                prefix = text[max(0, start - 35):start].casefold()

                if re.search(

                    r"\b(?:my\s+(?:friend|frd|brother|sister|guide)|"

                    r"name\s+is|siblings?\s+name\s+are)\s*$",

                    prefix,

                ):

                    continue

            entities.append(

                _entity(text, "LOCATION", start, end, source)

            )

    return entities

# ============================================================

# MODEL DETECTION

# ============================================================

def _detect_model_entities(text: str, protected_spans):

    entities = []

    # Custom model detects domain-specific PII.

    custom_doc = pii_nlp(text)

    for ent in custom_doc.ents:

        start, end = ent.start_char, ent.end_char

        if _overlaps(start, end, protected_spans):

            continue

        if not ent.text.strip():

            continue

        if not any(character.isalnum() for character in ent.text):

            continue

        label = ent.label_

        # Correct person predictions when the surrounding text

        # explicitly indicates a location.

        prefix = text[max(0, start - 45):start].casefold()

        if label == "PERSON" and re.search(

            r"\b(?:from|native\s+is|native\s+of|lives\s+in|located\s+in)\s+$",

            prefix,

        ):

            label = "LOCATION"

        if label in {"LOCATION", "GPE", "LOC"}:

            label = "LOCATION"

        entities.append(

            _entity(text, label, start, end, "spacy_pii")

        )

    # General English NER adds conventional PERSON and location labels.

    general_doc = general_nlp(text)

    for ent in general_doc.ents:

        if ent.label_ not in {"PERSON", "GPE", "LOC"}:

            continue

        if not any(character.isalnum() for character in ent.text):

            continue

        start, end = ent.start_char, ent.end_char

        if _overlaps(start, end, protected_spans):

            continue

        label = "PERSON" if ent.label_ == "PERSON" else "LOCATION"

        entities.append(

            _entity(text, label, start, end, "spacy_general")

        )

    return entities

# ============================================================

# CONTEXT-BASED PERSON CANDIDATES

# ============================================================

PERSON_CONTEXT_PATTERNS = [

    re.compile(

        r"\b(?:my\s+name\s+is|my\s+(?:friend|frd|brother|sister|guide)"

        r"\s+(?:name\s+is|is)|my\s+(?:mother|father)\s+name\s+is|"

        r"mhy\s+mother\s+name\s+is|i\s+am\s+daughter\s+of)\s+"

        r"([a-z][a-z'-]*)"

        r"(?:\s+(?!and\b|from\b|my\b|i\b|is\b|are\b)"

        r"[a-z][a-z'-]*){0,3}",

        re.IGNORECASE,

    ),

    re.compile(

        r"\b(?:my\s+siblings?\s+name\s+are|"

        r"i\s+have\s+\d+\s+siblings?\s+name)\s+"

        r"([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,5})",

        re.IGNORECASE,

    ),

]

def _detect_context_persons(text: str, protected_spans):

    entities = []

    for pattern in PERSON_CONTEXT_PATTERNS:

        for match in pattern.finditer(text):

            start, end = match.span(1)

            raw = text[start:end]

            parts = re.split(

                r"\s+(?:and|from|who|she|he|my|i|is|are|was|went|"

                r"lives|located|has|have|with)\s+",

                raw,

                maxsplit=1,

                flags=re.IGNORECASE,

            )

            candidate = parts[0].strip()

            candidate_end = start + len(candidate)

            if not candidate:

                continue

            if PLACEHOLDER_PATTERN.fullmatch(candidate):

                continue

            if re.search(

                r"\bsiblings?\s+name\s+are\b|"

                r"\bi\s+have\s+\d+\s+siblings?\s+name\b",

                match.group(0),

                re.IGNORECASE,

            ):

                offset = start

                for token in candidate.split():

                    token_start = text.find(

                        token, offset, candidate_end

                    )

                    if token_start < 0:

                        continue

                    token_end = token_start + len(token)

                    if (

                        not _is_placeholder(

                            text, token_start, token_end

                        )

                        and not _overlaps(

                            token_start, token_end, protected_spans

                        )

                    ):

                        entities.append(

                            _entity(

                                text,

                                "PERSON",

                                token_start,

                                token_end,

                                "context",

                            )

                        )

                    offset = token_end

            elif not _overlaps(

                start, candidate_end, protected_spans

            ):

                entities.append(

                    _entity(

                        text,

                        "PERSON",

                        start,

                        candidate_end,

                        "context",

                    )

                )

    # Detect the final name in a comma/and-separated person list.
    # This specifically catches the last name in text such as:
    # "... and keerthana and shalini".
    list_pattern = re.compile(
        r"(?:,|\band\b)\s*([A-Za-z][A-Za-z'-]*)\s*$",
        re.IGNORECASE,
    )

    match = list_pattern.search(text)

    if match:
        start, end = match.span(1)
        candidate = match.group(1).lower()

        has_person_context = re.search(
            r"\b(?:my\s+name\s+is|"
            r"my\s+(?:mother|father|friend|frd|brother|sister|guide)"
            r"(?:\s+name)?\s+is|"
            r"siblings?\s+name\s+are|"
            r"i\s+have\s+\d+\s+siblings?\s+name)\b",
            text[:start],
            re.IGNORECASE,
        )

        non_person_words = {
            "usa", "canada", "india", "udupi",
            "manglore", "mangalore", "davanagere",
        }

        if (
            has_person_context
            and candidate not in non_person_words
            and not _is_placeholder(text, start, end)
        ):
            entities.append(
                _entity(text, "PERSON", start, end, "context")
            )

    return entities

# ============================================================

# PLACEHOLDER FILTERING

# ============================================================

def _remove_placeholder_entities(text: str, entities):

    return [

        entity

        for entity in entities

        if not (

            entity["label"] == "PERSON"

            and _is_placeholder(

                text,

                entity["start"],

                entity["end"],

            )

        )

    ]

# ============================================================

# MERGE AND DEDUPLICATE

# ============================================================

def _merge_entities(entities):

    by_span = {}

    label_priority = {
        # Prefer explicit person-context detection over a conflicting
        # model LOCATION label for the exact same text span.
        "PERSON": 2,
        "LOCATION": 1,
        "DATE": 2,
        "DOB": 3,
        "PHONE": 3,
        "EMAIL": 3,
        "AADHAAR": 3,
        "PAN": 3,
    }

    for entity in entities:

        key = (

            entity["start"],

            entity["end"],

            _normalize(entity["text"]),

        )

        existing = by_span.get(key)

        if existing is None:

            by_span[key] = entity.copy()

            continue

        old_priority = label_priority.get(existing["label"], 2)

        new_priority = label_priority.get(entity["label"], 2)

        if new_priority > old_priority:

            by_span[key] = entity.copy()

        elif (

            existing["label"] in {"GPE", "LOC"}

            and entity["label"] == "LOCATION"

        ):

            by_span[key] = entity.copy()

    result = list(by_span.values())

    result.sort(key=lambda item: (item["start"], item["end"]))

    return result

# ============================================================

# BACKWARD-COMPATIBLE RESULT

# ============================================================

LEGACY_FIELDS = {

    "PHONE": "phones",

    "EMAIL": "emails",

    "AADHAAR": "aadhaars",

    "PAN": "pans",

    "DOB": "dobs",

    "PERSON": "persons",

    "LOCATION": "locations",

}

def _empty_result():

    return {

        "phones": [],

        "emails": [],

        "aadhaars": [],

        "pans": [],

        "dobs": [],

        "persons": [],

        "locations": [],

        "entities": [],

    }

def _build_result(entities):

    result = _empty_result()

    result["entities"] = entities

    seen = {

        field: set()

        for field in result

        if field != "entities"

    }

    for entity in entities:

        field = LEGACY_FIELDS.get(entity["label"])

        if field is None:

            continue

        value = entity["text"].strip()

        normalized = _normalize(value)

        if value and normalized not in seen[field]:

            seen[field].add(normalized)

            result[field].append(value)

    return result

# ============================================================

# MAIN DETECTOR

# ============================================================

def detect_pii(text: str):

    """

    Detect PII using structured regex, the trained PII model,

    general English NER, and context-based candidates.

    Returns the legacy fields plus a merged entity list.

    BERT, RoBERTa and Ollama are not integrated here.

    """

    if not isinstance(text, str) or not text.strip():

        return _empty_result()

    structured_entities, protected_spans = (

        _detect_structured_pii(text)

    )

    model_entities = _detect_model_entities(

        text,

        protected_spans,

    )

    context_entities = _detect_context_locations(
        text,
        protected_spans,
    ) or []

    context_person_entities = _detect_context_persons(
        text,
        protected_spans,
    ) or []

    entities = _merge_entities(

        structured_entities

        + model_entities

        + context_entities

        + context_person_entities

    )

    entities = _remove_placeholder_entities(text, entities)

    return _build_result(entities)

# ============================================================

# LOCAL TEST

# ============================================================

if __name__ == "__main__":

    samples = [

        "my name is mahimashree",

        "my friend is rahul",

        "MAHIMASHREE lives in mangalore",

        "my email is test@gmail.com",

        "my PAN is ABCDE1234F",

        "my Aadhaar is 7894 7894 7894",

        "I am from davanagere and my mother native is tamilnadu",

    ]

    for sample in samples:

        print("\nInput:", sample)

        print("Output:", detect_pii(sample))
