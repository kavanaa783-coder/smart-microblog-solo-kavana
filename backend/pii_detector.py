import re
import spacy


# ---------------- SPACY MODEL ---------------- #

try:
    nlp = spacy.load("en_core_web_trf")
except OSError as exc:
    raise RuntimeError(
        "The spaCy model 'en_core_web_trf' is missing. "
        "Install it with: python -m spacy download en_core_web_trf"
    ) from exc


# ---------------- REGEX PATTERNS ---------------- #

PHONE_PATTERN = re.compile(
    r"\b(?:\+91[- ]?)?[6-9]\d{9}\b"
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

AADHAAR_PATTERN = re.compile(
    r"\b\d{4}\s?\d{4}\s?\d{4}\b"
)

PAN_PATTERN = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    re.IGNORECASE
)

DOB_PATTERN = re.compile(
    r"\b(?:0?[1-9]|[12][0-9]|3[01])[\/.-]"
    r"(?:0?[1-9]|1[0-2])[\/.-]"
    r"(?:19|20)\d{2}\b"
)


# ---------------- KNOWN NAMES ---------------- #

KNOWN_NAMES = [
    "mahima",
    "mahimashree",
    "kavana",
    "kamakshi",
    "manyashree",
    "manya",
    "nivedha",
    "saritha",
    "sneha",
    "priyanka",
    "nandita",
    "ananya",
    "aravind",
    "madhumitha",
    "siddharth",
    "kavya",
    "rahul",
    "rohit",
    "priya",
    "anjali",
    "akash",
    "arjun",
    "kiran",
    "deepak",
    "pooja",
    "vikram",
    "ajay",
    "vijay",
    "suresh",
    "ramesh",
    "apoorva",
    "koushik",
]


# ---------------- MAIN DETECTOR ---------------- #

def detect_pii(text):

    result = {
        "phones": [],
        "emails": [],
        "aadhaars": [],
        "pans": [],
        "dobs": [],
        "persons": [],
        "locations": []
    }

    # ---------- REGEX DETECTION ---------- #

    result["phones"] = list(
        dict.fromkeys(PHONE_PATTERN.findall(text))
    )

    result["emails"] = list(
        dict.fromkeys(EMAIL_PATTERN.findall(text))
    )

    result["aadhaars"] = list(
        dict.fromkeys(AADHAAR_PATTERN.findall(text))
    )

    result["pans"] = list(
        dict.fromkeys(PAN_PATTERN.findall(text))
    )

    result["dobs"] = [
        match.group(0)
        for match in DOB_PATTERN.finditer(text)
    ]


    # ---------- SPACY ENTITY DETECTION ---------- #

    doc = nlp(text)

    dob_values = {
        value.lower()
        for value in result["dobs"]
    }

    # First pass:
    # collect PERSON entities and conceptual location entities.
    for ent in doc.ents:

        value = ent.text.strip()

        if not value:
            continue

        if value.lower() in dob_values:
            continue

        if ent.label_ == "PERSON":
            result["persons"].append(value)

        elif ent.label_ in {"GPE", "LOC", "FAC"}:
            result["locations"].append(value)


    # ---------- KNOWN NAME FALLBACK ---------- #

    lower_text = text.lower()

    for name in KNOWN_NAMES:

        pattern = rf"\b{re.escape(name)}\b"

        if re.search(pattern, lower_text):

            formatted_name = name.title()

            if formatted_name not in result["persons"]:

                result["persons"].append(formatted_name)


    # ---------- RESOLVE PERSON / LOCATION CONFLICT ---------- #

    person_values = {
        person.strip().lower()
        for person in result["persons"]
    }

    cleaned_locations = []

    for location in result["locations"]:

        location_value = location.strip()

        if not location_value:
            continue

        # If the same entity is already recognized as a person,
        # do not report it as a location.
        if location_value.lower() in person_values:
            continue

        cleaned_locations.append(location_value)


    result["locations"] = cleaned_locations


    # ---------- REMOVE DUPLICATES ---------- #

    result["phones"] = list(
        dict.fromkeys(result["phones"])
    )

    result["emails"] = list(
        dict.fromkeys(result["emails"])
    )

    result["aadhaars"] = list(
        dict.fromkeys(result["aadhaars"])
    )

    result["pans"] = list(
        dict.fromkeys(result["pans"])
    )

    result["dobs"] = list(
        dict.fromkeys(result["dobs"])
    )

    result["persons"] = list(
        dict.fromkeys(result["persons"])
    )

    result["locations"] = list(
        dict.fromkeys(result["locations"])
    )

    return result


# ---------------- TEST ---------------- #

if __name__ == "__main__":

    sample = """
    My name is Mahimashree.
    DOB: 23/12/2005
    Phone: 9876543210
    Email: mahima@gmail.com
    Aadhaar: 1234 5678 9012
    PAN: ABCDE1234F
    I live in Mangalore.
    """

    print(detect_pii(sample))