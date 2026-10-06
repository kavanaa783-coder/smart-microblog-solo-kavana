import json
import re
from pathlib import Path
from functools import lru_cache


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DICTIONARY_PATH = (
    BASE_DIR
    / "data"
    / "locations"
    / "processed"
    / "location_dictionary.json"
)

MIN_CONFIDENCE = 0.70


# ============================================================
# COMMON FALSE POSITIVES
# ============================================================

STOPWORDS = {
    "the", "a", "an",
    "i", "me", "my", "we", "you",
    "he", "she", "it", "they",

    "live", "life",
    "friend", "friends",
    "from", "to", "in", "at", "on",
    "near", "with", "for", "and", "or",
    "but", "of",

    "is", "am", "are", "was", "were",
    "be", "been", "being",

    "very", "happy", "sad", "good",
    "bad", "great", "best",

    "last", "next", "week", "month",
    "year", "today", "tomorrow",
    "yesterday", "summer", "winter",
    "spring", "autumn",

    "meeting", "company", "work",
    "works", "working",

    "trip", "travel", "travelled",
    "travelling",

    "visit", "visited", "visiting",

    "road", "street", "home", "house",
    "place", "area", "city", "town",
    "village", "valley", "river",
    "lake", "park", "station",

    "mit",
}


# ============================================================
# IMPORTANT WORLD LOCATIONS
# ============================================================

# These are deliberately explicit because these locations are
# extremely common in user-generated text and have many
# ambiguous GeoNames records.

KNOWN_LOCATIONS = {

    # ---------------- INDIA ----------------

    "india": "IN",
    "karnataka": "IN",
    "tamil nadu": "IN",

    "delhi": "IN",
    "new delhi": "IN",

    "mumbai": "IN",
    "bombay": "IN",

    "bengaluru": "IN",
    "bangalore": "IN",

    "mangalore": "IN",
    "mangaluru": "IN",

    "hyderabad": "IN",

    "chennai": "IN",
    "madras": "IN",

    "pune": "IN",
    "kochi": "IN",

    "mysore": "IN",
    "mysuru": "IN",

    "goa": "IN",

    # ---------------- UNITED KINGDOM ----------------

    "london": "GB",
    "manchester": "GB",
    "birmingham": "GB",
    "edinburgh": "GB",
    "glasgow": "GB",
    "liverpool": "GB",

    # ---------------- USA ----------------

    "united states": "US",
    "usa": "US",
    "us": "US",

    "new york": "US",
    "new york city": "US",

    "california": "US",
    "texas": "US",
    "florida": "US",
    "washington": "US",

    "los angeles": "US",
    "san francisco": "US",
    "chicago": "US",
    "boston": "US",
    "seattle": "US",

    # ---------------- FRANCE ----------------

    "france": "FR",
    "paris": "FR",
    "lyon": "FR",
    "marseille": "FR",

    # ---------------- UAE ----------------

    "united arab emirates": "AE",
    "uae": "AE",
    "dubai": "AE",
    "abu dhabi": "AE",

    # ---------------- SINGAPORE ----------------

    "singapore": "SG",

    # ---------------- CANADA ----------------

    "canada": "CA",
    "toronto": "CA",
    "vancouver": "CA",
    "montreal": "CA",
    "ottawa": "CA",

    # ---------------- AUSTRALIA ----------------

    "australia": "AU",
    "sydney": "AU",
    "melbourne": "AU",
    "perth": "AU",
    "brisbane": "AU",

    # ---------------- GERMANY ----------------

    "germany": "DE",
    "berlin": "DE",
    "munich": "DE",
    "frankfurt": "DE",

    # ---------------- ITALY ----------------

    "italy": "IT",
    "rome": "IT",
    "milan": "IT",
    "venice": "IT",

    # ---------------- JAPAN ----------------

    "japan": "JP",
    "tokyo": "JP",
    "osaka": "JP",
    "kyoto": "JP",

    # ---------------- CHINA ----------------

    "china": "CN",
    "beijing": "CN",
    "shanghai": "CN",

    # ---------------- SOUTH KOREA ----------------

    "south korea": "KR",
    "seoul": "KR",

    # ---------------- QATAR ----------------

    "qatar": "QA",
    "doha": "QA",

    # ---------------- SAUDI ARABIA ----------------

    "saudi arabia": "SA",
    "riyadh": "SA",

    # ---------------- EUROPE ----------------

    "netherlands": "NL",
    "amsterdam": "NL",

    "belgium": "BE",
    "brussels": "BE",

    "austria": "AT",
    "vienna": "AT",

    "switzerland": "CH",
    "zurich": "CH",

    "spain": "ES",
    "madrid": "ES",
    "barcelona": "ES",

    "portugal": "PT",
    "lisbon": "PT",

    "ireland": "IE",
    "dublin": "IE",
}


# ============================================================
# FEATURE PRIORITY
# ============================================================

FEATURE_SCORE = {

    "PCLI": 1.00,
    "PCL": 0.98,

    "ADM1": 0.95,
    "ADM2": 0.90,
    "ADM3": 0.85,
    "ADM4": 0.80,

    "PPLC": 1.00,
    "PPLA": 0.95,
    "PPLA2": 0.90,
    "PPLA3": 0.85,
    "PPLA4": 0.80,

    "PPL": 0.78,
    "PPLX": 0.70,
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip(
        ".,!?;:'\"()[]{}"
    )


# ============================================================
# LOAD GEONAMES
# ============================================================

@lru_cache(maxsize=1)
def load_dictionary():

    print(
        "Loading GeoNames location dictionary..."
    )

    with open(
        DICTIONARY_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        f"Loaded {len(data):,} entries."
    )

    return data


# ============================================================
# BUILD INDEX
# ============================================================

@lru_cache(maxsize=1)
def build_index():

    dictionary = load_dictionary()

    index = {}

    for name, records in dictionary.items():

        if not isinstance(name, str):
            continue

        key = normalize(name)

        if len(key) < 3:
            continue

        if key in STOPWORDS:
            continue

        if not isinstance(records, list):
            records = [records]

        valid = []

        for record in records:

            if not isinstance(record, dict):
                continue

            valid.append(record)

        if valid:
            index[key] = valid

    print(
        f"GeoNames index ready: "
        f"{len(index):,} names"
    )

    return index


# ============================================================
# GETTERS
# ============================================================

def get_country(record):

    return (
        record.get("country_code")
        or record.get("countryCode")
        or record.get("country")
    )


def get_feature(record):

    return (
        record.get("feature_code")
        or record.get("featureCode")
    )


def get_population(record):

    try:
        return int(
            record.get(
                "population",
                0
            )
        )
    except Exception:
        return 0


# ============================================================
# DIRECT KNOWN LOCATION MATCH
# ============================================================

def detect_known_locations(text):

    matches = []

    # Longest first.
    names = sorted(
        KNOWN_LOCATIONS.keys(),
        key=len,
        reverse=True
    )

    occupied = []

    for name in names:

        pattern = re.compile(
            rf"(?<![\w])"
            rf"{re.escape(name)}"
            rf"(?![\w])",
            re.IGNORECASE
        )

        for match in pattern.finditer(text):

            start = match.start()
            end = match.end()

            overlap = False

            for s, e in occupied:

                if start < e and end > s:
                    overlap = True
                    break

            if overlap:
                continue

            country = KNOWN_LOCATIONS[name]

            # Explicit known locations receive high confidence.
            confidence = 0.97

            # A location followed by another country/location
            # is still valid.
            matches.append({
                "text": match.group(),
                "start": start,
                "end": end,
                "label": "LOCATION",
                "source": "GeoNames-KnownLocation",
                "confidence": confidence,
                "country_code": country,
                "feature_class": None,
                "feature_code": None,
            })

            occupied.append(
                (start, end)
            )

    return matches


# ============================================================
# GEONAMES CANDIDATE SCORE
# ============================================================

def candidate_score(
    name,
    record,
    text,
    start,
    end
):

    country = get_country(
        record
    )

    feature = get_feature(
        record
    )

    score = 0.0

    # --------------------------------------------------------
    # Feature score
    # --------------------------------------------------------

    score += (
        FEATURE_SCORE.get(
            feature,
            0.40
        )
        * 0.45
    )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population = get_population(
        record
    )

    if population >= 10000000:
        score += 0.35

    elif population >= 1000000:
        score += 0.30

    elif population >= 100000:
        score += 0.22

    elif population >= 10000:
        score += 0.12

    elif population > 0:
        score += 0.05

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    before = text[
        max(0, start - 25):start
    ].lower()

    if re.search(
        r"\b(in|from|to|at|near|into|towards?)\s*$",
        before
    ):

        if feature in {
            "PPLC",
            "PPLA",
            "PPLA2",
            "PPLA3",
            "PPLA4",
            "PPL",
            "ADM1",
            "ADM2",
        }:

            score += 0.15

    # --------------------------------------------------------
    # Country exists
    # --------------------------------------------------------

    if country:
        score += 0.05

    return min(
        score,
        0.99
    )


# ============================================================
# GEONAMES DETECTOR
# ============================================================

def detect_geonames_locations(
    text,
    blocked_ranges
):

    index = build_index()

    candidates = []

    # IMPORTANT:
    #
    # We do not scan every tiny word.
    #
    # Only names >= 3 characters.
    #
    # Known locations are already handled above.

    for name, records in index.items():

        if name in KNOWN_LOCATIONS:
            continue

        pattern = re.compile(
            rf"(?<![\w])"
            rf"{re.escape(name)}"
            rf"(?![\w])",
            re.IGNORECASE
        )

        for match in pattern.finditer(text):

            start = match.start()
            end = match.end()

            # Check known matches.
            blocked = False

            for s, e in blocked_ranges:

                if start < e and end > s:

                    blocked = True
                    break

            if blocked:
                continue

            actual = match.group()

            if normalize(actual) in STOPWORDS:
                continue

            best_record = None
            best_score = 0.0

            for record in records:

                score = candidate_score(
                    name,
                    record,
                    text,
                    start,
                    end
                )

                if score > best_score:

                    best_score = score
                    best_record = record

            if best_record is None:
                continue

            if best_score < MIN_CONFIDENCE:
                continue

            candidates.append({
                "text": actual,
                "start": start,
                "end": end,
                "label": "LOCATION",
                "source": "GeoNames",
                "confidence": round(
                    best_score,
                    2
                ),
                "country_code": get_country(
                    best_record
                ),
                "feature_class": (
                    best_record.get(
                        "feature_class"
                    )
                    or best_record.get(
                        "featureClass"
                    )
                ),
                "feature_code": get_feature(
                    best_record
                ),
            })

    return candidates


# ============================================================
# FINAL DETECTOR
# ============================================================

def detect_locations(text):

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    if not text.strip():
        return []

    # --------------------------------------------------------
    # STEP 1
    # Known locations
    # --------------------------------------------------------

    known = detect_known_locations(
        text
    )

    blocked = [
        (
            item["start"],
            item["end"]
        )
        for item in known
    ]

    # --------------------------------------------------------
    # STEP 2
    # GeoNames fallback
    # --------------------------------------------------------

    other = detect_geonames_locations(
        text,
        blocked
    )

    all_matches = (
        known + other
    )

    # --------------------------------------------------------
    # STEP 3
    # Resolve overlaps
    # --------------------------------------------------------

    all_matches.sort(
        key=lambda x: (
            x["start"],
            -(x["end"] - x["start"]),
            -x["confidence"]
        )
    )

    final = []

    occupied = []

    for item in all_matches:

        start = item["start"]
        end = item["end"]

        overlap = False

        for s, e in occupied:

            if start < e and end > s:

                overlap = True
                break

        if overlap:
            continue

        final.append(item)

        occupied.append(
            (start, end)
        )

    final.sort(
        key=lambda x: x["start"]
    )

    return final


# ============================================================
# SIMPLE API
# ============================================================

def detect_location_entities(text):

    return detect_locations(text)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "SMART MICROBLOG PRIVACY GUARD"
    )
    print(
        "FINAL LOCATION DETECTOR TEST"
    )
    print("=" * 70)

    tests = [

        "I live in Mangalore, Karnataka.",

        "My friend moved from Bangalore to Mumbai.",

        "I am studying in Chennai, Tamil Nadu.",

        "She travelled from New York to London.",

        "We are planning a trip to Dubai and Singapore.",

        "He works in New Delhi.",

        "I visited Bengaluru and Hyderabad last month.",

        "My company is located in California.",

        "I travelled from Mumbai to New York City.",

        "I am from India.",

        "I live in Karnataka and work in Bengaluru.",

        "The meeting is near Mumbai.",

        "I went to Paris last summer.",

        "I work remotely.",

        "My friend is very happy today.",

        "I am travelling from Mangalore to Bengaluru.",

        "She lives in Delhi.",

        "He moved to New York City.",

        "We are visiting London next week.",

        "I work in Silicon Valley.",

        "I studied at MIT in Cambridge.",

        "She lives in Paris, France.",

        "I moved from London to Mumbai.",

        "I travelled from California to New York City.",

        "I am visiting Dubai from India.",
    ]

    for text in tests:

        print("\nTEXT:")
        print(text)

        results = detect_locations(
            text
        )

        print(
            "\nDETECTED LOCATIONS:"
        )

        if not results:

            print("None")

            continue

        for entity in results:

            print(
                f"  {entity['text']}"
                f" | {entity['label']}"
                f" | {entity['country_code']}"
                f" | confidence={entity['confidence']}"
            )

    print("\n")
    print("=" * 70)
    print(
        "LOCATION DETECTOR TEST COMPLETE"
    )
    print("=" * 70)