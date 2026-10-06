# ============================================================
# fast_location_detector.py
# Smart Microblog Privacy Guard
#
# Fast verification detector
# ============================================================

import json
import re
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DICTIONARY_PATH = (
    BASE_DIR
    / "data"
    / "locations"
    / "processed"
    / "small_location_dictionary.json"
)


# ============================================================
# COMMON WORDS THAT MUST NEVER BE LOCATIONS
# ============================================================

BLOCKLIST = {
    "the",
    "a",
    "an",
    "i",
    "he",
    "she",
    "we",
    "they",
    "you",
    "me",
    "my",
    "your",
    "our",
    "their",
    "friend",
    "company",
    "meeting",
    "live",
    "from",
    "to",
    "in",
    "at",
    "near",
    "very",
    "happy",
    "today",
    "tomorrow",
    "yesterday",
    "last",
    "next",
    "week",
    "month",
    "summer",
    "trip",
    "work",
    "works",
    "working",
    "studied",
    "study",
    "travelled",
    "traveling",
    "travelling",
}


# ============================================================
# NORMALIZE
# ============================================================

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SMART MICROBLOG PRIVACY GUARD")
print("FAST LOCATION DETECTOR")
print("=" * 70)

print("\nLoading small location index...")

with open(
    DICTIONARY_PATH,
    "r",
    encoding="utf-8"
) as f:

    location_dictionary = json.load(f)

print(
    f"Loaded {len(location_dictionary):,} locations."
)


# ============================================================
# SORT LONGEST FIRST
# ============================================================

LOCATION_NAMES = sorted(
    location_dictionary.keys(),
    key=lambda x: len(x),
    reverse=True
)


# ============================================================
# DETECTOR
# ============================================================

def detect_locations(text):

    if not isinstance(text, str):
        return []

    if not text.strip():
        return []

    normalized_text = normalize(text)

    results = []

    occupied = []

    for location in LOCATION_NAMES:

        # Never match blocked single words.
        if location in BLOCKLIST:
            continue

        # Do not search very short names.
        if len(location) < 3:
            continue

        pattern = re.compile(
            rf"(?<!\w){re.escape(location)}(?!\w)",
            re.IGNORECASE
        )

        for match in pattern.finditer(
            normalized_text
        ):

            start = match.start()
            end = match.end()

            # ------------------------------------------------
            # Prevent overlap
            # ------------------------------------------------

            overlap = False

            for old_start, old_end in occupied:

                if (
                    start < old_end
                    and end > old_start
                ):

                    overlap = True
                    break

            if overlap:
                continue

            metadata = location_dictionary[
                location
            ]

            if isinstance(metadata, list):

                metadata = (
                    metadata[0]
                    if metadata
                    else {}
                )

            if not isinstance(metadata, dict):
                metadata = {}

            country = metadata.get(
                "country_code"
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = 0.97

            if len(location.split()) >= 2:
                confidence = 0.99

            results.append({
                "text": match.group(),
                "start": start,
                "end": end,
                "label": "LOCATION",
                "country_code": country,
                "confidence": confidence,
                "source": "GeoNames"
            })

            occupied.append(
                (start, end)
            )

    results.sort(
        key=lambda x: x["start"]
    )

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

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

        "I studied at MIT in Cambridge."
    ]

    for text in tests:

        print("\nTEXT:")
        print(text)

        results = detect_locations(text)

        print("\nDETECTED:")

        if not results:

            print("None")

        else:

            for entity in results:

                print(
                    f"  {entity['text']}"
                    f" | {entity['label']}"
                    f" | {entity['country_code']}"
                    f" | confidence="
                    f"{entity['confidence']}"
                )

    print("\n" + "=" * 70)
    print("FAST LOCATION TEST COMPLETE")
    print("=" * 70)