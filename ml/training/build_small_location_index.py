# ============================================================
# build_small_location_index.py
# Smart Microblog Privacy Guard
#
# Purpose:
#   Create a small, fast location gazetteer from the already
#   prepared GeoNames data.
#
# IMPORTANT:
#   This does NOT modify the full GeoNames dictionary.
# ============================================================

import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "locations"
    / "processed"
)

FULL_DICTIONARY = (
    PROCESSED_DIR
    / "location_dictionary.json"
)

SMALL_DICTIONARY = (
    PROCESSED_DIR
    / "small_location_dictionary.json"
)


# ============================================================
# IMPORTANT LOCATIONS FOR FAST VERIFICATION
# ============================================================

IMPORTANT_LOCATIONS = {

    # --------------------------------------------------------
    # INDIA
    # --------------------------------------------------------

    "india",

    # Major Indian states / UTs
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "delhi",
    "jammu and kashmir",
    "ladakh",

    # Major Indian cities
    "mumbai",
    "delhi",
    "new delhi",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "chennai",
    "kolkata",
    "pune",
    "ahmedabad",
    "surat",
    "jaipur",
    "lucknow",
    "kanpur",
    "nagpur",
    "indore",
    "bhopal",
    "visakhapatnam",
    "patna",
    "vadodara",
    "coimbatore",
    "kochi",
    "thiruvananthapuram",
    "mysore",
    "mysuru",
    "mangalore",
    "mangaluru",
    "hubli",
    "hubballi",
    "belgaum",
    "belagavi",
    "shimoga",
    "shivamogga",
    "udupi",
    "dharwad",
    "tumkur",
    "tumakuru",

    # --------------------------------------------------------
    # COUNTRIES
    # --------------------------------------------------------

    "united states",
    "united states of america",
    "usa",
    "us",

    "united kingdom",
    "uk",
    "great britain",

    "canada",
    "australia",
    "india",
    "china",
    "japan",
    "south korea",
    "singapore",
    "malaysia",
    "indonesia",
    "thailand",
    "uae",
    "united arab emirates",
    "saudi arabia",
    "qatar",
    "nepal",
    "bhutan",
    "bangladesh",
    "sri lanka",
    "pakistan",

    "germany",
    "france",
    "italy",
    "spain",
    "portugal",
    "netherlands",
    "switzerland",
    "ireland",
    "russia",
    "ukraine",

    # --------------------------------------------------------
    # INTERNATIONAL MAJOR CITIES
    # --------------------------------------------------------

    "new york",
    "new york city",
    "los angeles",
    "chicago",
    "san francisco",
    "washington",
    "boston",
    "seattle",
    "las vegas",
    "silicon valley",

    "london",
    "paris",
    "berlin",
    "rome",
    "madrid",
    "barcelona",
    "amsterdam",
    "dublin",
    "zurich",

    "dubai",
    "abu dhabi",
    "doha",
    "riyadh",
    "singapore",
    "kuala lumpur",
    "bangkok",
    "jakarta",

    "tokyo",
    "osaka",
    "seoul",
    "beijing",
    "shanghai",
    "hong kong",

    "sydney",
    "melbourne",
    "toronto",
    "vancouver",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    return " ".join(
        text.lower().strip().split()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SMART MICROBLOG PRIVACY GUARD")
    print("BUILDING SMALL LOCATION INDEX")
    print("=" * 70)

    if not FULL_DICTIONARY.exists():

        raise FileNotFoundError(
            f"\nFull GeoNames dictionary not found:\n"
            f"{FULL_DICTIONARY}"
        )

    print("\nLoading full GeoNames dictionary...")

    with open(
        FULL_DICTIONARY,
        "r",
        encoding="utf-8"
    ) as f:

        dictionary = json.load(f)

    print(
        f"Full dictionary entries: "
        f"{len(dictionary):,}"
    )

    small_dictionary = {}

    print("\nSelecting important locations...")

    for name, metadata in dictionary.items():

        if not isinstance(name, str):
            continue

        normalized = normalize(name)

        if normalized not in IMPORTANT_LOCATIONS:
            continue

        # Keep the metadata structure.
        small_dictionary[normalized] = metadata

    # --------------------------------------------------------
    # Add curated names if GeoNames did not contain them
    # --------------------------------------------------------

    for name in IMPORTANT_LOCATIONS:

        if name not in small_dictionary:

            small_dictionary[name] = {
                "country_code": None,
                "feature_class": None,
                "feature_code": None,
                "source": "curated"
            }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        SMALL_DICTIONARY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            small_dictionary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("SMALL LOCATION INDEX CREATED")
    print("=" * 70)

    print(
        f"Entries: {len(small_dictionary):,}"
    )

    print(
        f"Saved to:\n{SMALL_DICTIONARY}"
    )


if __name__ == "__main__":
    main()