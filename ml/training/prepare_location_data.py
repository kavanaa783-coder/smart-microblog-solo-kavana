# ============================================================
# SMART MICROBLOG PRIVACY GUARD
# LOCATION DATA PREPARATION
# ============================================================
#
# Purpose:
#   Build a worldwide + India-focused location dictionary
#   from GeoNames datasets.
#
# Input:
#   ml/data/locations/raw/
#       IN.zip
#       cities500.zip
#       allCountries.zip
#       alternateNamesV2.zip
#
# Output:
#   ml/data/locations/processed/
#       indian_locations.json
#       worldwide_locations.json
#       location_dictionary.json
#
# ============================================================

import os
import json
import zipfile
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "locations",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "locations",
    "processed"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

IN_ZIP = os.path.join(RAW_DIR, "IN.zip")
CITIES_ZIP = os.path.join(RAW_DIR, "cities500.zip")
ALL_COUNTRIES_ZIP = os.path.join(RAW_DIR, "allCountries.zip")
ALTERNATE_ZIP = os.path.join(RAW_DIR, "alternateNamesV2.zip")


# ============================================================
# OUTPUT FILES
# ============================================================

INDIA_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "indian_locations.json"
)

WORLD_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "worldwide_locations.json"
)

FINAL_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "location_dictionary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# GeoNames feature classes useful for location detection.
#
# A = administrative regions
# P = populated places
#
# We intentionally avoid roads, buildings, mountains, etc.
# for the first version because they create a huge amount
# of noise for social-media PII detection.

VALID_FEATURE_CLASSES = {
    "A",
    "P"
}


# Administrative feature codes we especially care about.
VALID_ADMIN_CODES = {
    "ADM1",
    "ADM2",
    "ADM3",
    "ADM4",
    "PCLI",
    "PCL",
    "PPLA",
    "PPLA2",
    "PPLA3",
    "PPLA4",
    "PPLC",
    "PPL",
    "PPLX"
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_name(name):
    """
    Normalize a location name for dictionary lookup.
    """

    if not name:
        return ""

    name = name.strip().lower()

    # Normalize repeated whitespace.
    name = " ".join(name.split())

    return name


def add_location(location_dict, name, country_code=None,
                  feature_class=None, feature_code=None,
                  geoname_id=None):

    normalized = normalize_name(name)

    if not normalized:
        return

    # Ignore extremely short entries.
    if len(normalized) < 2:
        return

    if normalized not in location_dict:

        location_dict[normalized] = {
            "name": name.strip(),
            "country_code": country_code,
            "feature_class": feature_class,
            "feature_code": feature_code,
            "geoname_id": geoname_id
        }


def get_txt_file_from_zip(zf, preferred_names=None):
    """
    Safely select the actual GeoNames TXT file.

    Important:
    Some GeoNames ZIP files contain readme.txt.
    We must NEVER accidentally select readme.txt.
    """

    preferred_names = preferred_names or []

    files = zf.namelist()

    # --------------------------------------------------------
    # First: try exact preferred names
    # --------------------------------------------------------

    for preferred in preferred_names:

        for filename in files:

            if os.path.basename(filename).lower() == preferred.lower():

                if filename.lower().endswith(".txt"):

                    if "readme" not in filename.lower():

                        return filename

    # --------------------------------------------------------
    # Second: choose TXT file excluding readme
    # --------------------------------------------------------

    txt_files = []

    for filename in files:

        lower = filename.lower()

        if lower.endswith(".txt") and "readme" not in lower:

            txt_files.append(filename)

    if not txt_files:

        raise FileNotFoundError(
            "No usable .txt file found inside ZIP.\n"
            f"Files available: {files[:20]}"
        )

    # Prefer the largest TXT file because GeoNames datasets
    # normally have the main data file much larger than README.
    txt_files.sort(
        key=lambda x: zf.getinfo(x).file_size,
        reverse=True
    )

    return txt_files[0]


def file_size_mb(path):

    if not os.path.exists(path):
        return 0

    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print_header(
    "SMART MICROBLOG PRIVACY GUARD\n"
    "LOCATION DATA PREPARATION"
)

print()
print("Raw directory:")
print(RAW_DIR)

print()
print("Processed directory:")
print(PROCESSED_DIR)


print_header("CHECKING INPUT FILES")

required_files = [
    ("IN.zip", IN_ZIP),
    ("cities500.zip", CITIES_ZIP),
    ("allCountries.zip", ALL_COUNTRIES_ZIP),
    ("alternateNamesV2.zip", ALTERNATE_ZIP)
]

for name, path in required_files:

    if os.path.exists(path):

        print(f"[OK] {name}")

    else:

        print(f"[MISSING] {name}")
        print(f"Expected location: {path}")
        raise FileNotFoundError(
            f"Required file missing: {name}"
        )


# ============================================================
# 1. INDIAN LOCATIONS
# ============================================================

print_header("BUILDING INDIAN LOCATION DATA")

indian_locations = {}

with zipfile.ZipFile(IN_ZIP, "r") as z:

    internal_file = get_txt_file_from_zip(
        z,
        preferred_names=["IN.txt"]
    )

    print()
    print("Reading:")
    print(IN_ZIP)

    print("Internal file:", internal_file)

    records_scanned = 0
    records_accepted = 0

    with z.open(internal_file) as f:

        for raw_line in f:

            records_scanned += 1

            try:
                line = raw_line.decode(
                    "utf-8",
                    errors="replace"
                ).rstrip("\n\r")

                fields = line.split("\t")

                # GeoNames main file has 19 columns.
                if len(fields) < 19:
                    continue

                geoname_id = fields[0]
                name = fields[1]
                ascii_name = fields[2]
                alternate_names = fields[3]

                feature_class = fields[6]
                feature_code = fields[7]
                country_code = fields[8]

                # India dataset should contain IN.
                if country_code != "IN":
                    continue

                # Keep administrative and populated places.
                if feature_class not in VALID_FEATURE_CLASSES:
                    continue

                # Add primary name.
                add_location(
                    indian_locations,
                    name,
                    country_code,
                    feature_class,
                    feature_code,
                    geoname_id
                )

                # Add ASCII name.
                if ascii_name:

                    add_location(
                        indian_locations,
                        ascii_name,
                        country_code,
                        feature_class,
                        feature_code,
                        geoname_id
                    )

                # Add GeoNames alternate names stored in field 4.
                if alternate_names:

                    for alt in alternate_names.split(","):

                        add_location(
                            indian_locations,
                            alt,
                            country_code,
                            feature_class,
                            feature_code,
                            geoname_id
                        )

                records_accepted += 1

            except Exception:
                continue


print("Records scanned:", records_scanned)
print("Records accepted:", records_accepted)
print("Unique names:", len(indian_locations))


# ============================================================
# 2. WORLDWIDE CITIES
# ============================================================

print_header("BUILDING WORLDWIDE CITY DATA")

worldwide_locations = {}

with zipfile.ZipFile(CITIES_ZIP, "r") as z:

    internal_file = get_txt_file_from_zip(
        z,
        preferred_names=["cities500.txt"]
    )

    print()
    print("Reading:")
    print(CITIES_ZIP)

    print("Internal file:", internal_file)

    records_scanned = 0
    records_accepted = 0

    with z.open(internal_file) as f:

        for raw_line in f:

            records_scanned += 1

            try:

                line = raw_line.decode(
                    "utf-8",
                    errors="replace"
                ).rstrip("\n\r")

                fields = line.split("\t")

                if len(fields) < 19:
                    continue

                geoname_id = fields[0]
                name = fields[1]
                ascii_name = fields[2]
                alternate_names = fields[3]

                feature_class = fields[6]
                feature_code = fields[7]
                country_code = fields[8]

                if feature_class != "P":
                    continue

                # Primary name.
                add_location(
                    worldwide_locations,
                    name,
                    country_code,
                    feature_class,
                    feature_code,
                    geoname_id
                )

                # ASCII name.
                if ascii_name:

                    add_location(
                        worldwide_locations,
                        ascii_name,
                        country_code,
                        feature_class,
                        feature_code,
                        geoname_id
                    )

                # Alternate names.
                if alternate_names:

                    for alt in alternate_names.split(","):

                        add_location(
                            worldwide_locations,
                            alt,
                            country_code,
                            feature_class,
                            feature_code,
                            geoname_id
                        )

                records_accepted += 1

            except Exception:
                continue


print("Records scanned:", records_scanned)
print("Records accepted:", records_accepted)
print("Unique names:", len(worldwide_locations))


# ============================================================
# 3. WORLDWIDE ADMINISTRATIVE LOCATIONS
# ============================================================

print_header("ADDING WORLDWIDE ADMINISTRATIVE LOCATIONS")

with zipfile.ZipFile(ALL_COUNTRIES_ZIP, "r") as z:

    internal_file = get_txt_file_from_zip(
        z,
        preferred_names=["allCountries.txt"]
    )

    print()
    print("Reading:")
    print(ALL_COUNTRIES_ZIP)

    print("Internal file:", internal_file)

    records_scanned = 0
    records_accepted = 0

    with z.open(internal_file) as f:

        for raw_line in f:

            records_scanned += 1

            try:

                line = raw_line.decode(
                    "utf-8",
                    errors="replace"
                ).rstrip("\n\r")

                fields = line.split("\t")

                if len(fields) < 19:
                    continue

                geoname_id = fields[0]
                name = fields[1]
                ascii_name = fields[2]
                alternate_names = fields[3]

                feature_class = fields[6]
                feature_code = fields[7]
                country_code = fields[8]

                # We only need:
                #   populated places
                #   administrative regions
                if feature_class not in VALID_FEATURE_CLASSES:
                    continue

                if feature_class == "A":

                    if feature_code not in VALID_ADMIN_CODES:
                        continue

                # Avoid duplicating cities already collected.
                before = len(worldwide_locations)

                add_location(
                    worldwide_locations,
                    name,
                    country_code,
                    feature_class,
                    feature_code,
                    geoname_id
                )

                if ascii_name:

                    add_location(
                        worldwide_locations,
                        ascii_name,
                        country_code,
                        feature_class,
                        feature_code,
                        geoname_id
                    )

                if alternate_names:

                    for alt in alternate_names.split(","):

                        add_location(
                            worldwide_locations,
                            alt,
                            country_code,
                            feature_class,
                            feature_code,
                            geoname_id
                        )

                if len(worldwide_locations) > before:

                    records_accepted += 1

            except Exception:
                continue


print("Records scanned:", records_scanned)
print("Records accepted:", records_accepted)
print("Unique names:", len(worldwide_locations))


# ============================================================
# 4. ALTERNATE NAMES
# ============================================================

print_header("ADDING ALTERNATE NAMES")

# Build GeoNames ID -> location information.
geoname_to_location = {}

for location in worldwide_locations.values():

    gid = location.get("geoname_id")

    if gid:

        geoname_to_location[str(gid)] = location


for location in indian_locations.values():

    gid = location.get("geoname_id")

    if gid:

        geoname_to_location[str(gid)] = location


print(
    "GeoNames IDs available for alternate names:",
    len(geoname_to_location)
)


alternate_scanned = 0
alternate_matched = 0


with zipfile.ZipFile(ALTERNATE_ZIP, "r") as z:

    internal_file = get_txt_file_from_zip(
        z,
        preferred_names=["alternateNamesV2.txt"]
    )

    print()
    print("Reading:")
    print(ALTERNATE_ZIP)

    print("Internal file:", internal_file)

    with z.open(internal_file) as f:

        for raw_line in f:

            alternate_scanned += 1

            try:

                line = raw_line.decode(
                    "utf-8",
                    errors="replace"
                ).rstrip("\n\r")

                fields = line.split("\t")

                if len(fields) < 4:
                    continue

                geoname_id = str(fields[1])
                alternate_name = fields[3]

                # Ignore URLs and special identifiers.
                if not alternate_name:
                    continue

                if alternate_name.startswith("http://"):
                    continue

                if alternate_name.startswith("https://"):
                    continue

                location = geoname_to_location.get(
                    geoname_id
                )

                if location is None:
                    continue

                # Add alternate spelling.
                target_dict = (
                    indian_locations
                    if location.get("country_code") == "IN"
                    else worldwide_locations
                )

                before = len(target_dict)

                add_location(
                    target_dict,
                    alternate_name,
                    location.get("country_code"),
                    location.get("feature_class"),
                    location.get("feature_code"),
                    geoname_id
                )

                if len(target_dict) > before:

                    alternate_matched += 1

            except Exception:
                continue


print("Alternate names scanned:", alternate_scanned)
print("Alternate names matched:", alternate_matched)


# ============================================================
# 5. PREPARE OUTPUT
# ============================================================

print_header("PREPARING OUTPUT")


# Convert dictionaries to sorted lists.
indian_output = sorted(
    indian_locations.values(),
    key=lambda x: x["name"].lower()
)

worldwide_output = sorted(
    worldwide_locations.values(),
    key=lambda x: x["name"].lower()
)


# ============================================================
# SAVE INDIA
# ============================================================

with open(
    INDIA_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        indian_output,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("Saved:")
print(INDIA_OUTPUT)
print(
    f"Size: {file_size_mb(INDIA_OUTPUT):.2f} MB"
)


# ============================================================
# SAVE WORLDWIDE
# ============================================================

with open(
    WORLD_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        worldwide_output,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("Saved:")
print(WORLD_OUTPUT)
print(
    f"Size: {file_size_mb(WORLD_OUTPUT):.2f} MB"
)


# ============================================================
# 6. BUILD FINAL LOCATION DICTIONARY
# ============================================================

print_header("BUILDING FINAL LOCATION DICTIONARY")

final_dictionary = {}


# India first so India-specific information is retained.
for location in indian_output:

    normalized = normalize_name(
        location["name"]
    )

    if normalized:

        final_dictionary[normalized] = {
            "name": location["name"],
            "country_code": "IN",
            "feature_class": location["feature_class"],
            "feature_code": location["feature_code"],
            "geoname_id": location["geoname_id"]
        }


# Add worldwide locations that don't already exist.
for location in worldwide_output:

    normalized = normalize_name(
        location["name"]
    )

    if not normalized:
        continue

    if normalized not in final_dictionary:

        final_dictionary[normalized] = {
            "name": location["name"],
            "country_code": location["country_code"],
            "feature_class": location["feature_class"],
            "feature_code": location["feature_code"],
            "geoname_id": location["geoname_id"]
        }


with open(
    FINAL_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_dictionary,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("Saved:")
print(FINAL_OUTPUT)

print(
    f"Size: {file_size_mb(FINAL_OUTPUT):.2f} MB"
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print_header("LOCATION DATA PREPARATION COMPLETE")

print()
print(
    "Indian location records:",
    len(indian_output)
)

print(
    "Worldwide location records:",
    len(worldwide_output)
)

print(
    "Final dictionary entries:",
    len(final_dictionary)
)

print()
print("Output files:")

print("1.", INDIA_OUTPUT)
print("2.", WORLD_OUTPUT)
print("3.", FINAL_OUTPUT)


# ============================================================
# 8. BASIC VALIDATION
# ============================================================

print_header("BASIC VALIDATION")

test_locations = [
    "Bangalore",
    "Bengaluru",
    "Mangalore",
    "Mumbai",
    "Delhi",
    "Chennai",
    "Karnataka",
    "Tamil Nadu",
    "New York",
    "London",
    "Dubai",
    "Singapore"
]

for test in test_locations:

    key = normalize_name(test)

    if key in final_dictionary:

        data = final_dictionary[key]

        print(
            f"[FOUND] {test} -> "
            f"{data['country_code']} | "
            f"{data['feature_code']}"
        )

    else:

        print(
            f"[NOT FOUND] {test}"
        )


print()
print("=" * 70)
print("NEXT STEP:")
print("Build and evaluate the LOCATION detector.")
print("=" * 70)