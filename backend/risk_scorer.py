# ============================================================
# risk_scorer.py
# ============================================================
# PURPOSE:
# Calculates privacy risk from the unified entities produced
# by backend.pii_detector.
#
# The PII detector is ML-based.
# The risk engine is policy-based and independent of the model.
# ============================================================


# ============================================================
# RISK WEIGHTS
# ============================================================
#
# These represent privacy sensitivity, not ML confidence.
#
# Higher weight = greater potential privacy impact.
#
# The weights are deliberately kept here, separately from the
# ML detector, so the two responsibilities remain independent.
# ============================================================

RISK_WEIGHTS = {
    # Very sensitive identity / authentication information
    "AADHAAR": 100,
    "SSN": 100,
    "PASSWORD": 100,
    "API_KEY": 100,
    "BANK_ACCOUNT": 95,
    "CREDIT_CARD": 95,
    "PASSPORT": 90,

    # Financial / account information
    "PAN": 85,
    "DRIVER_LICENSE": 85,
    "UPI_ID": 75,
    "ACCOUNT": 70,

    # Direct contact information
    "PHONE": 60,
    "EMAIL": 55,

    # Device / network identifiers
    "MAC_ADDRESS": 50,
    "IP_ADDRESS": 45,
    "DEVICE_ID": 45,
    "CRYPTO_ADDRESS": 45,

    # Personal identity
    "PERSON": 20,
    "USERNAME": 20,

    # Personal / demographic information
    "DOB": 35,
    "AGE": 25,

    # Location / physical information
    "ADDRESS": 50,
    "LOCATION": 25,
    "VEHICLE_REG": 60,

    # Other potentially identifying information
    "DATE": 15,
    "URL": 10,
    "USER_AGENT": 20,
}


# ============================================================
# THRESHOLDS
# ============================================================

LOW_MAX = 29
MEDIUM_MAX = 69


# ============================================================
# RECOMMENDATIONS
# ============================================================

def _get_recommendation(risk_level):
    if risk_level == "LOW":
        return (
            "No major privacy risks detected. "
            "Your post appears safe to publish."
        )

    if risk_level == "MEDIUM":
        return (
            "Some personal information was detected. "
            "Please review your post before publishing."
        )

    return (
        "Highly sensitive personal information detected. "
        "Remove or mask the information before publishing."
    )


# ============================================================
# MAIN RISK ENGINE
# ============================================================

def calculate_risk(detected_entities):
    """
    Calculate privacy risk from the unified ML/regex entities.

    Expected input:

    {
        "entities": [
            {
                "text": "...",
                "label": "PERSON",
                "start": 0,
                "end": 10,
                "source": "spacy"
            }
        ]
    }

    The function also supports the old detector format as a
    temporary compatibility measure.
    """

    entities = detected_entities.get("entities", [])

    # --------------------------------------------------------
    # New unified entity format
    # --------------------------------------------------------

    if entities:

        score = 0
        counted_entities = []

        for entity in entities:

            label = entity.get("label")

            if not label:
                continue

            weight = RISK_WEIGHTS.get(label)

            if weight is None:
                continue

            score += weight

            counted_entities.append(
                {
                    "text": entity.get("text", ""),
                    "label": label,
                    "weight": weight,
                    "source": entity.get("source", "unknown"),
                }
            )

    # --------------------------------------------------------
    # Temporary compatibility with old detector format
    # --------------------------------------------------------

    else:

        legacy_mapping = {
            "phones": "PHONE",
            "emails": "EMAIL",
            "aadhaars": "AADHAAR",
            "pans": "PAN",
            "dobs": "DOB",
            "persons": "PERSON",
            "locations": "LOCATION",
        }

        score = 0
        counted_entities = []

        for field, label in legacy_mapping.items():

            values = detected_entities.get(field, [])

            weight = RISK_WEIGHTS[label]

            for value in values:

                score += weight

                counted_entities.append(
                    {
                        "text": value,
                        "label": label,
                        "weight": weight,
                        "source": "legacy",
                    }
                )

    # --------------------------------------------------------
    # Cap score
    # --------------------------------------------------------

    score = min(score, 100)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score <= LOW_MAX:
        risk_level = "LOW"

    elif score <= MEDIUM_MAX:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "recommendation": _get_recommendation(
            risk_level
        ),
        "risk_entities": counted_entities,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    sample = {
        "entities": [
            {
                "text": "Mahimashree",
                "label": "PERSON",
                "start": 0,
                "end": 11,
                "source": "spacy",
            },
            {
                "text": "mahima@gmail.com",
                "label": "EMAIL",
                "start": 20,
                "end": 36,
                "source": "regex",
            },
        ]
    }

    print(calculate_risk(sample))