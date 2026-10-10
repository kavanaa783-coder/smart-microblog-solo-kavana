"""Simple spaCy smoke test script.

This file is intentionally not a pytest test module.
"""

__test__ = False

import spacy


def main():
    nlp = spacy.load("en_core_web_sm")
    text = input("Enter text: ")

    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text, "->", ent.label_)


if __name__ == "__main__":
    main()
