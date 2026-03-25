#!/usr/bin/env python3

import csv
import os
import re
import unicodedata

try:
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    _STOP_RAW = set(_nltk_sw.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False
    _lemmatizer = None
    _STOP_RAW = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as",
        "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "through", "to", "from", "in", "on",
        "into", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "some", "such", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
        "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
        "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
        "shouldn", "wasn", "weren", "won", "wouldn",
    }

RAW_PATH = os.path.join("data", "unsupervised", "raw", "Student_Dataset.csv")
OUT_PATH = os.path.join("data", "unsupervised", "processed", "Student_Dataset.csv")
OUT_PATH_URGENCY = os.path.join("data", "unsupervised", "processed", "Student_Dataset_urgency.csv")
EMPTY_WORDS = set(["", " ", "\t", "\n"])

_MEDICAL_KEEP = {
    "no", "not", "nor", "never", "without",
    "more", "most", "very",
    "few", "further",
    "above", "below", "under", "over",
    "before", "after", "during",
    "both", "each", "again",
    "up", "down", "off", "out",
}

STOP_WORDS = _STOP_RAW - _MEDICAL_KEEP

STOP_WORDS |= {"feel", "felt", "get", "make", "like", "start", "notice", "even", "think", "look", "wa", "pain"}

# For urgency detection: keep words that indicate severity/urgency
_URGENCY_KEEP = {
    "no", "not", "nor", "never", "without",
    "more", "most", "very", "really", "extremely",
    "few", "further",
    "above", "below", "under", "over",
    "before", "during",
    "both", "each", "again",
    "up", "down", "off", "out",
    # Urgency-specific indicators
    "sudden", "suddenly", "acute", "severe", "severely", "serious",
    "urgent", "immediately", "emergency", "emergency", "critical",
    "worsening", "worse", "worsen", "rapidly", "sharp", "intense",
    "intense", "intensely", "immediately", "right", "straightaway",
    "alarming", "alarmed", "concerning", "concerning", "emergency",
    "hospitalization", "ambulance", "immediately", "now",
}

STOP_WORDS_URGENCY = _STOP_RAW - _URGENCY_KEEP

STOP_WORDS_URGENCY |= {"feel", "felt", "get", "make", "like", "notice", "even", "think", "look", "wa"}

COMMON_FIXES = {
    "headace":    "headache",
    "headach":    "headache",
    "stomack":    "stomach",
    "stomache":   "stomach",
    "diarhea":    "diarrhea",
    "diarrhoea":  "diarrhea",
    "diarrea":    "diarrhea",
    "nausia":     "nausea",
    "nausious":   "nauseous",
    "dizzyness":  "dizziness",
    "dizzines":   "dizziness",
    "breathin":   "breathing",
    "soarness":   "soreness",
    "sweling":    "swelling",
    "alergic":    "allergic",
    "alergy":     "allergy",
    "symtom":     "symptom",
    "symtoms":    "symptoms",
    "inflamation":"inflammation",
    "inflamatory":"inflammatory",
    "painfull":   "painful",
    "numbnes":    "numbness",
    "fatige":     "fatigue",
    "fatique":    "fatigue",
    "musle":      "muscle",
    "muscule":    "muscle",
    "temprature": "temperature",
    "temperture": "temperature",
    "coughing":   "cough",
    "feeverish":  "feverish",
    "fevered":    "feverish",
    "itchiness":  "itch",
    "itching":    "itch",
}

_SUFFIX_RULES = [
    ("ies", "y"), ("ves", "f"), ("ses", "s"), ("ches", "ch"),
    ("shes", "sh"), ("xes", "x"), ("zing", "ze"), ("ting", "te"),
    ("ning", "n"), ("ring", "r"), ("ling", "l"), ("ding", "d"),
    ("ing", ""), ("ed", ""), ("ly", ""), ("ness", ""), ("ment", ""),
    ("s", ""),
]


def _is_valid_color_marker(marker: str) -> bool:
    marker = (marker or "").strip().lower()
    if not marker.startswith("0x"):
        return False
    hex_part = marker[2:]
    if len(hex_part) != 6:
        return False
    try:
        int(hex_part, 16)
    except ValueError:
        return False
    return True


def _simple_lemmatize(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, repl in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(repl) >= 3:
            return word[: -len(suffix)] + repl
    return word


def _normalize_unicode(text: str) -> str:
    text = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = unicodedata.normalize("NFKD", text)
    return text


def _remove_special_chars(text: str) -> str:
    text = re.sub(r"(\w)-(\w)", r"\1\2", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def _fix_spelling(word: str) -> str:
    return COMMON_FIXES.get(word, word)


def cleaned_rows_generator(text: str) -> str:
    text = _normalize_unicode(text)
    text = text.lower()
    text = _remove_special_chars(text)
    if _HAS_NLTK:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    tokens = [_fix_spelling(tok) for tok in tokens]
    # Lemmatize BEFORE stop-word removal so inflected forms are caught too
    if _HAS_NLTK:
        tokens = [_lemmatizer.lemmatize(_lemmatizer.lemmatize(tok, pos="n"), pos="v")
                  for tok in tokens]
    else:
        tokens = [_simple_lemmatize(tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]

    return " ".join(tokens)


def cleaned_rows_generator_urgency(text: str) -> str:
    """Clean text specifically for urgency detection, keeping urgency/severity indicators"""
    text = _normalize_unicode(text)
    text = text.lower()
    text = _remove_special_chars(text)
    if _HAS_NLTK:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    tokens = [_fix_spelling(tok) for tok in tokens]
    # Lemmatize BEFORE stop-word removal so inflected forms are caught too
    if _HAS_NLTK:
        tokens = [_lemmatizer.lemmatize(_lemmatizer.lemmatize(tok, pos="n"), pos="v")
                  for tok in tokens]
    else:
        tokens = [_simple_lemmatize(tok) for tok in tokens]
    # Use URGENCY-specific stop words (keeps urgency/severity terms)
    tokens = [tok for tok in tokens if tok not in STOP_WORDS_URGENCY and len(tok) > 1]

    return " ".join(tokens)

def main():
    rows_kept = 0
    rows_dropped = 0
    rows_fixed = 0
    cleaned_rows = []
    cleaned_rows_urgency = []

    with open(RAW_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:
                # Row already has all 3 columns: id, 0x000000, text
                idx, marker, text = row
                if idx.strip().isdigit() and text.strip():
                    marker_clean = marker.strip().lower()
                    if not _is_valid_color_marker(marker_clean):
                        marker_clean = "0x000000"
                        rows_fixed += 1
                    text_cleaned = cleaned_rows_generator(text)
                    text_urgency = cleaned_rows_generator_urgency(text)
                    cleaned_rows.append([idx.strip(), marker_clean, text_cleaned.strip()])
                    cleaned_rows_urgency.append([idx.strip(), marker_clean, text_urgency.strip()])
                    rows_kept += 1
                else:
                    # Unexpected format – drop
                    rows_dropped += 1
                    print(f"[DROPPED] row {row[0]}: unexpected 3-column format -> {row}")

            elif len(row) == 2:
                # Row is missing the 0x000000 column: id, text
                idx, text = row
                
                if idx.strip().isdigit() and text.strip():
                    text_cleaned = cleaned_rows_generator(text)
                    text_urgency = cleaned_rows_generator_urgency(text)
                    cleaned_rows.append([idx.strip(), "0x000000", text_cleaned.strip()])
                    cleaned_rows_urgency.append([idx.strip(), "0x000000", text_urgency.strip()])
                    rows_fixed += 1
                else:
                    rows_dropped += 1
                    print(f"[DROPPED] row {row[0]}: missing text or bad id -> {row}")
            else:
                # Wrong number of columns – drop
                rows_dropped += 1
                print(f"[DROPPED] row (cols={len(row)}): {row[:3]}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Write standard cleaned output
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in cleaned_rows:
            writer.writerow(row)

    # Write urgency-specific cleaned output
    with open(OUT_PATH_URGENCY, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in cleaned_rows_urgency:
            writer.writerow(row)

    print("Cleaning Complete")
    print(f"\nDone!  kept={rows_kept}  fixed(added 0x000000)={rows_fixed}  dropped={rows_dropped}")
    print(f"Total rows written: {len(cleaned_rows)}")


if __name__ == "__main__":
    main()
