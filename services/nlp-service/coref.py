"""
coref.py – Coreference Resolution for the NLP service
======================================================

Replaces pronouns and repeated references in text with their canonical
named-entity antecedent before entity/relation extraction runs.

Example
-------
Input:  "Elon Musk founded Tesla. He later bought Twitter."
Output: "Elon Musk founded Tesla. Elon Musk later bought Twitter."

Strategy
--------
We use a two-tier approach:

1. **fastcoref** (preferred) — a lightweight, fast neural coref model
   (`FCoref`) that runs entirely on CPU with no GPU required.
   Install: `pip install fastcoref`

2. **Rule-based fallback** — if fastcoref is not installed, a deterministic
   pronoun resolver is used:
   - Scans each sentence for personal pronouns (he/she/they/it/his/her/…).
   - Walks backwards through the preceding sentences to find the most recent
     named entity whose gender/number matches.
   - Substitutes the pronoun with that entity's text.

The public API is a single function:

    resolve_coreferences(text: str) -> str

It always returns a string (the resolved text, or the original if resolution
fails or finds nothing to change).
"""

from __future__ import annotations
import re
import spacy

# ---------------------------------------------------------------------------
# Model loading (shared with extractor.py via the same process)
# ---------------------------------------------------------------------------
try:
    _nlp = spacy.load("en_core_web_trf")
except OSError:
    _nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# Tier 1: fastcoref neural model
# ---------------------------------------------------------------------------
_fastcoref_model = None
_FASTCOREF_AVAILABLE = False

try:
    from fastcoref import FCoref  # type: ignore
    _fastcoref_model = FCoref()   # downloads ~50 MB model on first run
    _FASTCOREF_AVAILABLE = True
except Exception:
    pass  # fall through to rule-based


def _resolve_with_fastcoref(text: str) -> str:
    """Use fastcoref to replace all coreferent mentions with their cluster head."""
    preds = _fastcoref_model.predict(texts=[text])
    clusters = preds[0].get_clusters(as_strings=False)  # list of list of (start, end) char spans

    if not clusters:
        return text

    # Build a mapping: every non-head span → head span text
    replacements: list[tuple[int, int, str]] = []
    for cluster in clusters:
        if not cluster:
            continue
        # Head = first (longest) mention in cluster
        head_start, head_end = cluster[0]
        head_text = text[head_start:head_end]
        for start, end in cluster[1:]:
            mention = text[start:end]
            # Only replace if the mention looks like a pronoun or short ref
            if mention.lower() != head_text.lower():
                replacements.append((start, end, head_text))

    if not replacements:
        return text

    # Apply replacements from right to left so char offsets stay valid
    replacements.sort(key=lambda x: x[0], reverse=True)
    text_list = list(text)
    for start, end, replacement in replacements:
        text_list[start:end] = list(replacement)

    return "".join(text_list)


# ---------------------------------------------------------------------------
# Tier 2: Rule-based pronoun resolver (fallback)
# ---------------------------------------------------------------------------

# Pronouns → (gender_hint, number)  where gender_hint: M/F/N/U(nknown)
_PRONOUN_MAP: dict[str, tuple[str, str]] = {
    "he":   ("M", "singular"), "him":  ("M", "singular"), "his":  ("M", "singular"),
    "himself": ("M", "singular"),
    "she":  ("F", "singular"), "her":  ("F", "singular"), "hers": ("F", "singular"),
    "herself": ("F", "singular"),
    "they": ("U", "plural"),   "them": ("U", "plural"),   "their": ("U", "plural"),
    "theirs": ("U", "plural"), "themselves": ("U", "plural"),
    "it":   ("N", "singular"), "its":  ("N", "singular"), "itself": ("N", "singular"),
}

# Expanded first-name gender lists for the rule-based heuristic
_MALE_NAMES = {
    "john","james","robert","michael","william","david","richard","joseph",
    "charles","thomas","elon","jeff","mark","tim","bill","george","kevin",
    "brian","edward","ronald","anthony","kenneth","jason","matthew","gary",
    "larry","jeffrey","frank","scott","eric","stephen","andrew","raymond",
    "gregory","samuel","patrick","jack","dennis","walter","peter","harold",
    "ryan","arthur","albert","joe","justin","terry","sean","henry","carl",
    "alex","adam","aaron","zachary","nathan","kyle","tyler","ethan","noah",
    "liam","mason","logan","lucas","oliver","elijah","aiden","jackson",
}
_FEMALE_NAMES = {
    "mary","patricia","jennifer","linda","barbara","elizabeth","jessica",
    "sarah","karen","lisa","susan","nancy","betty","margaret","sandra",
    "ashley","emily","dorothy","kimberly","donna","carol","michelle","amanda",
    "melissa","deborah","stephanie","rebecca","sharon","laura","cynthia",
    "kathleen","amy","angela","shirley","anna","brenda","pamela","emma",
    "nicole","helen","samantha","katherine","christine","debra","rachel",
    "carolyn","janet","catherine","maria","heather","diane","julia","alice",
    "amber","megan","victoria","hannah","sophia","olivia","ava","isabella",
    "mia","charlotte","amelia","harper","evelyn","abigail","ella","scarlett",
}

# Labels that represent people — gendered pronouns must only resolve to these
_PERSON_LABELS = {"PERSON"}
# Labels that "it/its" should resolve to (things, not people)
_THING_LABELS  = {"ORG", "PRODUCT", "GPE", "LOC", "WORK_OF_ART", "EVENT"}


def _gender_of_entity(ent_text: str, ent_label: str) -> str:
    """
    Heuristic gender for a named entity.
    - Non-PERSON entities are always gender-neutral (U) so that
      he/she pronouns never resolve to an ORG or GPE.
    """
    if ent_label != "PERSON":
        return "U"
    first = ent_text.split()[0].lower()
    if first in _MALE_NAMES:
        return "M"
    if first in _FEMALE_NAMES:
        return "F"
    return "U"


def _resolve_rule_based(text: str) -> str:
    """
    Deterministic pronoun → antecedent substitution.

    Key rules enforced:
    - Gendered pronouns (he/she/him/her/his/hers) → PERSON entities only.
    - Neutral pronoun (it/its)                    → non-PERSON entities only.
    - Plural pronouns (they/them/their)            → any entity type.
    - If no valid antecedent exists, the pronoun is left unchanged rather
      than substituting an entity of the wrong type.
    """
    doc = _nlp(text)
    sentences = list(doc.sents)

    # Collect named entities per sentence as (ent_text, ent_label)
    sent_entities: list[list[tuple[str, str]]] = [[] for _ in sentences]
    for ent in doc.ents:
        for idx, sent in enumerate(sentences):
            if ent.start >= sent.start and ent.end <= sent.end:
                sent_entities[idx].append((ent.text, ent.label_))
                break

    resolved_sentences: list[str] = []

    for s_idx, sent in enumerate(sentences):
        sent_text = sent.text

        for token in sent:
            pronoun_lower = token.text.lower()
            if pronoun_lower not in _PRONOUN_MAP:
                continue

            gender_needed, number_needed = _PRONOUN_MAP[pronoun_lower]

            # Determine which entity labels are valid for this pronoun
            if gender_needed in ("M", "F"):
                # he/she/him/her → must be a PERSON
                valid_labels = _PERSON_LABELS
            elif gender_needed == "N":
                # it/its → must NOT be a PERSON
                valid_labels = _THING_LABELS
            else:
                # they/them/their or unknown → any label
                valid_labels = _PERSON_LABELS | _THING_LABELS

            # Search backwards for a compatible antecedent
            antecedent: str | None = None
            for prev_idx in range(s_idx - 1, -1, -1):
                for ent_text, ent_label in reversed(sent_entities[prev_idx]):
                    # Must be a valid label type for this pronoun
                    if ent_label not in valid_labels:
                        continue
                    # Gender check (only meaningful for PERSON entities)
                    ent_gender = _gender_of_entity(ent_text, ent_label)
                    if gender_needed in ("U", "N"):
                        antecedent = ent_text
                        break
                    if ent_gender == gender_needed or ent_gender == "U":
                        antecedent = ent_text
                        break
                if antecedent:
                    break

            if antecedent:
                pattern = r'\b' + re.escape(token.text) + r'\b'
                sent_text = re.sub(pattern, antecedent, sent_text, count=1)

        resolved_sentences.append(sent_text)

    return " ".join(resolved_sentences)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_coreferences(text: str) -> str:
    """
    Resolve coreferences in *text* and return the resolved string.

    Uses fastcoref neural model if available, otherwise falls back to the
    rule-based pronoun resolver. On any error, returns the original text.
    """
    if not text or not text.strip():
        return text

    try:
        if _FASTCOREF_AVAILABLE:
            return _resolve_with_fastcoref(text)
        return _resolve_rule_based(text)
    except Exception as exc:
        # Never let coref errors break the pipeline
        print(f"[coref] Resolution failed, using original text. Reason: {exc}")
        return text