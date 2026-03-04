# services/nlp-service/test_entity_linker.py
from entity_linker import enrich_entities

entities = [
    {"text": "Elon Musk", "normalized_text": "elon musk", "label": "PERSON", "confidence": 0.9},
    {"text": "Tesla",     "normalized_text": "tesla",     "label": "ORG",    "confidence": 0.9},
    {"text": "Austin",    "normalized_text": "austin",    "label": "GPE",    "confidence": 0.9},
    {"text": "Chicago Bulls", "normalized_text": "chicago bulls", "label": "ORG", "confidence": 0.9},
]

enriched = enrich_entities(entities)
for e in enriched:
    print(f"\n{e['text']}")
    print(f"  Wikipedia: {e.get('wikipedia_url')}")
    print(f"  Wikidata:  {e.get('wikidata_id')}")
    print(f"  Desc:      {e.get('description')}")
    print(f"  Metadata:  {e.get('metadata')}")
    print(f"  Confidence:{e.get('link_confidence')}")