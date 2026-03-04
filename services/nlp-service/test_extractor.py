from extractor import extract_entities, extract_relationships

text = "Elon Musk is the CEO of Tesla. Tesla is headquartered in Austin. SpaceX acquired Starlink."

entities = extract_entities(text)
print("\n--- ENTITIES ---")
for e in entities:
    print(f"  {e['text']} [{e['label']}] confidence={e['confidence']}")

relationships = extract_relationships(text, entities)
print("\n--- RELATIONSHIPS ---")
for r in relationships:
    print(f"  {r['subject']} --[{r['type']}]--> {r['object']}  (source={r['source']}, confidence={r['confidence']})")