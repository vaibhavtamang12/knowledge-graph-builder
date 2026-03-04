from coref import resolve_coreferences

tests = [
    "Elon Musk founded Tesla. He later bought Twitter.",
    "Apple released a new iPhone. It sold millions of units.",
    "Sarah joined Google. She became the VP of Engineering.",
    "Jeff Bezos started Amazon. He stepped down as CEO in 2021.",
]

for text in tests:
    resolved = resolve_coreferences(text)
    print(f"  INPUT:    {text}")
    print(f"  RESOLVED: {resolved}")
    print()