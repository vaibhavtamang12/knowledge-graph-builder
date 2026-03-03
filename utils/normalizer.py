def normalize_entity(name: str) -> str:
    """
    Normalize entity names to prevent duplicates.
    """
    return " ".join(name.strip().lower().split())