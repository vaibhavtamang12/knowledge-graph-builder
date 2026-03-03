# shared/common/schemas.py

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid


# ==========================
# ENTITY TYPES (LOCKED)
# ==========================

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"
    DATE = "DATE"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"


# ==========================
# RELATION TYPES (LOCKED)
# ==========================

class RelationType(str, Enum):
    CEO_OF = "CEO_OF"
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    HEADQUARTERED_IN = "HEADQUARTERED_IN"
    ACQUIRED = "ACQUIRED"
    PARTNERED_WITH = "PARTNERED_WITH"
    COMPETES_WITH = "COMPETES_WITH"
    INVESTED_IN = "INVESTED_IN"
    MENTIONED_WITH = "MENTIONED_WITH"


# ==========================
# RAW TEXT MESSAGE
# ==========================

class RawTextMessage(BaseModel):
    message_id: str = str(uuid.uuid4())
    source: str
    text: str
    timestamp: datetime


# ==========================
# EXTRACTED ENTITY
# ==========================

class ExtractedEntity(BaseModel):
    entity_id: str = str(uuid.uuid4())
    text: str
    normalized_text: str
    entity_type: EntityType
    confidence: float


# ==========================
# EXTRACTED RELATIONSHIP
# ==========================

class ExtractedRelationship(BaseModel):
    relation_id: str = str(uuid.uuid4())
    subject_entity_id: str
    object_entity_id: str
    relation_type: RelationType
    confidence: float