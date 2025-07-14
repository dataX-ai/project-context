"""
Pydantic data models for type safety and validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RelationshipType(Enum):
    """Types of relationships between contexts"""
    DEPENDS_ON = "depends_on"
    AFFECTS = "affects"


class Node(BaseModel):
    """Knowledge graph node representing a context"""
    context_id: str = Field(..., description="Unique identifier (e.g., 'auth_jwt_token')")
    description: str = Field(..., description="One-two line description of the context")
    category: Optional[str] = Field(None, description="Context category")
    tags: List[str] = Field(default_factory=list, description="Context tags")
    updated_at: datetime = Field(default_factory=datetime.now)
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional node attributes")


class Relationship(BaseModel):
    """Knowledge graph relationship/edge"""
    source_id: str = Field(..., description="Source context ID")
    target_id: str = Field(..., description="Target context ID")
    rel_type: RelationshipType = Field(..., description="Relationship type")
    updated_at: datetime = Field(default_factory=datetime.now)
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional relationship attributes")