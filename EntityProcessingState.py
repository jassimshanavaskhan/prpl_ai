from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class EntityProcessingState:
    """Track processing state for entity responses"""
    entity_name: str
    component: str
    response: Optional[str] = None
    processed_at: Optional[datetime] = None
    status: str = "pending"  # pending, completed, failed
    retry_count: int = 0