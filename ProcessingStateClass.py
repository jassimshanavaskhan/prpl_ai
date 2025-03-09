import os
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field

@dataclass
class ProcessingState:
    """Track processing state for resumable operations"""
    processed_files: Set[str] = field(default_factory=set)
    current_component: str = ""
    processed_components: Set[str] = field(default_factory=set)
    last_update: datetime = field(default_factory=datetime.now)
    
    def save(self, path: str = "processing_state.json"):
        state = {
            "processed_files": list(self.processed_files),
            "current_component": self.current_component,
            "processed_components": list(self.processed_components),
            "last_update": self.last_update.isoformat()
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    @classmethod
    def load(cls, path: str = "processing_state.json") -> 'ProcessingState':
        if not os.path.exists(path):
            return cls()
        with open(path, 'r') as f:
            state = json.load(f)
        return cls(
            processed_files=set(state["processed_files"]),
            current_component=state["current_component"],
            processed_components=set(state["processed_components"]),
            last_update=datetime.fromisoformat(state["last_update"])
        )
