from typing import Dict, List, Set, Any, Optional, Tuple
from CodeEntityClass import CodeEntity
import json

def load_entities(filepath: str) -> Dict[str, CodeEntity]:
    """Load entities from file"""
    with open(filepath, 'r') as f:
        entities_dict = json.load(f)
    return {
        name: CodeEntity.from_dict(entity_data)
        for name, entity_data in entities_dict.items()
    }