from typing import Dict, List, Set, Any, Optional, Tuple
from CodeEntityClass import CodeEntity
import json


def save_entities(entities: Dict[str, CodeEntity], filepath: str):
    """Save entities to file using JSON serialization"""
    entities_dict = {
        name: entity.to_dict() 
        for name, entity in entities.items()
    }
    with open(filepath, 'w') as f:
        json.dump(entities_dict, f)
