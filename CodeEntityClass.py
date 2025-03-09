from dataclasses import dataclass, field
from typing import Dict, List, Any
import hashlib
from FunctionCallClass import FunctionCall


@dataclass
class CodeEntity:
    name: str
    type: str
    content: str
    file_path: str
    component: str
    description: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    structs_used: List[str] = field(default_factory=list)
    api_calls: List[str] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = field(init=False)

    def __hash__(self):
        return hash(self.content_hash)

    def __eq__(self, other):
        return isinstance(other, CodeEntity) and self.content_hash == other.content_hash
    
    def __post_init__(self):
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()
    
    def to_embedding_text(self) -> str:
        """Convert entity to text format for embedding"""
        text_parts = [
            f"Name: {self.name}",
            f"Type: {self.type}",
            f"Component: {self.component}",
            f"Description: {self.description}",
            "Function Calls: " + ", ".join(
                f"{call.function_name} ({call.component}) -> {call.return_type}" 
                for call in self.function_calls
            ),
            "Structs Used: " + ", ".join(self.structs_used),
            "API Calls: " + ", ".join(self.api_calls),
            "Context:",
            self.content
        ]
        return "\n".join(text_parts)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.type,
            "content": self.content,
            "file_path": str(self.file_path),
            "component": self.component,
            "description": self.description,
            "function_calls": [
                {
                    "function_name": fc.function_name,
                    "component": fc.component,
                    "parameters": fc.parameters,
                    "return_type": fc.return_type,
                    "is_api": fc.is_api,
                    "line_number": fc.line_number,
                    "context_before": fc.context_before,
                    "context_after": fc.context_after
                }
                for fc in self.function_calls
            ],
            "structs_used": list(self.structs_used),
            "api_calls": list(self.api_calls),
            "includes": list(self.includes),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeEntity':
        """Create instance from dictionary"""
        # Create a copy of data to modify
        data_copy = data.copy()
        
        # Remove content_hash if it exists since it's calculated in __post_init__
        data_copy.pop('content_hash', None)
        
        # Convert function calls back to FunctionCall objects
        function_calls_data = data_copy.pop("function_calls", [])
        
        # Create the entity without function calls first
        entity = cls(**data_copy)
        
        # Add function calls separately
        entity.function_calls = [FunctionCall(**fc) for fc in function_calls_data]
        
        return entity