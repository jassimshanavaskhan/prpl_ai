from typing import Dict, List, Set, Any, Optional, Tuple
from VectorStoreManager import VectorStoreManager
from CodeEntityClass import CodeEntity

class SequenceDiagramGenerator:
    def __init__(self, vector_store: VectorStoreManager, entities: Dict[str, CodeEntity]):
        self.vector_store = vector_store
        self.entities = entities
        self.visited = set()
        
    def generate(self, query: str, max_depth: int = 5) -> str:
        """Generate sequence diagram based on natural language query"""
        # Search for relevant functions
        relevant_functions = self.vector_store.search(
            query,
            store_type='function',
            k=3
        )
        # print("\n---->JAS13 Relevant functions fetching...")
        # print("\n --------------> relevant_functions : ", relevant_functions)
        if not relevant_functions:
            return "No relevant functions found for the query."
        
        # Start with the most relevant function
        # print("\n --------------> relevant_functions[0] : ", relevant_functions[0])
        # Access metadata through the document object
        start_function_name = relevant_functions[0]['document'].metadata['name']
        # print("\n --------------> start_function_name : ", start_function_name)
        start_function = self.entities.get(start_function_name)
        # print("\n---->JAS14 start_function : ", start_function)
        if not start_function:
            return "Could not find the starting function."
        
        # Generate diagram
        diagram_lines = ["sequenceDiagram"]
        self.visited.clear()
        self._add_sequence(start_function, diagram_lines, depth=0, max_depth=max_depth)
        
        return "\n".join(diagram_lines)
    
    def _add_sequence(self, entity: CodeEntity, diagram_lines: List[str], 
                     depth: int, max_depth: int, parent: str = None):
        """Recursively add sequence interactions"""
        if depth >= max_depth or entity.name in self.visited:
            return
        
        self.visited.add(entity.name)
        
        for call in entity.function_calls:
            called_entity = self.entities.get(call.function_name)
            if called_entity:
                # Add participants if not already added
                diagram_lines.append(
                    f"participant {entity.component} as {entity.component}"
                )
                diagram_lines.append(
                    f"participant {called_entity.component} as {called_entity.component}"
                )
                
                # Add interaction
                params_str = ", ".join(call.parameters)
                diagram_lines.append(
                    f"{entity.component}->>+{called_entity.component}: "
                    f"{call.function_name}({params_str})"
                )
                
                # Recurse
                self._add_sequence(called_entity, diagram_lines, 
                                 depth + 1, max_depth, entity.name)
                
                # Add return
                diagram_lines.append(
                    f"{called_entity.component}-->>-{entity.component}: return"
                )
