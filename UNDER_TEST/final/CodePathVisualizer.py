from typing import Dict, List, Optional
import google.generativeai as genai
from dataclasses import dataclass

@dataclass
class PathSelection:
    selected_path: Dict
    confidence_score: float
    reasoning: str
    question_asked: str
    user_response: str


# class CodePathVisualizer:
#     def __init__(self):
#         self.node_counter = 0
#         self.node_ids = {}
    
#     def _get_node_id(self, node_name: str) -> str:
#         """Generate or retrieve unique node ID for Mermaid diagram"""
#         if node_name not in self.node_ids:
#             self.node_ids[node_name] = f"node{self.node_counter}"
#             self.node_counter += 1
#         return self.node_ids[node_name]
    
#     def _get_node_style(self, node_type: str) -> str:
#         """Get node style based on entity type"""
#         styles = {
#             'CodeEntity': 'class',
#             'ODL': 'object',
#             'Object': 'object',
#             'Parameter': 'parameter',
#             'EventHandler': 'event',
#             'Validator': 'validator'
#         }
#         return styles.get(node_type, 'default')

#     def generate_mermaid(self, path_selection: PathSelection) -> str:
#         """Convert selected path to Mermaid diagram"""
#         mermaid_lines = ['graph TD;']
        
#         # Add style definitions
#         mermaid_lines.extend([
#             '    classDef default fill:#f9f,stroke:#333,stroke-width:2px;',
#             '    classDef class fill:#69c,stroke:#333,stroke-width:2px;',
#             '    classDef object fill:#fc9,stroke:#333,stroke-width:2px;',
#             '    classDef parameter fill:#9c6,stroke:#333,stroke-width:2px;',
#             '    classDef event fill:#c69,stroke:#333,stroke-width:2px;',
#             '    classDef validator fill:#96c,stroke:#333,stroke-width:2px;'
#         ])
        
#         # Process each step in the path
#         for step in path_selection.selected_path['chain']['path_sequence']:
#             from_id = self._get_node_id(step['from']['name'])
#             to_id = self._get_node_id(step['to']['name'])
            
#             # Add nodes with labels
#             from_style = self._get_node_style(step['from']['type'])
#             to_style = self._get_node_style(step['to']['type'])
            
#             mermaid_lines.extend([
#                 f'    {from_id}["{step["from"]["type"]}<br/>{step["from"]["name"]}"]:::{"default" if from_style == "default" else from_style}',
#                 f'    {to_id}["{step["to"]["type"]}<br/>{step["to"]["name"]}"]:::{"default" if to_style == "default" else to_style}'
#             ])
            
#             # Add relationship with line number if available
#             label = f' |{step["line_number"]}|' if step['line_number'] else ''
#             mermaid_lines.append(f'    {from_id} -->"{step["relationship"]}{label}" {to_id}')
        
#         return '\n'.join(mermaid_lines)

class CodePathVisualizer:
    def __init__(self):
        self.node_counter = 0
        self.node_ids = {}
    
    def _get_node_id(self, node_name: str) -> str:
        """Generate or retrieve unique node ID for Mermaid diagram"""
        if node_name not in self.node_ids:
            self.node_ids[node_name] = f"node{self.node_counter}"
            self.node_counter += 1
        return self.node_ids[node_name]
    
    def _get_node_style(self, node_type: str) -> str:
        """Get node style based on entity type"""
        styles = {
            'CodeEntity': 'codeEntity',
            'ODL': 'objectNode',
            'Object': 'objectNode',
            'Parameter': 'paramNode',
            'EventHandler': 'eventNode',
            'Validator': 'validatorNode'
        }
        return styles.get(node_type, 'defaultNode')

    def generate_mermaid(self, path_selection: PathSelection) -> str:
        """Convert selected path to Mermaid diagram"""
        # Track nodes by style for grouping
        nodes_by_style = {}
        
        # Build diagram parts
        nodes = []
        relationships = []
        
        # Process path sequence
        for step in path_selection.selected_path['chain']['path_sequence']:
            from_id = self._get_node_id(step['from']['name'])
            to_id = self._get_node_id(step['to']['name'])
            
            # Add nodes with their styles
            from_style = self._get_node_style(step['from']['type'])
            to_style = self._get_node_style(step['to']['type'])
            
            # Group nodes by style
            nodes_by_style.setdefault(from_style, set()).add(from_id)
            nodes_by_style.setdefault(to_style, set()).add(to_id)
            
            # Add node definitions
            nodes.extend([
                f'{from_id}["{step["from"]["type"]}<br/>{step["from"]["name"]}"]',
                f'{to_id}["{step["to"]["type"]}<br/>{step["to"]["name"]}"]'
            ])
            
            # Add relationship
            line_info = f'<br/>line:{step["line_number"]}' if step['line_number'] else ''
            relationships.append(
                f'{from_id} -->|"{step["relationship"]}{line_info}"| {to_id}'
            )
        
        # Combine all parts into final diagram
        mermaid_parts = [
            'graph TD',
            # '%% Styles',
            # 'classDef codeEntity fill:#69c,stroke:#333,stroke-width:2',
            # 'classDef objectNode fill:#fc9,stroke:#333,stroke-width:2',
            # 'classDef paramNode fill:#9c6,stroke:#333,stroke-width:2',
            # 'classDef eventNode fill:#c69,stroke:#333,stroke-width:2',
            # 'classDef validatorNode fill:#96c,stroke:#333,stroke-width:2',
            # 'classDef defaultNode fill:#f9f,stroke:#333,stroke-width:2',
            # '',
            '%% Nodes',
            *nodes,
            '',
            '%% Relationships',
            *relationships,
            '',
            '%% Styling'
        ]
        
        # Add style applications
        for style, node_ids in nodes_by_style.items():
            if node_ids:  # Only add if there are nodes with this style
                mermaid_parts.append(f'class {",".join(node_ids)} {style}')
        
        return '\n'.join(mermaid_parts)