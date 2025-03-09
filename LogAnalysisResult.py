from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from datetime import datetime

@dataclass
class EntityMention:
    name: str
    type: str  # 'function', 'api', 'component', 'struct'
    count: int
    context: str
    confidence: float  # 0.0 to 1.0

@dataclass
class ErrorPattern:
    pattern: str
    frequency: int
    examples: List[str]
    related_components: List[str]

@dataclass
class LogAnalysisResult:
    mentioned_entities: Dict[str, EntityMention]  # name -> EntityMention
    error_patterns: List[ErrorPattern]
    code_paths: List[List[str]]  # List of function call sequences
    system_state: Dict[str, str]
    log_entries: List[Dict]  # Original log entries that were analyzed
    timestamp_range: Optional[tuple[datetime, datetime]]
    
    @classmethod
    def empty(cls) -> 'LogAnalysisResult':
        """Create an empty analysis result"""
        return cls(
            mentioned_entities={},
            error_patterns=[],
            code_paths=[],
            system_state={},
            log_entries=[],
            timestamp_range=None
        )
    
    @classmethod
    def from_gemini_response(cls, response_text: str, log_results: List[Dict]) -> 'LogAnalysisResult':
        """Parse Gemini's response into a structured analysis result"""
        try:
            # Initialize containers
            mentioned_entities = {}
            error_patterns = []
            code_paths = []
            system_state = {}
            
            # Parse timestamps from log results
            timestamps = [
                datetime.fromisoformat(log['metadata']['timestamp'])
                for log in log_results
                if 'timestamp' in log['metadata']
            ]
            timestamp_range = (min(timestamps), max(timestamps)) if timestamps else None
            
            # Parse the response text sections
            sections = cls._split_response_into_sections(response_text)
            
            # Process each section
            if 'ENTITIES' in sections:
                mentioned_entities = cls._parse_entities_section(sections['ENTITIES'])
            
            if 'ERROR_PATTERNS' in sections:
                error_patterns = cls._parse_error_patterns(sections['ERROR_PATTERNS'])
            
            if 'CODE_PATHS' in sections:
                code_paths = cls._parse_code_paths(sections['CODE_PATHS'])
            
            if 'SYSTEM_STATE' in sections:
                system_state = cls._parse_system_state(sections['SYSTEM_STATE'])
            
            return cls(
                mentioned_entities=mentioned_entities,
                error_patterns=error_patterns,
                code_paths=code_paths,
                system_state=system_state,
                log_entries=log_results,
                timestamp_range=timestamp_range
            )
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            return cls.empty()
    
    @staticmethod
    def _split_response_into_sections(response_text: str) -> Dict[str, str]:
        """Split the response text into labeled sections"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line.upper().endswith(':'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[:-1].upper()
                current_content = []
            elif line and current_section:
                current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    @staticmethod
    def _parse_entities_section(section_text: str) -> Dict[str, EntityMention]:
        """Parse the entities section of the response"""
        entities = {}
        current_entity = None
        entity_type = None
        context_lines = []
        
        for line in section_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):  # New entity
                if current_entity:  # Save previous entity
                    entities[current_entity] = EntityMention(
                        name=current_entity,
                        type=entity_type or 'unknown',
                        count=1,  # Default count
                        context='\n'.join(context_lines),
                        confidence=0.8  # Default confidence
                    )
                
                # Parse new entity
                parts = line[2:].split('(')
                current_entity = parts[0].strip()
                entity_type = parts[1].rstrip(')').strip() if len(parts) > 1 else None
                context_lines = []
            elif line and current_entity:
                context_lines.append(line)
        
        # Save last entity
        if current_entity:
            entities[current_entity] = EntityMention(
                name=current_entity,
                type=entity_type or 'unknown',
                count=1,
                context='\n'.join(context_lines),
                confidence=0.8
            )
        
        return entities
    
    @staticmethod
    def _parse_error_patterns(section_text: str) -> List[ErrorPattern]:
        """Parse the error patterns section of the response"""
        patterns = []
        lines = section_text.split('\n')
        current_pattern = None
        examples = []
        components = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                if current_pattern:  # Save previous pattern
                    patterns.append(ErrorPattern(
                        pattern=current_pattern,
                        frequency=1,
                        examples=examples,
                        related_components=components
                    ))
                current_pattern = line[2:]
                examples = []
                components = []
            elif line.startswith('  Example:'):
                examples.append(line.replace('  Example:', '').strip())
            elif line.startswith('  Component:'):
                components.append(line.replace('  Component:', '').strip())
        
        # Save last pattern
        if current_pattern:
            patterns.append(ErrorPattern(
                pattern=current_pattern,
                frequency=1,
                examples=examples,
                related_components=components
            ))
        
        return patterns
    
    @staticmethod
    def _parse_code_paths(section_text: str) -> List[List[str]]:
        """Parse the code paths section of the response"""
        paths = []
        for line in section_text.split('\n'):
            if line.strip().startswith('- '):
                path = [
                    func.strip()
                    for func in line[2:].split('->')
                ]
                paths.append(path)
        return paths
    
    @staticmethod
    def _parse_system_state(section_text: str) -> Dict[str, str]:
        """Parse the system state section of the response"""
        state = {}
        for line in section_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                state[key.strip()] = value.strip()
        return state
    
    def get_mentioned_functions(self) -> List[str]:
        """Get list of mentioned function names"""
        return [
            name 
            for name, mention in self.mentioned_entities.items()
            if mention.type.lower() == 'function'
        ]
    
    def get_mentioned_components(self) -> List[str]:
        """Get list of mentioned component names"""
        components = set()
        # Add components from entity mentions
        for mention in self.mentioned_entities.values():
            if mention.type.lower() == 'component':
                components.add(mention.name)
        # Add components from error patterns
        for pattern in self.error_patterns:
            components.update(pattern.related_components)
        return list(components)
    
    def get_summary(self) -> str:
        """Generate a summary of the analysis results"""
        summary_parts = []
        
        # Add timestamp range if available
        if self.timestamp_range:
            start, end = self.timestamp_range
            summary_parts.append(f"Time Range: {start} to {end}")
        
        # Add mentioned functions
        functions = self.get_mentioned_functions()
        if functions:
            summary_parts.append(f"Functions: {', '.join(functions)}")
        
        # Add components
        components = self.get_mentioned_components()
        if components:
            summary_parts.append(f"Components: {', '.join(components)}")
        
        # Add error patterns
        if self.error_patterns:
            summary_parts.append("Error Patterns:")
            for pattern in self.error_patterns:
                summary_parts.append(f"- {pattern.pattern}")
                if pattern.examples:
                    summary_parts.append(f"  Example: {pattern.examples[0]}")
        
        # Add code paths
        if self.code_paths:
            summary_parts.append("Code Paths:")
            for path in self.code_paths[:2]:  # Limit to first 2 paths
                summary_parts.append(f"- {' -> '.join(path)}")
        
        # Add system state
        if self.system_state:
            summary_parts.append("System State:")
            for key, value in list(self.system_state.items())[:3]:  # Limit to first 3 states
                summary_parts.append(f"- {key}: {value}")
        
        return "\n".join(summary_parts)