# odl_def_parser.py
import re
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum, auto
from neo4j import GraphDatabase
from typing import Dict, Any, Optional
from dataclasses import asdict

class ParameterType(Enum):
    BOOL = auto()
    STRING = auto()
    CSV_STRING = auto()
    OBJECT = auto()
    OBJECT_ARRAY = auto()
    DATETIME = auto()
    UINT32 = auto()
    UINT64 = auto()
    INT32 = auto()
    VARIANT = auto()
    VOID = auto()

@dataclass
class Parameter:
    name: str
    param_type: ParameterType
    attributes: List[str]
    default_value: Any
    validators: List[Dict[str, Any]]
    counted_with: str = None
    userflags: List[str] = None
    actions: Dict[str, str] = None  # For actions like 'read'

@dataclass
class EventHandler:
    event_type: str
    callback: str
    filter_expr: str = None




def map_type_string_to_enum(type_str: str) -> ParameterType:
    """Map string type to ParameterType enum"""
    type_mapping = {
        'bool': ParameterType.BOOL,
        'string': ParameterType.STRING,
        'csv_string': ParameterType.CSV_STRING,
        'object': ParameterType.OBJECT,
        'datetime': ParameterType.DATETIME,
        'uint32': ParameterType.UINT32,
        'uint64': ParameterType.UINT64,
        'int32': ParameterType.INT32,
        'variant': ParameterType.VARIANT,
        'void': ParameterType.VOID
    }
    return type_mapping.get(type_str.lower(), ParameterType.STRING)

class ODLDefinitionParser:
    def __init__(self, content: str, filepath: str = None):
        self.content = ' '.join(line.strip() for line in content.split('\n'))
        self.filepath = filepath
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: List[EventHandler] = []

    def extract_all_sections(self, section_name: str) -> List[str]:
        """Extract all sections with the given name from the content"""
        sections = []
        current_pos = 0
        
        while True:
            # Find the next section marker
            start_match = re.search(f'%{section_name}\\s*{{', self.content[current_pos:])
            if not start_match:
                break
                
            start_pos = current_pos + start_match.end()
            brace_count = 1
            current_pos = start_pos
            
            # Find matching closing brace
            while brace_count > 0 and current_pos < len(self.content):
                if self.content[current_pos] == '{':
                    brace_count += 1
                elif self.content[current_pos] == '}':
                    brace_count -= 1
                current_pos += 1
                
            if brace_count == 0:
                section_content = self.content[start_pos:current_pos-1].strip()
                sections.append(section_content)
            else:
                raise ValueError(f"Unmatched braces in {section_name} section")
        
        return sections

    def extract_section(self, section_name: str) -> str:
        """Extract content between section marker and matching closing brace"""
        start_match = re.search(f'%{section_name}\\s*{{', self.content)
        if not start_match:
            return ""
        
        start_pos = start_match.end()
        brace_count = 1
        current_pos = start_pos
        
        while brace_count > 0 and current_pos < len(self.content):
            if self.content[current_pos] == '{':
                brace_count += 1
            elif self.content[current_pos] == '}':
                brace_count -= 1
            current_pos += 1
            
        return self.content[start_pos:current_pos-1].strip()
    
    def parse_userflags(self, param_text: str) -> List[str]:
        """Extract userflags from parameter definition"""
        flags = []
        flag_match = re.search(r'userflags\s+%(\w+)', param_text)
        if flag_match:
            flags.append(flag_match.group(1))
        return flags

    def parse_parameter_type(self, line: str) -> ParameterType:
        """Determine parameter type from definition line"""
        line = line.lower()
        if "bool" in line:
            return ParameterType.BOOL
        elif "uint64" in line:
            return ParameterType.UINT64
        elif "datetime" in line:
            return ParameterType.DATETIME
        elif "csv_string" in line:
            return ParameterType.CSV_STRING
        elif "string" in line:
            return ParameterType.STRING
        elif "object" in line and "[]" in line:
            return ParameterType.OBJECT_ARRAY
        elif "object" in line:
            return ParameterType.OBJECT
        return ParameterType.STRING

    def parse_parameter_attributes(self, attributes_text: str) -> List[str]:
        """Extract parameter attributes from the attributes group"""
        attributes = []
        if not attributes_text:
            return attributes
            
        # Split the attributes text by whitespace and process each attribute
        for attr in attributes_text.split():
            if attr.startswith('%'):
                attr = attr[1:]  # Remove the % prefix
                if attr in ["persistent", "protected", "read-only", "unique", "key", "volatile", "async"]:
                    attributes.append(attr)
        return attributes

    def extract_default_value(self, param_block: str) -> str:
        """Extract default value with improved pattern matching"""
        # Check for explicit default statement first
        default_stmt = re.search(r'default\s+"([^"]*)"', param_block)
        if default_stmt:
            return default_stmt.group(1).strip()
        
        # Then check for assignment
        default_assign = re.search(r'=\s*"([^"]*)"', param_block)
        if default_assign:
            return default_assign.group(1).strip()
        
        return ""

    def parse_parameter_validators(self, param_block: str) -> List[Dict[str, Any]]:
        """Extract validators with expanded pattern matching"""
        validators = []
        
        # First check for traditional check_enum with square brackets
        enum_matches = re.finditer(r'on\s+action\s+validate\s+call\s+check_enum\s*\[(.*?)\]', param_block)
        for match in enum_matches:
            values = [v.strip('" ') for v in match.group(1).split(',')]
            validators.append({"type": "check_enum", "values": values})
        
        # Check for check_is_in with quotes
        is_in_matches = re.finditer(r'on\s+action\s+validate\s+call\s+check_is_in\s+"([^"]+)"', param_block)
        for match in is_in_matches:
            validators.append({"type": "check_is_in", "reference": match.group(1)})
        
        # Check for check_maximum with direct number
        max_matches = re.finditer(r'on\s+action\s+validate\s+call\s+check_maximum\s+(\d+)', param_block)
        for match in max_matches:
            validators.append({"type": "check_maximum", "value": int(match.group(1))})
        
        # Check for check_is_empty_or_enum
        empty_enum_matches = re.finditer(r'on\s+action\s+validate\s+call\s+check_is_empty_or_enum\s*\[(.*?)\]', param_block)
        for match in empty_enum_matches:
            values = [v.strip('" ') for v in match.group(1).split(',')]
            validators.append({"type": "check_is_empty_or_enum", "values": values})
        
        return validators
    
    def parse_actions(self, param_block: str) -> Dict[str, str]:
        """Extract other actions like read"""
        actions = {}
        action_match = re.finditer(r'on\s+action\s+(\w+)\s+call\s+(\w+)', param_block)
        for match in action_match:
            action_type = match.group(1)
            if action_type != 'validate':  # Skip validate actions as they're handled separately
                actions[action_type] = match.group(2)
        return actions

    def parse_object_block(self, content: str, current_pos: int) -> tuple[str, int]:
        """Parse a block within braces, handling nested braces"""
        brace_count = 1
        start_pos = current_pos
        
        while brace_count > 0 and current_pos < len(content):
            if content[current_pos] == '{':
                brace_count += 1
            elif content[current_pos] == '}':
                brace_count -= 1
            current_pos += 1
                
        return content[start_pos:current_pos-1], current_pos

    def parse_object_content(self, object_name: str, content: str) -> Dict[str, Any]:
        """
        Parse object content including nested objects with improved parameter capture.
        Handles both simple object declarations and objects with content blocks.
        
        Args:
            object_name: Name of the object being parsed
            content: Content string to parse
            
        Returns:
            Dictionary containing parsed object definition
        """
        object_def = {
            "name": object_name,
            "parameters": {},
            "nested_objects": {},
            "is_array": "[]" in object_name,
            "actions": []  # New field to store actions
        }
        
        # Handle empty content for simple object declarations
        if not content.strip():
            return object_def
            
        if object_def["is_array"]:
            counted_match = re.search(r'counted\s+with\s+(\w+)', content)
            if counted_match:
                object_def["counted_with"] = counted_match.group(1)

        # Add action pattern to capture action definitions
        action_pattern = r'on\s+action\s+(\w+)\s+call\s+(\w+)\s*;'
        
        param_pattern = r"""
        ((?:%(?:persistent|protected|read-only|volatile|async|unique|key)\s+)*)((?:csv_string|string|variant|uint32|uint64|int32|datetime|void|bool))\s+((?:"?\${[^}]+}[^"]*"|'[^']*'|\w+))(?:\s*\((?:%(?:in|out|inout)\s+%?(?:(?:strict|mandatory))?\s*(?:csv_string|string|variant|uint32|uint64|int32|datetime|htable|list|bool)\s+\w+)(?:\s*,\s*%(?:in|out|inout)\s+%?(?:(?:strict|mandatory))?\s*(?:csv_string|string|variant|uint32|uint64|int32|datetime|htable|list|bool)\s+\w+)*\s*\))?(?:\s*=\s*((?:"[^"]*"|\d+|"${[^}]+}"|true|false|"?\$\([^)]+\)")))?(?:\s*{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*((?:default\s+([^;]+));)?[^}]*})?(?:\s*userflags\s+%\w+)?;?
        """

        param_pattern = re.compile(param_pattern, re.VERBOSE)
        object_pattern = r'(?:(?:%persistent|%read-only)\s+)*object\s+((?:[\'"]?\$\{[^}]+}[^\'\"]*[\'"]|[^{\s]+))(?:\[\])?\s*{'
        simple_object_pattern = r'(?:(?:%persistent|%read-only)\s+)*object\s+((?:[\'"]?\$\{[^}]+}[^\'\"]*[\'"]|[^{\s]+))(?:\[\])?\s*;'
        
        current_pos = 0
        while current_pos < len(content):
            # Try to find either a parameter, object, or action
            param_match = param_pattern.search(content[current_pos:])
            object_match = re.search(object_pattern, content[current_pos:])
            simple_object_match = re.search(simple_object_pattern, content[current_pos:])
            action_match = re.search(action_pattern, content[current_pos:])
            
            matches = []
            if param_match:
                matches.append((param_match.start(), 'param', param_match))
            if object_match:
                matches.append((object_match.start(), 'object', object_match))
            if simple_object_match:
                matches.append((simple_object_match.start(), 'simple_object', simple_object_match))
            if action_match:
                matches.append((action_match.start(), 'action', action_match))
            
            if not matches:
                current_pos += 1
                continue
                
            matches.sort(key=lambda x: x[0])
            match_type = matches[0][1]
            match = matches[0][2]
            
            if match_type == 'action':
                action = {
                    'event': match.group(1),
                    'handler': match.group(2)
                }
                object_def['actions'].append(action)
                current_pos += match.end()
            elif match_type in ['object', 'simple_object']:
                nested_name = match.group(1)
                current_pos += match.end()
                
                if match_type == 'simple_object':
                    # For simple objects without content blocks
                    object_def["nested_objects"][nested_name] = self.parse_object_content(nested_name, "")
                else:
                    # For objects with content blocks
                    nested_content, new_pos = self.parse_object_block(content, current_pos)
                    object_def["nested_objects"][nested_name] = self.parse_object_content(nested_name, nested_content)
                    current_pos = new_pos
            else:  # param match
                full_param_start = current_pos + match.start()
                full_param_text = match.group(0)
                
                attributes_text = match.group(1).strip() if match.group(1) else ""
                type_str = match.group(2)
                param_name = match.group(3)
                
                if param_name:
                    parameter = Parameter(
                        name=param_name,
                        param_type=map_type_string_to_enum(type_str),
                        attributes=self.parse_parameter_attributes(attributes_text),
                        default_value=self.extract_default_value(full_param_text),
                        validators=self.parse_parameter_validators(full_param_text),
                        userflags=self.parse_userflags(full_param_text),
                        actions=self.parse_actions(full_param_text)
                    )
                    object_def["parameters"][param_name] = parameter
                
                current_pos += match.end()
        
        return object_def

    def parse_populate_section(self, content: str):
        """Parse populate section and extract event handlers"""
        event_pattern = r'on\s+event\s+"([^"]+)"\s+call\s+(\w+)(?:\s+filter\s+\'([^\']+)\')?'
        
        for match in re.finditer(event_pattern, content):
            self.event_handlers.append(EventHandler(
                event_type=match.group(1),
                callback=match.group(2),
                filter_expr=match.group(3)
            ))

    def parse_select_content(self, select_name: str, select_content: str):
        """Parse content within a select block"""
        # Remove any leading/trailing whitespace
        select_content = select_content.strip()
        
        # Parse objects within the select block
        current_pos = 0
        while current_pos < len(select_content):
            obj_match =  re.search(r'(?:(?:%persistent|%read-only)\s+)*object\s+((?:[\'"]?\$\{[^}]+}[^\'\"]*[\'"]|[^{\s]+))(?:\[\])?\s*{?',select_content[current_pos:])
            if not obj_match:
                break
                
            object_name = f"{select_name}.{obj_match.group(1)}"  # Prefix with select name
            current_pos += obj_match.end()
            object_content, new_pos = self.parse_object_block(select_content, current_pos)
            self.objects[object_name] = self.parse_object_content(object_name, object_content)
            current_pos = new_pos

    def parse(self) -> Dict[str, Any]:
        """Parse the complete definition ODL file, handling multiple define sections"""
        # Get all define sections
        define_sections = self.extract_all_sections('define')
        
        # Parse each define section
        for define_content in define_sections:
            current_pos = 0
            while True:
                # Find the start of the next select statement
                select_match = re.search(r'select\s+(\S+)\s*{', define_content[current_pos:])
                if not select_match:
                    break
                    
                select_name = select_match.group(1)
                content_start = current_pos + select_match.end()
                
                # Find the matching closing brace
                brace_count = 1
                content_end = content_start
                
                while brace_count > 0 and content_end < len(define_content):
                    if define_content[content_end] == '{':
                        brace_count += 1
                    elif define_content[content_end] == '}':
                        brace_count -= 1
                    content_end += 1
                    
                if brace_count == 0:
                    # Extract the content between braces
                    select_content = define_content[content_start:content_end-1].strip()
                    self.parse_select_content(select_name, select_content)
                    current_pos = content_end
                else:
                    raise ValueError(f"Unmatched braces in select statement: {select_name}")

            # Parse non-select objects in this define section
            current_pos = 0
            while current_pos < len(define_content):
                # Skip any select statements we've already processed
                select_match = re.search(r'select\s+(\S+)\s*{', define_content[current_pos:])
                if select_match and select_match.start() == 0:
                    # Find the end of this select block
                    select_content_start = current_pos + select_match.end()
                    brace_count = 1
                    content_end = select_content_start
                    
                    while brace_count > 0 and content_end < len(define_content):
                        if define_content[content_end] == '{':
                            brace_count += 1
                        elif define_content[content_end] == '}':
                            brace_count -= 1
                        content_end += 1
                    
                    current_pos = content_end
                    continue

                obj_match = re.search(r'(?:(?:%persistent|%read-only)\s+)*object\s+((?:[\'"]?\$\{[^}]+}[^\'\"]*[\'"]|[^{\s]+))(?:\[\])?\s*{', 
                                    define_content[current_pos:])
                
                if not obj_match:
                    break
                    
                object_name = obj_match.group(1)
                current_pos += obj_match.end()
                object_content, new_pos = self.parse_object_block(define_content, current_pos)
                self.objects[object_name] = self.parse_object_content(object_name, object_content)
                current_pos = new_pos

        # Parse all populate sections
        populate_sections = self.extract_all_sections('populate')
        for populate_content in populate_sections:
            self.parse_populate_section(populate_content)
        
        return {
            "filepath": self.filepath,
            "objects": self.objects,
            "event_handlers": self.event_handlers
        }