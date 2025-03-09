from CFunctionParserClass import CFunctionParser
from FunctionCallClass import FunctionCall
from logger import logger
from CodeEntityClass import CodeEntity
import re
import networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple

class EnhancedCodeParser:
    def __init__(self):
        self.call_graph = nx.DiGraph()
        self._compile_patterns()
        self.known_struct_types = set()  # Initialize empty set for known struct types
        self.type_definitions = {}  # Store typedef mappings
        
    def _compile_patterns(self):
        # Your existing patterns...
        self.patterns = {
            'function': re.compile(
                r'^(?:static\s+)?'  # Optional static
                r'(?:inline\s+)?'   # Optional inline
                r'(?:extern\s+)?'   # Optional extern
                r'(?!if\b|for\b|while\b|switch\b)'  # Exclude control statements
                r'(?:\w+\s*\*?\s+)+' # Return type(s)
                r'(\w+)\s*\(',      # Function name and opening parenthesis
                re.MULTILINE
            ),
            'union': re.compile(
                r'union\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*(\w+);',
                re.MULTILINE | re.DOTALL
            ),
            #-------------
            #Updated struct pattern to handle both direct struct definitions and typedefs
            'struct': re.compile(
                r'(?:typedef\s+)?struct\s*'  # Optional typedef and struct keyword
                r'(_?\w+)?\s*'               # Optional struct name with possible underscore
                r'\{'                        # Opening brace
                r'([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'  # Struct body
                r'\s*(_struct_pack_)?'       # Optional _struct_pack_ attribute
                r'(?:\s*(\w+))?;',           # Optional typedef name
                re.MULTILINE | re.DOTALL
            ),
            # New pattern for separate typedef statements
            'typedef_declaration': re.compile(
                r'typedef\s+struct\s+(_?\w+)\s+'    # struct name
                r'(\w+)\s*,\s*'                     # type name
                r'\*\s*([P_]\w+);',                 # pointer type name
                re.MULTILINE
            ),
            # Updated member line pattern to handle more cases
            'member_line': re.compile(
                r'^\s*'                     # Start of line
                r'(?:/\*[^*]*\*/\s*)?'      # Optional comment
                r'(?:struct\s+)?'           # Optional struct keyword
                r'([a-zA-Z_]\w*(?:_t)?)'    # Type name
                r'(?:\s*\*)*'               # Optional pointer asterisks
                r'\s+'                      # Required whitespace
                r'([a-zA-Z_]\w*)'           # Member name
                r'(?:\[([^\]]+)\])*'        # Optional array dimensions
                r'(?:\s*:\s*(\d+))?'        # Optional bit field
                r'\s*;'                     # End semicolon
                r'\s*(?:/\*\s*([^*]*)\*/)?',  # Optional trailing comment
                re.MULTILINE
            ),
            'doc_comment': re.compile(
                r'/\*\s*([^*]*)\*/',
                re.MULTILINE | re.DOTALL
            ),
            #============================================
            'api_call': re.compile(
                r'(?:CCSP_|RDK_|RBUS_|TR181_|CcspCommon_|DM_|PSM_)(\w+)\s*\([^)]*\)',
                re.MULTILINE
            ),
            'include': re.compile(
                r'#include\s*[<"]([^>"]+)[>"]',
                re.MULTILINE
            ),
            'struct_usage': re.compile(
                r'struct\s+(\w+)\s+\w+',
                re.MULTILINE
            ),
            # 'function_call': re.compile(
            #     r'\b(\w+)\s*\(((?:[^()]*|\([^()]*\))*)\)',
            #     re.MULTILINE
            # )
            'function_call': re.compile(
                r'\b([a-zA-Z_]\w*)\s*\(((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*)\)',
                re.MULTILINE
            )
        }
        
    def _find_matching_brace(self, text: str, start: int) -> int:
        """Helper function to find matching closing brace."""
        count = 1
        pos = start
        while pos < len(text) and count > 0:
            if text[pos] == '{':
                count += 1
            elif text[pos] == '}':
                count -= 1
            pos += 1
            if count == 0:
                return pos
        return -1

    def find_functions(self, content: str) -> List[Dict[str, str]]:
        """Find complete C functions including their bodies."""
        functions = []
        
        for match in self.patterns['function'].finditer(content):
            start_pos = match.start()
            
            # Find opening brace
            opening_brace = content.find('{', start_pos)
            if opening_brace == -1:
                continue
                
            # Find matching closing brace
            end_pos = self._find_matching_brace(content, opening_brace + 1)
            if end_pos == -1:
                continue
                
            # Get complete function including declaration and body
            complete_function = content[start_pos:end_pos]
            
            # Extract parameters string
            params_start = content.find('(', start_pos)
            params_end = content.find(')', params_start)
            params_str = content[params_start + 1:params_end].strip() if params_start != -1 and params_end != -1 else ""
            
            functions.append({
                'name': match.group(1),
                'content': complete_function,
                'parameters': params_str,
                'start_line': content[:start_pos].count('\n') + 1,
                'end_line': content[:end_pos].count('\n') + 1
            })
        
        return functions
    
    def get_context(self, content: str, match_start: int, match_end: int, context_lines: int = 40) -> Tuple[str, str]:
        """
        Get context before and after a match with proper line counting and boundary handling.
        
        Args:
            content (str): The full source code content
            match_start (int): Starting position of the match
            match_end (int): Ending position of the match
            context_lines (int): Number of context lines to include before and after
        
        Returns:
            Tuple[str, str]: Context before and after the match
        """
        # Split content into lines and get line numbers
        lines = content.split('\n')
        
        # Find the line number where the match starts
        current_pos = 0
        current_line = 0
        for i, line in enumerate(lines):
            next_pos = current_pos + len(line) + 1  # +1 for newline
            if current_pos <= match_start < next_pos:
                current_line = i
                break
            current_pos = next_pos
        
        # Calculate line ranges for context
        start_line = max(0, current_line - context_lines)
        end_line = min(len(lines), current_line + context_lines + 1)
        
        # Get the matching line
        matching_line = lines[current_line] if 0 <= current_line < len(lines) else ""
        
        # Extract context before
        context_before = []
        for i in range(start_line, current_line):
            if i >= 0 and i < len(lines):
                context_before.append(lines[i])
        
        # Extract context after
        context_after = []
        for i in range(current_line + 1, end_line):
            if i < len(lines):
                context_after.append(lines[i])
        
        return (
            '\n'.join(context_before),
            '\n'.join(context_after)
        )

    def _analyze_struct_relationships(self, entity: CodeEntity, content: str):
        """Analyze relationships between structures and other code elements."""
        try:
            # Get structure members from metadata
            members = entity.metadata.get('members', [])
            
            # Track referenced structures
            referenced_structs = set()
            
            # Analyze each member
            for member in members:
                member_type = member['type']
                
                # Check if member type references another struct
                if 'struct' in member_type:
                    struct_name = member_type.replace('struct', '').strip()
                    if struct_name:
                        referenced_structs.add(struct_name)
                
                # Check for typedef'd struct types
                elif member_type in self.known_struct_types:
                    referenced_structs.add(member_type)
            
            # Add referenced structs to entity
            entity.structs_used = list(referenced_structs)
            
            # Add relationship metadata
            entity.metadata['struct_relationships'] = {
                'referenced_structs': list(referenced_structs),
                'member_count': len(members),
                'has_bitfields': any(m.get('bit_field_size') is not None for m in members),
                'has_arrays': any(m.get('array_dimensions') for m in members),
                'has_pointers': any(m.get('is_pointer', False) for m in members)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing struct relationships for {entity.name}: {str(e)}")

    def parse_file(self, file_path: str, component_name: str) -> List[CodeEntity]:
        """Parse a file and extract code entities including functions and structures."""
        try:
            logger.info(f"Starting to parse file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # # Pre-process: handle and store type definitions
            # self._process_type_definitions(content)
            # Process typedef declarations first
            self._process_typedef_declarations(content)
            
            entities = []
            includes = [m.group(1) for m in self.patterns['include'].finditer(content)]
            logger.info(f"Found {len(includes)} includes in {file_path}")
            
            # Step 1: Parse Functions (only for .c files)
            if file_path.lower().endswith('.c'):
                logger.info("Starting function parsing for .c file")
                functions = self.find_functions(content)
                for func in functions:
                    entity = self._create_function_entity_from_dict(
                        func, content, file_path, component_name, includes
                    )
                    if entity:
                        self._analyze_function_interactions(entity, content)
                        entities.append(entity)
                        logger.info(f"Successfully appended {entity.type} {entity.name} to entities")
                logger.info(f"Parsed {len(functions)} functions")
            else:
                logger.info(f"Skipping function parsing for non .c file: {file_path}")
                
            # Step 2: Parse Structures
            logger.info("Starting structure parsing")
            
            # Process typedef structs
            # typedef_matches = list(self.patterns['struct'].finditer(content))
            # for match in typedef_matches:
            #     logger.info(f"Struct Match from Mtaches : {match.group(0)}")
            #     entity = self._create_struct_entity(match, content, file_path, component_name, includes)
            #     if entity:
            #         entities.append(entity)
            struct_matches = list(self.patterns['struct'].finditer(content))
            for match in struct_matches:
                # logger.info(f"Struct Match from Mtaches : {match.group(0)}")
                entity = self._create_struct_entity(match, content, file_path, component_name, includes)
                if entity:
                    entities.append(entity)
                    logger.info(f"Successfully appended {entity.type} {entity.name} to entities")
            logger.info(f"Parsed {len(struct_matches)} Structures")
            logger.info(f"returing entities back to process single file.. ")
            return entities
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return []

    def _process_typedef_declarations(self, content: str):
        """
        Process all typedef declarations for structs and their pointer types.
        Handles various typedef patterns and maintains proper type relationships.
        """
        try:
            # Pattern for direct struct typedefs - note the careful handling of underscore prefix
            struct_typedef_pattern = re.compile(
                r'typedef\s+struct\s+'               # typedef struct prefix
                r'(_?\w+)?\s*'                      # Capture full struct name including underscore
                r'\{'                                # Opening brace
                r'([^{}]*(?:\{[^{}]*\}[^{}]*)*)'    # Struct body (handles nested braces)
                r'\}\s*'                            # Closing brace
                r'(\w+)_t\s*;',                     # Typedef name ending with _t
                re.MULTILINE | re.DOTALL
            )
            
            # Pattern for existing struct typedefs
            existing_struct_typedef_pattern = re.compile(
                r'typedef\s+struct\s+(_?\w+)\s+(\w+)_t\s*;'  # typedef struct _name name_t;
            )
            
            # Pattern for pointer typedefs
            pointer_typedef_pattern = re.compile(
                r'typedef\s+(?:struct\s+)?(_?\w+)\s*\*\s*([P_]\w+)\s*;'  # typedef type *Ptype;
            )

            # Process direct struct typedefs
            for match in struct_typedef_pattern.finditer(content):
                struct_name, struct_body, typedef_base = match.groups()
                
                if struct_name:
                    # Keep the original struct name with underscore if present
                    actual_struct_name = struct_name
                else:
                    actual_struct_name = f"_anonymous_{len(self.type_definitions)}"
                
                typedef_name = f"{typedef_base}_t"
                
                # Store typedef information
                self.type_definitions[typedef_name] = {
                    'base_type': f'struct {actual_struct_name}',
                    'is_pointer': False,
                    'is_anonymous': not bool(struct_name),
                    'original_definition': match.group(0),
                    'category': 'struct',
                    'struct_body': struct_body.strip()
                }
                self.known_struct_types.add(typedef_name)
                
                # Store original struct information
                if actual_struct_name and not actual_struct_name.startswith('_anonymous_'):
                    self.type_definitions[actual_struct_name] = {
                        'base_type': f'struct {actual_struct_name}',
                        'is_pointer': False,
                        'is_original': True,
                        'typedef_names': [typedef_name],
                        'category': 'struct',
                        'struct_body': struct_body.strip()
                    }
                    self.known_struct_types.add(actual_struct_name)

            # Process existing struct typedefs
            for match in existing_struct_typedef_pattern.finditer(content):
                struct_name, typedef_base = match.groups()
                typedef_name = f"{typedef_base}_t"
                
                # Store typedef information
                self.type_definitions[typedef_name] = {
                    'base_type': f'struct {struct_name}',  # Keep original struct name with underscore
                    'is_pointer': False,
                    'original_definition': match.group(0),
                    'category': 'struct'
                }
                self.known_struct_types.add(typedef_name)
                
                # Update or create the original struct entry
                if struct_name in self.type_definitions:
                    if 'typedef_names' not in self.type_definitions[struct_name]:
                        self.type_definitions[struct_name]['typedef_names'] = []
                    self.type_definitions[struct_name]['typedef_names'].append(typedef_name)
                else:
                    self.type_definitions[struct_name] = {
                        'base_type': f'struct {struct_name}',
                        'is_pointer': False,
                        'is_original': True,
                        'typedef_names': [typedef_name],
                        'category': 'struct'
                    }
                    self.known_struct_types.add(struct_name)

            # Process pointer typedefs
            for match in pointer_typedef_pattern.finditer(content):
                base_type, ptr_type_name = match.groups()
                
                # Handle both struct and non-struct pointer typedefs
                full_base_type = f'struct {base_type}' if not base_type.startswith('struct ') else base_type
                
                self.type_definitions[ptr_type_name] = {
                    'base_type': full_base_type,
                    'is_pointer': True,
                    'original_definition': match.group(0),
                    'category': 'pointer'
                }
                
                # If base type is a struct, add to known struct types
                if base_type in self.known_struct_types or base_type.startswith('struct '):
                    self.known_struct_types.add(ptr_type_name)

        except Exception as e:
            logger.error(f"Error processing typedef declarations: {str(e)}")

    def _parse_struct_members(self, struct_content: str) -> List[Dict]:
        """Parse structure members with improved comment and type handling."""
        members = []
        
        # Remove multi-line comments while preserving newlines
        content = re.sub(r'/\*.*?\*/', '', struct_content, flags=re.DOTALL)
        
        # Process each line
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            try:
                member = self._parse_member_line(line)
                if member:
                    members.append(member)
                            
            except Exception as e:
                logger.error(f"Error parsing member line '{line}': {str(e)}")
                continue
                    
        return members

    
    def _create_member_dict(self, match, original_line: str) -> Dict:
        """Create a member dictionary with improved documentation handling."""
        try:
            type_name, member_name, array_dim, bit_field, doc = match.groups()
            
            # Clean up type name and check for pointers
            clean_type = type_name.strip()
            pointer_count = original_line.count('*')
            
            member = {
                'name': member_name,
                'type': clean_type,
                'is_pointer': pointer_count > 0,
                'pointer_count': pointer_count,
                'original_line': original_line.strip()
            }
            
            # Handle array dimensions
            if array_dim:
                member['array_dimensions'] = array_dim.strip()
                try:
                    member['array_size'] = int(array_dim.strip())
                except ValueError:
                    member['array_size'] = array_dim.strip()
            
            # Handle bit fields
            if bit_field:
                member['bit_field_size'] = int(bit_field)
                
            # Handle documentation
            if doc:
                member['documentation'] = doc.strip()
            
            return member
            
        except Exception as e:
            logger.error(f"Error creating member dict: {str(e)}")
            raise

    def _create_struct_entity(self, match: re.Match, content: str, file_path: str, component_name: str, includes: List[str]) -> Optional[CodeEntity]:
        try:
            struct_name = match.group(1)
            struct_body = match.group(2)
            struct_pack = match.group(3)
            typedef_name = match.group(4)
            
            # Get documentation comment before the struct
            start_pos = content.find(match.group(0))
            prev_content = content[:start_pos].rstrip()
            doc_match = list(self.patterns['doc_comment'].finditer(prev_content))
            documentation = doc_match[-1].group(1).strip() if doc_match else ""
            
            # Parse members including nested structures and unions
            parsed_data = self._parse_complex_struct(struct_body)
            members = parsed_data['members']
            structs_used = parsed_data['structs_used']
            function_calls = parsed_data['function_calls']
            
            # Determine the final type name
            final_name = typedef_name if typedef_name else struct_name
            if not final_name:
                return None
                
            # Store in type definitions
            self.type_definitions[final_name] = {
                'original_type': 'struct',
                'struct_name': struct_name,
                'has_struct_pack': bool(struct_pack),
                'members': members,
                'documentation': documentation
            }
            self.known_struct_types.add(final_name)
            
            return CodeEntity(
                name=final_name,
                type='struct',
                content=match.group(0),
                file_path=file_path,
                component=component_name,
                includes=includes,
                structs_used=list(structs_used),
                function_calls=function_calls,
                metadata={
                    'struct_name': struct_name,
                    'members': members,
                    'has_struct_pack': bool(struct_pack),
                    'documentation': documentation,
                    'line_number': content[:start_pos].count('\n') + 1,
                    'member_count': len(members),
                    'has_arrays': any('array_dimensions' in m for m in members),
                    'has_pointers': any(m.get('is_pointer', False) for m in members),
                    'has_unions': any(m.get('is_union', False) for m in members)
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating struct entity: {str(e)}")
            return None

    def _parse_complex_struct(self, struct_content: str) -> Dict:
        """Parse complex structure content including nested structures and unions."""
        result = {
            'members': [],
            'structs_used': set(),
            'function_calls': []
        }
        
        # Remove C-style comments while preserving newlines
        content = re.sub(r'/\*.*?\*/', '', struct_content, flags=re.DOTALL)
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
        # Track nested level and current context
        nested_level = 0
        current_union = None
        current_struct = None
        buffer = []
        
        def parse_member_type(member_line: str) -> Optional[str]:
            """Extract struct type from a member line."""
            # Match struct/union type pattern
            match = re.match(r'^\s*struct\s+(\w+)', member_line)
            if match:
                return match.group(1)
            return None
        
        def process_union_members(members: List[str]) -> Tuple[set, List[Dict]]:
            """Process union members to extract structs used and format members."""
            structs = set()
            formatted_members = []
            for member in members:
                member = member.strip()
                if not member or member == '{':
                    continue
                
                # Extract struct types
                struct_type = parse_member_type(member)
                if struct_type:
                    structs.add(struct_type)
                
                # Parse the member line to get detailed information
                parsed_member = self._parse_member_line(member)
                if parsed_member:
                    formatted_member = {
                        'type': 'field',
                        'field_type': parsed_member['type'],
                        'name': parsed_member.get('name', ''),
                        'is_pointer': parsed_member.get('is_pointer', False),
                        'pointer_count': parsed_member.get('pointer_count', 0),
                        'array_dimensions': parsed_member.get('array_dimensions', None),
                        'bit_field_size': parsed_member.get('bit_field_size', None),
                        'content': member.rstrip(';'),
                        'original_line': member
                    }
                    formatted_members.append(formatted_member)
                
                # Check for struct types in the line
                if 'struct' in member:
                    struct_matches = re.finditer(r'struct\s+(\w+)', member)
                    for match in struct_matches:
                        structs.add(match.group(1))
            
            return structs, formatted_members

        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Handle union start
            if line.startswith('union'):
                current_union = {
                    'members': [],
                    'name': self._extract_union_name(line),
                    'structs_used': set()
                }
                nested_level += 1
                buffer = []
                i += 1
                continue
            
            # Handle nested struct start
            if line.startswith('struct') and '{' in line:
                current_struct = {
                    'members': [],
                    'name': self._extract_struct_name(line),
                    'structs_used': set()
                }
                nested_level += 1
                buffer = []
                i += 1
                continue
            
            # Handle closing braces
            if '}' in line:
                nested_level -= 1
                if current_union is not None:
                    union_name = self._extract_member_name(line)
                    structs_used, formatted_members = process_union_members(current_union['members'])
                    member = {
                        'name': union_name,
                        'type': 'union',
                        'is_union': True,
                        'members': formatted_members,
                        'union_name': current_union['name'],
                        'structs_used': structs_used
                    }
                    result['members'].append(member)
                    result['structs_used'].update(structs_used)
                    current_union = None
                elif current_struct is not None:
                    struct_name = self._extract_member_name(line)
                    structs_used, formatted_members = process_union_members(current_struct['members'])
                    member = {
                        'name': struct_name,
                        'type': 'struct',
                        'is_struct': True,
                        'members': formatted_members,
                        'struct_name': current_struct['name'],
                        'structs_used': structs_used
                    }
                    result['members'].append(member)
                    result['structs_used'].update(structs_used)
                    current_struct = None
                i += 1
                continue
            
            # Process regular members
            if nested_level == 0:
                parsed_member = self._parse_member_line(line)
                if parsed_member:
                    member = {
                        'type': 'field',
                        'field_type': parsed_member['type'],
                        'name': parsed_member['name'],
                        'is_pointer': parsed_member.get('is_pointer', False),
                        'pointer_count': parsed_member.get('pointer_count', 0),
                        'array_dimensions': parsed_member.get('array_dimensions', None),
                        'bit_field_size': parsed_member.get('bit_field_size', None),
                        'content': line.rstrip(';'),
                        'original_line': line
                    }
                    result['members'].append(member)
                    
                    # Track struct usage
                    if self._is_struct_type(parsed_member['type']):
                        struct_type = self._extract_struct_type(parsed_member['type'])
                        result['structs_used'].add(struct_type)
                    
                    # Track function pointers
                    if self._is_function_pointer(line):
                        func_call = self._create_function_pointer_call(line)
                        if func_call:
                            result['function_calls'].append(func_call)
            
            # Add line to current context buffer
            else:
                if current_union is not None:
                    current_union['members'].append(line)
                elif current_struct is not None:
                    current_struct['members'].append(line)
            
            i += 1
        
        return result


    def _is_struct_type(self, type_name: str) -> bool:
        """Check if a type is a struct type."""
        return (
            type_name.startswith('struct ') or
            type_name in self.known_struct_types or
            type_name.endswith('_t') or
            type_name.endswith('_st')
        )

    def _extract_struct_type(self, type_name: str) -> str:
        """Extract the actual struct name from a type."""
        if type_name.startswith('struct '):
            return type_name.split(' ')[1]
        return type_name

    def _create_union_member(self, union_data: Dict, name: str) -> Dict:
        """Create a member entry for a union."""
        return {
            'name': name,
            'type': 'union',
            'is_union': True,
            'members': union_data['members'],
            'union_name': union_data['name']
        }

    def _create_nested_struct_member(self, struct_data: Dict, name: str) -> Dict:
        """Create a member entry for a nested struct."""
        return {
            'name': name,
            'type': 'struct',
            'is_struct': True,
            'members': struct_data['members'],
            'struct_name': struct_data['name']
        }

    def _is_function_pointer(self, line: str) -> bool:
        """Check if a line contains a function pointer declaration."""
        return ('(' in line and '*' in line and ')' in line and 
                not line.strip().startswith('//'))

    def _create_function_pointer_call(self, line: str) -> Optional[FunctionCall]:
        """Create a FunctionCall object from a function pointer declaration."""
        try:
            # Match function pointer pattern
            # Example: void (*callback)(int param1, char* param2);
            match = re.match(
                r'(\w+)\s*\(\s*\*\s*(\w+)\s*\)\s*\((.*?)\)\s*;',
                line.strip()
            )
            
            if match:
                return_type, func_name, params_str = match.groups()
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                return FunctionCall(
                    function_name=func_name,
                    component='',  # Component info not available at this level
                    parameters=params,
                    parameter_functions=[],
                    return_type=return_type,
                    is_api=False,
                    line_number=0,
                    context_before='',
                    context_after=''
                )
                
        except Exception as e:
            logger.error(f"Error creating function pointer call: {str(e)}")
        
        return None

    def _extract_union_name(self, line: str) -> str:
        """Extract union name from declaration line."""
        match = re.search(r'union\s+(\w+)?\s*{', line)
        return match.group(1) if match and match.group(1) else ''

    def _extract_struct_name(self, line: str) -> str:
        """Extract struct name from declaration line."""
        match = re.search(r'struct\s+(\w+)?\s*{', line)
        return match.group(1) if match and match.group(1) else ''

    def _extract_member_name(self, line: str) -> str:
        """Extract member name from closing brace line."""
        match = re.search(r'}\s*(\w+)', line)
        return match.group(1) if match else ''
    
    def _parse_member_line(self, line: str) -> Optional[Dict]:
        """Parse a single member line of a structure."""
        try:
            # Skip preprocessor directives and empty lines
            if line.startswith('#') or not line.strip():
                return None
                
            # Remove trailing comments
            line = re.sub(r'/\*.*?\*/', '', line)  # Remove C-style comments
            line = re.sub(r'//.*$', '', line)      # Remove C++ style comments
            line = line.strip()
            
            if not line or line.endswith('{'):  # Skip struct/union opening lines
                return None
                
            # Match basic member pattern
            # Captures: type, name, array dimensions, bit field size
            pattern = r'''
                (\w+\s*[\w\s\*]*)\s+    # Type (including pointer asterisks)
                (\w+)                    # Name
                (?:\[([^\]]*)\])?       # Optional array dimensions
                (?::\s*(\d+))?          # Optional bit field
                \s*;                     # Ending semicolon
            '''
            match = re.match(pattern, line.strip(), re.VERBOSE)
            
            if not match:
                return None
                
            type_name, member_name, array_dim, bit_field = match.groups()
            
            # Clean up type name and check for pointers
            type_name = type_name.strip()
            pointer_count = type_name.count('*')
            type_name = re.sub(r'\s*\*\s*', '', type_name)  # Remove pointer asterisks for clean type name
            
            member = {
                'name': member_name,
                'type': type_name,
                'is_pointer': pointer_count > 0,
                'pointer_count': pointer_count,
                'original_line': line.strip()
            }
            
            # Handle array dimensions
            if array_dim:
                member['array_dimensions'] = array_dim.strip()
                try:
                    # Try to convert to integer if it's a numeric size
                    member['array_size'] = int(array_dim.strip())
                except ValueError:
                    # Keep as string if it's a symbolic constant
                    member['array_size'] = array_dim.strip()
            
            # Handle bit fields
            if bit_field:
                member['bit_field_size'] = int(bit_field)
                
            # Special handling for struct/union types
            if type_name.startswith('struct ') or type_name.startswith('union '):
                member['type_category'] = type_name.split()[0]  # 'struct' or 'union'
                member['type_name'] = ' '.join(type_name.split()[1:])
            
            # Handle function pointers
            if '(' in line and '*' in line and ')' in line:
                func_ptr_match = re.match(
                    r'(\w+)\s*\(\s*\*\s*(\w+)\s*\)\s*\((.*?)\)\s*;',
                    line.strip()
                )
                if func_ptr_match:
                    return_type, func_name, params_str = func_ptr_match.groups()
                    member.update({
                        'is_function_pointer': True,
                        'function_return_type': return_type,
                        'function_name': func_name,
                        'function_parameters': [p.strip() for p in params_str.split(',') if p.strip()]
                    })
            
            return member
            
        except Exception as e:
            logger.error(f"Error parsing member line '{line}': {str(e)}")
            return None

    #================================================================================== MODIFIED

    #==============================================================================================

    def _is_embedded_struct(self, content: str, start_pos: int) -> bool:
        """Check if a struct definition is embedded within another struct or union."""
        # Look backwards for the nearest struct or union keyword
        previous_content = content[:start_pos].strip()
        last_struct = previous_content.rfind('struct')
        last_union = previous_content.rfind('union')
        
        if last_struct == -1 and last_union == -1:
            return False
            
        # Find the last opening brace before this position
        last_brace = previous_content.rfind('{')
        
        # If we found a brace and it's after the struct/union keyword,
        # this is likely an embedded definition
        return last_brace > max(last_struct, last_union)
    #===================================================================================================

    def _create_function_entity_from_dict(
        self, 
        func_dict: Dict[str, str], 
        content: str, 
        file_path: str, 
        component_name: str, 
        includes: List[str]
    ) -> Optional[CodeEntity]:
        """Create a function entity from the dictionary returned by find_functions."""
        try:
            name = func_dict['name']
            if self._is_common_function(name):
                return None
                
            function_content = func_dict['content']
            start_pos = content.find(function_content)
            end_pos = start_pos + len(function_content)
            context_before, context_after = self.get_context(
                content, start_pos, end_pos
            )
            
            # Parse function signature
            return_type = self._extract_return_type(function_content)
            parameters = self._parse_parameters(func_dict['parameters'])
            
            # Find structs used in function
            structs_used = self._find_structs_in_function(function_content)
            
            return CodeEntity(
                name=name,
                type='function',
                content=function_content,
                file_path=file_path,
                component=component_name,
                includes=includes,
                structs_used=list(structs_used),  # Add structs_used to the entity
                metadata={
                    'return_type': return_type,
                    'parameters': parameters,
                    'line_number': func_dict['start_line'],
                    'end_line': func_dict['end_line'],
                    'context_before': context_before,
                    'context_after': context_after
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating function entity from dict: {str(e)}")
            return None
        
    def _normalize_struct_name(self, name: str) -> str:
        """
        Normalize struct names by removing trailing 's' if it would create a duplicate.
        Returns the base name that should be used.
        """
        # If name ends with 's' and removing it would match an existing pattern
        # like hints/hint, nodes/node, etc., return the base form
        if name.endswith('s'):
            base = name[:-1]
            if (
                # Check common struct name patterns where plurals might occur
                base.endswith('_hint') or
                base.endswith('_node') or
                base.endswith('_addr') or
                base.endswith('_info')
            ):
                return base
        return name


    def _find_structs_in_function(self, function_content: str) -> Set[str]:
        """
        Analyze function content to find struct usage patterns including typedef'd types.
        Returns a set of struct names used in the function.
        """
        # Use an intermediate dictionary to track original names and their occurrences
        struct_occurrences = {}
        
        try:
            # Remove comments to avoid false positives
            content = self._remove_comments(function_content)
            
            patterns = [
                # Direct struct declarations and parameters
                r'\bstruct\s+([a-zA-Z_]\w+)\b',
                
                # Typedef'd types with various suffixes
                r'\b([a-zA-Z_]\w+_[ts]t)\b',  # Matches _t and _st suffixes
                r'\b([a-zA-Z_]\w+Struct)\b',   # Matches CustomStruct pattern
                r'\b([a-zA-Z_]\w+Info)\b',     # Matches InfoStruct pattern
                r'\b(P[A-Z][a-zA-Z_]\w+)\b',   # Matches pointer typedef pattern (e.g., PCustomStruct)
                
                # Casting and sizeof
                r'(?:\(\s*struct|sizeof\s*\(\s*struct)\s+([a-zA-Z_]\w+)\b',
                
                # Variable declarations with typedef'd types
                r'\b([a-zA-Z_]\w+(?:_t|_st|Struct|Info))\s+\w+\s*[;=]',
            ]
            
            # Find all potential struct names
            for pattern in patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    struct_name = match.group(1)
                    if struct_name:
                        # Check if the name exists in type definitions
                        if struct_name in self.type_definitions:
                            base_type = self.type_definitions[struct_name].get('base_type', '')
                            if base_type.startswith('struct '):
                                # Add both typedef name and original struct name
                                struct_occurrences[struct_name] = struct_occurrences.get(struct_name, 0) + 1
                                original_struct = base_type.replace('struct ', '')
                                struct_occurrences[original_struct] = struct_occurrences.get(original_struct, 0) + 1
                        else:
                            # Track occurrences of both original and normalized names
                            normalized_name = self._normalize_struct_name(struct_name)
                            struct_occurrences[struct_name] = struct_occurrences.get(struct_name, 0) + 1
                            if normalized_name != struct_name:
                                struct_occurrences[normalized_name] = struct_occurrences.get(normalized_name, 0) + 1

            # Handle pointer member access with typedef'd types
            member_access_pattern = r'(\w+)->(\w+)'
            for match in re.finditer(member_access_pattern, content):
                var_name = match.group(1)
                # Enhanced variable declaration pattern to include typedef'd types
                var_decl_pattern = rf'''(?:
                    struct\s+([a-zA-Z_]\w+)|
                    ([a-zA-Z_]\w+_[ts]t)|
                    ([a-zA-Z_]\w+Struct)|
                    ([a-zA-Z_]\w+Info)|
                    (P[A-Z][a-zA-Z_]\w+)
                )\s*\*?\s*{var_name}\b'''
                var_decl = re.search(var_decl_pattern, content, re.VERBOSE)
                if var_decl:
                    # Get the first non-None group
                    struct_type = next((g for g in var_decl.groups() if g), None)
                    if struct_type:
                        if struct_type in self.type_definitions:
                            base_type = self.type_definitions[struct_type].get('base_type', '')
                            if base_type.startswith('struct '):
                                original_struct = base_type.replace('struct ', '')
                                struct_occurrences[original_struct] = struct_occurrences.get(original_struct, 0) + 1
                        normalized_name = self._normalize_struct_name(struct_type)
                        struct_occurrences[struct_type] = struct_occurrences.get(struct_type, 0) + 1
                        if normalized_name != struct_type:
                            struct_occurrences[normalized_name] = struct_occurrences.get(normalized_name, 0) + 1

            # Choose the most appropriate name form
            final_structs = set()
            processed_bases = set()
            
            for name in struct_occurrences:
                # Skip if it's a pointer typedef that we've already processed the base type for
                if name.startswith('P') and name[1:] in struct_occurrences:
                    continue
                    
                base_name = self._normalize_struct_name(name)
                if base_name not in processed_bases:
                    # If we have both forms, use the one with more occurrences
                    if base_name != name and base_name in struct_occurrences:
                        if struct_occurrences[name] >= struct_occurrences[base_name]:
                            final_structs.add(name)
                        else:
                            final_structs.add(base_name)
                    else:
                        final_structs.add(name)
                    processed_bases.add(base_name)

            return final_structs
            
        except Exception as e:
            logger.error(f"Error finding structs in function: {str(e)}")
            return set()

    def _analyze_function_interactions(self, entity: CodeEntity, content: str):
        """Analyze function calls using the token-based parser"""
        parser = CFunctionParser()
        function_calls = parser.parse_function_calls(entity.content)
        
        processed_functions = set()
        
        print(f"Found {len(function_calls)} potential function calls")
        
        # for func_name, params, position in function_calls:
        for func_name, params, position, param_functions in function_calls:
            # Skip if already processed or common function
            if func_name in processed_functions or self._is_common_function(func_name):
                continue
                
            processed_functions.add(func_name)
            
            # Determine if it's an API call
            is_api = any(func_name.startswith(prefix) for prefix in 
                        ('CCSP_', 'RDK_', 'RBUS_', 'TR181_', 'CcspCommon_', 'DM_', 'PSM_'))
            
            call = FunctionCall(
                function_name=func_name,
                component="Unknown",
                parameters=self._parse_parameters(params),  # Get just the parameters list
                parameter_functions=param_functions,      # Add the parameter functions
                return_type="Unknown",
                is_api=is_api,
                line_number=entity.content[:position].count('\n') + 1
            )
            
            entity.function_calls.append(call)
            if is_api:
                entity.api_calls.append(func_name)

    def _parse_member_declaration(self, declaration: str) -> Optional[Dict[str, str]]:
        """Parse a single struct member declaration."""
        try:
            # Handle basic types, pointers, arrays, and bit fields
            declaration = declaration.strip()
            
            # Skip empty declarations
            if not declaration:
                return None
                
            # Handle bit fields
            bit_field_match = re.match(r'(.*?)\s*:\s*(\d+)$', declaration)
            bit_field_size = None
            if bit_field_match:
                declaration = bit_field_match.group(1)
                bit_field_size = int(bit_field_match.group(2))
                
            # Split into words
            parts = declaration.split()
            
            # Handle various declaration patterns
            if len(parts) < 2:
                return None
                
            # Last part contains the member name (possibly with array dimensions)
            name_part = parts[-1]
            
            # Extract array dimensions if present
            array_dims = []
            while '[' in name_part:
                array_start = name_part.rfind('[')
                array_end = name_part.rfind(']')
                if array_end > array_start:
                    dim = name_part[array_start + 1:array_end].strip()
                    if dim:
                        array_dims.insert(0, dim)
                    name_part = name_part[:array_start]
                else:
                    break
                    
            # Type is everything before the name
            type_part = ' '.join(parts[:-1])
            
            return {
                'name': name_part.strip(),
                'type': type_part.strip(),
                'is_pointer': '*' in type_part,
                'array_dimensions': array_dims,
                'bit_field_size': bit_field_size
            }
            
        except Exception as e:
            logger.error(f"Error parsing member declaration '{declaration}': {str(e)}")
            return None

    def _remove_comments(self, content: str) -> str:
        """Remove C-style comments from the content."""
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        return content

    def get_context_struct(self, content: str, start_pos: int, end_pos: int, context_lines: int = 3) -> Tuple[str, str]:
        """Get the context before and after a code segment."""
        # Find line boundaries
        line_start = content.rfind('\n', 0, start_pos) + 1
        if line_start == 0:
            line_start = 0
            
        line_end = content.find('\n', end_pos)
        if line_end == -1:
            line_end = len(content)
            
        # Get context before
        context_start = content.rfind('\n', 0, line_start)
        for _ in range(context_lines - 1):
            prev_start = content.rfind('\n', 0, context_start)
            if prev_start == -1:
                break
            context_start = prev_start
        if context_start == -1:
            context_start = 0
            
        # Get context after
        context_end = line_end
        for _ in range(context_lines):
            next_end = content.find('\n', context_end + 1)
            if next_end == -1:
                context_end = len(content)
                break
            context_end = next_end
            
        return content[context_start:line_start].strip(), content[line_end:context_end].strip()
    #===================================================================================================

    @staticmethod
    def _is_common_function(name: str) -> bool:
        # common_functions = {
        #     'printf', 'scanf', 'malloc', 'free', 'strlen', 'strcpy',
        #     'strcmp', 'memcpy', 'memset', 'fopen', 'fclose',
        #     'main', 'if', 'for', 'while', 'switch'
        # }
        common_functions = {
            'printf', 'scanf', 'malloc', 'free', 'strlen', 'strcpy','strcmp_s','ERR_CHK','CcspTraceWarning','ccspWifiDbgPrint',
            'strcmp', 'memcpy', 'memset', 'fopen', 'fclose','AnscCopyString','strcat','AnscSizeOfString','CcspTraceInfo',
            'main', 'if', 'for', 'while', 'switch','wifi_util_dbg_print','snprintf','strncmp','defined','remove','CcspTraceError',
            'UNREFERENCED_PARAMETER','return','strncat','fprintf','strncpy','strtok','wifi_util_error_print','CcspWifiTrace',
        }
        return name in common_functions

    @staticmethod
    def _extract_return_type(function_content: str) -> str:
        # Simple return type extraction - can be enhanced
        first_line = function_content.split('(')[0]
        return_type = ' '.join(first_line.split()[:-1])
        return return_type.strip()

    @staticmethod
    def _parse_parameters(params_str: str) -> List[str]:
        if not params_str.strip():
            return []
        
        params = []
        current_param = []
        paren_count = 0
        
        for char in params_str:
            if char == '(' :
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                params.append(''.join(current_param).strip())
                current_param = []
                continue
            
            current_param.append(char)
        
        if current_param:
            params.append(''.join(current_param).strip())
        
        return params

    @staticmethod
    def _parse_struct_fields(struct_content: str) -> List[Dict[str, str]]:
        fields = []
        for line in struct_content.split(';'):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    fields.append({
                        'type': ' '.join(parts[:-1]),
                        'name': parts[-1]
                    })
        return fields
