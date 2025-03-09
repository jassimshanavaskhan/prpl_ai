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
            # 'struct': re.compile(
            #     r'struct\s+(\w+)\s*\{([^}]*)\}',
            #     re.MULTILINE | re.DOTALL
            # ),
            #============================================
            # 'struct': re.compile(
            #     r'typedef\s+struct\s*(?:\w+\s*)?\{'  # Start with typedef struct
            #     r'([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'   # Match struct body including nested
            #     r'\s*(__attribute__\s*\(\([^)]+\)\))?\s*'  # Match attributes
            #     r'(\w+);',                            # Capture typedef name
            #     re.MULTILINE | re.DOTALL
            # ),
            # # Updated member line pattern to better handle arrays
            # 'member_line': re.compile(
            #     r'^\s*'                   # Start of line with optional whitespace
            #     r'(?:struct\s+)?'         # Optional struct keyword
            #     r'([a-zA-Z_]\w*(?:_t)?)'  # Type name (including _t suffix)
            #     r'(?:\s*\*)*'             # Optional pointer asterisks
            #     r'\s+'                     # Required whitespace
            #     r'([a-zA-Z_]\w*)'         # Member name
            #     r'(?:\[([^\]]+)\])*'      # Optional array dimensions (multiple allowed)
            #     r'(?:\s*:\s*(\d+))?'      # Optional bit field
            #     r'\s*;'                   # End semicolon
            #     r'\s*(?://[/\s]*<?([^>]*))?',  # Optional documentation
            #     re.MULTILINE
            # ),
            #-------------
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



    # def parse_file(self, file_path: str, component_name: str) -> List[CodeEntity]:
    #     """Parse a file and extract code entities."""
    #     # print("\ncomponent name ",component_name)
    #     try:
    #         with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    #             content = f.read()
            
    #         entities = []
            
    #         print("\nJAS53 ============= >> Started Parsing the Includes from the file ")
    #         # Parse includes first
    #         includes = [m.group(1) for m in self.patterns['include'].finditer(content)] #['unistd.h', 'stdlib.h', 'stdio.h', 'string.h', 'stdarg.h', 'getopt.h', 'rbus/rbus.h', 'signal.h', 'wifi_hal.h', 'collection.h', 'wifi_monitor.h', 'sys/time.h', 'sys/stat.h']
    #         # print("includes : ",includes)
    #         # print("\n len(includes) : ",len(includes))
    #         print("\nJAS53 ============= >> Total Prased includes : ",len(includes))
            
    #         # Find and parse functions using the new method
    #         functions = self.find_functions(content)
    #         print("\nJAS54 ============= >> Total Found Functions from file : ",len(functions))
    #         # print("\n functions : ",functions)
    #         for func in functions:
    #             entity = self._create_function_entity_from_dict(
    #                 func, content, file_path, component_name, includes
    #             )
    #             if entity:
    #                 self._analyze_function_interactions(entity, content)
    #                 entities.append(entity)
            
    #         # Parse structs
    #         for match in self.patterns['struct'].finditer(content):
    #             entity = self._create_struct_entity(
    #                 match, content, file_path, component_name, includes
    #             )
    #             if entity:
    #                 entities.append(entity)
            
    #         return entities
            
    #     except Exception as e:
    #         logger.error(f"Error parsing file {file_path}: {str(e)}")
    #         return []
    #=============================================================================================
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

    # def parse_file(self, file_path: str, component_name: str) -> List[CodeEntity]:
    #     """Parse a file and extract code entities including functions and structures."""
    #     try:
    #         logger.info(f"Starting to parse file: {file_path}")
    #         with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    #             content = f.read()
            
    #         entities = []
            
    #         # Parse includes first
    #         includes = [m.group(1) for m in self.patterns['include'].finditer(content)]
    #         logger.info(f"Found {len(includes)} includes in {file_path}")
            
    #         # Step 1: Parse Functions
    #         logger.info("Starting function parsing")
    #         functions = self.find_functions(content)
    #         for func in functions:
    #             entity = self._create_function_entity_from_dict(
    #                 func, content, file_path, component_name, includes
    #             )
    #             if entity:
    #                 self._analyze_function_interactions(entity, content)
    #                 entities.append(entity)
    #         logger.info(f"Parsed {len(functions)} functions")
            
    #         # Step 2: Parse Structures
    #         logger.info("Starting structure parsing")
            
    #         # First pass: Find all struct definitions
    #         struct_matches = []
    #         position = 0
    #         while True:
    #             match = self.patterns['struct'].search(content, position)
    #             if not match:
    #                 break
                    
    #             # Get the complete struct definition
    #             start_pos = match.start()
    #             struct_content = match.group(0)
                
    #             # Store the match and update position
    #             struct_matches.append(match)
    #             position = match.end()
            
    #         logger.info(f"Found {len(struct_matches)} potential structures")
            
    #         # Second pass: Process each structure
    #         for match in struct_matches:
    #             try:
    #                 # Create structure entity
    #                 entity = self._create_struct_entity(
    #                     match, content, file_path, component_name, includes
    #                 )
                    
    #                 if entity:
    #                     # Add structure relationships
    #                     self._analyze_struct_relationships(entity, content)
    #                     entities.append(entity)
    #                     logger.info(f"structure Entity : {entity}")
    #                     logger.debug(f"----> Successfully parsed struct: {entity.name}")
    #             except Exception as e:
    #                 logger.error(f"Error processing structure match: {str(e)}")
    #                 continue
            
    #         return entities
                
    #     except Exception as e:
    #         logger.error(f"Error parsing file {file_path}: {str(e)}")
    #         return []
    #===================================================================================
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
            
            # Step 1: Parse Functions
            logger.info("Starting function parsing")
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
        """Process separate typedef declarations for structs and their pointer types."""
        for match in self.patterns['typedef_declaration'].finditer(content):
            struct_name = match.group(1)
            type_name = match.group(2)
            ptr_type_name = match.group(3)
            
            self.type_definitions[type_name] = {
                'base_type': f'struct {struct_name}',
                'is_pointer': False
            }
            self.type_definitions[ptr_type_name] = {
                'base_type': f'struct {struct_name}',
                'is_pointer': True
            }
            self.known_struct_types.update([type_name, ptr_type_name])

    # def _parse_struct_members(self, struct_content: str) -> List[Dict]:
    #     """Parse structure members with improved array handling."""
    #     members = []
        
    #     # Remove C-style comments while preserving newlines
    #     content = re.sub(r'/\*.*?\*/', '', struct_content, flags=re.DOTALL)
    #     # Remove C++-style comments, preserving newlines
    #     content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
    #     # Split into lines and process each member
    #     lines = [line.strip() for line in content.split('\n') if line.strip()]
        
    #     current_union = None
        
    #     for line in lines:
    #         if not line or line.startswith('}'): 
    #             if current_union:
    #                 members.append(current_union)
    #                 current_union = None
    #             continue
            
    #         # Handle union start
    #         if line.startswith('union'):
    #             current_union = {
    #                 'type': 'union',
    #                 'members': []
    #             }
    #             continue
            
    #         try:
    #             # Parse regular member
    #             member_match = self.patterns['member_line'].match(line)
    #             if member_match:
    #                 member = self._create_member_dict(member_match, line)
    #                 if current_union:
    #                     current_union['members'].append(member)
    #                 else:
    #                     members.append(member)
                        
    #         except Exception as e:
    #             logger.error(f"Error parsing member line '{line}': {str(e)}")
    #             continue
                
    #     return members
    def _parse_struct_members(self, struct_content: str) -> List[Dict]:
        """Parse structure members with improved comment and type handling."""
        members = []
        
        # Remove multi-line comments while preserving newlines
        content = re.sub(r'/\*.*?\*/', '', struct_content, flags=re.DOTALL)
        
        # Process each line
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            try:
                member_match = self.patterns['member_line'].match(line)
                if member_match:
                    member = self._create_member_dict(member_match, line)
                    if member:
                        members.append(member)
                        
            except Exception as e:
                logger.error(f"Error parsing member line '{line}': {str(e)}")
                continue
                
        return members

    # def _create_member_dict(self, match, original_line: str) -> Dict:
    #     """Create a member dictionary with improved type handling."""
    #     try:
    #         type_name, member_name, array_dim, bit_field, doc = match.groups()
            
    #         # Clean up type name and check for pointers
    #         clean_type = type_name.strip()
    #         pointer_count = original_line.count('*')
            
    #         member = {
    #             'name': member_name,
    #             'type': clean_type,
    #             'is_pointer': pointer_count > 0,
    #             'pointer_count': pointer_count,
    #             'original_line': original_line.strip()
    #         }
            
    #         # Handle array dimensions
    #         if array_dim:
    #             member['array_dimensions'] = array_dim.strip()
    #             # Store array size if it's a numeric constant
    #             try:
    #                 member['array_size'] = int(array_dim.strip())
    #             except ValueError:
    #                 member['array_size'] = array_dim.strip()  # Store as string if not numeric
            
    #         # Handle bit fields
    #         if bit_field:
    #             member['bit_field_size'] = int(bit_field)
                
    #         # Handle documentation
    #         if doc:
    #             member['documentation'] = doc.strip()
                
    #         # Type categorization
    #         if clean_type.endswith('_t'):
    #             member['is_typedef'] = True
    #             # Don't try to access base_type from type_definitions
    #             if clean_type in self.type_definitions:
    #                 member['type_info'] = self.type_definitions[clean_type]
    #         else:
    #             member['is_typedef'] = False
                
    #         # Basic type classification
    #         basic_types = {'BOOL', 'INT', 'UINT', 'CHAR', 'UCHAR', 'SHORT', 'USHORT', 'LONG', 'ULONG'}
    #         if clean_type in basic_types:
    #             member['type_category'] = 'basic'
    #         elif clean_type.endswith('_t'):
    #             member['type_category'] = 'custom'
    #         else:
    #             member['type_category'] = 'other'
                
    #         return member
            
    #     except Exception as e:
    #         logger.error(f"Error creating member dict: {str(e)}")
    #         raise
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

    # def _create_struct_entity(self, match: re.Match, content: str, file_path: str, component_name: str, includes: List[str]) -> Optional[CodeEntity]:
    #     """Create a structure entity with improved typedef and attribute handling."""
    #     try:
    #         struct_body = match.group(1)
    #         attributes = match.group(2) if match.group(2) else None
    #         typedef_name = match.group(3) if match.group(3) else None
            
    #         if not typedef_name:
    #             return None
                
    #         # Parse attributes
    #         attr_list = []
    #         if attributes:
    #             attr_pattern = re.compile(r'__attribute__\s*\(\(([^)]+)\)\)')
    #             attr_match = attr_pattern.search(attributes)
    #             if attr_match:
    #                 attr_list = [attr.strip() for attr in attr_match.group(1).split(',')]
            
    #         # Parse members
    #         members = self._parse_struct_members(struct_body)
            
    #         # Store typedef information
    #         self.type_definitions[typedef_name] = {
    #             'original_type': 'struct',
    #             'attributes': attr_list,
    #             'members': members
    #         }
    #         self.known_struct_types.add(typedef_name)
            
    #         # Get struct context
    #         start_pos = content.find(match.group(0))
    #         end_pos = start_pos + len(match.group(0))
    #         context_before = content[max(0, start_pos-200):start_pos].strip()
    #         context_after = content[end_pos:min(len(content), end_pos+200)].strip()
            
    #         return CodeEntity(
    #             name=typedef_name,
    #             type='struct',
    #             content=match.group(0),
    #             file_path=file_path,
    #             component=component_name,
    #             includes=includes,
    #             metadata={
    #                 'members': members,
    #                 'attributes': attr_list,
    #                 'is_typedef': True,
    #                 'line_number': content[:start_pos].count('\n') + 1,
    #                 'end_line': content[:end_pos].count('\n') + 1,
    #                 'context_before': context_before,
    #                 'context_after': context_after,
    #                 'member_count': len(members),
    #                 'has_arrays': any('array_dimensions' in m for m in members),
    #                 'has_pointers': any(m.get('is_pointer', False) for m in members),
    #                 'has_bit_fields': any('bit_field_size' in m for m in members)
    #             }
    #         )
            
    #     except Exception as e:
    #         logger.error(f"Error creating struct entity: {str(e)}")
    #         return None

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
            
            # Parse members
            members = self._parse_struct_members(struct_body)
            
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
                metadata={
                    'struct_name': struct_name,
                    'members': members,
                    'has_struct_pack': bool(struct_pack),
                    'documentation': documentation,
                    'line_number': content[:start_pos].count('\n') + 1,
                    'member_count': len(members),
                    'has_arrays': any('array_dimensions' in m for m in members),
                    'has_pointers': any(m.get('is_pointer', False) for m in members)
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating struct entity: {str(e)}")
            return None
    # #==============================================================================================

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
            # print("\n---------------------------------")
            # print("\n name : ",name)
            if self._is_common_function(name):
                return None
                
            function_content = func_dict['content']
            # print("\n---------------------------------")
            # print("\n function_content : ",function_content)
            # Get context
            # print("\ncontent : ",content)
            start_pos = content.find(function_content)
            print("\nstart_pos : ",start_pos)
            end_pos = start_pos + len(function_content)
            print("\nend_pos : ",end_pos)
            context_before, context_after = self.get_context(
                content, start_pos, end_pos
            )
            # print("\ncontext_before : ",context_before)
            # print("\ncontext_after : ",context_after)
            
            # Parse function signature
            return_type = self._extract_return_type(function_content)
            # print("\nretrun type : ",return_type)
            parameters = self._parse_parameters(func_dict['parameters'])
            # print("\nparameters : ",parameters)
            # print("\n---------------------------")
            # print("\nname : ",name)
            # print("\nfile_path : ",file_path)
            # print("\ncomponent : ",component_name)
            # print("\nincludes : ",includes)
            # print("\nreturn_type : ",return_type)
            # print("\nparameters : ",parameters)
            # print("\nfunc_dict['start_line'] : ",func_dict['start_line'])
            # print("\nfunc_dict['end_line'] : ",func_dict['end_line'])
            # print("\n---------------------------")
            return CodeEntity(
                name=name,
                type='function',
                content=function_content,
                file_path=file_path,
                component=component_name,
                includes=includes,
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
        

    # def _analyze_function_interactions(self, entity: CodeEntity, content: str):
    #     """Analyze function calls, API calls, and struct usage within a function"""
    #     function_content = entity.content
        
    #     # Analyze function calls
    #     print("Analyzing Function Calls...")
    #     for match in self.patterns['function_call'].finditer(function_content):
    #         called_func = match.group(1)
    #         if not self._is_common_function(called_func):
    #             print("\nMatch Found : ",match.group(1))
    #             params = self._parse_parameters(match.group(2))
                
    #             # Determine if it's an API call
    #             is_api = bool(self.patterns['api_call'].match(called_func))
                
    #             call = FunctionCall(
    #                 function_name=called_func,
    #                 component="Unknown",  # Will be updated later
    #                 parameters=params,
    #                 return_type="Unknown",  # Will be updated later
    #                 is_api=is_api
    #             )
                
    #             entity.function_calls.append(call)
    #             if is_api:
    #                 entity.api_calls.append(called_func)
        
    #     # Analyze struct usage
    #     for match in self.patterns['struct_usage'].finditer(function_content):
    #         struct_name = match.group(1)
    #         if struct_name not in entity.structs_used:
    #             entity.structs_used.append(struct_name)



    def _analyze_function_interactions(self, entity: CodeEntity, content: str):
        """Analyze function calls using the token-based parser"""
        parser = CFunctionParser()
        function_calls = parser.parse_function_calls(entity.content)
        
        processed_functions = set()
        
        print(f"Found {len(function_calls)} potential function calls")
        
        for func_name, params, position in function_calls:
            # Skip if already processed or common function
            if func_name in processed_functions or self._is_common_function(func_name):
                continue
                
            processed_functions.add(func_name)
            
            # Determine if it's an API call
            is_api = any(func_name.startswith(prefix) for prefix in 
                        ('CCSP_', 'RDK_', 'RBUS_', 'TR181_', 'CcspCommon_', 'DM_', 'PSM_'))
            
            # Create function call object
            call = FunctionCall(
                function_name=func_name,
                component="Unknown",
                parameters=self._parse_parameters(params),
                return_type="Unknown",
                is_api=is_api,
                line_number=entity.content[:position].count('\n') + 1
            )
            
            entity.function_calls.append(call)
            if is_api:
                entity.api_calls.append(func_name)

    #========================================================================

    # def _create_struct_entity(self, match, content: str, file_path: str, 
    #                         component_name: str) -> CodeEntity:
    #     name = match.group(1)
    #     struct_content = match.group(0)
        
    #     return CodeEntity(
    #         name=name,
    #         type='struct',
    #         content=struct_content,
    #         file_path=file_path,
    #         component=component_name,
    #         metadata={
    #             'fields': self._parse_struct_fields(match.group(2)),
    #             'line_number': content[:match.start()].count('\n') + 1
    #         }
    #     )

    #===================================================================================================
    # def _create_struct_entity(self, match: re.Match, content: str, file_path: str, component_name: str, includes: List[str]) -> Optional[CodeEntity]:
    #     """Create a structure entity from a regex match, supporting typedef structs."""
    #     try:
    #         # Extract the full struct content
    #         struct_content = match.group(0)
    #         struct_body = match.group(1)
            
    #         # Look for typedef name after the closing brace
    #         typedef_name = match.group(2) if len(match.groups()) > 1 else None
            
    #         # Look for struct name after 'struct' keyword
    #         struct_pattern = re.compile(r'(?:typedef\s+)?struct\s+(\w+)?\s*\{')
    #         struct_match = struct_pattern.search(struct_content)
    #         struct_name = struct_match.group(1) if struct_match and struct_match.group(1) else None
            
    #         # Use typedef name if available, otherwise use struct name
    #         final_name = typedef_name or struct_name
            
    #         if not final_name:
    #             logger.warning(f"Skipping anonymous struct in {file_path}")
    #             return None
            
    #         # Parse attributes
    #         attributes = []
    #         attr_pattern = re.compile(r'__attribute__\s*\(\(([^)]+)\)\)')
    #         attr_match = attr_pattern.search(struct_content)
    #         if attr_match:
    #             attributes = [attr.strip() for attr in attr_match.group(1).split(',')]
            
    #         # Parse members
    #         members = self._parse_struct_members(struct_body)
            
    #         # Get struct context
    #         start_pos = content.find(struct_content)
    #         end_pos = start_pos + len(struct_content)
    #         context_before, context_after = self.get_context_struct(content, start_pos, end_pos)
            
    #         # Store typedef mapping if this is a typedef
    #         if typedef_name:
    #             self.type_definitions[typedef_name] = {
    #                 'original_type': f"struct {struct_name}" if struct_name else 'struct',
    #                 'attributes': attributes
    #             }
    #             self.known_struct_types.add(typedef_name)
            
    #         # Create and return the entity
    #         return CodeEntity(
    #             name=final_name,
    #             type='struct',
    #             content=struct_content,
    #             file_path=file_path,
    #             component=component_name,
    #             includes=includes,
    #             metadata={
    #                 'members': members,
    #                 'attributes': attributes,
    #                 'is_typedef': bool(typedef_name),
    #                 'original_struct_name': struct_name,
    #                 'line_number': content[:start_pos].count('\n') + 1,
    #                 'end_line': content[:end_pos].count('\n') + 1,
    #                 'context_before': context_before,
    #                 'context_after': context_after
    #             }
    #         )
            
    #     except Exception as e:
    #         logger.error(f"Error creating struct entity: {str(e)}")
    #         return None

    # def _parse_struct_members(self, struct_content: str) -> List[Dict[str, str]]:
    #     """Parse the members of a struct definition."""
    #     # Remove comments first
    #     content = self._remove_comments(struct_content)
        
    #     # Extract the content between braces
    #     brace_pattern = re.compile(r'{(.*?)}', re.DOTALL)
    #     match = brace_pattern.search(content)
    #     if not match:
    #         return []
            
    #     members_content = match.group(1)
        
    #     # Split into individual member declarations
    #     members = []
    #     for line in members_content.split(';'):
    #         line = line.strip()
    #         if not line:
    #             continue
                
    #         # Parse member declaration
    #         member = self._parse_member_declaration(line)
    #         if member:
    #             members.append(member)
                
    #     return members
    #===============================================================================
    # def _parse_struct_members(self, struct_content: str) -> List[Dict]:
    #     """Parse structure members including arrays, unions, and nested structures."""
    #     members = []
        
    #     # Remove comments
    #     content = re.sub(r'/\*.*?\*/', '', struct_content, flags=re.DOTALL)
    #     content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
    #     # Split into lines and process each member
    #     lines = [line.strip() for line in content.split(';') if line.strip()]
        
    #     for line in lines:
    #         if not line or line.startswith('}'): continue
            
    #         try:
    #             # Check for union
    #             if 'union' in line:
    #                 union_match = self.patterns['union'].search(line)
    #                 if union_match:
    #                     members.append({
    #                         'name': 'u',  # Common name for union in the examples
    #                         'type': 'union',
    #                         'members': self._parse_struct_members(union_match.group(1))
    #                     })
    #                     continue

    #             # Regular member parsing
    #             parts = line.strip().split()
    #             if not parts: continue
                
    #             member = {}
                
    #             # Handle array declarations
    #             array_match = self.patterns['array_declaration'].search(line)
    #             if array_match:
    #                 member['name'] = array_match.group(1)
    #                 member['array_dimensions'] = array_match.group(2)
    #                 # Remove array part from type analysis
    #                 line = line.replace(array_match.group(0), array_match.group(1))
    #                 parts = line.strip().split()

    #             # Basic type and name
    #             if not member.get('name'):
    #                 member['name'] = parts[-1].strip()
                
    #             # Handle pointers
    #             member['is_pointer'] = '*' in line
                
    #             # Build type string
    #             type_parts = parts[:-1]  # Everything except the name
    #             member['type'] = ' '.join(type_parts).replace('*', '').strip()
                
    #             # Check for bit fields
    #             bit_field_match = re.search(r':\s*(\d+)', line)
    #             if bit_field_match:
    #                 member['bit_field_size'] = int(bit_field_match.group(1))
                
    #             # Add member documentation if present
    #             doc_match = re.search(r'/\*\*<\s*(.+?)\s*\*/', line)
    #             if doc_match:
    #                 member['documentation'] = doc_match.group(1).strip()
                
    #             members.append(member)
                
    #         except Exception as e:
    #             logger.error(f"Error parsing member line '{line}': {str(e)}")
    #             continue
                
    #     return members
    #===============================================================================

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
