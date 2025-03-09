# from dataclasses import dataclass, field
# from typing import Dict, List, Set, Any, Optional, Tuple
# from typing import Generator

# @dataclass
# class Token:
#     type: str
#     value: str
#     position: int

# class CFunctionParser:
#     def __init__(self):
#         self.keywords = {'if', 'while', 'for', 'switch', 'return'}
#         self.current_pos = 0
        
#     def tokenize(self, content: str) -> Generator[Token, None, None]:
#         """Tokenize C code content"""
#         i = 0
#         length = len(content)
        
#         while i < length:
#             char = content[i]
            
#             # Skip whitespace
#             if char.isspace():
#                 i += 1
#                 continue
                
#             # Skip comments
#             if char == '/' and i + 1 < length:
#                 if content[i + 1] == '/':  # Single line comment
#                     while i < length and content[i] != '\n':
#                         i += 1
#                     continue
#                 elif content[i + 1] == '*':  # Multi-line comment
#                     i += 2
#                     while i < length - 1 and not (content[i] == '*' and content[i + 1] == '/'):
#                         i += 1
#                     i += 2
#                     continue
            
#             # Identifiers and keywords
#             if char.isalpha() or char == '_':
#                 start = i
#                 while i < length and (content[i].isalnum() or content[i] == '_'):
#                     i += 1
#                 value = content[start:i]
#                 yield Token('IDENTIFIER', value, start)
#                 continue
            
#             # Operators and punctuation
#             if char in '(){},;':
#                 yield Token(char, char, i)
#                 i += 1
#                 continue
            
#             # Arrow operator
#             if char == '-' and i + 1 < length and content[i + 1] == '>':
#                 yield Token('ARROW', '->', i)
#                 i += 2
#                 continue
                
#             # Dot operator
#             if char == '.':
#                 yield Token('DOT', '.', i)
#                 i += 1
#                 continue
            
#             i += 1

#     def parse_function_calls(self, content: str) -> List[Tuple[str, str, int]]:
#         """Parse function calls returning (name, params, position)"""
#         tokens = list(self.tokenize(content))
#         calls = []
#         i = 0
        
#         while i < len(tokens):
#             token = tokens[i]
            
#             # Look for identifier followed by open parenthesis
#             if (token.type == 'IDENTIFIER' and 
#                 i + 1 < len(tokens) and 
#                 tokens[i + 1].type == '(' and
#                 token.value not in self.keywords):
                
#                 # Get function name
#                 func_name = token.value
#                 position = token.position
                
#                 # Handle member access (-> or .)
#                 if i > 1:
#                     prev_token = tokens[i - 1]
#                     if prev_token.type in ('ARROW', 'DOT'):
#                         if i > 2 and tokens[i - 2].type == 'IDENTIFIER':
#                             func_name = f"{tokens[i - 2].value}{prev_token.value}{func_name}"
                
#                 # Extract parameters
#                 i += 2  # Skip to first parameter
#                 param_tokens = []
#                 paren_count = 1
                
#                 while i < len(tokens) and paren_count > 0:
#                     if tokens[i].type == '(':
#                         paren_count += 1
#                     elif tokens[i].type == ')':
#                         paren_count -= 1
#                     if paren_count > 0:
#                         param_tokens.append(tokens[i])
#                     i += 1
                
#                 # Convert parameter tokens to string
#                 params = self._tokens_to_params(param_tokens)
#                 calls.append((func_name, params, position))
#             else:
#                 i += 1
                
#         return calls
    
#     def _tokens_to_params(self, tokens: List[Token]) -> str:
#         """Convert parameter tokens to string representation"""
#         return ''.join(t.value for t in tokens).strip()


from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from typing import Generator

@dataclass
class Token:
    type: str
    value: str
    position: int

class CFunctionParser:
    def __init__(self):
        self.keywords = {'if', 'while', 'for', 'switch', 'return'}
        self.current_pos = 0
        
    def tokenize(self, content: str) -> Generator[Token, None, None]:
        """Tokenize C code content"""
        i = 0
        length = len(content)
        
        while i < length:
            char = content[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
                
            # Skip comments
            if char == '/' and i + 1 < length:
                if content[i + 1] == '/':  # Single line comment
                    while i < length and content[i] != '\n':
                        i += 1
                    continue
                elif content[i + 1] == '*':  # Multi-line comment
                    i += 2
                    while i < length - 1 and not (content[i] == '*' and content[i + 1] == '/'):
                        i += 1
                    i += 2
                    continue
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                start = i
                while i < length and (content[i].isalnum() or content[i] == '_'):
                    i += 1
                value = content[start:i]
                token_type = 'KEYWORD' if value in self.keywords else 'IDENTIFIER'
                yield Token(token_type, value, start)
                continue
            
            # Operators and punctuation
            if char in '(){},;':
                yield Token(char, char, i)
                i += 1
                continue
            
            # Arrow operator
            if char == '-' and i + 1 < length and content[i + 1] == '>':
                yield Token('ARROW', '->', i)
                i += 2
                continue
                
            # Dot operator
            if char == '.':
                yield Token('DOT', '.', i)
                i += 1
                continue
            
            i += 1

    def is_function_declaration(self, tokens: List[Token], start_idx: int) -> bool:
        """
        Check if the identifier at start_idx is part of a function declaration.
        Look for patterns like:
        - return_type function_name(params) {
        - struct_name* function_name(params) {
        """
        if start_idx <= 0:
            return False
            
        # Look ahead for opening brace after parentheses
        i = start_idx
        while i < len(tokens):
            if tokens[i].type == '{':
                return True
            if tokens[i].type == ';':
                return False
            i += 1
            
        return False

    # def parse_function_calls(self, content: str) -> List[Tuple[str, str, int]]:
    #     """Parse function calls returning (name, params, position)"""
    #     tokens = list(self.tokenize(content))
    #     calls = []
    #     i = 0
        
    #     while i < len(tokens):
    #         token = tokens[i]
            
    #         # Look for identifier followed by open parenthesis
    #         if (token.type == 'IDENTIFIER' and 
    #             i + 1 < len(tokens) and 
    #             tokens[i + 1].type == '(' and
    #             token.value not in self.keywords):
                
    #             # Skip if this is a function declaration
    #             if not self.is_function_declaration(tokens, i):
    #                 # Get function name
    #                 func_name = token.value
    #                 position = token.position
                    
    #                 # Handle member access (-> or .)
    #                 if i > 1:
    #                     prev_token = tokens[i - 1]
    #                     if prev_token.type in ('ARROW', 'DOT'):
    #                         if i > 2 and tokens[i - 2].type == 'IDENTIFIER':
    #                             func_name = f"{tokens[i - 2].value}{prev_token.value}{func_name}"
                    
    #                 # Extract parameters
    #                 i += 2  # Skip to first parameter
    #                 param_tokens = []
    #                 paren_count = 1
                    
    #                 while i < len(tokens) and paren_count > 0:
    #                     if tokens[i].type == '(':
    #                         paren_count += 1
    #                     elif tokens[i].type == ')':
    #                         paren_count -= 1
    #                     if paren_count > 0:
    #                         param_tokens.append(tokens[i])
    #                     i += 1
                    
    #                 # Convert parameter tokens to string
    #                 params = self._tokens_to_params(param_tokens)
    #                 calls.append((func_name, params, position))
    #         i += 1
                
    #     return calls
    #================================================================================= ADDED
    def _identify_function_in_param(self,param: str) -> Optional[str]:
        """
        Identify if a parameter is a function call and extract the function name
        """
        # Remove whitespace
        param = param.strip()
        
        # Check for function call pattern (identifier followed by parentheses)
        if '(' in param and ')' in param:
            # Extract the function name before the parentheses
            func_name = param.split('(')[0].strip()
            # Ensure it's a valid identifier
            if func_name and func_name not in self.keywords:
                return func_name
        return None

    def _parse_function_parameters(self,params_str: str) -> Tuple[List[str], List[str]]:
        """
        Convert parameter string to list of parameters and identify function parameters
        Returns (parameters list, function parameters list)
        """
        if not params_str.strip():
            return [], []
            
        params = []
        param_functions = []
        current_param = []
        paren_count = 0
        
        for char in params_str:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                param = ''.join(current_param).strip()
                params.append(param)
                
                # Check if the parameter is a function call
                func_name = self._identify_function_in_param(param)
                if func_name:
                    param_functions.append(func_name)
                    
                current_param = []
                continue
                
            current_param.append(char)
        
        if current_param:
            param = ''.join(current_param).strip()
            params.append(param)
            
            # Check last parameter for function
            func_name = self._identify_function_in_param(param)
            if func_name:
                param_functions.append(func_name)
        
        return params, param_functions

    def parse_function_calls(self,content: str) -> List[Tuple[str, str, int, List[str]]]:
        """
        Parse function calls returning (name, params, position, parameter_functions)
        """
        tokens = list(self.tokenize(content))
        calls = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Look for identifier followed by open parenthesis
            if (token.type == 'IDENTIFIER' and 
                i + 1 < len(tokens) and
                tokens[i + 1].type == '(' and
                token.value not in self.keywords):
                
                # Skip if this is a function declaration
                if not self.is_function_declaration(tokens, i):
                    # Get function name
                    func_name = token.value
                    position = token.position
                    
                    # Handle member access (-> or .)
                    if i > 1:
                        prev_token = tokens[i - 1]
                        if prev_token.type in ('ARROW', 'DOT'):
                            if i > 2 and tokens[i - 2].type == 'IDENTIFIER':
                                func_name = f"{tokens[i - 2].value}{prev_token.value}{func_name}"
                    
                    # Extract parameters
                    i += 2  # Skip to first parameter
                    param_tokens = []
                    paren_count = 1
                    
                    while i < len(tokens) and paren_count > 0:
                        if tokens[i].type == '(':
                            paren_count += 1
                        elif tokens[i].type == ')':
                            paren_count -= 1
                        if paren_count > 0:
                            param_tokens.append(tokens[i])
                        i += 1
                    
                    # Convert parameter tokens to string
                    params = self._tokens_to_params(param_tokens)
                    params_list, param_functions = self._parse_function_parameters(params)
                    calls.append((func_name, params, position, param_functions))
            i += 1
            
        return calls
    #================================================================================= ADDED
        
    def _tokens_to_params(self, tokens: List[Token]) -> str:
        """Convert parameter tokens to string representation"""
        return ''.join(t.value for t in tokens).strip()