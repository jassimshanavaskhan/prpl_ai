from CodeEntityClass import *
def print_entities(entities: List[CodeEntity], indent: int = 0) -> None:
    """
    Print code entities in a structured, readable format.
    
    Args:
        entities: List of CodeEntity objects
        indent: Initial indentation level
    """
    def print_indented(text: str, level: int = 0):
        print("    " * level + text)
        
    def print_function_call(call: FunctionCall, level: int = 0):
        print_indented(f"Function Call: {call.function_name}", level)
        print_indented(f"Component: {call.component}", level + 1)
        print_indented(f"Return Type: {call.return_type}", level + 1)
        if call.parameters:
            print_indented("Parameters:", level + 1)
            for param in call.parameters:
                print_indented(f"- {param}", level + 2)
                
    def print_members(members: List[Dict], level: int = 0):
        for member in members:
            if member['type'] == 'field':
                print_indented(f"Field: {member.get('name', '')}", level)
                print_indented(f"Type: {member.get('field_type', '')}", level + 1)
                if member.get('is_pointer'):
                    print_indented(f"Pointer Count: {member.get('pointer_count', 0)}", level + 1)
                if member.get('array_dimensions'):
                    print_indented(f"Array Dimensions: {member.get('array_dimensions')}", level + 1)
                if member.get('bit_field_size'):
                    print_indented(f"Bit Field Size: {member.get('bit_field_size')}", level + 1)
            elif member['type'] == 'union':
                print_indented(f"Union: {member.get('name', '')}", level)
                if member.get('union_name'):
                    print_indented(f"Union Name: {member['union_name']}", level + 1)
                if member.get('members'):
                    print_indented("Members:", level + 1)
                    print_members(member['members'], level + 2)
                if member.get('structs_used'):
                    print_indented("Structs Used:", level + 1)
                    for struct in member['structs_used']:
                        print_indented(f"- {struct}", level + 2)
            elif member['type'] == 'struct':
                print_indented(f"Struct: {member.get('name', '')}", level)
                if member.get('struct_name'):
                    print_indented(f"Struct Name: {member['struct_name']}", level + 1)
                if member.get('members'):
                    print_indented("Members:", level + 1)
                    print_members(member['members'], level + 2)
                if member.get('structs_used'):
                    print_indented("Structs Used:", level + 1)
                    for struct in member['structs_used']:
                        print_indented(f"- {struct}", level + 2)

    for entity in entities:
        print_indented("=" * 80, indent)
        print_indented(f"Entity: {entity.name}", indent)
        print_indented(f"Type: {entity.type}", indent)
        print_indented(f"File: {entity.file_path}", indent)
        print_indented(f"Component: {entity.component}", indent)
        
        if entity.description:
            print_indented("Description:", indent)
            print_indented(entity.description, indent + 1)
            
        if entity.includes:
            print_indented("Includes:", indent)
            for include in entity.includes:
                print_indented(f"- {include}", indent + 1)
                
        if entity.function_calls:
            print_indented("Function Calls:", indent)
            for call in entity.function_calls:
                print_function_call(call, indent + 1)
                
        if entity.structs_used:
            print_indented("Structs Used:", indent)
            for struct in entity.structs_used:
                print_indented(f"- {struct}", indent + 1)
                
        if entity.api_calls:
            print_indented("API Calls:", indent)
            for api in entity.api_calls:
                print_indented(f"- {api}", indent + 1)
                
        if entity.metadata:
            print_indented("Metadata:", indent)
            for key, value in entity.metadata.items():
                if key == 'members':
                    print_indented("Members:", indent + 1)
                    print_members(value, indent + 2)
                else:
                    print_indented(f"{key}: {value}", indent + 1)
        
        print_indented("-" * 80, indent)