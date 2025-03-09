from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class FunctionCall:
    """
    Represents a function call within a code entity
    """
    function_name: str
    component: Optional[str] = None
    parameters: Optional[List[Any]] = None
    return_type: Optional[str] = None
    is_api: bool = False
    line_number: Optional[int] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None

@dataclass
class CodeEntity:
    """
    Represents a code entity with its comprehensive properties
    """
    name: str
    type: str
    content: str
    file_path: str
    component: Optional[str] = None
    description: Optional[str] = None
    function_calls: List[FunctionCall] = field(default_factory=list)
    structs_used: Optional[List[str]] = field(default_factory=list)
    api_calls: Optional[List[str]] = field(default_factory=list)
    includes: Optional[List[str]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    content_hash: Optional[str] = None

class Neo4jCodeEntityProcessor:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j driver connection
        
        :param uri: Neo4j database URI
        :param username: Neo4j username
        :param password: Neo4j password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            print("Neo4j Driver Initialized Successfully!")
        except Exception as e:
            print(f"Error initializing Neo4j driver: {e}")
            raise

    def prepare_code_entity_dict(self, code_entity: CodeEntity) -> Dict[str, Any]:
        """
        Prepare a dictionary representation of a code entity
        
        :param code_entity: Code entity object
        :return: Dictionary representation of the code entity
        """
        return {
            "name": code_entity.name,
            "type": code_entity.type,
            "content": code_entity.content,
            "file_path": code_entity.file_path,
            "component": code_entity.component or "",
            "description": code_entity.description or "",
            "function_calls": [
                {
                    "function_name": call.function_name,
                    "component": call.component or "",
                    "parameters": str(call.parameters or []),
                    "return_type": call.return_type or "",
                    "is_api": call.is_api,
                    "line_number": call.line_number,
                    "context_before": call.context_before or "",
                    "context_after": call.context_after or ""
                }
                for call in code_entity.function_calls
            ],
            "structs_used": code_entity.structs_used or [],
            "api_calls": code_entity.api_calls or [],
            "includes": code_entity.includes or [],
            "metadata": str(code_entity.metadata or {}),
            "content_hash": code_entity.content_hash or ""
        }

    def create_code_entities(self, code_entities: List[CodeEntity]):
        """
        Create code entity nodes in Neo4j for a list of code entities
        
        :param code_entities: List of code entity objects
        """
        with self.driver.session() as session:
            # Cypher query to create or merge code entity node
            create_node_query = """
            MERGE (e:CodeEntity {
                name: $name,
                type: $type,
                file_path: $file_path,
                component: $component
            })
            SET e.content = $content
            SET e.description = $description
            SET e.structs_used = $structs_used
            SET e.api_calls = $api_calls
            SET e.includes = $includes
            SET e.metadata = $metadata
            SET e.content_hash = $content_hash
            """
            
            # Process each code entity
            for code_entity in code_entities:
                code_entity_dict = self.prepare_code_entity_dict(code_entity)
                
                try:
                    # Execute the query
                    session.run(create_node_query, code_entity_dict)
                    print(f"Code Entity Node Created: {code_entity_dict['name']}")
                except Exception as e:
                    print(f"Error creating node for {code_entity_dict['name']}: {e}")

    def create_function_call_relationships(self, code_entities: List[CodeEntity]):
        """
        Create relationships between code entities based on function calls
        
        :param code_entities: List of code entity objects
        """
        with self.driver.session() as session:
            # Cypher query to create relationships between functions
            create_relationship_query = """
            MATCH (source:CodeEntity {name: $source_name})
            MERGE (target:CodeEntity {name: $target_name})
            MERGE (source)-[:CALLS {
                parameters: $parameters,
                is_api: $is_api,
                line_number: $line_number,
                context_before: $context_before,
                context_after: $context_after
            }]->(target)
            """
            
            # Process each code entity
            for code_entity in code_entities:
                code_entity_dict = self.prepare_code_entity_dict(code_entity)
                
                # Iterate through function calls and create relationships
                for call in code_entity_dict.get('function_calls', []):
                    try:
                        session.run(create_relationship_query, {
                            "source_name": code_entity_dict['name'],
                            "target_name": call['function_name'],
                            "parameters": call['parameters'],
                            "is_api": call['is_api'],
                            "line_number": call['line_number'],
                            "context_before": call['context_before'],
                            "context_after": call['context_after']
                        })
                        print(f"Function Call Relationship Created: {code_entity_dict['name']} -> {call['function_name']}")
                    except Exception as e:
                        print(f"Error creating relationship: {e}")

    def retrieve_function_call_paths(self, code_entities: List[CodeEntity]):
        """
        Retrieve function call paths for each code entity
        
        :param code_entities: List of code entity objects
        :return: Dictionary of function call paths for each entity
        """
        paths_result = {}
        
        with self.driver.session() as session:
            # Cypher query to find paths of function calls
            path_query = """
            MATCH path = (start:CodeEntity)-[:CALLS*1..5]->(end:CodeEntity {name: $function_name})
            RETURN 
                [node in nodes(path) | node.name] AS function_path,
                [rel in relationships(path) | {
                    parameters: rel.parameters,
                    is_api: rel.is_api,
                    line_number: rel.line_number
                }] AS relationship_details,
                length(path) AS path_length
            ORDER BY path_length
            LIMIT 10
            """
            
            # Process each code entity
            for code_entity in code_entities:
                code_entity_dict = self.prepare_code_entity_dict(code_entity)
                
                try:
                    # Execute the query
                    result = session.run(path_query, {"function_name": code_entity_dict['name']})
                    
                    # Store paths for the current entity
                    paths_result[code_entity_dict['name']] = []
                    
                    # Print and store retrieved paths
                    print(f"\nFunction Call Paths for {code_entity_dict['name']}:")
                    for record in result:
                        path_info = {
                            "path": ' -> '.join(record['function_path']),
                            "relationship_details": record['relationship_details'],
                            "path_length": record['path_length']
                        }
                        paths_result[code_entity_dict['name']].append(path_info)
                        
                        print(f"Path: {path_info['path']}")
                        print(f"Relationship Details: {path_info['relationship_details']}")
                        print(f"Path Length: {path_info['path_length']}")
                
                except Exception as e:
                    print(f"Error retrieving paths for {code_entity_dict['name']}: {e}")
        
        return paths_result

    def close(self):
        """
        Close the Neo4j driver connection
        """
        if self.driver:
            self.driver.close()
            print("Neo4j Driver Connection Closed.")

# def main():
#     # Neo4j connection details
#     NEO4J_URI = "bolt://localhost:7687"
#     NEO4J_USERNAME = "neo4j"
#     NEO4J_PASSWORD = "Jassim@123"

#     try:
#         # Initialize the processor
#         processor = Neo4jCodeEntityProcessor(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

#         # Create code entities (you'll replace this with your actual list)
#         code_entities =entities

#         # Create code entity nodes
#         processor.create_code_entities(code_entities)

#         # Create function call relationships
#         processor.create_function_call_relationships(code_entities)

#         # Retrieve and print function call paths
#         paths = processor.retrieve_function_call_paths(code_entities)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         # Ensure the connection is closed
#         processor.close()

# if __name__ == "__main__":
#     main()