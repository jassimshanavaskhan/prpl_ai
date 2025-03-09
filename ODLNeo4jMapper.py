# odl_to_neo_mapper.py
from neo4j import GraphDatabase
from typing import Dict, Any, Optional
from dataclasses import asdict
from ODLDefinitionParser import *
from pathlib import Path

class ODLNeo4jMapper:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()

    def clear_existing_data(self):
        """Clear all existing ODL-related nodes and relationships"""
        with self.driver.session() as session:
            session.run("""
                MATCH (n:ODL)-[r*0..]->(m)
                DETACH DELETE n, m
            """)

    def create_parameter_node(self, object_id: str, parameter: Parameter):
        """Create a Parameter node and connect its actions to CodeEntity with underscore prefix"""
        param_data = asdict(parameter)
        param_data["param_type"] = parameter.param_type.name
        
        with self.driver.session() as session:
            session.run("""
                MATCH (obj:Object) WHERE id(obj) = $object_id
                CREATE (param:Parameter {
                    name: $name,
                    param_type: $param_type,
                    attributes: $attributes,
                    default_value: $default_value,
                    userflags: $userflags,
                    counted_with: $counted_with
                })<-[:HAS_PARAMETER]-(obj)
                WITH param
                UNWIND $validators as validator
                CREATE (v:Validator {
                    type: validator.type,
                    values: validator.values
                })<-[:HAS_VALIDATOR]-(param)
                WITH param
                UNWIND $actions as action
                CREATE (a:Action {
                    type: action[0],
                    callback: action[1]
                })<-[:HAS_ACTION]-(param)
                WITH a, action[1] as callback
                // Match existing CodeEntity node with underscore prefix
                MATCH (code:CodeEntity)
                WHERE code.name = '_' + callback
                // Create relationship from action to code entity
                CREATE (a)-[:IMPLEMENTED_BY]->(code)
            """,
                object_id=object_id,
                name=param_data["name"],
                param_type=param_data["param_type"],
                attributes=param_data["attributes"],
                default_value=param_data["default_value"],
                userflags=param_data["userflags"],
                counted_with=param_data["counted_with"],
                validators=param_data["validators"],
                actions=[[k, v] for k, v in (param_data["actions"] or {}).items()]
            )

    def create_implementation_relationships(self):
        """Create relationships between existing Actions/EventHandlers and CodeEntity nodes"""
        with self.driver.session() as session:
            # Create relationships for EventHandlers
            session.run("""
                MATCH (h:EventHandler) 
                WHERE NOT EXISTS((h)-[:IMPLEMENTED_BY]->())
                MATCH (c:CodeEntity)
                WHERE c.name = '_' + h.callback
                CREATE (h)-[:IMPLEMENTED_BY]->(c)
            """)

            # Create relationships for Actions
            session.run("""
                MATCH (a:Action)
                WHERE NOT EXISTS((a)-[:IMPLEMENTED_BY]->())
                MATCH (c:CodeEntity)
                WHERE c.name = '_' + a.callback
                CREATE (a)-[:IMPLEMENTED_BY]->(c)
            """)

    def verify_callback_matches(self):
        """Verify and report any callbacks that don't have matching CodeEntity nodes"""
        with self.driver.session() as session:
            # Check EventHandlers
            unmatched_handlers = session.run("""
                MATCH (h:EventHandler)
                WHERE NOT EXISTS((h)-[:IMPLEMENTED_BY]->())
                RETURN h.callback as callback, 'EventHandler' as type
            """).data()

            # Check Actions
            unmatched_actions = session.run("""
                MATCH (a:Action)
                WHERE NOT EXISTS((a)-[:IMPLEMENTED_BY]->())
                RETURN a.callback as callback, 'Action' as type
            """).data()

            if unmatched_handlers or unmatched_actions:
                print("\nWarning: Found callbacks without matching CodeEntity nodes:")
                for item in unmatched_handlers + unmatched_actions:
                    print(f"- {item['type']}: {item['callback']} (expected CodeEntity: _{item['callback']})")
                    
            return unmatched_handlers + unmatched_actions

    def create_event_handler_node(self, odl_id: str, handler: EventHandler):
        """Create an EventHandler node and connect to CodeEntity with underscore prefix"""
        with self.driver.session() as session:
            # First create the event handler and connect to ODL
            session.run("""
                MATCH (odl:ODL) WHERE id(odl) = $odl_id
                CREATE (handler:EventHandler {
                    event_type: $event_type,
                    callback: $callback,
                    filter_expr: $filter_expr
                })<-[:HAS_EVENT_HANDLER]-(odl)
                WITH handler
                // Match existing CodeEntity node with underscore prefix
                MATCH (code:CodeEntity)
                WHERE code.name = '_' + $callback
                // Create relationship from handler to code entity
                CREATE (handler)-[:IMPLEMENTED_BY]->(code)
            """,
                odl_id=odl_id,
                event_type=handler.event_type,
                callback=handler.callback,
                filter_expr=handler.filter_expr
            )

    # def create_odl_node(self, filepath: str, content: str, component_name: str = None) -> str:
    #     """
    #     Create an ODL node with extended metadata and return its identifier
        
    #     Args:
    #         filepath (str): Full path to the ODL file
    #         content (str): File content
    #         component_name (str, optional): Name of the component associated with the ODL file
        
    #     Returns:
    #         str: Neo4j node identifier for the created ODL node
    #     """
    #     with self.driver.session() as session:
    #         # Extract filename from filepath
    #         filename = Path(filepath).name if filepath else None
            
    #         result = session.run("""
    #             CREATE (odl:ODL {
    #                 filepath: $filepath,
    #                 filename: $filename,
    #                 content: $content,
    #                 component: $component,
    #                 created_at: datetime()
    #             })
    #             RETURN id(odl) as odl_id
    #         """, 
    #             filepath=filepath, 
    #             filename=filename,
    #             content=content, 
    #             component=component_name
    #         )
    #         return result.single()["odl_id"]

    def create_odl_node(self, filepath: str, content: str, component_name: str = None) -> str:
        """
        Create an ODL node with extended metadata and return its identifier
        
        Args:
            filepath (str): Full path to the ODL file
            content (str): File content
            component_name (str, optional): Name of the component associated with the ODL file
        
        Returns:
            str: Neo4j node identifier for the created ODL node
        """
        with self.driver.session() as session:
            # Extract filename and name (without .odl extension)
            filename = Path(filepath).name if filepath else None
            name = Path(filepath).stem if filepath else None
            
            result = session.run("""
                CREATE (odl:ODL {
                    filepath: $filepath,
                    filename: $filename,
                    name: $name,
                    content: $content,
                    component: $component,
                    created_at: datetime()
                })
                RETURN id(odl) as odl_id
            """, 
                filepath=filepath, 
                filename=filename,
                name=name,
                content=content, 
                component=component_name
            )
            return result.single()["odl_id"]
        
    # Update map_definition_to_neo4j method to pass component name
    def map_definition_to_neo4j(self, definition_info: Dict[str, Any], odl_content: str, component_name: str = None):
        """
        Map the entire ODL definition to Neo4j including the ODL content
        
        Args:
            definition_info (Dict[str, Any]): Parsed ODL definition
            odl_content (str): Original ODL file content
            component_name (str, optional): Name of the component associated with the ODL file
        """
        # Create the main ODL node with content and additional metadata
        odl_id = self.create_odl_node(
            filepath=definition_info.get("filepath"), 
            content=odl_content, 
            component_name=component_name
        )
        
        # Rest of the method remains the same
        for object_name, object_info in definition_info["objects"].items():
            object_id = self.create_object_node(odl_id, object_name, object_info)
            
            for parameter in object_info["parameters"].values():
                self.create_parameter_node(object_id, parameter)
            
            self._handle_nested_objects(object_id, object_info["nested_objects"])
        
        for handler in definition_info["event_handlers"]:
            self.create_event_handler_node(odl_id, handler)
    # ======================================================================================== MODIFIED

    def create_object_node(self, odl_id: str, object_name: str, object_info: Dict[str, Any]) -> str:
        """Create an Object node and connect it to the ODL node"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (odl:ODL) WHERE id(odl) = $odl_id
                CREATE (obj:Object {
                    name: $name,
                    is_array: $is_array,
                    counted_with: $counted_with
                })<-[:HAS_OBJECT]-(odl)
                RETURN id(obj) as obj_id
            """, 
                odl_id=odl_id,
                name=object_name,
                is_array=object_info["is_array"],
                counted_with=object_info.get("counted_with")
            )
            return result.single()["obj_id"]

    def _handle_nested_objects(self, parent_object_id: str, nested_objects: Dict[str, Any]):
        """Recursively handle nested objects"""
        for object_name, object_info in nested_objects.items():
            # Create nested object node and connect to parent
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (parent:Object) WHERE id(parent) = $parent_id
                    CREATE (obj:Object {
                        name: $name,
                        is_array: $is_array,
                        counted_with: $counted_with
                    })<-[:HAS_NESTED_OBJECT]-(parent)
                    RETURN id(obj) as obj_id
                """,
                    parent_id=parent_object_id,
                    name=object_name,
                    is_array=object_info["is_array"],
                    counted_with=object_info.get("counted_with")
                )
                nested_object_id = result.single()["obj_id"]
            
            # Create parameters for nested object
            for parameter in object_info["parameters"].values():
                self.create_parameter_node(nested_object_id, parameter)
            
            # Recursively handle nested objects of this object
            self._handle_nested_objects(nested_object_id, object_info["nested_objects"])
