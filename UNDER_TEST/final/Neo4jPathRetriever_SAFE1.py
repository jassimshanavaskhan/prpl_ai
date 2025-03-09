from neo4j import GraphDatabase

class Neo4jPathRetriever:
    def __init__(self, uri: str, username: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            print("Neo4j Path Retriever Initialized Successfully!")
        except Exception as e:
            print(f"Error initializing Neo4j driver: {e}")
            raise

    def get_all_paths(self, node_name: str, max_depth: int, node_type: str = None) -> dict:
        """
        Retrieve all possible paths for a given node up to specified depth, removing redundant subpaths
        
        Args:
            node_name (str): Name of the starting node
            max_depth (int): Maximum depth to traverse
            node_type (str): Type of node ('CodeEntity', 'ODL', 'Object', 'Parameter', etc.)
            
        Returns:
            dict: Dictionary containing incoming and outgoing paths with their details
        """
        incoming_paths = self.get_incoming_paths(node_name, max_depth, node_type)
        outgoing_paths = self.get_outgoing_paths(node_name, max_depth, node_type)
        
        return {
            "incoming_paths": incoming_paths,
            "outgoing_paths": outgoing_paths
        }
    

    def _filter_subpaths(self, paths: list) -> list:
        """
        Filter out paths that are subpaths of longer paths
        
        Args:
            paths (list): List of path dictionaries
            
        Returns:
            list: Filtered list of paths with subpaths removed
        """
        # Sort paths by length in descending order
        sorted_paths = sorted(paths, key=lambda x: x['length'], reverse=True)
        paths_to_keep = []
        
        def is_subpath(shorter_path, longer_path) -> bool:
            """Check if shorter_path is a subpath of longer_path"""
            shorter_sequence = _get_path_signature(shorter_path)
            longer_sequence = _get_path_signature(longer_path)
            
            # Check if shorter sequence appears in longer sequence
            shorter_str = " -> ".join(shorter_sequence)
            longer_str = " -> ".join(longer_sequence)
            return shorter_str in longer_str

        def _get_path_signature(path) -> list:
            """Create a signature for a path that can be used for comparison"""
            return [
                f"{step['from']['type']}({step['from']['name']})-[{step['relationship']}]->{step['to']['type']}({step['to']['name']})"
                for step in path['path_sequence']
            ]

        # Iterate through paths and keep only non-redundant ones
        for current_path in sorted_paths:
            # For the longest paths, always keep them
            if not paths_to_keep:
                paths_to_keep.append(current_path)
                continue
                
            # Check if current path is a subpath of any kept path
            is_redundant = any(
                is_subpath(current_path, kept_path) 
                for kept_path in paths_to_keep
            )
            
            if not is_redundant:
                paths_to_keep.append(current_path)
        
        return sorted(paths_to_keep, key=lambda x: x['length'])


    def get_incoming_paths(self, node_name: str, max_depth: int, node_type: str = None) -> list:
        """Retrieve all incoming paths to the specified node"""
        with self.driver.session() as session:
            # Build the match condition based on node type
            if node_type:
                match_condition = f"end:{node_type}"
            else:
                match_condition = "end"
            
            query = f"""
            MATCH path = (start)-[*1..{max_depth}]->(end {{{self._get_name_property(node_type)}: $node_name}})
            WHERE ALL(rel IN relationships(path) 
                     WHERE type(rel) IN [
                         'CALLS', 'USES_AS_PARAMETER', 'HAS_OBJECT', 
                         'HAS_PARAMETER', 'HAS_EVENT_HANDLER', 
                         'HAS_NESTED_OBJECT', 'HAS_ACTION', 
                         'HAS_VALIDATOR', 'IMPLEMENTED_BY'
                     ])
            RETURN path,
                   [node in nodes(path) | 
                    CASE 
                        WHEN 'name' IN keys(node) THEN node.name 
                        WHEN 'callback' IN keys(node) THEN node.callback
                        WHEN 'event_type' IN keys(node) THEN node.event_type
                        ELSE 'Unknown'
                    END
                   ] as node_names,
                   [node in nodes(path) | labels(node)[0]] as node_types,
                   [rel in relationships(path) | type(rel)] as relationship_types,
                   [rel in relationships(path) | 
                    CASE 
                        WHEN 'line_number' IN keys(rel) THEN rel.line_number 
                        ELSE null 
                    END
                   ] as line_numbers,
                   length(path) as path_length
            ORDER BY path_length
            """

            result = session.run(query, node_name=node_name)
            return self._process_path_results(result)

    def get_outgoing_paths(self, node_name: str, max_depth: int, node_type: str = None) -> list:
        """Retrieve all outgoing paths from the specified node"""
        with self.driver.session() as session:
            # Build the match condition based on node type
            if node_type:
                match_condition = f"start:{node_type}"
            else:
                match_condition = "start"
            
            query = f"""
            MATCH path = (start {{{self._get_name_property(node_type)}: $node_name}})-[*1..{max_depth}]->(end)
            WHERE ALL(rel IN relationships(path) 
                     WHERE type(rel) IN [
                         'CALLS', 'USES_AS_PARAMETER', 'HAS_OBJECT', 
                         'HAS_PARAMETER', 'HAS_EVENT_HANDLER', 
                         'HAS_NESTED_OBJECT', 'HAS_ACTION', 
                         'HAS_VALIDATOR', 'IMPLEMENTED_BY'
                     ])
            RETURN path,
                   [node in nodes(path) | 
                    CASE 
                        WHEN 'name' IN keys(node) THEN node.name 
                        WHEN 'callback' IN keys(node) THEN node.callback
                        WHEN 'event_type' IN keys(node) THEN node.event_type
                        ELSE 'Unknown'
                    END
                   ] as node_names,
                   [node in nodes(path) | labels(node)[0]] as node_types,
                   [rel in relationships(path) | type(rel)] as relationship_types,
                   [rel in relationships(path) | 
                    CASE 
                        WHEN 'line_number' IN keys(rel) THEN rel.line_number 
                        ELSE null 
                    END
                   ] as line_numbers,
                   length(path) as path_length
            ORDER BY path_length
            """

            result = session.run(query, node_name=node_name)
            return self._process_path_results(result)
        
    def get_complete_chains(self, node_name: str, max_depth: int, node_type: str = None) -> list:
        """
        Retrieve all possible complete chains that include the specified node,
        finding the longest possible paths that pass through this node.
        
        Args:
            node_name (str): Name of the target node
            max_depth (int): Maximum depth to traverse in each direction
            node_type (str): Type of node ('CodeEntity', 'ODL', 'Object', 'Parameter', etc.)
            
        Returns:
            list: List of complete chains containing the target node
        """
        with self.driver.session() as session:
            # Build the match condition based on node type
            type_condition = f":{node_type}" if node_type else ""
            name_property = self._get_name_property(node_type)
            
            # Query to find complete chains passing through the target node
            query = f"""
            MATCH path = (start)-[r1*0..{max_depth}]->(middle{type_condition} {{{name_property}: $node_name}})-[r2*0..{max_depth}]->(end)
            WHERE 
                // Ensure we're not getting just the middle node
                (start <> middle OR end <> middle) AND
                // Filter relationship types for the first part of the path
                ALL(rel IN r1 WHERE type(rel) IN [
                    'CALLS', 'USES_AS_PARAMETER', 'HAS_OBJECT',
                    'HAS_PARAMETER', 'HAS_EVENT_HANDLER',
                    'HAS_NESTED_OBJECT', 'HAS_ACTION',
                    'HAS_VALIDATOR', 'IMPLEMENTED_BY'
                ]) AND
                // Filter relationship types for the second part of the path
                ALL(rel IN r2 WHERE type(rel) IN [
                    'CALLS', 'USES_AS_PARAMETER', 'HAS_OBJECT',
                    'HAS_PARAMETER', 'HAS_EVENT_HANDLER',
                    'HAS_NESTED_OBJECT', 'HAS_ACTION',
                    'HAS_VALIDATOR', 'IMPLEMENTED_BY'
                ])
            RETURN path,
                [node in nodes(path) |
                    CASE
                        WHEN 'name' IN keys(node) THEN node.name
                        WHEN 'callback' IN keys(node) THEN node.callback
                        WHEN 'event_type' IN keys(node) THEN node.event_type
                        ELSE 'Unknown'
                    END
                ] as node_names,
                [node in nodes(path) | labels(node)[0]] as node_types,
                [rel in relationships(path) | type(rel)] as relationship_types,
                [rel in relationships(path) |
                    CASE
                        WHEN 'line_number' IN keys(rel) THEN rel.line_number
                        ELSE null
                    END
                ] as line_numbers,
                length(path) as path_length
            ORDER BY path_length DESC
            """
            
            result = session.run(query, node_name=node_name)
            return self._process_path_results(result)

    def print_complete_chains(self, chains: list, node_name: str):
        """
        Print the complete chains in a readable format
        
        Args:
            chains (list): List of complete chains
            node_name (str): Name of the target node
        """
        print(f"\nComplete Chain Analysis for node: {node_name}")
        
        if not chains:
            print("No chains found")
            return
            
        for i, chain in enumerate(chains, 1):
            print(f"\nChain {i} (Length: {chain['length']}):")
            
            # Highlight the target node in the chain
            for step in chain["path_sequence"]:
                from_highlight = "**" if step['from']['name'] == node_name else ""
                to_highlight = "**" if step['to']['name'] == node_name else ""
                
                line_info = f":line {step['line_number']}" if step['line_number'] is not None else ""
                print(f"  {step['from']['type']}({from_highlight}{step['from']['name']}{from_highlight}) "
                    f"--[{step['relationship']}{line_info}]--> "
                    f"{step['to']['type']}({to_highlight}{step['to']['name']}{to_highlight})")

    def _get_name_property(self, node_type: str) -> str:
        """Determine which property to use as the name based on node type"""
        if node_type == 'EventHandler':
            return 'callback'
        if node_type == 'ODL':
            return 'filename'
        return 'name'

    def _process_path_results(self, result) -> list:
        """Process and format the path results from Neo4j queries, removing redundant subpaths"""
        # First, collect and format all paths
        all_paths = []
        for record in result:
            path_info = {
                "nodes": record["node_names"],
                "node_types": record["node_types"],
                "relationships": record["relationship_types"],
                "line_numbers": record["line_numbers"],
                "length": record["path_length"],
                "path_sequence": []
            }
            
            # Create a detailed path sequence
            for i in range(len(path_info["nodes"]) - 1):
                step = {
                    "from": {
                        "name": path_info["nodes"][i],
                        "type": path_info["node_types"][i]
                    },
                    "to": {
                        "name": path_info["nodes"][i + 1],
                        "type": path_info["node_types"][i + 1]
                    },
                    "relationship": path_info["relationships"][i],
                    "line_number": path_info["line_numbers"][i]
                }
                path_info["path_sequence"].append(step)
            
            all_paths.append(path_info)

        # Filter out subpaths
        filtered_paths = self._filter_subpaths(all_paths)
        return filtered_paths

    def print_paths(self, paths: dict, node_name: str):
        """
        Print the paths in a readable format
        
        Args:
            paths (dict): Dictionary containing incoming and outgoing paths
            node_name (str): Name of the central node
        """
        print(f"\nPath Analysis for node: {node_name}")
        print("\n=== Incoming Paths ===")
        self._print_path_direction(paths["incoming_paths"])
        
        print("\n=== Outgoing Paths ===")
        self._print_path_direction(paths["outgoing_paths"])

    def _print_path_direction(self, direction_paths: list):
        """Helper method to print paths for each direction"""
        if not direction_paths:
            print("No paths found")
            return
            
        for i, path in enumerate(direction_paths, 1):
            print(f"\nPath {i} (Length: {path['length']}):")
            for step in path["path_sequence"]:
                line_info = f":line {step['line_number']}" if step['line_number'] is not None else ""
                print(f"  {step['from']['type']}({step['from']['name']}) "
                      f"--[{step['relationship']}{line_info}]--> "
                      f"{step['to']['type']}({step['to']['name']})")

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            print("Neo4j Path Retriever Connection Closed.")

