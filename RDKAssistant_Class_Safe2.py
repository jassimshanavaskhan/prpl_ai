from neo4j_CodeContextRetriever import CodeContextRetriever
from neo4j_Class import Neo4jCodeEntityProcessor
from LoadEnities import load_entities
from SaveEntities import save_entities
from RenderDiagram import render_diagram
from SequenceDiagramGenerator import SequenceDiagramGenerator
from ImprovedRateLimitedGemini import ImprovedRateLimitedGeminiProcessor
import os
from EnhancedCodeParserClass import EnhancedCodeParser
from collections import defaultdict
from EnhancedVectorSearchClass import EnhancedVectorSearch
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from google.generativeai import configure
from CodeEntityClass import CodeEntity
from google.oauth2.service_account import Credentials
from logger import logger
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from tqdm import tqdm
from VectorStoreManager import VectorStoreManager
from ProcessingStateClass import ProcessingState
import pickle

# credential_path = "credentials.json"
# # Load the credentials from the JSON file
# credentials = Credentials.from_service_account_file(credential_path)

import os
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.auth import exceptions

# Load environment variables
load_dotenv()


# For development, you can still use local credentials file
if os.path.exists('credentials.json'):
    credentials = Credentials.from_service_account_file('credentials.json')
else:
    # For production, use environment variables
    credentials_info = {
        "type": os.getenv("GOOGLE_TYPE"),
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
        "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
    }
    credentials = Credentials.from_service_account_info(credentials_info)

class RDKAssistant:
    def __init__(self, code_base_path: str, gemini_api_key: str,):
        self.code_base_path = Path(code_base_path)
        self.parser = EnhancedCodeParser()
        self.entities: Dict[str, CodeEntity] = {}
        self.processing_state = ProcessingState.load()
        self.enhanced_search = None




        
        # Initialize embedding model
        # self.embedding_model = GooglePalmEmbeddings(google_api_key=gemini_api_key)
        #------------------------------
        configure(credentials=credentials)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            credentials=credentials
        )
        #-------------------------
        self.vector_store = VectorStoreManager(self.embedding_model)
        self.neo4j_context_retriever = CodeContextRetriever(
            neo4j_uri=os.environ.get('NEO4J_URI'), 
            neo4j_username=os.environ.get('NEO4J_USERNAME'), 
            neo4j_password=os.environ.get('NEO4J_PASSWORD'), 
            gemini_api_key=os.environ.get('GEMINI_API_KEY')
        )
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Initialize sequence generator
        self.sequence_generator = None
        


    def initialize(self, force_rebuild: bool = False, max_workers: int = 1):
        """Initialize the assistant by processing the codebase"""
        cache_file = Path("rdk_assistant_cache.json")
        vector_store_path = "vector_stores"
        #==========================================================
        self.enhanced_search = EnhancedVectorSearch(
            self.gemini_model,
            self.vector_store,
            self.entities
        )
        #==========================================================
        
        try:
            if not force_rebuild and cache_file.exists() and os.path.exists(vector_store_path):
                logger.info("Loading from cache...")
                self.entities = load_entities(cache_file)
                
                if self.vector_store.load_indices(vector_store_path):
                    logger.info("Vector stores loaded successfully")
                else:
                    logger.warning("Failed to load vector stores, rebuilding...")
                    self._process_codebase(max_workers)
            else:
                logger.info("Processing codebase...")
                self._process_codebase(max_workers)
            
            self.sequence_generator = SequenceDiagramGenerator(
                self.vector_store,
                self.entities
            )
            #==========================================
            logger.info(f"Loaded {len(self.entities)} entities")
            logger.debug(f"Entity names: {list(self.entities.keys())}")
            
            # Verify vector store initialization
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                raise RuntimeError("Vector store not properly initialized")
            #==========================================
            logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise
    

    def _process_codebase(self, max_workers: int):
        """Process all source files in the codebase with parallel processing"""
        try:
            source_files = []
            for ext in ['.c', '.cpp', '.cc']:
                source_files.extend(self.code_base_path.rglob(f'*{ext}'))
            
            total_files = len(source_files)
            logger.info(f"Found {total_files} source files")
            
            if self.processing_state.processed_files:
                source_files = [f for f in source_files 
                            if str(f) not in self.processing_state.processed_files]
                logger.info(f"Resuming processing with {len(source_files)} remaining files")
            
            processed_entities = []

            # Use tqdm for progress bar without threading
            for file_path in tqdm(source_files, desc="Processing files"):
                try:
                    # Call _process_single_file directly
                    entities = self._process_single_file(
                        str(file_path),
                        self._determine_component_name(file_path)
                    )
                    
                    if entities:
                        processed_entities.extend(entities)
                        # print("\nprocessed_entities : ", processed_entities)
                        
                    # Update processing state
                    self.processing_state.processed_files.add(str(file_path))
                    if len(self.processing_state.processed_files) % 100 == 0:
                        self.processing_state.save()
                        
                except Exception as exc:
                    print(f"Error processing {file_path}: {exc}")
            #================================================================================
            # Update entities dictionary
            for entity in processed_entities:
                self.entities[entity.name] = entity
            
            # Create vector store indices
            logger.info(f"passing entities from process codebase to save in vector store...")
            self.vector_store.create_indices(list(self.entities.values()))
            
            # Update function call component information
            self._update_function_call_components()
            
            # Save cache using JSON serialization instead of pickle
            save_entities(self.entities, "rdk_assistant_cache.json")
            
            # Save final processing state
            self.processing_state.save()
            
        except Exception as e:
            logger.error(f"Error processing codebase: {str(e)}")
            raise
    #---------------------------------------------------------------------

    def _update_function_call_components(self):
        """Update component information for function calls"""
        for entity in self.entities.values():
            for call in entity.function_calls:
                if called_entity := self.entities.get(call.function_name):
                    call.component = called_entity.component
    


    def _determine_component_name(self, file_path: Path) -> str:
        """Determine RDK component name from file path"""
        # Common RDK component names
        rdk_components = {
            'CcspCr', 'CcspCommonLibrary', 'CcspPsm', 'RdkWanManager',
            'RdkWifiManager', 'RdkCellularManager', 'CcspTr069Pa',
            'CcspLMLite', 'CcspEthAgent', 'Utopia', 'hal', 'webpa',
            'OneWifi','CcspWifiAgent','tr181-deviceinfo','CcspPandM','CcspEthAgent','tr181-dslite'
        }
        
        # Check path parts for component names
        parts = file_path.parts
        for part in parts:
            for component in rdk_components:
                if component.lower() in part.lower():
                    return component
        
        # If no known component found, use parent directory name
        return parts[-2] if len(parts) > 1 else 'Unknown'
    
    def _create_function_analysis_prompt(self, entity: CodeEntity) -> str:
        """Create analysis prompt for function entity"""
        return f"""
        Analyze this RDK function:
        
        Function Name: {entity.name}
        Component: {entity.component}
        Return Type: {entity.metadata.get('return_type', 'Unknown')}
        Parameters: {', '.join(entity.metadata.get('parameters', []))}
        
        Code:
        {entity.content}
        
        Please provide a concise analysis covering:
        1. Main purpose and functionality
        2. Key operations and data flow
        3. Interaction with other components (if any)
        4. Important parameters and return values
        5. Any specific RDK-related operations
        """

    def _create_struct_analysis_prompt(self, entity: CodeEntity) -> str:
        """Create analysis prompt for struct entity"""
        return f"""
        Analyze this RDK structure:
        
        Structure Name: {entity.name}
        Component: {entity.component}
        
        Definition:
        {entity.content}
        
        Please provide a concise analysis covering:
        1. Purpose of this structure
        2. Key fields and their significance
        3. Usage context in RDK
        4. Related components or interfaces
        5. Any specific RDK-related details
        """

    def _process_single_file(self, file_path: Path, component_name: str) -> List[CodeEntity]:
        try:
            entities = self.parser.parse_file(str(file_path), component_name)
            logger.info(f"Parsed {len(entities)} entities from {file_path}")
            
            processor = ImprovedRateLimitedGeminiProcessor(
                self.gemini_model,
                requests_per_minute=30,  # Conservative rate limit
                cooldown_period=120      # 2 minute cooldown
            )
            # import ipdb; ipdb.set_trace()
            processor.process_entities_batch(entities)
            logger.info(f"returing processed entities after adding gemini reponse to process codebase...")
            return entities
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    #========================================== ADDED ===========================================

    def handle_user_interaction(self):
        """Main interaction loop for the RDK Assistant"""
        while True:
            print("\nRDK Assistant Menu:")
            print("1. Generate Sequence Diagram")
            print("2. Ask a question")
            print("3. Add New Code to Database")
            print("4. Exit")

            choice = input("Enter your choice (1-4): ")

            if choice == "1":
                query = input("Enter your query for sequence diagram generation: ")
                max_depth = int(input("Enter maximum depth for sequence diagram (default=5): ") or "5")
                diagram = self.generate_sequence_diagram(query, max_depth)
                print("\nGenerated Sequence Diagram:")
                render_diagram(diagram, 'diagrams/sequence.png')
                print(diagram)

            elif choice == "2":
                # import ipdb; ipdb.set_trace()
                query = input("Enter your question: ")
                # response = self.handle_user_query(query)
                # print("\nResponse:")
                # print(response)
                #================================================== SAFE
                result= self.neo4j_context_retriever.process_query(query)
                # Print response
                # import ipdb; ipdb.set_trace()
                print("\nResponse:\n", result["response"])
                
                # Display diagram path if generated
                if result["diagram_path"]:
                    print(f"\nSequence Diagram generated: {result['diagram_path']}")
                
                print("\n" + "="*50 + "\n")

            elif choice == "3":
                # print("Analyze output here...")
                # pdb.set_trace()
                # print("Script will resume after exiting pdb...")
                path = input("Enter path to new code file or directory: ")
                self._process_new_code(Path(path))

            elif choice == "4":
                print("Exiting RDK Assistant...")
                break

            else:
                print("Invalid choice. Please try again.")
    #================================================================================
    def _process_new_code(self, path: Path):
        """Process new code files and add to database with Neo4j integration"""
        try:
            if path.is_file():
                files = [path]
            else:
                files = []
                for ext in ['.c', '.cpp', '.h', '.cc']:
                    files.extend(path.rglob(f'*{ext}'))
            
            # Neo4j connection details - replace with your actual credentials
            # NEO4J_URI = "bolt://localhost:7687"
            # NEO4J_USERNAME = "neo4j"
            # NEO4J_PASSWORD = "Jassim@123"
            # Initialize Neo4j processor
            neo4j_processor = Neo4jCodeEntityProcessor(
                uri=os.environ.get('NEO4J_URI'), 
                username=os.environ.get('NEO4J_USERNAME'), 
                password= os.environ.get('NEO4J_PASSWORD')
            )
            
            processed_entities = []
            
            for file_path in tqdm(files, desc="Processing new files"):
                component_name = self._determine_component_name(file_path)
                logger.info(f"===========================>>> {component_name}")
                entities = self._process_single_file(file_path, component_name)
                
                for entity in entities:
                    self.entities[entity.name] = entity
                    processed_entities.append(entity)
            
            # Neo4j Integration: Create Code Entity Nodes
            neo4j_processor.create_code_entities(processed_entities)
            
            # Neo4j Integration: Create Function Call Relationships
            neo4j_processor.create_function_call_relationships(processed_entities)
            
            # Optional: Retrieve Function Call Paths (for debugging or analysis)
            # paths = neo4j_processor.retrieve_function_call_paths(processed_entities)
            
            # Update vector stores
            self.vector_store.create_indices(list(self.entities.values()))
            
            # Save updated cache
            with open("rdk_assistant_cache.pkl", 'wb') as f:
                pickle.dump(self.entities, f)
            
            print(f"Successfully processed {len(files)} new files")
            
            # Close Neo4j connection
            neo4j_processor.close()
            
        except Exception as e:
            logger.error(f"Error processing new code: {str(e)}")
            print(f"Error processing new code: {str(e)}")
    #========================================================================== 
    # def _process_new_code(self, path: Path):
    #     """Process new code files and add to database"""
    #     try:
    #         if path.is_file():
    #             files = [path]
    #         else:
    #             files = []
    #             for ext in ['.c', '.cpp', '.h', '.cc']:
    #                 files.extend(path.rglob(f'*{ext}'))
            
    #         i=1
    #         for file_path  in files:
    #             # print("\nJAS50 ============= >> Listing all files in provided directory...")
    #             # print("\nJAS50 ============= >> ",i ,') ',file_path)
    #             i=i+1
    #         for file_path in tqdm(files, desc="Processing new files"):
    #             # print("\nJAS51 ============= >> Processing File : ",file_path)
    #             component_name = self._determine_component_name(file_path)
    #             # print("\nJAS51 ============= >> From Component : ",component_name)
    #             entities = self._process_single_file(file_path, component_name)
                
    #             for entity in entities:
    #                 self.entities[entity.name] = entity
            
    #         # Update vector stores
    #         self.vector_store.create_indices(list(self.entities.values()))
            
    #         # Save updated cache
    #         with open("rdk_assistant_cache.pkl", 'wb') as f:
    #             pickle.dump(self.entities, f)
            
    #         print(f"Successfully processed {len(files)} new files")
            
    #     except Exception as e:
    #         logger.error(f"Error processing new code: {str(e)}")
    #         print(f"Error processing new code: {str(e)}")

    #================================================================================== SAFE JAS
    # def handle_user_query(self, query: str) -> str:
    #     try:
    #         relevant_entities, response = self.enhanced_search.contextual_search(query)
    #         logger.info(f"-------------------------------")
    #         logger.info(f"response : {response}")
    #         logger.info(f"-------------------------------")
    #         logger.info(f"response : {relevant_entities}")
    #         logger.info(f"-------------------------------")
    #         return response
    #     except Exception as e:
    #         logger.error(f"Error handling user query: {str(e)}")
    #         return "I'm sorry, there was an error processing your query. Please try again later."
    def handle_user_query(self, query: str) -> str:
        """Handle user queries and provide relevant responses"""
        try:
            # Search for relevant entities based on the query
            relevant_entities = self.search_relevant_entities(query)
            # Generate a response using Gemini
            response = self.generate_response_from_entities(query, relevant_entities)

            return response
        except Exception as e:
            logger.error(f"Error handling user query: {str(e)}")
            return "I'm sorry, there was an error processing your query. Please try again later."
    #================================================================================== SAFE JAS
    # def handle_user_query(self, query: str) -> str:
    #     """Enhanced query handling with log analysis"""
    #     try:
    #         # First analyze logs
    #         log_analysis = self.analyze_logs(query)
            
    #         # Search for relevant entities using both query and log analysis
    #         relevant_entities = self.search_relevant_entities(query)
            
    #         # Generate response using both code entities and log context
    #         response = self.generate_response_from_entities(
    #             query, 
    #             relevant_entities,
    #             log_analysis
    #         )
            
    #         return response
    #     except Exception as e:
    #         logger.error(f"Error handling user query: {str(e)}")
    #         return "I'm sorry, there was an error processing your query. Please try again later."
    #================================================================================== added
    def generate_response_from_entities(self, query: str, entities: List[CodeEntity]) -> str:
        """Generate a response to the user query using the relevant entities"""
        try:
            # Create a prompt that combines the user query and the relevant entities
            # prompt = f"""
            # User Query: {query}

            # Relevant Code Entities:
            # {'\n'.join([f'- {entity.name} ({entity.type}) in {entity.component}' for entity in entities])}

            # Please provide a concise and informative response to the user's query, leveraging the context provided by the relevant code entities. Focus on explaining the functionality, interactions, and RDK-specific details.
            # """
            prompt = (
                f"User Query: {query}\n\n"
                f"Relevant Code Entities:\n"
                + "\n".join(
                    [f"- {entity.name} ({entity.type}) in {entity.component}" for entity in entities]
                )
                + "\n\nPlease provide a concise and informative response to the user's query, "
                "leveraging the context provided by the relevant code entities. Focus on explaining "
                "the functionality, interactions, and RDK-specific details."
            )


            # Generate the response using Gemini
            response = self.gemini_model.generate_content(prompt)

            return response.text
        except Exception as e:
            logger.error(f"Error generating response from entities: {str(e)}")
            # logger.error("Error generating response from entities: %s", str(e))
            # logger.error("Error generating response from entities: {}".format(str(e)))
            return "I'm sorry, I couldn't generate a response for your query. Please try again later."


    #
    #  def search_relevant_entities(self, query: str) -> List[CodeEntity]:
    #     """Search for relevant code entities based on the user query"""
    #     # Search the vector store for relevant functions, structs, and APIs
    #     relevant_functions = self.vector_store.search(query, 'function', k=3)
    #     relevant_structs = self.vector_store.search(query, 'struct', k=3)
    #     relevant_apis = self.vector_store.search(query, 'api', k=3)

    #     # Combine the results and deduplicate
    #     relevant_entities = set()
    #     for result in relevant_functions + relevant_structs + relevant_apis:
    #         entity = self.entities[result['metadata']['name']]
    #         relevant_entities.add(entity)

    #     return list(relevant_entities)
    #===============================================================================================================
    def search_relevant_entities(self, query: str) -> List[CodeEntity]:
        """Search for relevant code entities based on the user query"""
        try:
            # Search the vector store for relevant functions, structs, and APIs
            relevant_functions = self.vector_store.search(query, 'function', k=3)
            relevant_structs = self.vector_store.search(query, 'struct', k=3)
            relevant_apis = self.vector_store.search(query, 'api', k=3)

            # Add debugging information
            logger.debug(f"Available entities: {list(self.entities.keys())}")
            logger.debug(f"Search results: {relevant_functions + relevant_structs + relevant_apis}")

            # Combine the results and deduplicate
            relevant_entities = set()
            for result in relevant_functions + relevant_structs + relevant_apis:
                entity_name = result['metadata']['name']
                if entity_name in self.entities:
                    entity = self.entities[entity_name]
                    relevant_entities.add(entity)
                else:
                    logger.warning(f"Entity '{entity_name}' found in search results but not in entities dictionary")
                    # Optional: You might want to skip this entity or handle it differently
                    continue

            return list(relevant_entities)
        except Exception as e:
            logger.error(f"Error in search_relevant_entities: {str(e)}")
            return []
    # #=================================================================================================================================
    # def search_relevant_entities(self, query: str) -> List[CodeEntity]:
    #     """
    #     Search for relevant code entities using Neo4j first, 
    #     with Faiss as a fallback mechanism
        
    #     :param query: Search query string
    #     :return: List of relevant code entities
    #     """
    #     try:
    #         # First, attempt to search using Neo4j-based method
    #         neo4j_results = self.neo4j_context_retriever.search_functions(query)
            
    #         # If Neo4j search returns results, convert and return
    #         if neo4j_results:
    #             neo4j_entities = []
    #             for result in neo4j_results:
    #                 # Create a CodeEntity from Neo4j result
    #                 entity = CodeEntity(
    #                     name=result['name'],
    #                     content=result['content'],
    #                     file_path=result['file_path']
    #                 )
    #                 neo4j_entities.append(entity)
                
    #             logger.info(f"Found {len(neo4j_entities)} entities via Neo4j search")
    #             return neo4j_entities
            
    #         # Fallback to Faiss if Neo4j search yields no results
    #         logger.info("No results from Neo4j, falling back to Faiss")
            
    #         # Original Faiss-based search
    #         relevant_functions = self.vector_store.search(query, 'function', k=3)
    #         relevant_structs = self.vector_store.search(query, 'struct', k=3)
    #         relevant_apis = self.vector_store.search(query, 'api', k=3)

    #         # Combine and deduplicate Faiss results
    #         relevant_entities = set()
    #         for result in relevant_functions + relevant_structs + relevant_apis:
    #             entity_name = result['metadata']['name']
    #             if entity_name in self.entities:
    #                 entity = self.entities[entity_name]
    #                 relevant_entities.add(entity)
    #             else:
    #                 logger.warning(f"Entity '{entity_name}' found in search results but not in entities dictionary")
            
    #         logger.info(f"Found {len(relevant_entities)} entities via Faiss search")
    #         return list(relevant_entities)
        
    #     except Exception as e:
    #         logger.error(f"Error in search_relevant_entities: {str(e)}")
    #         return []


    def generate_sequence_diagram(self, query: str, max_depth: int = 5) -> str:
        try:
            # Get context from logs and vector store
            relevant_entities, context = self.enhanced_search.contextual_search(query)
            
            # Generate initial diagram
            if not self.sequence_generator:
                return "Error: RDK Assistant not properly initialized"
                
            diagram = self.sequence_generator.generate(query, max_depth)
            
            # Enhance diagram with additional context
            enhanced_diagram = self._enhance_diagram_with_context(
                diagram,
                query,
                relevant_entities,
                context
            )
            
            return enhanced_diagram
            
        except Exception as e:
            error_msg = f"Error generating sequence diagram: {str(e)}"
            logger.error(error_msg)
            return error_msg
    

    def _enhance_diagram_with_context(
        self,
        initial_diagram: str,
        query: str,
        relevant_entities: List[CodeEntity],
        context: str
    ) -> str:
        """
        Enhance sequence diagram by incorporating log context, relevant entities,
        and LLM insights for better visualization of the flow.
        
        Args:
            initial_diagram: Base sequence diagram generated from vector store
            query: Original user query
            relevant_entities: List of CodeEntity objects found through contextual search
            context: Context information from log analysis and initial LLM processing
        
        Returns:
            str: Enhanced sequence diagram in Mermaid format
        """
        try:
            # Extract component interactions from relevant entities
            component_interactions = defaultdict(set)
            api_flows = set()
            critical_paths = set()
            
            # Analyze entities for interactions
            for entity in relevant_entities:
                for call in entity.function_calls:
                    if call.is_api:
                        api_flows.add((entity.component, call.component, call.function_name))
                    component_interactions[entity.component].add(call.component)
                    
                    # Identify critical paths (e.g., error handling, initialization)
                    if any(keyword in call.function_name.lower() 
                        for keyword in ['init', 'error', 'handle', 'validate', 'sync']):
                        critical_paths.add((entity.component, call.component, call.function_name))

            # Create enhanced diagram prompt
            prompt = f"""
            Analyze and enhance this sequence diagram for the query: "{query}"
            
            Original diagram:
            {initial_diagram}
            
            Consider the following additional context:
            1. Component Interactions: {dict(component_interactions)}
            2. API Flows: {list(api_flows)}
            3. Critical Paths: {list(critical_paths)}
            4. Recent System Context: {context}
            
            Please enhance this diagram by:
            1. Adding missing critical component interactions
            2. Including error handling and recovery flows
            3. Showing API call sequences with proper error handling
            4. Adding participant notes for important state changes
            5. Highlighting synchronization points between components
            6. Including any relevant CCSP/RBUS message flows
            7. Adding TR-181 parameter interactions where applicable
            8. Showing initialization and cleanup sequences
            
            Additional requirements:
            - Maintain proper component lifecycle
            - Include proper transaction boundaries
            - Show retry mechanisms for critical operations
            - Indicate asynchronous operations with appropriate syntax
            - Add activation/deactivation boxes for long-running operations
            
            Return only the enhanced Mermaid sequence diagram without any explanation.
            """
            
            # Get enhanced diagram from LLM
            response = self.gemini_model.generate_content(prompt)
            enhanced_diagram = response.text.strip()
            
            # Clean up the diagram format
            if "```mermaid" in enhanced_diagram:
                enhanced_diagram = enhanced_diagram.split("```mermaid")[1].split("```")[0].strip()
                
            # Validate diagram syntax
            if not enhanced_diagram.startswith("sequenceDiagram"):
                enhanced_diagram = "sequenceDiagram\n" + enhanced_diagram
                
            # Add title if not present
            if "title" not in enhanced_diagram:
                title_line = f"    title {query}\n"
                enhanced_diagram = enhanced_diagram.replace(
                    "sequenceDiagram",
                    f"sequenceDiagram\n{title_line}"
                )
                
            # Add autonumber if not present
            if "autonumber" not in enhanced_diagram:
                enhanced_diagram = enhanced_diagram.replace(
                    "sequenceDiagram",
                    "sequenceDiagram\nautonumber"
                )
                
            return enhanced_diagram
            
        except Exception as e:
            logger.error(f"Error enhancing diagram with context: {str(e)}")
            return initial_diagram  # Return original diagram if enhancement fails