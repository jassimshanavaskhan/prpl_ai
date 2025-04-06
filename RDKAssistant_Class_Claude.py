def _process_files_for_vector_db(self, files: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Process files by chunking them directly without entity extraction
    and store in vector database
    
    Args:
        files: List of file paths to process
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS  # Or your preferred vector DB
        from langchain.embeddings import OpenAIEmbeddings  # Or your preferred embedding model
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        file_contents = {}
        
        # Read and chunk all files
        for file_path in tqdm(files, desc="Chunking files for vector DB"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_contents[str(file_path)] = content
                    
                # Create chunks for this file
                chunks = text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{
                        "source": str(file_path),
                        "file_type": file_path.suffix,
                        "component": self._determine_component_name(file_path)
                    }]
                )
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {str(e)}")
                continue
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()  # Replace with your preferred embedding model
        
        # Create vector store
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        
        # Save the vector store
        vector_store.save_local("code_chunks_vector_store")
        
        logger.info(f"Successfully created vector DB with {len(all_chunks)} chunks from {len(files)} files")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        print(f"Error creating vector database: {str(e)}")
        return None

def _process_new_code(self, path: Path):
    """Process new code files and add to database with Neo4j integration and parallel vector DB creation"""
    try:
        if path.is_file():
            files = [path]
        else:
            files = []
            code_extensions = ['.c', '.cpp', '.h', '.cc']
            code_extensions.append('.odl')
            
            for ext in code_extensions:
                files.extend(path.rglob(f'*{ext}'))
        
        # Separate ODL and code files
        odl_files = []
        code_files = []
        for file_path in files:
            if file_path.suffix == '.odl':
                odl_files.append(file_path)
            else:
                code_files.append(file_path)
        
        # Create the simple vector DB in parallel
        # This can run in a separate thread if you want true parallelism
        import threading
        vector_db_thread = threading.Thread(
            target=self._process_files_for_vector_db,
            args=(files,)  # Pass all files for chunking
        )
        vector_db_thread.start()
        
        # Continue with the existing entity-based processing
        
        # Neo4j connection details
        neo4j_processor = Neo4jCodeEntityProcessor(
            uri=os.environ.get('NEO4J_URI'), 
            username=os.environ.get('NEO4J_USERNAME'), 
            password=os.environ.get('NEO4J_PASSWORD')
        )
        
        # ODL Neo4j Mapper
        odl_neo4j_mapper = ODLNeo4jMapper(
            uri=os.environ.get('NEO4J_URI'), 
            username=os.environ.get('NEO4J_USERNAME'), 
            password=os.environ.get('NEO4J_PASSWORD')
        )
        
        processed_entities = []
        odl_entities = {}
        
        # Process code files first
        for file_path in tqdm(code_files, desc="Processing code files"):
            component_name = self._determine_component_name(file_path)
            logger.info(f"===========================>>> {component_name}")
            entities = self._process_single_file(file_path, component_name)
            
            for entity in entities:
                self.entities[entity.name] = entity
                processed_entities.append(entity)
        
        # Create code entity nodes first
        neo4j_processor.create_code_entities(processed_entities)
        
        # Process relationships for code entities
        neo4j_processor.process_all_relationships(processed_entities)
        
        # Now process ODL files after code entities exist
        for file_path in tqdm(odl_files, desc="Processing ODL files"):
            component_name = self._determine_component_name(file_path)
            logger.info(f"===========================>>> {component_name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            parser = ODLDefinitionParser(original_content, str(file_path))
            odl_result = parser.parse()
            odl_entities[str(file_path)] = odl_result
            
            # Map ODL to Neo4j
            odl_neo4j_mapper.map_definition_to_neo4j(
                odl_result, 
                original_content, 
                component_name=component_name
            )
        
        # Create implementation relationships after all ODL files are processed
        odl_neo4j_mapper.create_implementation_relationships()
        
        # Update vector stores with both code and ODL entities
        self.vector_store.create_indices(
            list(self.entities.values()),
            odl_entities=odl_entities
        )
        
        # Save updated cache
        with open("rdk_assistant_cache.pkl", 'wb') as f:
            pickle.dump(self.entities, f)
        
        # Wait for the vector DB thread to complete
        vector_db_thread.join()
        
        print(f"Successfully processed {len(files)} files ({len(code_files)} code files, {len(odl_files)} ODL files)")
        neo4j_processor.close()
        odl_neo4j_mapper.close()
        
    except Exception as e:
        logger.error(f"Error processing new code: {str(e)}")
        print(f"Error processing new code: {str(e)}")