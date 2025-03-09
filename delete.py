

def _clear_databases(self):
    """Interactive method to clear Neo4j and Vector Store databases"""
    print("\n--- Database Clearing Options ---")
    print("1. Clear Neo4j Database")
    print("2. Clear Vector Store Indices")
    print("3. Clear Both Neo4j and Vector Store")
    print("4. Cancel")

    choice = input("Enter your choice (1-4): ")

    try:
        # Neo4j Clearing
        if choice in ["1", "3"]:
            print("\nWarning: This will delete ALL nodes and relationships in Neo4j!")
            confirm = input("Are you sure? (yes/no): ").lower()
            
            if confirm == "yes":
                # Create a Neo4j session to clear the database
                with self.neo4j_processor.driver.session() as session:
                    # Comprehensive database clearing
                    session.run("""
                        MATCH (n)
                        DETACH DELETE n
                    """)
                print("Neo4j Database cleared successfully!")
            else:
                print("Neo4j database clearing cancelled.")

        # Vector Store Clearing
        if choice in ["2", "3"]:
            print("\nWarning: This will delete all vector store indices!")
            confirm = input("Are you sure? (yes/no): ").lower()
            
            if confirm == "yes":
                # Reset vector stores
                for store_type in self.vector_store.vector_stores:
                    self.vector_store.vector_stores[store_type] = None
                
                # Optional: Remove saved index files
                import shutil
                if os.path.exists("vector_stores"):
                    shutil.rmtree("vector_stores")
                
                print("Vector Store indices cleared successfully!")
            else:
                print("Vector Store clearing cancelled.")

        if choice == "4":
            print("Database clearing cancelled.")

    except Exception as e:
        print(f"Error during database clearing: {e}")
        import traceback
        traceback.print_exc()