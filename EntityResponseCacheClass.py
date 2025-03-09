from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from CodeEntityClass import CodeEntity
from pathlib import Path
import json
from EntityProcessingState import EntityProcessingState


class EntityResponseCache:
    def __init__(self, cache_dir: str = "entity_responses"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.state_file = self.cache_dir / "processing_state.json"
        self.processing_states: Dict[str, EntityProcessingState] = {}
        self.load_state()

    def get_cache_path(self, entity: CodeEntity) -> Path:
        """Get cache file path for an entity"""
        # Use content hash to ensure uniqueness
        return self.cache_dir / f"{entity.content_hash}.json"

    def load_state(self):
        """Load processing state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
                for key, data in state_data.items():
                    self.processing_states[key] = EntityProcessingState(
                        entity_name=data['entity_name'],
                        component=data['component'],
                        response=data.get('response'),
                        processed_at=datetime.fromisoformat(data['processed_at']) if data.get('processed_at') else None,
                        status=data['status'],
                        retry_count=data['retry_count']
                    )

    def save_state(self):
        """Save processing state to file"""
        state_data = {}
        for key, state in self.processing_states.items():
            state_data[key] = {
                'entity_name': state.entity_name,
                'component': state.component,
                'response': state.response,
                'processed_at': state.processed_at.isoformat() if state.processed_at else None,
                'status': state.status,
                'retry_count': state.retry_count
            }
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def get_response(self, entity: CodeEntity) -> Optional[str]:
        """Get cached response for an entity"""
        cache_path = self.get_cache_path(entity)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get('response')
        return None

    def save_response(self, entity: CodeEntity, response: str):
        """Save response for an entity"""
        cache_path = self.get_cache_path(entity)
        with open(cache_path, 'w') as f:
            json.dump({
                'entity_name': entity.name,
                'component': entity.component,
                'content_hash': entity.content_hash,
                'response': response,
                'processed_at': datetime.now().isoformat()
            }, f, indent=2)
        
        # Update processing state
        self.processing_states[entity.content_hash] = EntityProcessingState(
            entity_name=entity.name,
            component=entity.component,
            response=response,
            processed_at=datetime.now(),
            status="completed"
        )
        self.save_state()

    def mark_failed(self, entity: CodeEntity, retry_count: int):
        """Mark an entity as failed"""
        self.processing_states[entity.content_hash] = EntityProcessingState(
            entity_name=entity.name,
            component=entity.component,
            status="failed",
            retry_count=retry_count
        )
        self.save_state()

    # def should_process(self, entity: CodeEntity, max_retries: int = 3) -> bool:
    #     """Determine if an entity should be processed"""
    #     state = self.processing_states.get(entity.content_hash)
    #     if not state:
    #         return True
    #     if state.status == "completed":
    #         return False
    #     if state.status == "failed" and state.retry_count >= max_retries:
    #         return False
    #     return 
    #     True
    def should_process(self, entity: CodeEntity, max_retries: int = 3) -> bool:
        state = self.processing_states.get(entity.content_hash)
        
        print(f"Checking processing for entity: {entity.name}")
        print(f"Content hash: {entity.content_hash}")
        
        if not state:
            print("No previous state found. Processing allowed.")
            return True
        
        print(f"Current state: {state.__dict__}")
        
        if state.status == "completed":
            print(f"Entity {entity.name} already completed. Skipping.")
            return False
        
        if state.status == "failed" and state.retry_count >= max_retries:
            print(f"Entity {entity.name} failed {state.retry_count} times. Max retries reached. Skipping.")
            return False
        
        print(f"Entity {entity.name} can be processed.")
        return True