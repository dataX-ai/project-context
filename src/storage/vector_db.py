"""
LanceDB vector database operations for semantic search and embeddings
"""

import lancedb
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from .models import Node


class VectorDB:
    """LanceDB vector database operations"""
    
    def __init__(self, settings):
        self.settings = settings
        self.db = None
        self.table = None
        self.embedding_model = None
        self.db_path = Path(settings.data_dir) / "lancedb"
        self.table_name = "project_contexts"
        
    async def initialize(self):
        """Initialize LanceDB connection and embedding model"""
        # Ensure data directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create table if it doesn't exist
        await self._ensure_table_exists()
    
    async def _ensure_table_exists(self):
        """Create table if it doesn't exist"""
        try:
            # Try to open existing table
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Create new table with initial schema
            initial_data = pd.DataFrame({
                'context_id': pd.Series([], dtype='string'),
                'content': pd.Series([], dtype='string'),
                'description': pd.Series([], dtype='string'),
                'category': pd.Series([], dtype='string'),
                'tags': pd.Series([], dtype='object'),  # Array of strings
                'updated_at': pd.Series([], dtype='string'),
                'vector': pd.Series([], dtype='object')  # Array of floats
            })
            self.table = self.db.create_table(self.table_name, initial_data)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformer"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_context(self, context_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add context to vector database
        
        Args:
            context_id: Unique context identifier (must match graph node)
            content: Text content to embed and store
            metadata: Additional metadata to store with the context
            
        Raises:
            ValueError: If context_id already exists
        """
        # Check for duplicates
        if self.context_exists(context_id):
            raise ValueError(f"Context ID '{context_id}' already exists in vector database")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Generate embedding
        vector = self._generate_embedding(content)
        
        # Prepare row data
        row_data = pd.DataFrame([{
            'context_id': context_id,
            'content': content,
            'description': metadata.get('description', ''),
            'category': metadata.get('category', ''),
            'tags': metadata.get('tags', []),
            'updated_at': metadata.get('updated_at', ''),
            'vector': vector
        }])
        
        # Add to table
        self.table.add(row_data)
    
    def add_context_from_node(self, node: Node, content: str):
        """Add context from a Node object
        
        Args:
            node: Node object from graph
            content: Text content to embed
        """
        metadata = {
            "description": node.description,
            "category": node.category or "",
            "tags": node.tags,
            "updated_at": node.updated_at.isoformat() if node.updated_at else "",
            **node.attributes
        }
        
        self.add_context(node.context_id, content, metadata)
    
    def search_similar(self, query: str, n_results: int = 10, where: str = None) -> List[Dict[str, Any]]:
        """Search for similar contexts using semantic similarity
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional SQL-like filter condition
            
        Returns:
            List of matching contexts with metadata and similarity scores
        """
        if not self.table:
            return []
        
        # Generate query embedding
        query_vector = self._generate_embedding(query)
        
        try:
            # Perform vector search
            search_builder = self.table.search(query_vector).limit(n_results)
            
            if where:
                search_builder = search_builder.where(where)
            
            results = search_builder.to_pandas()
            
            # Format results
            formatted_results = []
            for _, row in results.iterrows():
                formatted_results.append({
                    "context_id": row['context_id'],
                    "content": row['content'],
                    "metadata": {
                        "description": row['description'],
                        "category": row['category'],
                        "tags": row['tags'],
                        "updated_at": row['updated_at']
                    },
                    "distance": row.get('_distance', None)
                })
            
            return formatted_results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get specific context by ID
        
        Args:
            context_id: Context identifier
            
        Returns:
            Context data or None if not found
        """
        if not self.table:
            return None
        
        try:
            # Query by context_id
            results = self.table.search().where(f"context_id = '{context_id}'").to_pandas()
            
            if len(results) > 0:
                row = results.iloc[0]
                return {
                    "context_id": row['context_id'],
                    "content": row['content'],
                    "metadata": {
                        "description": row['description'],
                        "category": row['category'], 
                        "tags": row['tags'],
                        "updated_at": row['updated_at']
                    }
                }
        except Exception as e:
            print(f"Get context error: {e}")
        
        return None
    
    def update_context(self, context_id: str, content: str = None, metadata: Dict[str, Any] = None):
        """Update existing context
        
        Args:
            context_id: Context identifier
            content: New content (optional)
            metadata: New metadata (optional)
            
        Raises:
            ValueError: If context doesn't exist
        """
        if not self.context_exists(context_id):
            raise ValueError(f"Context ID '{context_id}' not found in vector database")
        
        # For LanceDB, we need to delete and re-add (no direct update)
        current = self.get_context(context_id)
        if not current:
            raise ValueError(f"Failed to retrieve context '{context_id}'")
        
        # Delete existing
        self.delete_context(context_id)
        
        # Prepare new data
        new_content = content if content is not None else current["content"]
        new_metadata = metadata if metadata is not None else current["metadata"]
        
        # Re-add with new data
        self.add_context(context_id, new_content, new_metadata)
    
    def delete_context(self, context_id: str):
        """Delete context from vector database
        
        Args:
            context_id: Context identifier to delete
        """
        if self.context_exists(context_id):
            try:
                # LanceDB delete using filter
                self.table.delete(f"context_id = '{context_id}'")
            except Exception as e:
                print(f"Delete error: {e}")
    
    def context_exists(self, context_id: str) -> bool:
        """Check if context exists in vector database
        
        Args:
            context_id: Context identifier to check
            
        Returns:
            True if context exists, False otherwise
        """
        if not self.table:
            return False
        
        try:
            results = self.table.search().where(f"context_id = '{context_id}'").limit(1).to_pandas()
            return len(results) > 0
        except Exception:
            return False
    
    def get_all_context_ids(self) -> List[str]:
        """Get all context IDs in the vector database
        
        Returns:
            List of all context IDs
        """
        if not self.table:
            return []
        
        try:
            # Get all records and extract context_ids
            all_data = self.table.to_pandas()
            return all_data['context_id'].tolist()
        except Exception:
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics
        
        Returns:
            Statistics about the vector database
        """
        if not self.table:
            return {"count": 0}
        
        try:
            count = len(self.table.to_pandas())
            return {
                "count": count,
                "table_name": self.table_name,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        except Exception:
            return {"count": 0} 