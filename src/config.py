"""
Configuration management for Project Context MCP Server
"""

from pathlib import Path
# from pydantic_settings import BaseSettings


class Settings:
    """Configuration settings for the MCP server"""
    
    def __init__(self):
        # TODO: Implement with pydantic-settings
        # Data directories
        self.data_dir = Path("./data")
        self.vector_store_path = self.data_dir / "vector_store"
        self.graph_file = self.data_dir / "knowledge_graph.pkl"
        self.metadata_db = self.data_dir / "contexts.db"
        
        # Embedding model
        self.embedding_model = "all-MiniLM-L6-v2"
        
        # Similarity thresholds
        self.duplicate_threshold = 0.8
        self.similarity_threshold = 0.6
        
        # Auto-categorization
        self.category_confidence_threshold = 0.7 