"""
Configuration management for Project Context MCP Server
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration settings for the MCP server"""
    
        # Data directories
    data_dir: Path = Path("./data")
    vector_store_path: Optional[Path] = None
    graph_file: Optional[Path] = None
    metadata_db: Optional[Path] = None
        
        # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
        
        # Similarity thresholds
    duplicate_threshold: float = 0.8
    similarity_threshold: float = 0.6
        
        # Auto-categorization
    category_confidence_threshold: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    
    # Neo4j Graph Database Configuration (read from .env)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    # Azure OpenAI Configuration (read from .env)
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_deployment_name: Optional[str] = None
    azure_openai_embedding_deployment: Optional[str] = None
    
    # Azure OpenAI Embedding Configuration (separate endpoint)
    azure_openai_embedding_endpoint: Optional[str] = None
    azure_openai_embedding_api_key: Optional[str] = None
    azure_openai_embedding_api_version: str = "2023-05-15"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set up dependent paths after initialization
        if self.vector_store_path is None:
            self.vector_store_path = self.data_dir / "vector_store"
        if self.graph_file is None:
            self.graph_file = self.data_dir / "knowledge_graph.pkl"
        if self.metadata_db is None:
            self.metadata_db = self.data_dir / "contexts.db"


# Global settings instance
settings = Settings() 