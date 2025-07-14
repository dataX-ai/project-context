"""
Neo4j Knowledge Graph Builder for Design Conversations
"""

import os
from typing import Optional
from neo4j import GraphDatabase
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from .models import NEO4J_SCHEMA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings


class GraphBuilder:
    """
    Neo4j Knowledge Graph Builder for extracting entities and relationships
    from software design conversations using Azure OpenAI
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        azure_embedding_deployment: Optional[str] = None,
        azure_embedding_endpoint: Optional[str] = None,
        azure_embedding_api_key: Optional[str] = None,
        azure_embedding_api_version: Optional[str] = None,
    ):
        """
        Initialize the graph builder with Azure OpenAI
        
        Args:
            neo4j_uri: Neo4j database URI (optional, reads from .env NEO4J_URI or defaults to bolt://localhost:7687)
            neo4j_user: Neo4j username (optional, reads from .env NEO4J_USER or defaults to neo4j)
            neo4j_password: Neo4j password (optional, reads from .env NEO4J_PASSWORD or defaults to password)
            azure_endpoint: Azure OpenAI endpoint (optional, reads from .env AZURE_OPENAI_ENDPOINT)
            azure_api_key: Azure OpenAI API key (optional, reads from .env AZURE_OPENAI_API_KEY)
            azure_api_version: Azure OpenAI API version (optional, reads from .env AZURE_OPENAI_API_VERSION)
            azure_deployment_name: Azure OpenAI deployment name (optional, reads from .env AZURE_OPENAI_DEPLOYMENT_NAME)
            azure_embedding_deployment: Azure OpenAI embedding deployment (optional, reads from .env AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        """
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri or "bolt://localhost:7687"
        self.neo4j_user = neo4j_user or settings.neo4j_user or "neo4j"
        self.neo4j_password = neo4j_password or settings.neo4j_password or "password"
        
        # Set up Azure OpenAI credentials
        self.azure_endpoint = azure_endpoint or settings.azure_openai_endpoint
        self.azure_api_key = azure_api_key or settings.azure_openai_api_key
        self.azure_api_version = azure_api_version or settings.azure_openai_api_version
        self.azure_deployment_name = azure_deployment_name or settings.azure_openai_deployment_name
        self.azure_embedding_deployment = azure_embedding_deployment or settings.azure_openai_embedding_deployment
        
        # Set up Azure OpenAI embedding credentials (separate endpoint)
        self.azure_embedding_endpoint = azure_embedding_endpoint or settings.azure_openai_embedding_endpoint
        self.azure_embedding_api_key = azure_embedding_api_key or settings.azure_openai_embedding_api_key
        self.azure_embedding_api_version = azure_embedding_api_version or settings.azure_openai_embedding_api_version
        
        # Initialize components
        self.driver = None
        self.llm = None
        self.embedder = None
        self.pipeline = None
        
        self._setup_components()
    
    def _setup_components(self):
        """Set up Neo4j driver, Azure OpenAI LLM, embedder, and pipeline"""
        
        # Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Validate Azure OpenAI configuration
        if not all([self.azure_endpoint, self.azure_api_key, self.azure_deployment_name]):
            raise ValueError("Azure OpenAI configuration incomplete. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME in .env")
        
        # LLM setup - Using your Azure OpenAI deployment
        self.llm = AzureOpenAILLM(
            model_name=self.azure_deployment_name,  # Your deployment: gpt-4.1-nano
            azure_endpoint=self.azure_endpoint,
            api_version=self.azure_api_version,
            api_key=self.azure_api_key,
            model_params={
                "temperature": 0.0,
                "max_tokens": 1500,
            }
        )
        
        # Embedder setup - Using Azure OpenAI embeddings (separate endpoint)
        embedding_deployment = self.azure_embedding_deployment or "text-embedding-ada-002"
        embedding_endpoint = self.azure_embedding_endpoint or self.azure_endpoint
        embedding_api_key = self.azure_embedding_api_key or self.azure_api_key
        embedding_api_version = self.azure_embedding_api_version or self.azure_api_version
        
        self.embedder = AzureOpenAIEmbeddings(
            model=embedding_deployment,
            azure_endpoint=embedding_endpoint,
            api_version=embedding_api_version,
            api_key=embedding_api_key,
        )
        
        # Custom prompt for extraction (COMMENTED OUT - using default for now)
        # self.custom_prompt = """
        # You are extracting a knowledge graph from software design discussions.
        # 
        # Focus on:
        # - Components (software modules, services, systems)
        # - Technologies (frameworks, databases, protocols, tools)
        # - Decisions (architectural choices, design patterns)
        # 
        # For each relationship, return:
        # - source: the source entity name
        # - target: the target entity name  
        # - type: one of USES, DECIDES, DEPENDS_ON
        # - strength: float from 0.0 to 1.0 indicating confidence/importance
        # 
        # Guidelines for strength:
        # - 0.9-1.0: Critical dependency or explicit decision
        # - 0.7-0.8: Important relationship
        # - 0.5-0.6: Moderate relationship
        # - 0.3-0.4: Weak or implied relationship
        # 
        # Text: {text}
        # """
        
        # Pipeline setup
        self.pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            from_pdf=False,
            schema=NEO4J_SCHEMA,
            # prompt_template=self.custom_prompt,  # Using default template for now
            # perform_entity_resolution=False  # Keep versioned entities distinct
        )
    
    async def build_graph_from_text(self, text: str) -> None:
        """
        Extract entities and relationships from text and build Neo4j graph
        
        Args:
            text: The conversation or design text to process
        """
        print(f"🔍 Processing text: {text[:100]}...")
        try:
            await self.pipeline.run_async(text=text)
            print("✅ Pipeline execution completed")
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def build_graph_from_text_sync(self, text: str) -> None:
        """
        Synchronous version of build_graph_from_text
        
        Args:
            text: The conversation or design text to process
        """
        import asyncio
        asyncio.run(self.build_graph_from_text(text))
    
    def query_graph(self, cypher_query: str, parameters: Optional[dict] = None):
        """
        Execute a Cypher query against the Neo4j graph
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record for record in result]
    
    def get_components(self):
        """Get all Component nodes"""
        query = "MATCH (c:Component) RETURN c.name as name"
        return self.query_graph(query)
    
    def get_technologies(self):
        """Get all Technology nodes"""
        query = "MATCH (t:Technology) RETURN t.name as name"
        return self.query_graph(query)
    
    def get_decisions(self):
        """Get all Decision nodes"""
        query = "MATCH (d:Decision) RETURN d.summary as summary"
        return self.query_graph(query)
    
    def get_all_relationships(self):
        """
        Get all relationships in the graph
        
        Returns:
            List of relationships with source, target, and type
        """
        query = """
        MATCH (a)-[r]->(b) 
        RETURN labels(a)[0] as source_type, a.name as source_name,
               type(r) as relationship_type,
               labels(b)[0] as target_type, b.name as target_name
        ORDER BY relationship_type, source_name
        """
        return self.query_graph(query)
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage function
def example_usage():
    """Example of how to use the GraphBuilder"""
    
    # Sample conversation text
    conversation_text = """
    The API Gateway component uses Redis for caching user sessions.
    The AuthService component decides to implement OAuth 2.0 for authentication.
    The API Gateway component depends on the AuthService for user validation.
    The Frontend component uses React framework for the user interface.
    The Database component decides to use PostgreSQL for data persistence.
    """
    
    # Initialize graph builder (uses environment variables)
    with GraphBuilder() as graph_builder:
        
        # Build graph from conversation
        graph_builder.build_graph_from_text_sync(conversation_text)
        
        # Query the results
        components = graph_builder.get_components()
        technologies = graph_builder.get_technologies()
        relationships = graph_builder.get_all_relationships()
        
        print("Components:", components)
        print("Technologies:", technologies)
        print("All relationships:", relationships)


if __name__ == "__main__":
    example_usage() 