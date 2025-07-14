"""
Sentence transformers & embedding logic
"""

# from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Sentence transformers wrapper for embeddings"""
    
    def __init__(self, settings):
        self.settings = settings
        # TODO: Initialize sentence transformer model
        # self.model = SentenceTransformer(settings.embedding_model)
    
    async def embed_text(self, text: str):
        """Generate embedding for text"""
        # TODO: Implement text embedding
        return [0.1] * 384  # Placeholder embedding 