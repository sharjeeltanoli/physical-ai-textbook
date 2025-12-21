import os
import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient

class CodeRetriever:
    def __init__(self):
        load_dotenv()
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.EMBED_MODEL = "embed-english-v3.0"

        self.cohere_client = cohere.Client(self.COHERE_API_KEY)
        self.qdrant_client = QdrantClient(
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY
        )

    def retrieve(self, query, collection_name, top_k=5):
        """
        Retrieves relevant code snippets based on the query.
        """
        print(f"Retrieving code for query: '{query}'")

        # Embed the query
        query_vector = self.cohere_client.embed(
            model=self.EMBED_MODEL,
            input_type="search_query",
            texts=[query]
        ).embeddings[0]

        # Search the vector database
        search_results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        # Format and return the results
        retrieved_code = ""
        for result in search_results.points:
            retrieved_code += f"Source URL: {result.payload['url']}\n"
            retrieved_code += f"```\n{result.payload['text']}\n```\n\n"
            
        return retrieved_code