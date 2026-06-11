import chromadb
from chromadb.config import Settings
from typing import List, Optional


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "task_history"


class VectorStore:
    """
    Wraps ChromaDB to store and retrieve task embeddings.
    Uses ChromaDB's built-in sentence-transformer for embeddings —
    no external API calls, runs locally.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def store_task_embedding(self, task_id: str, task_description: str):
        """
        Stores an embedding of the task description, linked to its task_id.
        ChromaDB auto-generates the embedding using its default model.
        """
        try:
            self.collection.add(
                ids=[task_id],
                documents=[task_description]
            )
        except Exception as e:
            
            print(f"--- Vector store error (non-fatal): {e} ---")

    def find_similar_tasks(
        self, query: str, n_results: int = 3
    ) -> List[str]:
        """
        Returns the task_ids of the N most semantically similar past tasks.
        Empty list if no past tasks exist or search fails.
        """
        
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            return results["ids"][0] if results["ids"] else []
        except Exception as e:
            print(f"--- Vector search error (non-fatal): {e} ---")
            return []

    def count(self) -> int:
        """Returns the number of stored task embeddings."""
        return self.collection.count()