# src/rag.py
import chromadb
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from tqdm import tqdm

class MemoryRAG:
    def __init__(
        self,
        collection_name="memory",
        persist_directory="data/chroma",
        embedding_model="all-MiniLM-L6-v2",
        batch_size=128
    ):
        self.batch_size = batch_size

        self.client = chromadb.Client(
            chromadb.config.Settings(
                persist_directory=persist_directory
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedder = SentenceTransformer(embedding_model)

    def add_texts(self, texts):
        """
        Batch-safe ingestion into vector ChromaDB
        """
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i : i + self.batch_size]

            embeddings = self.embedder.encode(
                batch,
                show_progress_bar=False
            ).tolist()

            ids = [str(uuid4()) for _ in batch]

            self.collection.add(
                documents=batch,
                embeddings=embeddings,
                ids=ids
            )

        self.client.persist()

    def retrieve(self, query, k=5):
        embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=k
        )

        return results["documents"][0]
