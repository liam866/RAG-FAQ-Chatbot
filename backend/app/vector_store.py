import chromadb
import httpx
import hashlib
import logging
from typing import List, Tuple
from .schemas import DocumentChunk

logger = logging.getLogger(__name__)

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, ollama_base_url: str, model: str):
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.http_client = httpx.Client(timeout=60.0)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        url = f"{self.ollama_base_url}/api/embeddings"
        embeddings = []
        for text in input:
            try:
                payload = {"model": self.model, "prompt": text}
                response = self.http_client.post(url, json=payload)
                response.raise_for_status()
                embedding = response.json().get("embedding")
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.error("Ollama embedding response did not contain 'embedding' field.")
                    embeddings.append([0.0] * 384)
            except httpx.RequestError as e:
                logger.error(f"Could not connect to Ollama for embedding. Error: {e}")
                embeddings.append([0.0] * 384)
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama returned an error status for embedding: {e.response.status_code}")
                embeddings.append([0.0] * 384)
        return embeddings

class VectorStore:
    def __init__(self, path: str, ollama_base_url: str, embed_model: str, collection_name: str = "chroma_data"):
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = OllamaEmbeddingFunction(ollama_base_url, embed_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def _generate_chunk_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def sync_documents(self, chunks: List[DocumentChunk]):
        logger.info("Starting document synchronization...")
        
        db_entries = self.collection.get(include=["metadatas"])
        db_chunks_map = {
            entry_id: metadata for entry_id, metadata 
            in zip(db_entries["ids"], db_entries["metadatas"])
        }
        
        file_chunk_ids = set()
        chunks_to_add = []
        chunks_to_update = []

        for chunk in chunks:
            chunk_id = f"{chunk.file}-{chunk.start_line}"
            file_chunk_ids.add(chunk_id)
            new_hash = self._generate_chunk_hash(chunk.text)
            
            if chunk_id not in db_chunks_map:
                chunks_to_add.append(chunk)
            elif new_hash != db_chunks_map[chunk_id].get("chunk_hash"):
                chunks_to_update.append(chunk)

        ids_to_delete = [chunk_id for chunk_id in db_chunks_map if chunk_id not in file_chunk_ids]

        if ids_to_delete:
            logger.info(f"Deleting {len(ids_to_delete)} stale chunks.")
            self.collection.delete(ids=ids_to_delete)

        if chunks_to_add:
            logger.info(f"Adding {len(chunks_to_add)} new chunks.")
            self.collection.add(
                ids=[f"{chunk.file}-{chunk.start_line}" for chunk in chunks_to_add],
                documents=[chunk.text for chunk in chunks_to_add],
                metadatas=[{
                    "file": chunk.file,
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_hash": self._generate_chunk_hash(chunk.text)
                } for chunk in chunks_to_add]
            )

        if chunks_to_update:
            logger.info(f"Updating {len(chunks_to_update)} modified chunks.")
            self.collection.update(
                ids=[f"{chunk.file}-{chunk.start_line}" for chunk in chunks_to_update],
                documents=[chunk.text for chunk in chunks_to_update],
                metadatas=[{
                    "file": chunk.file,
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_hash": self._generate_chunk_hash(chunk.text)
                } for chunk in chunks_to_update]
            )
        logger.info("Document synchronization complete.")

    def retrieve_with_distances(self, query_text: str, n_results: int) -> List[Tuple[DocumentChunk, float]]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        retrieved_data = []
        if results and results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                meta = results["metadatas"][0][i]
                retrieved_data.append((
                    DocumentChunk(
                        text=doc_text,
                        file=meta.get("file"),
                        heading=meta.get("heading"),
                        start_line=meta.get("start_line"),
                        end_line=meta.get("end_line")
                    ),
                    distance
                ))
        return retrieved_data

    def query(self, query_text: str, n_results: int, threshold: float = 0.7) -> List[DocumentChunk]:
        # Use the retrieve_with_distances method and then apply the threshold
        retrieved_data = self.retrieve_with_distances(query_text, n_results)
        
        filtered_chunks = []
        for chunk, distance in retrieved_data:
            if distance <= threshold:
                filtered_chunks.append(chunk)
            else:
                logger.info(f"Filtered out chunk with distance {distance} (above threshold {threshold}).")
        return filtered_chunks
