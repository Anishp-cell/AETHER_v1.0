import os
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_FILE = os.path.join(MEMORY_DIR, "faiss_index.bin")
DEFAULT_DATA_FILE = os.path.join(MEMORY_DIR, "documents.npy")

class KnowledgeDatabase:
    """
    Tier 3 Memory Engine for AETHER:
    - Persistent semantic document and note retrieval using FAISS + SentenceTransformer.
    - Distance-threshold gated search prevents false-positive context injection.
    - Supports runtime file ingestion (txt, md, py, json, code).
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", index_file=DEFAULT_INDEX_FILE, data_file=DEFAULT_DATA_FILE):
        print(f"[KnowledgeRAG] Initializing Embedding Model: {model_name} on CPU...")
        self.embedding_model = SentenceTransformer(model_name, device="cpu")
        self.index_file = index_file
        self.data_file = data_file
        
        # 384-dimensional MiniLM embeddings
        self.dimension = 384 
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[Dict[str, str]] = []
        
        self.load_index()

    def add_document(self, text: str, source: str = "user_note"):
        """
        Splits, embeds, and indexes text into the FAISS vector database.
        """
        if not text or len(text.strip()) < 10:
            return
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        chunks = splitter.split_text(text)
        
        if not chunks:
            return

        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)
        
        for chunk in chunks:
            self.documents.append({"text": chunk, "source": source})
            
        self.save_index()
        print(f"[KnowledgeRAG] Added {len(chunks)} fragment(s) from source '{source}' (Total records: {self.index.ntotal}).")

    def ingest_note(self, title: str, content: str):
        """Helper to save a user-provided note or reference."""
        full_text = f"Title: {title}\nContent:\n{content}"
        self.add_document(full_text, source=f"note:{title}")

    def ingest_file(self, file_path: str) -> bool:
        """Reads and embeds a local document file into knowledge memory."""
        if not os.path.exists(file_path):
            print(f"[KnowledgeRAG Error] File not found: {file_path}")
            return False
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.add_document(content, source=os.path.basename(file_path))
            return True
        except Exception as e:
            print(f"[KnowledgeRAG Error] Ingestion failed for {file_path}: {e}")
            return False

    def search(self, query: str, top_k: int = 2, max_distance: float = 1.25) -> str:
        """
        Searches vector memory with strict L2 distance gating.
        Discards chunks if distance > max_distance to prevent hallucinated context.
        """
        if self.index.ntotal == 0:
            return ""
            
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.documents):
                    # Strict threshold filter: lower L2 distance means higher semantic match
                    if dist <= max_distance:
                        doc = self.documents[idx]
                        results.append(f"[{doc.get('source', 'doc')}]: {doc['text']}")
                    else:
                        pass # Filtered out as irrelevant
                        
            if results:
                return "\n".join(results)
            return ""
        except Exception as e:
            print(f"[KnowledgeRAG Error] Search failed: {e}")
            return ""

    def save_index(self):
        """Saves FAISS index and documents metadata persistently."""
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            np.save(self.data_file, self.documents)
        except Exception as e:
            print(f"[KnowledgeRAG Error] Failed to save index: {e}")

    def load_index(self):
        """Loads FAISS index and document records from disk."""
        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            try:
                self.index = faiss.read_index(self.index_file)
                self.documents = np.load(self.data_file, allow_pickle=True).tolist()
                print(f"[KnowledgeRAG] Loaded existing Knowledge Base ({self.index.ntotal} records).")
            except Exception as e:
                print(f"[KnowledgeRAG Warning] Failed to load index, starting fresh: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
        else:
            print("[KnowledgeRAG] Initialized fresh Knowledge Base.")
