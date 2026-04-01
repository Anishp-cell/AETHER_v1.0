import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class KnowledgeDatabase:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_file="memory/faiss_index.bin", data_file="memory/documents.npy"):
        """
        Initializes the Local RAG Database using FAISS and Sentence Transformers.
        """
        print(f"[Memory Layer] Loading Embedding Model: {model_name} on CPU...")
        self.embedding_model = SentenceTransformer(model_name, device="cpu")
        self.index_file = index_file
        self.data_file = data_file
        
        # We use a 384-dimensional miniLM model; it's extremely fast and accurate.
        self.dimension = 384 
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []  # To store the actual text chunks mapped to the FAISS index
        
        self.load_index()

    def add_document(self, text: str, source: str = "user_input"):
        """
        Chunks the document, embeds it, and saves it to the FAISS index.
        """
        print(f"[Memory Layer] Processing document from: {source}")
        
        # Break text into paragraphs/chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        
        if not chunks:
            return

        # Create embeddings and add to FAISS
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)
        
        for chunk in chunks:
            self.documents.append({"text": chunk, "source": source})
            
        self.save_index()
        print(f"[Memory Layer] Added {len(chunks)} fragments to AETHER's Knowledge Bank.")

    def search(self, query: str, top_k: int = 2):
        """
        Searches the vector database for the most relevant chunks based on semantic similarity.
        """
        if self.index.ntotal == 0:
            return ""
            
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx]["text"])
                
        # Return a compiled context string
        if results:
            return "\n".join(results)
        return ""

    def save_index(self):
        """Saves memory to disk persistently."""
        faiss.write_index(self.index, self.index_file)
        np.save(self.data_file, self.documents)

    def load_index(self):
        """Loads memory from disk if it exists."""
        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            self.index = faiss.read_index(self.index_file)
            self.documents = np.load(self.data_file, allow_pickle=True).tolist()
            print(f"[Memory Layer] Loaded existing Knowledge Base with {self.index.ntotal} records.")
        else:
            print("[Memory Layer] No existing Knowledge Base found. Starting fresh.")

if __name__ == "__main__":
    # Ensure memory folder exists
    os.makedirs("memory", exist_ok=True)
    
    db = KnowledgeDatabase()
    
    print("\n--- Testing Knowledge RAG ---")
    demo_text = "AETHER is governed by a strict local-first policy router. The micro-expert model evaluates routing confidence without altering the base LLM weights."
    db.add_document(demo_text, source="AETHER_Core_Concept.txt")
    
    query = "What evaluates routing confidence?"
    print(f"\nQuery: {query}")
    context = db.search(query)
    print(f"\nRetrieved Context:\n{context}")
