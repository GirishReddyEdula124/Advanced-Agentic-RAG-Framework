"""
Infrastructure Layer: Vector DB Setup and Configuration.
Supports multiple backends like FAISS (local) and ChromaDB (distributed/persistent).
"""

import os
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class VectorDBInitializer:
    """
    Manages vector database lifecycle: storage, persistence, and retrieval setup.
    """

    def __init__(self, db_type: str = "faiss", persist_directory: Optional[str] = None):
        self.db_type = db_type.lower()
        self.persist_directory = persist_directory or os.getenv("VECTOR_DB_PATH", "./vector_storage")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def create_or_load(self, documents: Optional[List[Document]] = None) -> Any:
        """
        Creates a new vector store or loads an existing one from disk.
        """
        if self.db_type == "faiss":
            return self._handle_faiss(documents)
        elif self.db_type == "chroma":
            return self._handle_chroma(documents)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _handle_faiss(self, documents: Optional[List[Document]]) -> FAISS:
        """FAISS specific persistence logic."""
        faiss_path = os.path.join(self.persist_directory, "faiss_index")
        
        if documents:
            # Create new index
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            os.makedirs(self.persist_directory, exist_ok=True)
            vectorstore.save_local(faiss_path)
            return vectorstore
        
        # Load existing index
        if os.path.exists(os.path.join(faiss_path, "index.faiss")):
            return FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
        
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

    def _handle_chroma(self, documents: Optional[List[Document]]) -> Chroma:
        """ChromaDB specific persistence logic."""
        if documents:
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

def initialize_production_db(raw_docs: List[Document]):
    """Utility function to bootstrap production DB."""
    initializer = VectorDBInitializer(db_type="faiss")
    try:
        vectorstore = initializer.create_or_load(documents=raw_docs)
        print(f"Successfully initialized {initializer.db_type} vector store.")
        return vectorstore
    except Exception as e:
        print(f"Failed to initialize Vector DB: {str(e)}")
        return None

if __name__ == "__main__":
    # Example documents for testing initialization
    test_docs = [
        Document(page_content="RAG Architecture includes Retrieval and Generation.", metadata={"source": "whitepaper_v1"}),
        Document(page_content="Agents use Tool-Calling for expanded capabilities.", metadata={"source": "whitepaper_v2"}),
    ]
    
    # Bootstrap
    db = initialize_production_db(test_docs)
    if db:
        print("Test Verification: OK")
