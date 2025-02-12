"""Document storage and management module."""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from returns.result import Result, Success, Failure

from ..config import Config
from .exceptions import StateError


@dataclass
class Document:
    """Container for document data."""

    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: str


class DocumentStore:
    """Manages document storage and retrieval."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the document store.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.documents: Dict[str, Document] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self._load_documents()

    async def add_document(
        self, content: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> Result[str, Exception]:
        """Add a document to the store.

        Args:
            content: Document content
            embedding: Document embedding vector
            metadata: Optional document metadata

        Returns:
            Result containing document ID

        Raises:
            ValueError: If content or embedding is invalid
        """
        try:
            if not content or not content.strip():
                return Failure(ValueError("Document content cannot be empty"))
            if not embedding or len(embedding) == 0:
                return Failure(ValueError("Document embedding cannot be empty"))

            # Generate document ID
            doc_id = str(hash(content))

            # Create document
            doc = Document(
                id=doc_id, content=content, embedding=embedding, metadata=metadata or {}, timestamp=datetime.now().isoformat()
            )

            # Add to store
            self.documents[doc_id] = doc

            # Update embeddings matrix
            if self.embeddings_matrix is None:
                self.embeddings_matrix = np.array([embedding])
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, embedding])

            # Save to disk
            await self._save_documents()

            return Success(doc_id)

        except Exception as e:
            self.logger.exception("Failed to add document")
            return Failure(StateError(f"Failed to add document: {str(e)}"))

    async def get_document(self, doc_id: str) -> Result[Document, Exception]:
        """Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Result containing document

        Raises:
            ValueError: If document not found
        """
        try:
            if doc_id not in self.documents:
                return Failure(ValueError(f"Document {doc_id} not found"))
            return Success(self.documents[doc_id])
        except Exception as e:
            self.logger.exception("Failed to get document")
            return Failure(StateError(f"Failed to get document: {str(e)}"))

    async def search_similar(
        self, query_embedding: List[float], top_k: int = 3
    ) -> Result[List[Tuple[Document, float]], Exception]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            Result containing list of (document, similarity) tuples

        Raises:
            ValueError: If query embedding is invalid
        """
        try:
            if not query_embedding or len(query_embedding) == 0:
                return Failure(ValueError("Query embedding cannot be empty"))

            if not self.documents:
                return Success([])

            # Calculate similarities
            query_vector = np.array(query_embedding)
            similarities = np.dot(self.embeddings_matrix, query_vector) / (
                np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_vector)
            )

            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            doc_ids = list(self.documents.keys())

            results = [(self.documents[doc_ids[i]], float(similarities[i])) for i in top_indices]

            return Success(results)

        except Exception as e:
            self.logger.exception("Failed to search documents")
            return Failure(StateError(f"Failed to search documents: {str(e)}"))

    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result[bool, Exception]:
        """Update a document.

        Args:
            doc_id: Document ID
            content: New content (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)

        Returns:
            Result indicating success

        Raises:
            ValueError: If document not found
        """
        try:
            if doc_id not in self.documents:
                return Failure(ValueError(f"Document {doc_id} not found"))

            doc = self.documents[doc_id]

            if content is not None:
                if not content.strip():
                    return Failure(ValueError("Document content cannot be empty"))
                doc.content = content

            if embedding is not None:
                if not embedding:
                    return Failure(ValueError("Document embedding cannot be empty"))
                doc.embedding = embedding
                # Update embeddings matrix
                doc_index = list(self.documents.keys()).index(doc_id)
                self.embeddings_matrix[doc_index] = embedding

            if metadata is not None:
                doc.metadata.update(metadata)

            doc.timestamp = datetime.now().isoformat()

            # Save changes
            await self._save_documents()

            return Success(True)

        except Exception as e:
            self.logger.exception("Failed to update document")
            return Failure(StateError(f"Failed to update document: {str(e)}"))

    async def delete_document(self, doc_id: str) -> Result[bool, Exception]:
        """Delete a document.

        Args:
            doc_id: Document ID

        Returns:
            Result indicating success

        Raises:
            ValueError: If document not found
        """
        try:
            if doc_id not in self.documents:
                return Failure(ValueError(f"Document {doc_id} not found"))

            # Remove from store
            doc_index = list(self.documents.keys()).index(doc_id)
            del self.documents[doc_id]

            # Update embeddings matrix
            if self.embeddings_matrix is not None:
                self.embeddings_matrix = np.delete(self.embeddings_matrix, doc_index, axis=0)

            # Save changes
            await self._save_documents()

            return Success(True)

        except Exception as e:
            self.logger.exception("Failed to delete document")
            return Failure(StateError(f"Failed to delete document: {str(e)}"))

    async def _load_documents(self) -> None:
        """Load documents from disk."""
        try:
            if os.path.exists(Config.DOCUMENT_STORE_PATH):
                with open(Config.DOCUMENT_STORE_PATH, "r") as f:
                    data = json.load(f)

                self.documents = {doc_id: Document(**doc_data) for doc_id, doc_data in data.items()}

                if self.documents:
                    self.embeddings_matrix = np.array([doc.embedding for doc in self.documents.values()])

        except Exception as e:
            self.logger.error(f"Failed to load documents: {str(e)}")
            self.documents = {}
            self.embeddings_matrix = None

    async def _save_documents(self) -> None:
        """Save documents to disk."""
        try:
            data = {doc_id: asdict(doc) for doc_id, doc in self.documents.items()}

            with open(Config.DOCUMENT_STORE_PATH, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save documents: {str(e)}")
            raise StateError(f"Failed to save documents: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_documents": len(self.documents),
            "embedding_size": len(next(iter(self.documents.values())).embedding) if self.documents else 0,
            "total_size_mb": os.path.getsize(Config.DOCUMENT_STORE_PATH) / (1024 * 1024)
            if os.path.exists(Config.DOCUMENT_STORE_PATH)
            else 0,
        }
