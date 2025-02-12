"""
Base Service Module

Provides base classes and utilities for AI services.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from .config import Config, AIServiceConfig
import json
import pytest


class DocumentStore:
    """Document store for managing documents and their embeddings"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, text: str, embedding: List[float]) -> None:
        """Add a document and its embedding"""
        if text is None or not text.strip():
            raise ValueError("Document text cannot be empty")
        if not embedding:
            raise ValueError("Embedding cannot be empty")
        self.documents.append(text)
        self.embeddings.append(embedding)

    def save(self, path: str) -> None:
        """Save document store to disk"""
        data = {
            "documents": self.documents,
            "embeddings": [e.tolist() if hasattr(e, "tolist") else e for e in self.embeddings],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load document store from disk"""
        with open(path, "r") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if not self.embeddings:
            return []

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]


class BaseMockService:
    """Base class for mock services"""

    @staticmethod
    def create_mock_embedding(size: int = 1536) -> List[float]:
        """Create mock embedding vector"""
        return [0.1] * size

    @staticmethod
    def create_mock_completion(content: str = "Mock response") -> Mock:
        """Create mock completion response"""
        message = Mock()
        message.content = content
        choice = Mock()
        choice.message = message
        response = Mock()
        response.choices = [choice]
        return response

    @staticmethod
    def create_mock_embedding_response(embedding: List[float]) -> Mock:
        """Create mock embedding response"""
        response = Mock()
        response.data = [Mock(embedding=embedding)]
        return response


class BaseAIService:
    """Base class for AI services"""

    def __init__(self, config: AIServiceConfig, test_mode: bool = False):
        """Initialize base service

        Args:
            config: Service configuration
            test_mode: Whether to use mock services
        """
        self.config = config
        self.test_mode = test_mode
        self.doc_store = DocumentStore()

        if test_mode:
            self._setup_base_mocks()

    def _setup_base_mocks(self):
        """Set up base mock services"""
        self.mock_embedding = [0.1] * 1536  # OpenAI's embedding size
        self.mock_documents = []
        self.mock_embeddings = []

    def index_document(self, text: str) -> None:
        """Index a document with its embedding"""
        if text is None:
            raise ValueError("Document text cannot be None")
        if not text.strip():
            raise ValueError("Document text cannot be empty")

        embedding = self.generate_embedding(text)
        self.doc_store.add_document(text, embedding)

    def semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document, similarity_score) tuples
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self.generate_embedding(query)
        return self.doc_store.search(query_embedding, top_k)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        raise NotImplementedError("Subclasses must implement generate_embedding")

    def generate_completion(self, prompt: str, context: str = None) -> str:
        """Generate completion with optional context"""
        raise NotImplementedError("Subclasses must implement generate_completion")

    def setup_agent(self):
        """Set up agent workflow"""
        raise NotImplementedError("Subclasses must implement setup_agent")

    def run_agent(self, query: str) -> str:
        """Run agent workflow"""
        raise NotImplementedError("Subclasses must implement run_agent")
