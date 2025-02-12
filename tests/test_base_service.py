import pytest
from unittest.mock import Mock, patch
import json
from src.genai_agents.base_service import BaseAIService, BaseMockService, DocumentStore
from src.genai_agents.config import AIServiceConfig, Config
import numpy as np


class TestService(BaseAIService):
    """Test implementation of BaseAIService"""

    def __init__(self, config: AIServiceConfig, test_mode: bool = False):
        super().__init__(config, test_mode)
        self.mock_embedding = [0.1] * 10  # Override the default size

    def generate_embedding(self, text: str):
        if self.test_mode:
            return self.mock_embedding
        return [0.1] * 10

    def generate_completion(self, prompt: str, context: str = None):
        return "Test completion"

    def setup_agent(self):
        pass

    def run_agent(self, query: str):
        return "Test agent response"

    async def ask_with_rag(self, query: str, context: str = None):
        if self.test_mode:
            return json.dumps(
                {
                    "deprecated": ["mock_deprecated_func"],
                    "new_features": ["mock_new_feature"],
                    "security": ["mock_security_update"],
                    "performance": ["mock_performance_improvement"],
                    "breaking_changes": ["mock_breaking_change"],
                }
            )
        return json.dumps({"deprecated": [], "new_features": [], "security": [], "performance": [], "breaking_changes": []})


@pytest.fixture
def config():
    return AIServiceConfig(api_key="test_key", model_name="test_model", embedding_model="test_embedding_model")


@pytest.fixture
def service(config):
    return TestService(config, test_mode=False)


@pytest.fixture
def mock_service(config):
    return TestService(config, test_mode=True)


def test_base_service_initialization(service):
    """Test base service initialization"""
    assert service.config.api_key == "test_key"
    assert service.config.model_name == "test_model"
    assert service.config.embedding_model == "test_embedding_model"
    assert not service.test_mode
    assert hasattr(service, "doc_store")


def test_mock_service_initialization(mock_service):
    """Test mock service initialization"""
    assert mock_service.test_mode
    assert hasattr(mock_service, "mock_embedding")
    assert len(mock_service.mock_embedding) == 10


def test_index_document(service):
    """Test document indexing"""
    doc = "Test document"
    service.index_document(doc)
    assert doc in service.doc_store.documents
    assert len(service.doc_store.embeddings) == 1


def test_semantic_search(service):
    """Test semantic search functionality"""
    # Index some test documents
    docs = ["First document", "Second document", "Third document"]
    for doc in docs:
        service.index_document(doc)

    # Test search
    results = service.semantic_search("test query")
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_mock_service_semantic_search(mock_service):
    """Test mock service semantic search"""
    docs = ["First document", "Second document", "Third document"]
    for doc in docs:
        mock_service.index_document(doc)

    results = mock_service.semantic_search("test query")
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_base_mock_service():
    """Test BaseMockService functionality"""
    # Test embedding response creation
    embedding = [0.1] * 10
    response = BaseMockService.create_mock_embedding_response(embedding)
    assert hasattr(response, "data")
    assert hasattr(response.data[0], "embedding")
    assert response.data[0].embedding == embedding

    # Test completion response creation
    text = "Test completion"
    response = BaseMockService.create_mock_completion(text)
    assert hasattr(response, "choices")
    assert hasattr(response.choices[0], "message")
    assert response.choices[0].message.content == text


@pytest.mark.asyncio
async def test_ask_with_rag(mock_service):
    """Test ask with RAG functionality"""
    # Index some test documents
    docs = ["First document", "Second document", "Third document"]
    for doc in docs:
        mock_service.index_document(doc)

    # Test RAG query
    response = await mock_service.ask_with_rag("test query", "test context")
    assert isinstance(response, str)
    # Verify it's valid JSON
    data = json.loads(response)
    assert isinstance(data, dict)
    assert "deprecated" in data
    assert "new_features" in data


def test_error_handling(service):
    """Test error handling in base service"""
    # Test document indexing with invalid input
    with pytest.raises(ValueError):
        service.index_document(None)

    # Test semantic search with invalid input
    with pytest.raises(ValueError):
        service.semantic_search("")


@pytest.mark.asyncio
async def test_deployed_service():
    """Test service in deployed environment"""
    pytest.importorskip("openai")

    if not Config.DEPLOY_TESTS:
        pytest.skip("Skipping deployed environment test")

    config = AIServiceConfig(api_key=Config.OPENAI_API_KEY, model_name="gpt-4", embedding_model="text-embedding-3-large")
    service = TestService(config, test_mode=False)

    # Test real embedding generation
    embedding = service.generate_embedding("Test text")
    assert isinstance(embedding, list)
    assert len(embedding) > 0

    # Test real completion generation
    completion = service.generate_completion("Test prompt")
    assert isinstance(completion, str)
    assert len(completion) > 0


def test_document_store_persistence(service, tmp_path):
    """Test document store persistence"""
    # Create a DocumentStore instance
    service.doc_store = DocumentStore()

    # Add some documents
    docs = ["First document", "Second document", "Third document"]
    for doc in docs:
        service.index_document(doc)

    # Save document store
    store_path = tmp_path / "doc_store.json"
    service.doc_store.save(store_path)

    # Create a new service and load the document store
    new_service = TestService(service.config, test_mode=service.test_mode)
    new_service.doc_store = DocumentStore()
    new_service.doc_store.load(store_path)

    # Verify documents and embeddings
    assert len(new_service.doc_store.documents) == len(docs)
    assert len(new_service.doc_store.embeddings) == len(docs)

    # Test search after loading
    results = new_service.semantic_search("test")
    assert isinstance(results, list)
    assert len(results) > 0
