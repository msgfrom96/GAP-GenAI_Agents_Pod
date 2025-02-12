import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
import numpy as np
from src.genai_agents.google import GoogleAIService, GoogleAISystem
from src.genai_agents.config import Config, AIServiceConfig
import pandas as pd
import logging


@pytest.fixture
def mock_genai():
    with patch("google.generativeai") as mock:
        mock.configure = Mock()
        mock.embed_content = Mock(return_value={"embedding": [0.1] * 768})
        mock.GenerativeModel = Mock()
        yield mock


@pytest.fixture
def service(mock_genai):
    config = AIServiceConfig(api_key="test_key", model_name="gemini-pro", embedding_model="models/text-embedding-004")
    return GoogleAIService(test_mode=True)


@pytest.fixture
def system(mock_genai):
    system = GoogleAISystem(test_mode=True)
    system.logger = logging.getLogger()
    return system


def test_service_initialization(service):
    """Test service initialization"""
    assert service.test_mode
    assert hasattr(service, "model")
    assert hasattr(service, "llm")


def test_generate_embedding(service):
    """Test embedding generation"""
    text = "Test text"
    embedding = service.generate_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 768  # Google's embedding size


def test_generate_completion(service):
    """Test completion generation"""
    prompt = "Test prompt"
    completion = service.generate_completion(prompt)
    assert isinstance(completion, str)
    assert "Mock Gemini response" in completion

    # Test with context
    completion = service.generate_completion(prompt, context="Test context")
    assert isinstance(completion, str)
    assert "Mock Gemini response" in completion


@pytest.mark.asyncio
async def test_ask_with_rag(service):
    """Test RAG functionality"""
    response = await service.aask_with_rag("test query")
    assert isinstance(response, str)
    data = json.loads(response)
    assert "deprecated" in data
    assert "new_features" in data


def test_system_initialization(system):
    """Test system initialization"""
    assert system.test_mode
    assert hasattr(system, "model")
    assert hasattr(system, "embedding_model")
    assert hasattr(system, "documents")
    assert hasattr(system, "document_embeddings")


def test_prepare_tuning_data(system):
    """Test preparation of tuning data"""
    texts = ["Text 1", "Text 2", "Text 3"]
    labels = ["Label 1", "Label 2", "Label 3"]
    train_df, test_df = system.prepare_tuning_data(texts, labels)

    assert len(train_df) + len(test_df) == len(texts)
    assert all(col in train_df.columns for col in ["input_text", "label"])
    assert all(col in test_df.columns for col in ["input_text", "label"])


@pytest.mark.asyncio
async def test_create_tuned_model(system):
    """Test model tuning"""

    # Prepare test data
    df = pd.DataFrame({"input_text": ["Text 1", "Text 2"], "label": ["Label 1", "Label 2"]})

    model_id = await system.acreate_tuned_model(df)
    assert isinstance(model_id, str)
    assert model_id in system.tuned_models
    assert model_id in system.model_versions


def test_list_tuned_models(system):
    """Test listing tuned models"""
    models = system.list_tuned_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model_id, str) for model_id in models)


def test_predict_with_tuned_model(system):
    """Test prediction with tuned model"""
    # Set active model
    system.active_tuned_model = "mock-model-v2"

    prediction = system.predict_with_tuned_model("Test input")
    assert isinstance(prediction, str)
    assert "Mock tuned model response" in prediction


def test_evaluate_tuned_model(system):
    """Test model evaluation"""

    # Prepare test data
    test_data = pd.DataFrame({"input_text": ["Text 1", "Text 2"], "label": ["Label 1", "Label 2"]})

    # Set active model
    system.active_tuned_model = "mock-model-v2"

    metrics = system.evaluate_tuned_model(test_data)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "total_samples" in metrics


@pytest.mark.asyncio
async def test_batch_embedding(system):
    """Test batch embedding processing"""
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = await system.aget_embedding(texts[0])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 768


def test_document_management(system):
    """Test document management"""
    # Test adding single document
    system.add_document("Test document")
    assert len(system.documents) == 1
    assert len(system.document_embeddings) == 1

    # Test batch document addition
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    system.add_documents_batch(docs)
    assert len(system.documents) == 4  # 1 + 3
    assert len(system.document_embeddings) == 4


def test_similar_documents(system):
    """Test similar document search"""
    # Add test documents
    docs = ["First document", "Second document", "Third document"]
    system.add_documents_batch(docs)

    results = system.find_similar_documents("test query")
    assert isinstance(results, list)
    assert len(results) == min(3, len(docs))  # Default top_k is 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_classifier_building(system):
    """Test neural classifier building"""
    classifier = system.build_classifier(num_classes=3)
    assert classifier is not None
    assert len(classifier.layers) > 0


@pytest.mark.asyncio
async def test_classifier_training(system):
    """Test classifier training"""
    # Build classifier
    system.build_classifier(num_classes=2)

    # Prepare training data
    texts = ["Text 1", "Text 2", "Text 3"]
    labels = np.array([0, 1, 0])

    await system.train_classifier_async(texts, labels)
    # Since we're using mocks, just verify the classifier exists
    assert system.classifier is not None


@pytest.mark.asyncio
async def test_deployed_system():
    """Test system in deployed environment"""
    pytest.importorskip("google.generativeai")
    import os

    if not Config.DEPLOY_TESTS:
        pytest.skip("Skipping deployed environment test")

    system = GoogleAISystem(test_mode=False)

    # Test real embedding generation
    embedding = await system.aget_embedding("Test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 768

    # Test real document search
    system.add_document("Test document")
    results = system.find_similar_documents("test")
    assert isinstance(results, list)
    assert len(results) > 0


def test_error_handling(system):
    """Test error handling"""
    # Test invalid model ID
    with pytest.raises(ValueError):
        system.load_tuned_model("nonexistent-model")

    # Test prediction without active model
    system.active_tuned_model = None
    with pytest.raises(ValueError):
        system.predict_with_tuned_model("test")

    # Test evaluation without active model
    test_data = pd.DataFrame({"input_text": [], "label": []})
    with pytest.raises(ValueError):
        system.evaluate_tuned_model(test_data)


def test_model_versioning(system):
    """Test model versioning functionality"""
    versions = system.list_model_versions()
    assert isinstance(versions, dict)
    assert "active" in versions
    assert "available" in versions
    assert "performance" in versions

    # Test model reversion
    if versions["available"]:
        old_version = versions["available"][0]
        system.revert_model_version(old_version)
        assert system.active_tuned_model == old_version
