import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np
from src.genai_agents.openai import OpenAIService, OpenAISystem
from src.genai_agents.config import Config, AIServiceConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
import pandas as pd
import os
import asyncio
import re


@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock:
        mock.return_value.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        mock.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mock response"))]
        )
        yield mock


@pytest.fixture
def service():
    return OpenAIService(test_mode=True)


@pytest.fixture
def system():
    return OpenAISystem(test_mode=True)


def test_service_initialization(service):
    """Test service initialization"""
    assert service.test_mode is True
    assert hasattr(service, "client")
    assert hasattr(service, "llm")


def test_generate_embedding(service):
    """Test embedding generation"""
    text = "Test text"
    embedding = service.generate_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # OpenAI's embedding size


def test_generate_completion(service):
    """Test completion generation"""
    prompt = "Test prompt"
    completion = service.generate_completion(prompt)
    assert isinstance(completion, str)
    assert "Mock response" in completion

    # Test with context
    completion = service.generate_completion(prompt, context="Test context")
    assert isinstance(completion, str)
    assert "Mock response" in completion


@pytest.mark.asyncio
async def test_ask_with_rag(service):
    """Test RAG functionality"""
    response = await service.ask_with_rag("test query", "test context")
    assert isinstance(response, str)
    try:
        data = json.loads(response)
        assert isinstance(data, dict)
        assert "deprecated" in data
        assert "new_features" in data
    except json.JSONDecodeError:
        pytest.fail("Response is not a valid JSON")


def test_setup_agent(service):
    """Test agent setup"""
    service.setup_agent()
    assert hasattr(service, "workflow")


def test_run_agent(system):
    """Test agent execution"""
    # First test without agent_executor
    with pytest.raises(RuntimeError, match=re.escape("Agent executor not initialized. Call initialize_agent() first.")):
        system.run_agent("test query")

    # Now test with agent_executor
    system.initialize_agent()
    with patch.object(system, "agent_executor") as mock_executor:
        mock_executor.invoke.return_value = {"output": "Test agent response"}
        result = system.run_agent("What can you tell me about AI?")
        assert isinstance(result, str)
        assert "Test agent response" in result
        mock_executor.invoke.assert_called_once_with({"input": "What can you tell me about AI?"})

    # Test error handling
    with patch.object(system, "agent_executor") as mock_executor:
        mock_executor.invoke.side_effect = Exception("Test error")
        with pytest.raises(RuntimeError) as exc_info:
            system.run_agent("test query")
        assert "Failed to run agent" in str(exc_info.value)


def test_system_initialization(system):
    """Test system initialization"""
    assert system.test_mode is True
    assert hasattr(system, "model")
    assert hasattr(system, "embedding_model")
    assert hasattr(system, "documents")
    assert hasattr(system, "document_embeddings")


def test_prepare_tuning_data(system):
    """Test preparation of tuning data"""
    texts = ["Text 1", "Text 2", "Text 3"]
    labels = ["Label 1", "Label 2", "Label 3"]
    train_df, test_df = system.prepare_training_data(texts, labels)

    assert len(train_df) + len(test_df) == len(texts)
    assert "messages" in train_df.columns
    assert "messages" in test_df.columns


def test_train_custom_model(system):
    """Test model tuning"""
    df = pd.DataFrame(
        {
            "messages": [
                [{"role": "user", "content": "Text 1"}, {"role": "assistant", "content": "Label 1"}],
                [{"role": "user", "content": "Text 2"}, {"role": "assistant", "content": "Label 2"}],
            ]
        }
    )

    job_id = system.train_custom_model(df)
    assert isinstance(job_id, str)


def test_get_training_status(system):
    """Test getting training status"""
    # Test in test mode
    status = system.get_training_status("mock-job-id")
    assert isinstance(status, dict)
    assert "status" in status
    assert "fine_tuned_model" in status
    assert status["status"] == "succeeded"
    assert status["fine_tuned_model"] == "mock-model-id"

    # Test with non-test mode
    system.test_mode = False
    with patch.object(system, "client") as mock_client:
        mock_job = MagicMock()
        mock_job.status = "running"
        mock_job.fine_tuned_model = None
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

        status = system.get_training_status("real-job-id")
        assert isinstance(status, dict)
        assert status["status"] == "running"
        assert status["fine_tuned_model"] is None
        mock_client.fine_tuning.jobs.retrieve.assert_called_once_with("real-job-id")


def test_load_custom_model(system):
    """Test loading custom model"""
    system.load_custom_model("mock-model-id")
    assert system.fine_tuned_model == "mock-model-id"


def test_generate_with_custom_model(system):
    """Test generation with custom model"""
    # Test without loaded model
    with pytest.raises(ValueError, match=re.escape("No fine-tuned model loaded. Call load_custom_model() first.")):
        system.generate_with_custom_model("test")

    # Test with loaded model in test mode
    system.load_custom_model("mock-model-id")
    response = system.generate_with_custom_model("test prompt")
    assert isinstance(response, str)
    assert "Mock response for: test prompt" in response

    # Test with loaded model in non-test mode
    system.test_mode = False
    with patch.object(system, "client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Generated response"))]
        )
        response = system.generate_with_custom_model("test prompt")
        assert isinstance(response, str)
        assert "Generated response" in response
        mock_client.chat.completions.create.assert_called_once()


def test_evaluate_model_performance(system):
    """Test model evaluation"""
    test_data = pd.DataFrame(
        {
            "messages": [
                [{"role": "user", "content": "Text 1"}, {"role": "assistant", "content": "Label 1"}],
                [{"role": "user", "content": "Text 2"}, {"role": "assistant", "content": "Label 2"}],
            ]
        }
    )

    # Test without loaded model
    with pytest.raises(ValueError, match=re.escape("No fine-tuned model loaded. Call load_custom_model() first.")):
        system.evaluate_model_performance(test_data)

    # Test with loaded model in test mode
    system.load_custom_model("mock-model-id")
    result = system.evaluate_model_performance(test_data)
    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "total_samples" in result
    assert result["total_samples"] == len(test_data)


def test_generate_embeddings(system):
    """Test embedding generation"""
    embedding = system.generate_embeddings("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 1536


def test_index_document(system):
    """Test document indexing"""
    # Test with test mode
    test_doc = "test document"
    system.index_document(test_doc)
    assert len(system.documents) == 1
    assert len(system.document_embeddings) == 1
    assert system.documents[0] == test_doc
    assert len(system.document_embeddings[0]) == 1536  # OpenAI's embedding size

    # Test with non-test mode
    system.test_mode = False
    with patch.object(system, "client") as mock_client:
        mock_embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=mock_embedding)])

        test_doc = "another test document"
        system.index_document(test_doc)
        assert len(system.documents) == 2
        assert len(system.document_embeddings) == 2
        assert system.documents[1] == test_doc
        assert system.document_embeddings[1] == mock_embedding
        mock_client.embeddings.create.assert_called_once_with(model=system.embedding_model, input=test_doc)

    # Test error cases
    with pytest.raises(ValueError, match=re.escape("Document text cannot be None or empty.")):
        system.index_document(None)

    with pytest.raises(ValueError, match=re.escape("Document text cannot be None or empty.")):
        system.index_document("")


def test_semantic_search(system):
    """Test semantic search"""
    # Add test documents
    docs = ["First document", "Second document", "Third document"]
    for doc in docs:
        system.index_document(doc)

    results = system.semantic_search("test query")
    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_create_classifier_model(system):
    """Test classifier creation"""
    # Test with default parameters
    system.create_classifier_model(num_classes=3)
    assert hasattr(system, "classifier")
    assert len(system.classifier.layers) > 0

    # Test with custom embedding size
    system.create_classifier_model(num_classes=2, embedding_size=768)
    assert system.classifier.layers[0].input_shape == (None, 768)
    assert system.classifier.layers[-1].units == 2


def test_train_classification_model(system):
    """Test classifier training"""
    texts = ["Text 1", "Text 2", "Text 3"]
    labels = [0, 1, 0]
    system.create_classifier_model(num_classes=2)
    system.train_classification_model(texts, labels)
    assert system.classifier is not None


def test_classify_content(system):
    """Test content classification"""
    # Test without classifier
    with pytest.raises(ValueError, match=re.escape("Classifier not trained. Call train_classification_model() first.")):
        system.classify_content("test text")

    # Test with classifier in test mode
    system.test_mode = True
    system.create_classifier_model(num_classes=2)
    result = system.classify_content("test text")
    assert isinstance(result, dict)
    assert len(result) == 2
    assert 0 in result and 1 in result
    assert abs(result[0] + result[1] - 1.0) < 1e-6  # Probabilities sum to 1

    # Test with classifier in non-test mode
    system.test_mode = False
    with patch.object(system, "generate_embedding") as mock_embed, patch.object(system, "classifier") as mock_classifier:
        mock_embed.return_value = [0.1] * 1536
        mock_classifier.predict.return_value = np.array([[0.7, 0.3]])

        result = system.classify_content("test text")
        assert isinstance(result, dict)
        assert len(result) == 2
        assert result[0] == 0.7
        assert result[1] == 0.3
        mock_embed.assert_called_once_with("test text")
        mock_classifier.predict.assert_called_once()

    # Test error handling
    with patch.object(system, "generate_embedding") as mock_embed:
        mock_embed.side_effect = Exception("Test error")
        with pytest.raises(RuntimeError) as exc_info:
            system.classify_content("test text")
        assert "Failed to classify content" in str(exc_info.value)


def test_generate_rag_response(system):
    """Test RAG response generation"""
    # Test in test mode
    system.test_mode = True
    response = system.generate_rag_response("test question")
    assert isinstance(response, str)
    assert "Mock RAG response" in response

    # Test with no documents
    system.test_mode = False
    system.documents = []  # Ensure documents list is empty
    response = system.generate_rag_response("test question")
    assert "No documents available for search" in response

    # Test with documents but no context
    system.documents = ["test document 1", "test document 2"]  # Add test documents
    system.document_embeddings = [[0.1] * 1536, [0.2] * 1536]  # Add corresponding embeddings

    with patch.object(system, "semantic_search") as mock_search, patch.object(system, "client") as mock_client:
        mock_search.return_value = [("test document 1", 0.9)]
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Generated response"))]
        )

        response = system.generate_rag_response("test question")
        assert isinstance(response, str)
        assert "Generated response" in response
        mock_search.assert_called_once_with("test question", top_k=2)

    # Test with explicit context
    with patch.object(system, "client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Context response"))]
        )

        response = system.generate_rag_response("test question", context="test context")
        assert isinstance(response, str)
        assert "Context response" in response

    # Test error handling
    with patch.object(system, "client") as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("Test error")
        response = system.generate_rag_response("test question", context="test context")
        assert "An error occurred" in response


def test_enable_function_calling(system):
    """Test function calling setup"""
    system.enable_function_calling()
    assert hasattr(system, "available_functions")
    assert "search_documents" in system.available_functions


def test_process_function_call(system):
    """Test function call processing"""
    with patch.object(system, "client") as mock_client:
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test function call response"))]
        mock_client.chat.completions.create.return_value = mock_completion

        system.enable_function_calling()
        result = system.process_function_call("Find documents about AI")
        assert isinstance(result, str)


def test_initialize_agent(system):
    """Test agent initialization"""
    system.initialize_agent()
    assert hasattr(system, "agent_executor")


def test_save_load_agent_state(system, tmp_path):
    """Test agent state persistence"""
    state = {
        "messages": [HumanMessage(content="test query")],
        "next_step": "search",
        "current_task": "initial_search",
        "documents": ["doc1", "doc2"],
        "results": ["result1", "result2"],
    }

    # Save state
    state_path = tmp_path / "agent_state.json"
    system.save_agent_state(state, str(state_path))

    # Load state
    loaded_state = system.load_agent_state(str(state_path))
    assert isinstance(loaded_state, dict)
    assert loaded_state["next_step"] == state["next_step"]
    assert loaded_state["current_task"] == state["current_task"]
    assert loaded_state["documents"] == state["documents"]
    assert loaded_state["results"] == state["results"]


def test_error_handling(system):
    """Test error handling"""
    # Test invalid model ID
    with pytest.raises(ValueError, match=re.escape("Model ID cannot be None or empty.")):
        system.load_custom_model(None)

    # Test prediction without model
    with pytest.raises(ValueError, match=re.escape("No fine-tuned model loaded. Call load_custom_model() first.")):
        system.generate_with_custom_model("test")

    # Test evaluation without model
    test_data = pd.DataFrame({"messages": []})
    with pytest.raises(ValueError, match=re.escape("No fine-tuned model loaded. Call load_custom_model() first.")):
        system.evaluate_model_performance(test_data)

    # Test empty document
    with pytest.raises(ValueError, match=re.escape("Document text cannot be None or empty.")):
        system.index_document(None)

    # Test invalid classifier input
    with pytest.raises(ValueError, match=re.escape("Classifier not trained. Call train_classification_model() first.")):
        system.classify_content("test")


@pytest.mark.asyncio
async def test_deployed_service():
    """Test service in deployed environment"""
    openai = pytest.importorskip("openai")

    if not Config.DEPLOY_TESTS:
        pytest.skip("Skipping deployed environment test")

    api_key = Config.OPENAI_API_KEY
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping deployed test")

    service = OpenAIService(test_mode=False)

    # Test real embedding generation
    embedding = service.generate_embedding("Test text")
    assert isinstance(embedding, list)
    assert len(embedding) > 0

    # Test real completion generation
    completion = service.generate_completion("Test prompt")
    assert isinstance(completion, str)
    assert len(completion) > 0
