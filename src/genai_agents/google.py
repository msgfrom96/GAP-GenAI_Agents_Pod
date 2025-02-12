"""
Google AI Integration Module

This module provides an integrated interface for Google AI services.
"""

import os
import json
import numpy as np
import pandas as pd
import time
import asyncio
import logging
from typing import List, Dict, Any, Annotated, TypedDict, Union, Optional
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from functools import wraps
from datetime import datetime
from google.api_core import retry
from google.generativeai.types import GenerateContentResponse
from unittest.mock import Mock
import logging
from .base_service import BaseAIService, BaseMockService
from .config import Config, AIServiceConfig
from .agent_utils import AgentWorkflow, AgentTools

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure the Gemini API
GOOGLE_API_KEY = Config.GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Define state types for LangGraph
class AgentState(TypedDict):
    """Type definition for agent state management in LangGraph workflow."""

    messages: List[Union[HumanMessage, AIMessage]]
    next_step: str
    current_task: str
    documents: List[str]
    results: List[str]
    agent_scratchpad: str


# Add rate limiting decorator
def rate_limit(max_concurrent: int = 3):
    """Decorator to limit API call concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class GoogleAIService(BaseAIService):
    """Google AI service implementation"""

    def __init__(self, test_mode: bool = False):
        """Initialize Google AI service

        Args:
            test_mode: Whether to use mock services
        """
        config = Config.load_google_config()
        super().__init__(config, test_mode)

        if not self.test_mode:
            genai.configure(api_key=config.api_key)
            self.model = genai.GenerativeModel(config.model_name)
            self.llm = ChatGoogleGenerativeAI(model=config.model_name)

    def _setup_base_mocks(self):
        """Set up mock services"""
        super()._setup_base_mocks()
        self.mock_embedding = [0.1] * 768  # Google's embedding size
        self.model = Mock()
        mock_response = Mock()
        mock_response.text = "Mock Gemini response"
        self.model.generate_content = Mock(return_value=mock_response)

        def mock_embed_content(*args, **kwargs):
            return {"embedding": self.mock_embedding}

        genai.embed_content = Mock(side_effect=mock_embed_content)

        self.llm = Mock()
        self.llm.invoke = Mock(return_value="Mock LangChain response")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.test_mode:
            return self.mock_embedding

        response = genai.embed_content(model=self.config.embedding_model, content=text, task_type="retrieval_document")
        return response["embedding"]

    def generate_completion(self, prompt: str, context: str = None) -> str:
        """Generate completion with optional context"""
        if self.test_mode:
            return f"Mock Gemini response for: {prompt[:50]}..."

        if context:
            prompt = f"Context: {context}\n\nQuestion: {prompt}"

        response = self.model.generate_content(prompt)
        return response.text

    def setup_agent(self):
        """Set up agent workflow"""
        tools = AgentTools.create_document_search(self.semantic_search)
        prompt = AgentWorkflow.create_base_prompt()
        agent_tools = [
            {"type": t["type"], "function": {k: v for k, v in t["function"].items() if k != "implementation"}} for t in tools
        ]
        agent = create_openai_tools_agent(self.llm, agent_tools, prompt)
        self.workflow = AgentWorkflow.setup_workflow(agent, tools, prompt)

    def run_agent(self, query: str) -> str:
        """Run agent workflow"""
        if not hasattr(self, "workflow"):
            self.setup_agent()

        state = {
            "messages": [HumanMessage(content=query)],
            "next_step": "search",
            "current_task": "initial_search",
            "documents": self.doc_store.documents,
            "results": [],
            "agent_scratchpad": "",
            "error": None,
        }

        try:
            final_state = self.workflow.invoke(state)

            # Check for errors
            if final_state.get("error"):
                logger.error(f"Agent workflow failed: {final_state['error']}")
                return f"An error occurred: {final_state['error']}"

            # Process results
            results = final_state.get("results", [])
            if not results:
                return "No results found."

            # Format results
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    if "output" in result:
                        formatted_results.append(result["output"])
                    elif "document" in result:
                        formatted_results.append(f"{result['document']} (relevance: {result.get('relevance', 'N/A')})")
                else:
                    formatted_results.append(str(result))

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error running agent workflow: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

    async def aask_with_rag(self, query: str) -> str:
        """Async version of RAG query"""
        if self.test_mode:
            # Return a valid JSON mock response
            return json.dumps(
                {
                    "deprecated": ["mock_deprecated_func"],
                    "new_features": ["mock_new_feature"],
                    "security": ["mock_security_update"],
                    "performance": ["mock_performance_improvement"],
                    "breaking_changes": ["mock_breaking_change"],
                }
            )

        try:
            query_embedding = await self.aget_embedding(query, "retrieval_query")
            similarities = cosine_similarity([query_embedding], self.doc_store.embeddings)[0]

            top_indices = np.argsort(similarities)[-2:][::-1]
            context = "\n\n".join([self.doc_store.documents[i] for i in top_indices])

            # Construct a prompt that explicitly asks for JSON output
            json_prompt = f"""
            Based on the following context, provide a response in valid JSON format with these keys:
            - deprecated: list of deprecated functions/methods
            - new_features: list of new features
            - security: list of security updates
            - performance: list of performance improvements
            - breaking_changes: list of breaking changes

            Context: {context}
            Question: {query}
            
            Response must be valid JSON.
            """

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.model.generate_content(json_prompt, generation_config={"output_format": "json"})
            )

            # Parse the response to ensure it's valid JSON
            result = json.loads(response.text)
            return json.dumps(result)  # Return serialized JSON string

        except Exception as e:
            logger.error(f"Error in aask_with_rag: {str(e)}")
            # Return a valid JSON error response
            return json.dumps(
                {
                    "deprecated": [],
                    "new_features": [],
                    "security": [],
                    "performance": [],
                    "breaking_changes": [],
                    "error": str(e),
                }
            )


class GoogleAISystem:
    """
    Integrated GenAI system combining multiple Google AI capabilities.

    This class provides a unified interface for working with embeddings,
    document search, classification, RAG, agent-based workflows,
    and model fine-tuning.

    Args:
        test_mode: Whether to use mock services for testing (default: False)
    """

    def __init__(self, test_mode: bool = False):
        """Initialize the GenAI system with required models and storage."""
        self.test_mode = test_mode or Config.TEST_MODE

        if self.test_mode:
            self._setup_mocks()
        else:
            self._setup_real_services()

    def _setup_mocks(self):
        """Initialize mock services for testing purposes."""
        # Mock embedding vectors for consistent testing
        self.mock_embedding = [0.1] * 768  # Google's embedding size
        self.mock_documents = []
        self.mock_embeddings = []

        # Mock generative model
        self.model = Mock()
        mock_response = Mock()
        mock_response.text = "This is a mock response from Gemini model."
        self.model.generate_content = Mock(return_value=mock_response)

        # Mock embeddings
        def mock_embed_content(*args, **kwargs):
            return {"embedding": self.mock_embedding}

        genai.embed_content = Mock(side_effect=mock_embed_content)

        # Mock chat model for LangChain
        self.llm = Mock()
        self.llm.invoke = Mock(return_value="Mock LangChain response")

        # Mock document storage
        self.documents = []
        self.document_embeddings = []

        # Mock classifier
        self.classifier = Mock()
        self.classifier.predict = Mock(return_value=np.array([[0.8, 0.2]]))

        # Mock agent executor
        self.agent = Mock()
        self.agent.plan = Mock(return_value=AgentFinish(return_values={"output": "Mock agent response"}, log=""))

        # Mock LangGraph workflow
        self.graph = Mock()
        self.graph.invoke = Mock(
            return_value={
                "messages": [AIMessage(content="Mock workflow response")],
                "next_step": END,
                "current_task": "completed",
                "documents": [],
                "results": ["Mock workflow result"],
            }
        )

        # Mock tuned models
        self.tuned_models = {
            "mock-model-v1": Mock(generate_content=Mock(return_value=Mock(text="Mock tuned model response v1"))),
            "mock-model-v2": Mock(generate_content=Mock(return_value=Mock(text="Mock tuned model response v2"))),
        }
        self.active_tuned_model = "mock-model-v2"
        self.model_versions = {
            "mock-model-v1": {"accuracy": 0.85, "training_time": "2h"},
            "mock-model-v2": {"accuracy": 0.87, "training_time": "3h"},
        }

        # Other configurations
        self.embedding_cache = {}
        self.chunk_size = 512
        self.rate_limit_semaphore = asyncio.Semaphore(5)
        self.embedding_model = "models/text-embedding-004"
        self.model_performance = {
            "mock-model-v1": {"accuracy": 0.85, "latency": 1.0, "throughput": 1000},
            "mock-model-v2": {"accuracy": 0.87, "latency": 0.9, "throughput": 1200},
        }

    def _setup_real_services(self):
        """Initialize real Google AI services and configure system parameters."""
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable must be set when not using mock data.")

        self.model = genai.GenerativeModel("gemini-pro")
        self.embedding_model = "models/text-embedding-004"
        self.documents = []
        self.document_embeddings = []
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.tuned_models = {}
        self.active_tuned_model = None
        self.embedding_cache = {}
        self.chunk_size = 512
        self.model_versions = {}
        self.rate_limit_semaphore = asyncio.Semaphore(5)
        self.model_performance = {}

    def prepare_tuning_data(self, texts: List[str], labels: List[str], train_size: float = 0.8) -> tuple:
        """
        Prepare data for model fine-tuning.

        Args:
            texts: List of input texts
            labels: List of corresponding labels/categories
            train_size: Fraction of data to use for training

        Returns:
            Tuple of (train_df, test_df)
        """
        data = pd.DataFrame({"input_text": texts, "label": labels})

        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)

        # Split into train and test
        train_size = int(len(data) * train_size)
        train_df = data[:train_size]
        test_df = data[train_size:]

        return train_df, test_df

    def create_tuned_model(
        self,
        training_data: pd.DataFrame,
        model_name: str = "models/gemini-1.5-pro-001-tuning",
        batch_size: int = 16,
        epochs: int = 3,
    ) -> str:
        """
        Create a fine-tuned model using the provided training data.

        Args:
            training_data: DataFrame with 'input_text' and 'label' columns
            model_name: Base model to tune
            batch_size: Training batch size
            epochs: Number of training epochs

        Returns:
            ID of the tuned model
        """
        if self.test_mode:
            model_id = f"custom-model-{int(time.time())}"
            self.tuned_models[model_id] = Mock()
            self.active_tuned_model = model_id
            self.model_versions[model_id] = {
                "created_at": datetime.now().isoformat(),
                "base_model": model_name,
                "status": "PENDING",
                "training_samples": len(training_data),
            }
            return model_id

        try:
            model_id = f"custom-model-{int(time.time())}"
            tuning_op = genai.create_tuned_model(
                model_name,
                training_data=training_data,
                input_key="input_text",
                output_key="label",
                id=model_id,
                display_name=f"Custom tuned model {model_id}",
                batch_size=batch_size,
                epoch_count=epochs,
            )
            self.tuned_models[model_id] = tuning_op
            self.active_tuned_model = model_id
            self.model_versions[model_id] = {
                "created_at": datetime.now().isoformat(),
                "base_model": model_name,
                "status": "PENDING",
                "training_samples": len(training_data),
            }
            return model_id
        except Exception as e:
            self.logger.error(f"Error creating tuned model: {str(e)}")
            raise RuntimeError("Failed to create tuned model") from e

    @rate_limit()
    async def acreate_tuned_model(
        self,
        training_data: pd.DataFrame,
        model_name: str = "models/gemini-1.5-pro-001-tuning",
        batch_size: int = 16,
        epochs: int = 3,
    ) -> str:
        """
        Asynchronously create a fine-tuned model.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.create_tuned_model, training_data, model_name, batch_size, epochs)

    def load_tuned_model(self, model_id: str) -> None:
        """
        Load a previously fine-tuned model.

        Args:
            model_id: ID of the tuned model to load
        """
        try:
            if model_id not in self.tuned_models:
                raise ValueError(f"Model {model_id} not found in tuned models")
            self.active_tuned_model = model_id
        except Exception as e:
            logger.error(f"Error loading tuned model: {str(e)}")
            raise

    def list_tuned_models(self) -> List[str]:
        """Return list of available tuned model IDs"""
        return list(self.tuned_models.keys())

    def set_active_tuned_model(self, model_id: str) -> None:
        """Set active model from available tuned models"""
        if model_id not in self.tuned_models:
            raise ValueError(f"Model {model_id} not found")
        self.active_tuned_model = model_id

    def predict_with_tuned_model(self, text: str) -> str:
        """
        Generate predictions using the fine-tuned model.

        Args:
            text: Input text to classify/process

        Returns:
            Model's prediction
        """
        if not self.active_tuned_model:
            raise ValueError("No tuned model loaded. Call load_tuned_model first.")

        try:
            model = self.tuned_models[self.active_tuned_model]
            response = model.generate_content(text)
            return response.text
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError("Prediction failed") from e

    def evaluate_tuned_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the fine-tuned model on test data.

        Args:
            test_data: DataFrame with 'input_text' and 'label' columns

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.active_tuned_model:
            raise ValueError("No tuned model loaded. Call load_tuned_model first.")

        try:
            predictions = []
            for text in test_data["input_text"]:
                pred = self.predict_with_tuned_model(text)
                predictions.append(pred)

            accuracy = sum(p == l for p, l in zip(predictions, test_data["label"])) / len(predictions)

            return {"accuracy": accuracy, "total_samples": len(predictions)}
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise RuntimeError("Evaluation failed") from e

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    @retry.Retry()  # Add retry logic
    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        cache_key = hash(text + task_type)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        response = genai.embed_content(model=self.embedding_model, content=text, task_type=task_type)
        embedding = response["embedding"]
        self.embedding_cache[cache_key] = embedding
        return embedding

    @rate_limit()
    async def aget_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        cache_key = hash(text + task_type)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, genai.embed_content, self.embedding_model, text, task_type)
        self.embedding_cache[cache_key] = response["embedding"]
        return response["embedding"]

    def add_document(self, text: str) -> None:
        """Process documents in chunks"""
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.documents.append(chunk)
            embedding = self.get_embedding(chunk)
            self.document_embeddings.append(embedding)

    def add_documents_batch(self, texts: List[str]) -> None:
        """
        Add multiple documents to the knowledge base with batch processing.
        """
        try:
            self.documents.extend(texts)
            embeddings = self.get_embeddings_batch(texts)
            self.document_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Batch document addition failed: {str(e)}")
            raise

    def find_similar_documents(self, query: str, top_k: int = 3) -> List[tuple]:
        """
        Find documents most similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            query_embedding = self.get_embedding(query, task_type="retrieval_query")
            similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.documents[i], similarities[i]) for i in top_indices]
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise

    def build_classifier(self, num_classes: int, embedding_size: int = 768) -> keras.Sequential:
        """
        Build a neural classifier for embedded text.

        Args:
            num_classes: Number of classification categories
            embedding_size: Size of input embeddings

        Returns:
            Compiled Keras model
        """
        try:
            self.classifier = keras.Sequential(
                [
                    keras.layers.Input(shape=(embedding_size,)),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            self.classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return self.classifier
        except Exception as e:
            logger.error(f"Classifier build failed: {str(e)}")
            raise

    async def train_classifier_async(self, texts: List[str], labels: List[int], epochs: int = 10) -> None:
        """
        Asynchronously train the classifier on text data.

        Args:
            texts: List of training texts
            labels: Corresponding class labels
            epochs: Number of training epochs
        """
        try:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.get_embeddings_batch, texts, "classification"
            )
            X = np.array(embeddings)
            y = np.array(labels)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.classifier.fit(X, y, epochs=epochs, validation_split=0.2)
            )
        except Exception as e:
            self.logger.error(f"Async training failed: {str(e)}")
            raise

    def setup_function_calling(self) -> None:
        """Configure available functions for the model."""
        self.available_functions = {
            "search_documents": {
                "name": "search_documents",
                "description": "Search through documents to find relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "top_k": {"type": "integer", "description": "Number of results to return"},
                    },
                    "required": ["query"],
                },
            }
        }

    def process_with_function_calling(self, user_input: str) -> str:
        """
        Process user input using function calling.

        Args:
            user_input: User's text input

        Returns:
            Response from function execution or model
        """
        response = self.model.generate_content(user_input, tools=[self.available_functions["search_documents"]])

        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call

            if function_call.name == "search_documents":
                args = json.loads(function_call.args)
                results = self.find_similar_documents(args["query"], args.get("top_k", 3))
                return f"Found these relevant documents: {results}"

        return response.text

    def setup_agent(self) -> None:
        """Configure the LangGraph agent workflow."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search through documents to find relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "The search query"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_sentiment",
                    "description": "Analyze the sentiment of a text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string", "description": "The text to analyze"}},
                        "required": ["text"],
                    },
                },
            },
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant that can search through documents and analyze text."),
                ("user", "{input}\n{agent_scratchpad}"),
            ]
        )

        self.agent = create_openai_tools_agent(self.llm, tools, prompt)

        def search_node(state: AgentState) -> AgentState:
            """Process node that handles AgentAction and AgentFinish transitions."""
            result = self.agent.invoke({"input": state["messages"][-1].content, "agent_scratchpad": ""})

            if isinstance(result, AgentFinish):
                state["results"].extend(result.return_values["output"])
                return {**state, "next_step": END}

            if isinstance(result, AgentAction):
                args = json.loads(result.tool_input)
                search_results = self.find_similar_documents(args["query"])
                state["results"].extend(search_results)
                state["next_step"] = "analyze"
            return state

        def analyze_node(state: AgentState) -> AgentState:
            if not state["results"]:
                state["next_step"] = END
                return state

            context = "\n".join([doc for doc, _ in state["results"]])
            response = self.llm.invoke(f"Analyze this information: {context}")
            state["messages"].append(AIMessage(content=str(response)))
            state["next_step"] = END
            return state

        workflow = StateGraph(AgentState)
        workflow.add_node("search", search_node)
        workflow.add_node("analyze", analyze_node)
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", END)
        workflow.set_entry_point("search")

        self.graph = workflow.compile()

    def run_agent(self, user_input: str) -> str:
        """
        Run the agent workflow with user input.

        Args:
            user_input: User's query or command

        Returns:
            Agent's response after workflow completion
        """
        state = {
            "messages": [HumanMessage(content=user_input)],
            "next_step": "search",
            "current_task": "initial_search",
            "documents": self.documents.copy(),
            "results": [],
            "agent_scratchpad": "",
            "error": None,
        }

        final_state = self.graph.invoke(state)

        if final_state["messages"]:
            return str(final_state["messages"][-1].content)
        return "No results found."

    def save_agent_state(self, state: AgentState, filename: str) -> None:
        """Save agent state to JSON file"""
        with open(filename, "w") as f:
            json.dump(
                {
                    "messages": [m.dict() for m in state["messages"]],
                    "next_step": state["next_step"],
                    "current_task": state["current_task"],
                    "documents": state["documents"],
                    "results": state["results"],
                },
                f,
            )

    def load_agent_state(self, filename: str) -> AgentState:
        """Load agent state from JSON file"""
        with open(filename, "r") as f:
            data = json.load(f)
            return {
                "messages": [HumanMessage(**m) if m["type"] == "human" else AIMessage(**m) for m in data["messages"]],
                "next_step": data["next_step"],
                "current_task": data["current_task"],
                "documents": data["documents"],
                "results": data["results"],
            }

    @retry.Retry()
    def get_embeddings_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Batch process with caching"""
        if self.test_mode:
            return [self.mock_embedding] * len(texts)

        uncached = []
        cache_keys = []
        embeddings = []

        for text in texts:
            key = hash(text + task_type)
            cache_keys.append(key)
            if key in self.embedding_cache:
                embeddings.append(self.embedding_cache[key])
            else:
                uncached.append(text)
                embeddings.append(None)  # Placeholder

        if uncached:
            response = genai.embed_content(model=self.embedding_model, content=uncached, task_type=task_type)
            for i, emb in enumerate(response["embedding"]):
                self.embedding_cache[cache_keys[i]] = emb
                embeddings[i] = emb

        return embeddings

    def list_model_versions(self) -> dict:
        """List available model versions"""
        return {
            "active": self.active_tuned_model,
            "available": list(self.tuned_models.keys()),
            "performance": self.model_performance,
        }

    def revert_model_version(self, version: str):
        """Rollback to previous model version"""
        if version in self.tuned_models:
            self.active_tuned_model = version
            logging.info(f"Reverted to model version {version}")


def main():
    """Interactive CLI to select GenAI functionalities"""
    import argparse

    parser = argparse.ArgumentParser(description="Google GenAI Integration CLI")
    parser.add_argument("--rag", action="store_true", help="Run RAG Q&A test")
    parser.add_argument("--search", action="store_true", help="Run document search test")
    parser.add_argument("--function", action="store_true", help="Test function calling")
    parser.add_argument("--agent", action="store_true", help="Run agent workflow")
    parser.add_argument("--tune", action="store_true", help="Test model fine-tuning")
    parser.add_argument("--classify", action="store_true", help="Test classification")
    parser.add_argument("--add-docs", action="store_true", help="Add sample documents to knowledge base")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--live", action="store_true", help="Use live API instead of mock mode")
    args = parser.parse_args()

    # Use mock mode by default unless --live is specified
    genai_system = GoogleAISystem(test_mode=not args.live)

    print(f"Running in {'live' if args.live else 'mock'} mode")

    # Sample documents
    documents = [
        "The Mars rover Perseverance landed in February 2021.",
        "Neural networks are a key component of deep learning.",
        "Python is a popular programming language for AI development.",
    ]

    if args.add_docs or args.all:
        print("\n=== Adding Documents ===")
        genai_system.add_documents_batch(documents)
        print("Added 3 sample documents")

    if args.search or args.all:
        print("\n=== Document Search Test ===")
        query = "When did the Mars mission land?"
        similar = genai_system.find_similar_documents(query)
        print(f"Similar documents for '{query}':")
        for doc, score in similar:
            print(f"- {doc} (similarity: {score:.2f})")

    if args.rag or args.all:
        print("\n=== RAG Q&A Test ===")
        question = "What Mars rover are we talking about?"
        answer = asyncio.run(genai_system.aask_with_rag(question))
        print(f"Question: {question}\nAnswer: {answer}")

    if args.function or args.all:
        print("\n=== Function Calling Test ===")
        genai_system.setup_function_calling()
        result = genai_system.process_with_function_calling("Find me documents about space")
        print(f"Function calling result: {result}")

    if args.agent or args.all:
        print("\n=== Agent Workflow Test ===")
        genai_system.setup_agent()
        agent_response = genai_system.run_agent("What can you tell me about AI from the documents?")
        print(f"Agent response: {agent_response}")

    if args.tune or args.all:
        print("\n=== Model Tuning Test ===")
        try:
            train_df, test_df = genai_system.prepare_tuning_data(["Sample text 1", "Sample text 2"], ["Label 1", "Label 2"])
            model_id = genai_system.create_tuned_model(train_df)
            print(f"Started tuning job ID: {model_id}")
        except Exception as e:
            print(f"Tuning test failed: {str(e)}")

    if args.classify or args.all:
        print("\n=== Classification Test ===")
        try:
            genai_system.build_classifier(num_classes=3)
            print("Built classifier model")
        except Exception as e:
            print(f"Classification setup failed: {str(e)}")


if __name__ == "__main__":
    main()
