"""
OpenAI Integration Module

This module provides an integrated interface for:
- Text embeddings and similarity analysis
- Document Q&A with RAG (Retrieval Augmented Generation)
- Neural classification with embeddings
- Function calling capabilities
- Agent workflows with LangGraph
- Model fine-tuning and customization
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, TypedDict, Union, Optional, Tuple
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from unittest.mock import Mock, MagicMock
import asyncio
import logging
from .config import Config, AIServiceConfig
from .base_service import BaseAIService, BaseMockService
from .agent_utils import AgentWorkflow, AgentTools
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType

OPENAI_API_KEY = Config.OPENAI_API_KEY
OPENAI_ORGANIZATION = Config.OPENAI_ORGANIZATION
OPENAI_PROJECT = Config.OPENAI_PROJECT
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG, project=OPENAI_PROJECT)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Type definition for agent state management in LangGraph workflow.

    Attributes:
        messages: List of chat messages in the conversation
        next_step: Name of the next workflow step to execute
        current_task: Description of the current task being processed
        documents: List of documents available for processing
        results: Accumulated results from workflow execution
    """

    messages: List[Union[HumanMessage, AIMessage]]
    next_step: str
    current_task: str
    documents: List[str]
    results: List[Dict]


class OpenAIService(BaseAIService):
    """OpenAI service implementation"""

    def __init__(self, test_mode: bool = False):
        """Initialize OpenAI service

        Args:
            test_mode: Whether to use mock services
        """
        config = Config.load_openai_config()
        super().__init__(config, test_mode)

        if not self.test_mode:
            self.client = OpenAI(api_key=config.api_key, organization=Config.OPENAI_ORG, project=Config.OPENAI_PROJECT)
            self.llm = ChatOpenAI(model=config.model_name, openai_api_key=config.api_key)

    def _setup_base_mocks(self):
        """Set up mock services"""
        super()._setup_base_mocks()
        self.client = Mock()
        self.client.embeddings.create = Mock(return_value=BaseMockService.create_mock_embedding_response(self.mock_embedding))
        self.client.chat.completions.create = Mock(return_value=BaseMockService.create_mock_completion("Mock OpenAI response"))
        self.llm = Mock()
        self.llm.invoke = Mock(return_value="Mock LangChain response")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.test_mode:
            return self.mock_embedding

        response = self.client.embeddings.create(model=self.config.embedding_model, input=text)
        return response.data[0].embedding

    def generate_completion(self, prompt: str, context: str = None) -> str:
        """Generate completion with optional context"""
        if self.test_mode:
            return f"Mock response for: {prompt[:50]}..."

        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(model=self.config.model_name, messages=messages)
        return response.choices[0].message.content

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
        """Run agent workflow with user query.

        Args:
            query: User's query to process

        Returns:
            Agent's response after workflow completion
        """
        if self.test_mode:
            return "Mock agent response"

        try:
            if not hasattr(self, "agent_executor"):
                self.initialize_agent()

            result = self.agent_executor.invoke({"input": query})
            return result.get("output", "No response generated")
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return f"An error occurred: {str(e)}"

    async def ask_with_rag(self, prompt: str, context: str) -> str:
        """Ask a question with retrieval-augmented generation."""
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
            # Construct a prompt that explicitly asks for JSON output
            json_prompt = f"""
            Based on the following context, provide a response in valid JSON format with these keys:
            - deprecated: list of deprecated functions/methods
            - new_features: list of new features
            - security: list of security updates
            - performance: list of performance improvements
            - breaking_changes: list of breaking changes

            Context: {context}
            Question: {prompt}
            
            Response must be valid JSON.
            """

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that always responds in valid JSON format.",
                        },
                        {"role": "user", "content": json_prompt},
                    ],
                    response_format={"type": "json_object"},  # Enforce JSON response
                ),
            )

            # Parse the response to ensure it's valid JSON
            result = json.loads(response.choices[0].message.content)
            return json.dumps(result)  # Return serialized JSON string

        except Exception as e:
            logger.error(f"Error in ask_with_rag: {str(e)}")
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


class OpenAISystem(BaseAIService):
    """
    Integrated OpenAI system combining multiple AI capabilities.

    Provides a unified interface for:
    - Text embeddings and semantic search
    - Document indexing and retrieval
    - Neural network classification
    - Retrieval Augmented Generation (RAG)
    - Function calling and agent workflows
    - Model fine-tuning and management

    Args:
        model: Base model name to use for completions (default: 'gpt-4-turbo-preview')
        test_mode: Whether to use mock services for testing (default: False)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, test_mode: bool = False):
        """Initialize OpenAI system with configuration.

        Args:
            config: Configuration dictionary
            test_mode: Whether to run in test mode
        """
        if config is None:
            config = {}

        # Set test mode in config and as instance variable
        config["test_mode"] = test_mode or Config.TEST_MODE

        super().__init__(config)

        # Set test mode after super().__init__() to ensure it's not overridden
        self.test_mode = config["test_mode"]

        # Set API key
        self.api_key = config.get("api_key") or Config.OPENAI_API_KEY
        if not self.api_key and not self.test_mode:
            raise ValueError("API key must be set when not using mock data.")

        # Initialize common attributes
        self.tools = []  # Initialize empty tools list
        self.agent_executor = None
        self.documents = []  # Initialize empty documents list
        self.document_embeddings = []  # Initialize empty embeddings list
        self.embedding_cache = {}
        self.chunk_size = 512
        self.model_versions = {}
        self.rate_limit_semaphore = asyncio.Semaphore(5)

        if self.test_mode:
            self._setup_mocks()
        else:
            self._setup_real_services()

    def _setup_real_services(self):
        """Set up real OpenAI services."""
        self.client = OpenAI(api_key=self.api_key)
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.embedding_model = self.config.get("embedding_model", "text-embedding-ada-002")
        self.fine_tuned_model = None
        self.active_tuned_model = None
        self.llm = ChatOpenAI(model_name=self.model, temperature=0.7, api_key=self.api_key)

    def _setup_mocks(self):
        """Set up mock services for testing."""
        # Mock OpenAI client
        mock_client = MagicMock(spec=OpenAI)
        mock_embeddings = MagicMock()
        mock_chat = MagicMock()
        mock_files = MagicMock()
        mock_fine_tuning = MagicMock()

        # Set up mock responses
        mock_embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])

        mock_chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Generated response"))]
        )

        mock_files.create.return_value = MagicMock(id="mock-file-id")
        mock_files.list.return_value = MagicMock(data=[])

        mock_fine_tuning.jobs.create.return_value = MagicMock(id="mock-job-id")
        mock_fine_tuning.jobs.retrieve.return_value = MagicMock(status="succeeded", fine_tuned_model="mock-model-id")

        # Attach mock services to client
        mock_client.embeddings = mock_embeddings
        mock_client.chat = mock_chat
        mock_client.files = mock_files
        mock_client.fine_tuning = mock_fine_tuning
        self.client = mock_client

        # Set up other attributes
        self.model = "gpt-3.5-turbo"
        self.embedding_model = "text-embedding-ada-002"
        self.fine_tuned_model = None
        self.active_tuned_model = None

        # Create a proper mock for ChatOpenAI
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm.invoke.return_value = "Mock LangChain response"
        mock_llm.model_name = "gpt-3.5-turbo"
        mock_llm.temperature = 0.7
        self.llm = mock_llm

        # Set up mock embeddings
        self.mock_embedding = [0.1] * 1536

        # Set up mock classifier
        self.classifier = MagicMock()
        self.classifier.predict.return_value = np.array([[0.8, 0.2]])

        # Set up mock model versions
        self.model_versions = {
            "mock-model-v1": {"accuracy": 0.85, "training_time": "2h"},
            "mock-model-v2": {"accuracy": 0.87, "training_time": "3h"},
        }
        self.active_tuned_model = "mock-model-v2"

        # Initialize document storage
        self.documents = []
        self.document_embeddings = []

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.test_mode:
            return self.mock_embedding

        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        return self.generate_embedding(text)

    def generate_completion(self, prompt: str, context: str = None) -> str:
        """Generate completion with optional context."""
        if self.test_mode:
            return f"Mock response for: {prompt[:50]}..."

        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content

    def save_agent_state(self, state: Dict, filename: str) -> None:
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

    def load_agent_state(self, filename: str) -> Dict:
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

    def prepare_training_data(self, texts: List[str], labels: List[str], train_size: float = 0.8) -> tuple:
        """Prepare training data for fine-tuning.

        Args:
            texts: List of input text samples
            labels: Corresponding list of target labels
            train_size: Proportion of data to use for training (0-1)

        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        data = pd.DataFrame(
            {
                "messages": [
                    [{"role": "user", "content": text}, {"role": "assistant", "content": label}]
                    for text, label in zip(texts, labels)
                ]
            }
        )

        data = data.sample(frac=1).reset_index(drop=True)
        train_size = int(len(data) * train_size)
        return data[:train_size], data[train_size:]

    def train_custom_model(self, training_data: pd.DataFrame) -> str:
        """Train a custom model using fine-tuning."""
        if self.test_mode:
            return "mock-model-id"

        # Convert training data to JSONL
        training_file = "training_data.jsonl"
        training_data.to_json(training_file, orient="records", lines=True)

        # Upload training file
        with open(training_file, "rb") as f:
            file_response = self.client.files.create(file=f, purpose="fine-tune")

        # Create fine-tuning job
        job = self.client.fine_tuning.jobs.create(training_file=file_response.id, model=self.model)

        return job.id

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of fine-tuning job."""
        if self.test_mode:
            return {"status": "succeeded", "fine_tuned_model": "mock-model-id"}

        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {"status": job.status, "fine_tuned_model": job.fine_tuned_model if job.status == "succeeded" else None}

    def load_custom_model(self, model_id: str) -> None:
        """Load a fine-tuned model."""
        if not model_id or not str(model_id).strip():
            raise ValueError("Model ID cannot be None or empty.")

        if self.test_mode:
            self.fine_tuned_model = "mock-model-id"
            return

        try:
            # Verify model exists
            response = self.client.models.retrieve(model_id)
            self.fine_tuned_model = model_id
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate_with_custom_model(self, prompt: str) -> str:
        """Generate text with custom fine-tuned model."""
        if not self.fine_tuned_model:
            raise ValueError("No fine-tuned model loaded. Call load_custom_model() first.")

        if self.test_mode:
            return f"Mock response for: {prompt}"

        try:
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model, messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating with custom model: {str(e)}")
            raise RuntimeError(f"Failed to generate with custom model: {str(e)}")

    def evaluate_model_performance(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not self.fine_tuned_model:
            raise ValueError("No fine-tuned model loaded. Call load_custom_model() first.")

        if self.test_mode:
            return {"accuracy": 0.85, "total_samples": len(test_data)}

        try:
            predictions = []
            for _, row in test_data.iterrows():
                messages = row["messages"]
                input_text = next(m["content"] for m in messages if m["role"] == "user")
                true_label = next(m["content"] for m in messages if m["role"] == "assistant")

                pred = self.generate_with_custom_model(input_text)
                predictions.append(pred == true_label)

            accuracy = sum(predictions) / len(predictions)
            return {"accuracy": accuracy, "total_samples": len(predictions)}
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")

    def create_classifier_model(self, num_classes: int = 2, embedding_size: int = 1536) -> None:
        """Create a neural network classifier.

        Args:
            num_classes: Number of output classes
            embedding_size: Size of input embeddings
        """
        if self.test_mode:
            # In test mode, create a mock classifier
            mock_classifier = MagicMock()
            mock_classifier.layers = [MagicMock(input_shape=(None, embedding_size)), MagicMock(units=num_classes)]
            self.classifier = mock_classifier
            return

        try:
            # Create a simple feed-forward neural network
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(embedding_size,)),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            # Compile the model
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            self.classifier = model

        except Exception as e:
            logger.error(f"Error creating classifier model: {str(e)}")
            raise RuntimeError(f"Failed to create classifier model: {str(e)}")

    def train_classification_model(self, texts: List[str], labels: List[int]) -> None:
        """Train neural classifier on text embeddings."""
        if self.test_mode:
            return

        embeddings = [self.generate_embeddings(text) for text in texts]
        X = np.array(embeddings)
        y = np.array(labels)

        if not hasattr(self, "classifier"):
            self.create_classifier_model(num_classes=len(set(labels)))

        self.classifier.fit(X, y, epochs=10, batch_size=32, verbose=0)

    def classify_content(self, text: str) -> Dict[str, float]:
        """Classify content using the trained model.

        Args:
            text: Text to classify

        Returns:
            Dictionary mapping class indices to probabilities

        Raises:
            ValueError: If classifier is not trained
        """
        if not hasattr(self, "classifier") or self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classification_model() first.")

        if self.test_mode:
            # Return mock probabilities in test mode
            return {0: 0.8, 1: 0.2}

        try:
            # Generate embedding for the text
            embedding = self.generate_embedding(text)

            # Get model predictions
            predictions = self.classifier.predict(np.array([embedding]), verbose=0)

            # Convert predictions to dictionary
            return {i: float(prob) for i, prob in enumerate(predictions[0])}

        except Exception as e:
            logger.error(f"Error classifying content: {str(e)}")
            raise RuntimeError(f"Failed to classify content: {str(e)}")

    def generate_rag_response(self, question: str, context: str = None) -> str:
        """Generate answer using Retrieval Augmented Generation.

        Args:
            question: Question to answer
            context: Optional additional context to use. If None, will use similar documents.

        Returns:
            Generated answer with context from documents
        """
        if self.test_mode:
            return f"Mock RAG response for question: {question}"

        try:
            if context is None:
                if not self.documents:
                    return "No documents available for search."
                similar_docs = self.semantic_search(question, top_k=2)
                if not similar_docs:
                    return "No relevant documents found."
                context = "\n\n".join([doc for doc, _ in similar_docs])

            # Ensure we have documents or context
            if not context:
                return "No context available for response generation."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer questions based on the context below. If unsure, say 'I don't know'.",
                    },
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in generate_rag_response: {str(e)}")
            return f"An error occurred: {str(e)}"

    def semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.documents:
            return []

        if self.test_mode:
            return [(doc, 0.9) for doc in self.documents[:top_k]]

        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Calculate similarities
            similarities = []
            for doc, doc_embedding in zip(self.documents, self.document_embeddings):
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((doc, float(similarity)))

            # Sort by similarity and return top_k
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

    def enable_function_calling(self) -> None:
        """Enable document search function calling capability."""
        self.available_functions = {
            "search_documents": {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search through documents to find relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {"type": "integer", "description": "Number of results to return"},
                        },
                        "required": ["query"],
                    },
                },
            }
        }

    def process_function_call(self, user_input: str) -> str:
        """Process user input with function calling.

        Args:
            user_input: Natural language query requiring document search

        Returns:
            Search results or direct response
        """
        if self.test_mode:
            return f"Mock function call response for: {user_input}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_input}],
                tools=[self.available_functions["search_documents"]],
                tool_choice={"type": "function", "function": {"name": "search_documents"}},
            )

            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                results = self.semantic_search(args["query"], args.get("top_k", 3))
                return f"Relevant documents: {[doc[0] for doc in results]}"

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in process_function_call: {str(e)}")
            return f"An error occurred: {str(e)}"

    def initialize_agent(self):
        """Initialize the agent with the current system configuration"""
        self.agent_executor = initialize_agent(tools=self.tools, llm=self.llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        return self.agent_executor

    def run_agent(self, input_text: str) -> str:
        """Run the agent with the given input text.

        Args:
            input_text (str): The input text to process

        Returns:
            str: The agent's response

        Raises:
            ValueError: If the agent is not initialized
        """
        if not hasattr(self, "agent_executor") or self.agent_executor is None:
            raise RuntimeError("Agent executor not initialized. Call initialize_agent() first.")

        try:
            result = self.agent_executor.invoke({"input": input_text})
            return result.get("output", "No response generated")
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            raise RuntimeError(f"Failed to run agent: {str(e)}")

    def execute_agent_workflow(self, user_input: str) -> str:
        """Execute full agent workflow for a user query.

        Args:
            user_input: Natural language query to process

        Returns:
            Formatted results from workflow execution
        """
        state = {
            "messages": [HumanMessage(content=user_input)],
            "next_step": "search",
            "current_task": "initial_search",
            "documents": self.documents.copy(),
            "results": [],
            "agent_scratchpad": "",  # Initialize empty scratchpad
        }

        final_state = self.graph.invoke(state)
        return str(final_state["results"]) if final_state["results"] else "No results found."

    def list_model_versions(self) -> dict:
        """List available model versions and performance metrics.

        Returns:
            Dictionary with active model, available versions, and performance data
        """
        return {
            "active": self.active_tuned_model,
            "available": list(self.model_versions.keys()),
            "performance": self.evaluate_model_performance(
                pd.DataFrame(
                    {
                        "messages": [
                            [
                                {"role": "user", "content": "This is a test message."},
                                {"role": "assistant", "content": self.active_tuned_model},
                            ]
                        ]
                    }
                )
            ),
        }

    def revert_model_version(self, version: str) -> None:
        """Revert to a previous model version.

        Args:
            version: Model version identifier to revert to
        """
        if version in self.model_versions:
            self.active_tuned_model = version
            self.logger.info(f"Reverted to model version {version}")

    def index_document(self, document: str) -> None:
        """Index a document for semantic search.

        Args:
            document: Document text to index
        """
        if not document:
            raise ValueError("Document text cannot be None or empty.")

        if self.test_mode:
            # In test mode, just store the document with a mock embedding
            self.documents.append(document)
            self.document_embeddings.append(self.mock_embedding)
            return

        try:
            # Generate embedding for the document
            embedding = self.generate_embedding(document)

            # Store document and its embedding
            self.documents.append(document)
            self.document_embeddings.append(embedding)

        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise RuntimeError(f"Failed to index document: {str(e)}")


def main():
    """Demonstration of IntegratedOpenAI system capabilities.

    Example usage:
    - Document indexing and semantic search
    - RAG-based question answering
    - Function calling implementation
    - Agent workflow execution
    """
    openai_system = OpenAISystem(test_mode=True)

    documents = [
        "The Mars rover Perseverance landed in February 2021.",
        "Neural networks are a key component of deep learning.",
        "Python is a popular programming language for AI development.",
    ]

    for doc in documents:
        openai_system.index_document(doc)

    query = "When did the Mars mission land?"
    similar = openai_system.semantic_search(query)
    print(f"Similar documents for '{query}':")
    for doc, score in similar:
        print(f"- {doc} (similarity: {score:.2f})")

    question = "What Mars rover are we talking about?"
    answer = openai_system.generate_rag_response(question)
    print(f"\nQuestion: {question}\nAnswer: {answer}")

    openai_system.enable_function_calling()
    result = openai_system.process_function_call("Find me documents about Mars")
    print(f"\nFunction calling result: {result}")

    openai_system.initialize_agent()
    agent_response = openai_system.execute_agent_workflow("What can you tell me about Mars from the documents?")
    print(f"\nAgent response: {agent_response}")


if __name__ == "__main__":
    main()
