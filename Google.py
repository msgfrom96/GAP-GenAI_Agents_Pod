"""
Google GenAI Integration Module

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Define state types for LangGraph
class AgentState(TypedDict):
    """Type definition for agent state management in LangGraph workflow."""
    messages: List[Union[HumanMessage, AIMessage]]
    next_step: str
    current_task: str
    documents: List[str]
    results: List[Dict]

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

class IntegratedGenAI:
    """
    Integrated GenAI system combining multiple Google AI capabilities.
    
    This class provides a unified interface for working with embeddings,
    document search, classification, RAG, agent-based workflows,
    and model fine-tuning.
    """

    def __init__(self):
        """Initialize the GenAI system with required models and storage."""
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = "models/text-embedding-004"
        self.documents = []
        self.document_embeddings = []
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.tuned_models = {}  # Dictionary for version management
        self.active_tuned_model = None
        self.embedding_cache = {}
        self.chunk_size = 512  # For document chunking
        self.model_versions = {}  # Track tuned models
        self.rate_limit_semaphore = asyncio.Semaphore(5)  # API rate limiting
        
    def prepare_tuning_data(self, texts: List[str], labels: List[str], 
                          train_size: float = 0.8) -> tuple:
        """
        Prepare data for model fine-tuning.

        Args:
            texts: List of input texts
            labels: List of corresponding labels/categories
            train_size: Fraction of data to use for training

        Returns:
            Tuple of (train_df, test_df)
        """
        data = pd.DataFrame({
            "input_text": texts,
            "label": labels
        })
        
        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        
        # Split into train and test
        train_size = int(len(data) * train_size)
        train_df = data[:train_size]
        test_df = data[train_size:]
        
        return train_df, test_df

    def create_tuned_model(self, training_data: pd.DataFrame, 
                          model_name: str = "models/gemini-1.5-pro-001-tuning",
                          batch_size: int = 16,
                          epochs: int = 3) -> str:
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
        model_id = f"custom-model-{int(time.time())}"
        
        try:
            tuning_op = genai.create_tuned_model(
                model_name,
                training_data=training_data,
                input_key="input_text",
                output_key="label",
                id=model_id,
                display_name=f"Custom tuned model {model_id}",
                batch_size=batch_size,
                epoch_count=epochs
            )
            self.tuned_models[model_id] = tuning_op
            self.active_tuned_model = model_id
            self.model_versions[model_id] = {
                "created_at": datetime.now().isoformat(),
                "base_model": model_name,
                "status": "PENDING",
                "training_samples": len(training_data)
            }
            return model_id
        except Exception as e:
            self.model_versions[model_id]["status"] = "FAILED"
            logger.error(f"Error creating tuned model: {str(e)}")
            raise RuntimeError("Failed to create tuned model") from e

    @rate_limit()
    async def acreate_tuned_model(self, training_data: pd.DataFrame, 
                                     model_name: str = "models/gemini-1.5-pro-001-tuning",
                                     batch_size: int = 16,
                                     epochs: int = 3) -> str:
        """
        Asynchronously create a fine-tuned model.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.create_tuned_model, training_data, model_name, batch_size, epochs
        )

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
            for text in test_data['input_text']:
                pred = self.predict_with_tuned_model(text)
                predictions.append(pred)
            
            accuracy = sum(p == l for p, l in zip(predictions, test_data['label'])) / len(predictions)
            
            return {
                'accuracy': accuracy,
                'total_samples': len(predictions)
            }
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise RuntimeError("Evaluation failed") from e

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        return [text[i:i+self.chunk_size] 
                for i in range(0, len(text), self.chunk_size)]

    @retry.Retry()  # Add retry logic
    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        cache_key = hash(text + task_type)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        response = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type=task_type
        )
        embedding = response["embedding"]
        self.embedding_cache[cache_key] = embedding
        return embedding

    @rate_limit()
    async def aget_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        cache_key = hash(text + task_type)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            genai.embed_content,
            self.embedding_model,
            text,
            task_type
        )
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
            similarities = cosine_similarity(
                [query_embedding], 
                self.document_embeddings
            )[0]
            
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
            self.classifier = keras.Sequential([
                keras.layers.Input(shape=(embedding_size,)),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            self.classifier.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
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
                None, self.classifier.fit, X, y, epochs, 0.2
            )
        except Exception as e:
            logger.error(f"Async training failed: {str(e)}")
            raise

    @rate_limit()
    async def aask_with_rag(self, question: str) -> str:
        """Async version of RAG query"""
        query_embedding = await self.aget_embedding(question, "retrieval_query")
        similarities = cosine_similarity(
            [query_embedding], 
            self.document_embeddings
        )[0]
        
        top_indices = np.argsort(similarities)[-2:][::-1]
        context = "\n\n".join([self.documents[i] for i in top_indices])
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.model.generate_content,
            f"Context: {context}\nQuestion: {question}"
        )
        return response.text

    def setup_function_calling(self) -> None:
        """Configure available functions for the model."""
        self.available_functions = {
            "search_documents": {
                "name": "search_documents",
                "description": "Search through documents to find relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
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
        response = self.model.generate_content(
            user_input,
            tools=[self.available_functions["search_documents"]]
        )
        
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            
            if function_call.name == "search_documents":
                args = json.loads(function_call.args)
                results = self.find_similar_documents(
                    args["query"], 
                    args.get("top_k", 3)
                )
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
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_sentiment",
                    "description": "Analyze the sentiment of a text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to analyze"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that can search through documents and analyze text."),
            ("user", "{input}")
        ])
        
        self.agent = create_openai_tools_agent(self.llm, tools, prompt)

        def search_node(state: AgentState) -> AgentState:
            messages = state["messages"]
            action = self.agent.plan(messages)
            
            if isinstance(action, AgentFinish):
                state["next_step"] = END
                return state
            
            if action.tool == "search_documents":
                results = self.find_similar_documents(action.tool_input["query"])
                state["results"].extend(results)
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
            "results": []
        }
        
        final_state = self.graph.invoke(state)
        
        if final_state["messages"]:
            return str(final_state["messages"][-1].content)
        return "No results found."

    def save_agent_state(self, state: AgentState, filename: str) -> None:
        """Save agent state to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "messages": [m.dict() for m in state["messages"]],
                "next_step": state["next_step"],
                "current_task": state["current_task"],
                "documents": state["documents"],
                "results": state["results"]
            }, f)

    def load_agent_state(self, filename: str) -> AgentState:
        """Load agent state from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
            return {
                "messages": [HumanMessage(**m) if m['type'] == 'human' 
                            else AIMessage(**m) for m in data["messages"]],
                "next_step": data["next_step"],
                "current_task": data["current_task"],
                "documents": data["documents"],
                "results": data["results"]
            }

    @retry.Retry()
    def get_embeddings_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Batch process with caching"""
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
            response = genai.embed_content(
                model=self.embedding_model,
                content=uncached,
                task_type=task_type
            )
            for i, emb in enumerate(response["embedding"]):
                self.embedding_cache[cache_keys[i]] = emb
                embeddings[i] = emb
                
        return embeddings

    def list_model_versions(self) -> dict:
        """List available model versions"""
        return {
            'active': self.active_tuned_model,
            'available': list(self.tuned_models.keys()),
            'performance': self.model_performance
        }

    def revert_model_version(self, version: str):
        """Rollback to previous model version"""
        if version in self.tuned_models:
            self.active_tuned_model = version
            self.logger.info(f"Reverted to model version {version}")

def main():
    """Interactive CLI to select GenAI functionalities"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google GenAI Integration CLI')
    parser.add_argument('--rag', action='store_true', help='Run RAG Q&A test')
    parser.add_argument('--search', action='store_true', help='Run document search test')
    parser.add_argument('--function', action='store_true', help='Test function calling')
    parser.add_argument('--agent', action='store_true', help='Run agent workflow')
    parser.add_argument('--tune', action='store_true', help='Test model fine-tuning')
    parser.add_argument('--classify', action='store_true', help='Test classification')
    parser.add_argument('--add-docs', action='store_true', 
                      help='Add sample documents to knowledge base')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    args = parser.parse_args()

    genai_system = IntegratedGenAI()
    
    # Sample documents
    documents = [
        "The Mars rover Perseverance landed in February 2021.",
        "Neural networks are a key component of deep learning.",
        "Python is a popular programming language for AI development."
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
        answer = genai_system.ask_with_rag(question)
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
            train_df, test_df = genai_system.prepare_tuning_data(
                ["Sample text 1", "Sample text 2"],
                ["Label 1", "Label 2"]
            )
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
