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
from typing import List, Dict, Any, TypedDict, Union, Optional
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
from unittest.mock import Mock
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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

class IntegratedOpenAI:
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

    def __init__(self, model: str = "gpt-4-turbo-preview", test_mode: bool = False):
        self.test_mode = test_mode or (os.getenv("TEST_MODE") == "1")
        
        if self.test_mode:
            self._setup_mocks()
        else:
            self._setup_real_services()

    def _setup_mocks(self):
        """Initialize mock services for testing purposes.
        
        Creates mocked versions of:
        - Embedding generation
        - Chat completions
        - Classification model
        - Agent executor
        """
        self.client = Mock()
        self.client.embeddings.create = Mock(return_value=Mock(data=[Mock(embedding=[0.1]*1536)]))
        self.client.chat.completions.create = Mock(return_value=Mock(choices=[Mock(message=Mock(content="Test response"))]))
        self.classifier = Mock()
        self.classifier.predict = Mock(return_value=np.array([[0.8, 0.2]]))
        self.agent_executor = Mock()
        self.agent_executor.invoke = Mock(return_value={"output": "Mock response"})
        
    def _setup_real_services(self):
        """Initialize production services and configure system parameters.
        
        Sets up:
        - OpenAI API client
        - Document storage and embeddings
        - Language model instances
        - Model version tracking
        - Rate limiting safeguards
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4-turbo-preview"
        self.embedding_model = "text-embedding-3-large"
        self.documents = []
        self.document_embeddings = []
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", openai_api_key=OPENAI_API_KEY)
        self.tuned_models = {}
        self.active_tuned_model = None
        self.embedding_cache = {}
        self.chunk_size = 512
        self.model_versions = {}
        self.rate_limit_semaphore = asyncio.Semaphore(5)

    def prepare_training_data(self, texts: List[str], labels: List[str], 
                           train_size: float = 0.8) -> tuple:
        """Prepare training data for fine-tuning.
        
        Args:
            texts: List of input text samples
            labels: Corresponding list of target labels
            train_size: Proportion of data to use for training (0-1)
            
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        data = pd.DataFrame({
            "messages": [
                [{"role": "user", "content": text},
                 {"role": "assistant", "content": label}]
                for text, label in zip(texts, labels)
            ]
        })
        
        data = data.sample(frac=1).reset_index(drop=True)
        train_size = int(len(data) * train_size)
        return data[:train_size], data[train_size:]

    def train_custom_model(self, training_data: pd.DataFrame,
                         base_model: str = "gpt-3.5-turbo",
                         model_suffix: Optional[str] = None) -> str:
        """Start a fine-tuning job for a custom model.
        
        Args:
            training_data: Prepared training data in OpenAI format
            base_model: Base model to fine-tune
            model_suffix: Optional suffix for the trained model name
            
        Returns:
            ID of the initiated fine-tuning job
        """
        file_response = self.client.files.create(
            file=training_data.to_json(orient='records', lines=True),
            purpose='fine-tune'
        )
        
        job = self.client.fine_tuning.jobs.create(
            training_file=file_response.id,
            model=base_model,
            suffix=model_suffix
        )
        return job.id

    def get_training_status(self, job_id: str) -> Dict:
        """Retrieve status of a fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job to check
            
        Returns:
            Dictionary containing job status, trained tokens, and resulting model
        """
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            'status': job.status,
            'trained_tokens': job.trained_tokens,
            'model': job.fine_tuned_model
        }

    def load_custom_model(self, model_id: str) -> None:
        """Load a fine-tuned model for use in generation.
        
        Args:
            model_id: ID of the fine-tuned model to load
        """
        self.fine_tuned_model = model_id

    def generate_with_custom_model(self, text: str) -> str:
        """Generate text using the loaded custom model.
        
        Args:
            text: Input prompt to generate from
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If no custom model is loaded
        """
        if not self.fine_tuned_model:
            raise ValueError("No custom model loaded")
            
        response = self.client.chat.completions.create(
            model=self.fine_tuned_model,
            messages=[{"role": "user", "content": text}]
        )
        return response.choices[0].message.content

    def evaluate_model_performance(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate custom model performance on test data.
        
        Args:
            test_data: DataFrame containing test samples
            
        Returns:
            Dictionary with accuracy and sample count
            
        Raises:
            ValueError: If no custom model is loaded
        """
        if not self.fine_tuned_model:
            raise ValueError("No custom model loaded")
            
        predictions = []
        labels = []
        
        for row in test_data.itertuples():
            messages = row.messages
            input_text = messages[0]['content']
            true_label = messages[1]['content']
            
            pred = self.generate_with_custom_model(input_text)
            predictions.append(pred)
            labels.append(true_label)
            
        accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(predictions)
        return {'accuracy': accuracy, 'total_samples': len(predictions)}

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding

    def index_document(self, text: str) -> None:
        """Index a document for semantic search.
        
        Args:
            text: Document content to index
        """
        self.documents.append(text)
        self.document_embeddings.append(self.generate_embeddings(text))

    def semantic_search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Perform semantic search across indexed documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        query_embedding = self.generate_embeddings(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

    def create_classifier_model(self, num_classes: int, embedding_size: int = 1536) -> keras.Sequential:
        """Create a neural network classifier model.
        
        Args:
            num_classes: Number of output classes
            embedding_size: Size of input embeddings
            
        Returns:
            Compiled Keras Sequential model
        """
        self.classifier = keras.Sequential([
            keras.layers.Input(shape=(embedding_size,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        self.classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.classifier

    def train_classification_model(self, texts: List[str], labels: List[int], epochs: int = 10) -> None:
        """Train the classification model on labeled data.
        
        Args:
            texts: List of training texts
            labels: Corresponding class labels
            epochs: Number of training epochs
        """
        embeddings = [self.generate_embeddings(text) for text in texts]
        self.classifier.fit(np.array(embeddings), np.array(labels), epochs=epochs, validation_split=0.2)

    def classify_content(self, text: str) -> np.ndarray:
        """Classify input text using the trained model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Array of class probabilities
        """
        embedding = self.generate_embeddings(text)
        return self.classifier.predict(np.array([embedding]))[0]

    def generate_rag_response(self, question: str) -> str:
        """Generate answer using Retrieval Augmented Generation.
        
        Args:
            question: Question to answer
            
        Returns:
            Generated answer with context from documents
        """
        similar_docs = self.semantic_search(question, top_k=2)
        context = "\n\n".join([doc for doc, _ in similar_docs])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer questions based on the context below. If unsure, say 'I don't know'."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content

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
                            "top_k": {"type": "integer", "description": "Number of results to return"}
                        },
                        "required": ["query"]
                    }
                }
            }
        }

    def process_function_call(self, user_input: str) -> str:
        """Process user input with function calling.
        
        Args:
            user_input: Natural language query requiring document search
            
        Returns:
            Search results or direct response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_input}],
            tools=[self.available_functions["search_documents"]],
            tool_choice={"type": "function", "function": {"name": "search_documents"}}
        )
        
        if response.choices[0].message.tool_calls:
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            results = self.semantic_search(args["query"], args.get("top_k", 3))
            return f"Relevant documents: {[doc[0] for doc in results]}"
            
        return response.choices[0].message.content

    def initialize_agent(self) -> None:
        """Initialize LangGraph agent workflow with document search capability.
        
        Configures:
        - Agent prompt template
        - Workflow nodes for search and processing
        - State transition logic
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that can search through documents."),
            ("user", "{input}")
        ])
        
        agent = create_openai_tools_agent(self.llm, [], prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=[], handle_parsing_errors=True)

        def search_node(state: AgentState) -> Dict[str, Any]:
            """Process node that handles AgentAction and AgentFinish transitions."""
            result = self.agent_executor.invoke({"input": state["messages"][-1].content})
            
            if isinstance(result, AgentFinish):
                state["results"].extend(result.return_values["output"])
                return {**state, "next_step": END}
            
            if isinstance(result, AgentAction):
                args = json.loads(result.tool_input)
                search_results = self.semantic_search(args["query"])
                state["results"].extend(search_results)
                return {**state, "next_step": "process_results"}
            
            return state

        workflow = StateGraph(AgentState)
        workflow.add_node("search", search_node)
        workflow.add_node("process_results", lambda state: state)
        workflow.set_entry_point("search")
        workflow.add_edge("search", "process_results")
        workflow.add_edge("process_results", END)
        self.graph = workflow.compile()

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
            "results": []
        }
        
        final_state = self.graph.invoke(state)
        return str(final_state["results"]) if final_state["results"] else "No results found."
    
    def list_model_versions(self) -> dict:
        """List available model versions and performance metrics.
        
        Returns:
            Dictionary with active model, available versions, and performance data
        """
        return {
            'active': self.active_tuned_model,
            'available': list(self.tuned_models.keys()),
            'performance': self.model_performance
        }

    def revert_model_version(self, version: str) -> None:
        """Revert to a previous model version.
        
        Args:
            version: Model version identifier to revert to
        """
        if version in self.tuned_models:
            self.active_tuned_model = version
            self.logger.info(f"Reverted to model version {version}")

def main():
    """Demonstration of IntegratedOpenAI system capabilities.
    
    Example usage:
    - Document indexing and semantic search
    - RAG-based question answering
    - Function calling implementation
    - Agent workflow execution
    """
    openai_system = IntegratedOpenAI()
    
    documents = [
        "The Mars rover Perseverance landed in February 2021.",
        "Neural networks are a key component of deep learning.",
        "Python is a popular programming language for AI development."
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
