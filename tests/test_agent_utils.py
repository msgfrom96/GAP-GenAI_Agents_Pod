import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from src.genai_agents.agent_utils import AgentWorkflow, AgentTools, get_initial_state, AgentState
from src.genai_agents.config import Config


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke = Mock(return_value="Mock LangChain response")
    return llm


@pytest.fixture
def mock_semantic_search():
    def search_func(query: str, top_k: int = 3):
        return [("Document 1", 0.9), ("Document 2", 0.8), ("Document 3", 0.7)]

    return search_func


def test_create_document_search():
    """Test creation of document search tools"""
    mock_search = lambda x: [("Doc", 0.9)]
    tools = AgentTools.create_document_search(mock_search)

    assert isinstance(tools, list)
    assert len(tools) > 0
    assert tools[0]["type"] == "function"
    assert "search_documents" in tools[0]["function"]["name"]
    assert "parameters" in tools[0]["function"]


def test_create_base_prompt():
    """Test creation of base prompt template"""
    prompt = AgentWorkflow.create_base_prompt()
    assert prompt is not None

    # Test prompt formatting
    formatted = prompt.format(input="test query", agent_scratchpad="")
    assert "test query" in str(formatted)


@pytest.mark.asyncio
async def test_setup_workflow(mock_llm, mock_semantic_search):
    """Test workflow setup and execution"""
    tools = AgentTools.create_document_search(mock_semantic_search)
    prompt = AgentWorkflow.create_base_prompt()
    workflow = AgentWorkflow.setup_workflow(mock_llm, tools, prompt)

    assert workflow is not None

    # Test workflow execution
    state: AgentState = {
        "messages": [HumanMessage(content="test query")],
        "next_step": "search",
        "current_task": "initial_search",
        "documents": ["doc1", "doc2"],
        "results": [],
        "agent_scratchpad": "",
        "error": None,
    }

    final_state = workflow.invoke(state)
    assert final_state is not None
    assert "results" in final_state


@pytest.mark.asyncio
async def test_workflow_with_agent_action(mock_llm, mock_semantic_search):
    """Test workflow handling of AgentAction"""
    # Mock the agent to return an action first
    action = AgentAction(tool="search_documents", tool_input='{"query": "test"}', log="")
    finish = AgentFinish(return_values={"output": "Final result"}, log="")

    # Mock the LLM responses
    mock_llm.invoke = Mock(return_value=action)  # First call returns action
    mock_llm.bind = Mock(return_value=mock_llm)

    # Create a mock agent that returns our mocked responses
    mock_agent = Mock()
    mock_agent.invoke = Mock(side_effect=[action, finish])

    tools = AgentTools.create_document_search(mock_semantic_search)
    prompt = AgentWorkflow.create_base_prompt()

    # Mock the create_openai_tools_agent function to return our mock agent
    with patch("src.genai_agents.agent_utils.create_openai_tools_agent", return_value=mock_agent):
        workflow = AgentWorkflow.setup_workflow(mock_llm, tools, prompt)

        state: AgentState = {
            "messages": [HumanMessage(content="test query")],
            "next_step": "search",
            "current_task": "initial_search",
            "documents": ["doc1", "doc2"],
            "results": [],
            "agent_scratchpad": "",
            "error": None,
            "intermediate_steps": [],
        }

        final_state = workflow.invoke(state)
        assert isinstance(final_state, dict)
        assert "results" in final_state
        assert len(final_state["results"]) > 0
        assert "Final result" in str(final_state["results"])


def test_workflow_error_handling(mock_llm, mock_semantic_search):
    """Test workflow error handling"""
    # Mock the agent to raise an exception
    mock_llm.invoke.side_effect = Exception("Test error")

    tools = AgentTools.create_document_search(mock_semantic_search)
    prompt = AgentWorkflow.create_base_prompt()
    workflow = AgentWorkflow.setup_workflow(mock_llm, tools, prompt)

    state: AgentState = {
        "messages": [HumanMessage(content="test query")],
        "next_step": "search",
        "current_task": "initial_search",
        "documents": ["doc1", "doc2"],
        "results": [],
        "agent_scratchpad": "",
        "error": None,
    }

    final_state = workflow.invoke(state)
    assert final_state.get("error") is not None


def test_tool_implementation():
    """Test the implementation of search tool"""
    mock_search = Mock(return_value=[("Doc", 0.9)])
    tools = AgentTools.create_document_search(mock_search)

    # Get the implementation function
    search_func = tools[0]["function"]["implementation"]

    # Test with minimal parameters
    result = search_func({"query": "test"})
    assert isinstance(result, list)
    mock_search.assert_called_once()

    # Test with top_k parameter
    mock_search.reset_mock()
    result = search_func({"query": "test", "top_k": 5})
    assert isinstance(result, list)
    mock_search.assert_called_once_with("test", 5)


def test_workflow_state_transitions():
    """Test workflow state transitions"""
    mock_llm = Mock()
    mock_search = Mock(return_value=[("Doc", 0.9)])

    tools = AgentTools.create_document_search(mock_search)
    prompt = AgentWorkflow.create_base_prompt()
    workflow = AgentWorkflow.setup_workflow(mock_llm, tools, prompt)

    # Test initial state
    state = get_initial_state("test query", ["doc1", "doc2"])
    assert state["next_step"] == "search"
    assert state["current_task"] == "initial_search"
    assert isinstance(state["messages"], list)
    assert isinstance(state["results"], list)


@pytest.mark.asyncio
async def test_deployed_workflow():
    """Test workflow in deployed environment"""
    pytest.importorskip("openai")

    if not Config.DEPLOY_TESTS:
        pytest.skip("Skipping deployed environment test")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI()
    mock_search = lambda x: [("Test document", 0.9)]

    tools = AgentTools.create_document_search(mock_search)
    prompt = AgentWorkflow.create_base_prompt()
    workflow = AgentWorkflow.setup_workflow(llm, tools, prompt)

    state: AgentState = {
        "messages": [HumanMessage(content="What is in the test document?")],
        "next_step": "search",
        "current_task": "initial_search",
        "documents": ["Test document content"],
        "results": [],
        "agent_scratchpad": "",
        "error": None,
    }

    final_state = workflow.invoke(state)
    assert final_state is not None
    assert len(final_state["results"]) > 0
