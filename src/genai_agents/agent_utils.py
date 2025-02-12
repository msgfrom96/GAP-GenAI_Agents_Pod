"""
Agent Utilities Module

Provides common functionality for agent workflows.
"""

from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_tools_agent
import json
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Type definition for agent state"""

    messages: List[HumanMessage | AIMessage]
    next_step: str
    current_task: str
    documents: List[str]
    results: List[Any]
    agent_scratchpad: str
    error: Optional[str]


class AgentTools:
    """Utility class for creating agent tools"""

    @staticmethod
    def create_document_search(search_func) -> list:
        """Create document search tools"""

        def search_documents(args):
            query = args["query"]
            top_k = args.get("top_k", 3)
            return search_func(query, top_k)

        return [
            {
                "type": "function",
                "function": {
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
                    "implementation": search_documents,
                },
            }
        ]


class AgentWorkflow:
    """Utility class for creating agent workflows"""

    @staticmethod
    def create_base_prompt() -> ChatPromptTemplate:
        """Create base prompt template"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant that can search through documents."),
                ("user", "{input}\n{agent_scratchpad}"),
            ]
        )

    @staticmethod
    def setup_workflow(llm, tools, prompt):
        """Set up workflow with document search capability"""
        agent = create_openai_tools_agent(llm, tools, prompt)

        def search_node(state: AgentState) -> AgentState:
            """Process node that handles AgentAction and AgentFinish transitions."""
            try:
                result = agent.invoke(
                    {
                        "input": state["messages"][-1].content,
                        "agent_scratchpad": state.get("agent_scratchpad", ""),
                        "intermediate_steps": state.get("intermediate_steps", []),
                    }
                )

                if isinstance(result, AgentFinish):
                    state["results"] = [result.return_values["output"]]
                    state["next_step"] = END
                    return state

                if isinstance(result, AgentAction):
                    tool = next((t for t in tools if t["function"]["name"] == result.tool), None)
                    if tool:
                        args = json.loads(result.tool_input)
                        search_results = tool["function"]["implementation"](args)
                        if isinstance(search_results, str):
                            state["results"].append(search_results)
                        elif isinstance(search_results, list):
                            state["results"].extend(search_results)

                    # Add intermediate step
                    if "intermediate_steps" not in state:
                        state["intermediate_steps"] = []
                    state["intermediate_steps"].append(result)

                    # Generate summary after search
                    summary = "Summary of results:\n" + "\n".join(str(r) for r in state["results"])
                    state["results"] = [summary]
                    state["next_step"] = "process_results"
                    return state

                return state
            except Exception as e:
                logger.error(f"Error in search node: {str(e)}")
                state["error"] = str(e)
                state["next_step"] = END
                return state

        def process_results(state: AgentState) -> AgentState:
            """Process search results and generate final response."""
            try:
                if not state["results"]:
                    state["next_step"] = END
                    return state

                # Format results for better readability
                formatted_results = []
                for result in state["results"]:
                    if isinstance(result, tuple):
                        doc, score = result
                        formatted_results.append(f"{doc} (relevance: {score:.2f})")
                    elif isinstance(result, dict):
                        if "output" in result:
                            formatted_results.append(result["output"])
                        elif "document" in result:
                            formatted_results.append(f"{result['document']} (relevance: {result.get('relevance', 'N/A')})")
                    else:
                        formatted_results.append(str(result))

                # Generate final response
                result = agent.invoke(
                    {
                        "input": f"Based on these results:\n{formatted_results}\nProvide a final response.",
                        "agent_scratchpad": state.get("agent_scratchpad", ""),
                        "intermediate_steps": state.get("intermediate_steps", []),
                    }
                )

                if isinstance(result, AgentFinish):
                    state["results"] = [result.return_values["output"]]
                else:
                    state["results"] = formatted_results

                state["next_step"] = END
                return state
            except Exception as e:
                logger.error(f"Error in process_results: {str(e)}")
                state["error"] = str(e)
                state["next_step"] = END
                return state

        workflow = StateGraph(AgentState)
        workflow.add_node("search", search_node)
        workflow.add_node("process_results", process_results)
        workflow.set_entry_point("search")
        workflow.add_edge("search", "process_results")
        workflow.add_edge("process_results", END)

        return workflow.compile()


def get_initial_state(query: str, documents: List[str]) -> AgentState:
    """Create initial state for workflow"""
    return {
        "messages": [HumanMessage(content=query)],
        "next_step": "search",
        "current_task": "initial_search",
        "documents": documents,
        "results": [],
        "agent_scratchpad": "",
    }


def search_node(state: AgentState) -> AgentState:
    """Process node that handles AgentAction and AgentFinish transitions."""
    try:
        result = agent.invoke(
            {
                "input": state["messages"][-1].content,
                "agent_scratchpad": state.get("agent_scratchpad", ""),
                "intermediate_steps": state.get("intermediate_steps", []),
            }
        )

        if isinstance(result, AgentFinish):
            state["results"] = [result.return_values["output"]]
            state["next_step"] = END
            return state

        if isinstance(result, AgentAction):
            tool = next((t for t in tools if t["function"]["name"] == result.tool), None)
            if tool:
                args = json.loads(result.tool_input)
                search_results = tool["function"]["implementation"](args)
                if isinstance(search_results, str):
                    state["results"].append(search_results)
                elif isinstance(search_results, list):
                    state["results"].extend(search_results)

            # Add intermediate step
            if "intermediate_steps" not in state:
                state["intermediate_steps"] = []
            state["intermediate_steps"].append(result)

            # Generate summary after search
            summary = "Summary of results:\n" + "\n".join(str(r) for r in state["results"])
            state["results"] = [summary]
            state["next_step"] = "process_results"
            return state

        return state
    except Exception as e:
        logger.error(f"Error in search node: {str(e)}")
        state["error"] = str(e)
        state["next_step"] = END
        return state


def process_results(state: AgentState) -> AgentState:
    """Process search results and generate final response."""
    try:
        if not state["results"]:
            state["next_step"] = END
            return state

        # Format results for better readability
        formatted_results = []
        for result in state["results"]:
            if isinstance(result, tuple):
                doc, score = result
                formatted_results.append(f"{doc} (relevance: {score:.2f})")
            elif isinstance(result, dict):
                if "output" in result:
                    formatted_results.append(result["output"])
                elif "document" in result:
                    formatted_results.append(f"{result['document']} (relevance: {result.get('relevance', 'N/A')})")
            else:
                formatted_results.append(str(result))

        # Generate final response
        result = agent.invoke(
            {
                "input": f"Based on these results:\n{formatted_results}\nProvide a final response.",
                "agent_scratchpad": state.get("agent_scratchpad", ""),
                "intermediate_steps": state.get("intermediate_steps", []),
            }
        )

        if isinstance(result, AgentFinish):
            state["results"] = [result.return_values["output"]]
        else:
            state["results"] = formatted_results

        state["next_step"] = END
        return state
    except Exception as e:
        logger.error(f"Error in process_results: {str(e)}")
        state["error"] = str(e)
        state["next_step"] = END
        return state
