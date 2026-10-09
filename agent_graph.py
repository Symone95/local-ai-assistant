"""
Costruzione del grafo LangGraph dell'assistente.

Vive fuori da app_with_mcp.py perche' lo usano due interfacce: la web app Streamlit e la
modalita' vocale ([voice/](voice/)). Qui si definiscono nodi ed edge; la compilazione resta
a chi lo usa, che decide il checkpointer.
"""
from langgraph.graph import StateGraph, END

from dto.agent_state import AgentState
from mcp_integration.client import mcp_tool_node
from nodes import router_node, tool_node, direct_llm_answer, llm_node


def decide_path(state: AgentState):
    """Sceglie il nodo successivo al router in base al tool deciso dal planner."""
    tool = state["tool_plan"]["tool"].strip()

    if tool == "none":
        return "direct_llm_answer"

    if tool.startswith("mcp_"):
        return "mcp_tool"

    return "tool"


def build_graph() -> StateGraph:
    """
    Diagramma:

                  router
             /      |      \\
          tool  mcp_tool   direct_llm_answer
             \\     /             |
               llm              END
                |
               END
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("tool", tool_node)
    graph.add_node("direct_llm_answer", direct_llm_answer)
    graph.add_node("llm", llm_node)
    graph.add_node("mcp_tool", mcp_tool_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        decide_path,
        {
            "tool": "tool",
            "mcp_tool": "mcp_tool",
            "direct_llm_answer": "direct_llm_answer",
        },
    )

    graph.add_edge("tool", "llm")
    graph.add_edge("mcp_tool", "llm")
    graph.add_edge("llm", END)
    graph.add_edge("direct_llm_answer", END)

    return graph
