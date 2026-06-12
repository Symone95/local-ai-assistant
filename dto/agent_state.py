from typing import TypedDict, List, Dict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    sources: Dict
    context: str
    current_file: str  # <--- Traccia l'ultimo file aperto/nominato
    selected_doc: List
    tool_plan: Dict
    tool_result: Dict
    final_answer: str