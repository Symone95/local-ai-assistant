from rag_engine import build_chat_history, rewrite_query_with_memory
from tools import tool_planner, execute_tool
from dto.agent_state import AgentState
import json

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3")


def router_node(state: AgentState):
    plan = tool_planner(state["query"])

    try:
        plan = json.loads(plan)
    except:
        plan = {"tool": "none"}
    state["tool_plan"] = {"tool": plan["tool"]}
    return state


def tool_node(state: AgentState):
    print("CHIAMO TOOL NODE")
    tool_name = state["tool_plan"]["tool"]
    query = state["query"]
    messages = state.get("messages", [])
    result = execute_tool(tool_name, query, selected_doc=state.get("selected_doc"), messages=messages)
    return {"tool_result": result}


def direct_llm_answer(state: AgentState):
    """
    Risposta diretta senza usare tools o RAG.
    Serve per small talk o domande generiche.
    """
    print("CHIAMO DLA")
    query = state["query"]
    messages = state.get("messages", "")
    chat_history = build_chat_history(messages) if messages else ""
    standalone_query = rewrite_query_with_memory(query, chat_history)

    prompt = f"""
Sei un assistente AI utile e intelligente.
Rispondi normalmente alla domanda dell'utente simpaticamente senza essere troppo conciso.

Conversazione:
{chat_history}

Domanda: {standalone_query}
"""
    response = llm.invoke(prompt)
    return {
            "final_answer": response.content,
            "messages": [response]
        }


def llm_node(state: AgentState):
    # Recupero la history dei messaggi per passarla alla conversazione in modo tale che abbia un ricordo di quanto detto fin'ora
    chat_history = build_chat_history(state["messages"]) if state["messages"] else ""

    prompt = f"""
Usa questi dati per rispondere all'utente.

Regole:
- Alla fine mostra sempre le fonti usate con nome del file e pagina in cui hai trovato le informazioni fornite
- Rispondi in modo chiaro e diretto.
- Non inventare dati.
- Non parlare dello strumento.
- Non fare meta-commenti.
- Mantieni la stessa lingua usata dall'utente nella risposta

Contesto:
{state.get('context', '').strip()}

Conversazione:
{chat_history}

Domanda dell'utente: {state['query']}

Lo strumento ha restituito questi dati:
{state.get('tool_result', '')}
        
Rispondi in modo chiaro facendo riferimento al risultato dei tool utilizzati.
"""

    response = llm.invoke(prompt)
    return {
            "final_answer": response.content,
            "messages": [response]
        }