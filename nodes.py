from rag_engine import build_chat_history, rewrite_query_with_memory
from tools import tool_planner, execute_tool
from dto.agent_state import AgentState
import json

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5-coder:3b",  # llama3")
                 num_ctx=4096)              # con questo dico di non andare oltre i 4k di token


def router_node(state: AgentState):
    # Passa anche i messaggi al tool_planner per considerare il contesto conversazionale
    messages = state.get("messages", [])
    plan = tool_planner(state["query"], messages=messages)

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

    # Se abbiamo generato un PDF NON serve far parlare l'LLM
    if result and "pdf" in result:
        return {
            "final_answer": "Ho creato il report PDF! Scaricalo qui sotto 👇",
            "pdf_path": result["pdf"]
        }

    return {"tool_result": result}


def direct_llm_answer(state: AgentState):
    """
    Risposta diretta senza usare tools o RAG.
    Serve per small talk o domande generiche.
    """
    print("CHIAMO DLA")
    query = state["query"]
    messages = state.get("messages", [])
    chat_history = build_chat_history(messages) if messages else ""
    standalone_query = rewrite_query_with_memory(query, chat_history)

    prompt = f"""
Sei un assistente AI utile, simpatico e intelligente.

COMPORTAMENTO:
- Rispondi naturalmente e conversazionalmente
- Sii utile senza essere troppo conciso
- Mantieni una personalità amichevole
- Mantieni la stessa lingua usata dall'utente
- Ricorda il contesto della conversazione precedente
- Se la domanda è correlata ai messaggi precedenti, fai riferimento ad essi
- Se è un nuovo argomento, rispondi semplicemente senza forzare connessioni

COSA NON FARE:
- Non inventare informazioni
- Non fare meta-commenti
- Non essere robotico o formale eccessivamente

Conversazione precedente:
{chat_history if chat_history else "(nessuna conversazione precedente)"}

Domanda attuale: {standalone_query}
"""
    response = llm.invoke(prompt)
    return {
            "final_answer": response.content,
            "messages": [response]
        }


def llm_node(state: AgentState):
    # Recupero la history dei messaggi per passarla alla conversazione in modo tale che abbia un ricordo di quanto detto fin'ora
    chat_history = build_chat_history(state["messages"]) if state["messages"] else ""
    tool_result = state.get('tool_result', '')
    context = state.get('context', '').strip()

    prompt = f"""
Sei un assistente AI che elabora informazioni da documenti e tool.

COMPORTAMENTO OBBLIGATORIO:
1. Rispondi direttamente e chiaramente alla domanda dell'utente
2. Usa SOLO le informazioni fornite dal tool/contesto
3. Se il risultato del tool è rilevante, fai riferimento ad esso esplicitamente
4. Se il contesto è vuoto o il tool non ha trovato nulla, comunica chiaramente

CITAZIONI E FONTI:
- SEMPRE alla fine della risposta, elenca le fonti usate
- Formato: "📚 Fonti: [nome_file.pdf] (pagina X)"
- Se hai usato dati da contesto ricercato, cita sempre il file di provenienza
- Se il tool non ha restituito fonti, indica chiaramente che l'informazione proviene da conoscenza generale

LINGUA:
- Mantieni la stessa lingua usata dall'utente in tutta la risposta
- Detecta automaticamente se è italiano, inglese, ecc.

DIVIETI ASSOLUTI:
- Non inventare dati o fonti che non conosci
- Non fare meta-commenti sul processo ("il tool mi ha detto", "ho cercato")
- Non essere robotico - sii naturale e conversazionale
- Non ignorare il contesto conversazionale se rilevante
- Non aggiungere informazioni non fondate

CONTESTO CONVERSAZIONALE PRECEDENTE:
{chat_history if chat_history else "(nessuna conversazione precedente)"}

INFORMAZIONI DAL TOOL:
Contesto/Dati recuperati:
{context if context else "(nessun contesto disponibile)"}

Risultato dello strumento:
{tool_result if tool_result else "(nessun risultato disponibile)"}

DOMANDA ATTUALE DELL'UTENTE:
{state['query']}

---
Rispondi ora fornendo una risposta chiara e completa, terminando SEMPRE con le fonti.
"""

    response = llm.invoke(prompt)
    return {
            "final_answer": response.content,
            "messages": [response]
        }
