from rag_engine import build_chat_history, rewrite_query_with_memory
from tools import tool_planner, execute_tool, TOOLS
from utils.general import loads_json_loose
from dto.agent_state import AgentState
import json
import re

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5-coder:3b",  # llama3"
                 num_ctx=4096,              # con questo dico di non andare oltre i 4k di token
                )


KNOWN_TOOLS = {t["name"] for t in TOOLS}


def parse_tool_plan(raw: str) -> dict:
    """
    Estrae il piano JSON dalla risposta del planner.

    L'LLM spesso incapsula il JSON in un blocco markdown (```json ... ```) o vi premette una frase:
    un json.loads() diretto fallisce e il piano viene scartato. Qui si isola il primo oggetto JSON
    presente nel testo e si valida il nome del tool contro il registro.
    """
    if not raw:
        return {"tool": "none"}

    plan = loads_json_loose(raw)
    if plan is None:
        print(f"⚠️  Piano del tool non interpretabile, uso 'none'. Risposta grezza: {raw[:200]!r}")
        return {"tool": "none"}

    if not isinstance(plan, dict) or "tool" not in plan:
        return {"tool": "none"}

    tool = str(plan.get("tool", "none")).strip()
    if tool != "none" and tool not in KNOWN_TOOLS:
        print(f"⚠️  Il planner ha scelto un tool inesistente ({tool!r}), uso 'none'.")
        return {"tool": "none"}

    plan["tool"] = tool
    return plan


def router_node(state: AgentState):
    # Passa anche i messaggi e il contesto al tool_planner per considerare il file temporaneo caricato
    messages = state.get("messages", [])
    context = state.get("context", "")
    plan = tool_planner(state["query"], messages=messages, context=context)

    plan = parse_tool_plan(plan)

    print("TOOL PLANNER HA DECISO DI USARE IL TOOL: %s" % plan["tool"])
    # Conserva anche query e params: servono ai tool MCP, che altrimenti ricevono args vuoti
    state["tool_plan"] = {
        "tool": plan["tool"],
        "query": plan.get("query") or state["query"],
        "args": plan.get("params") or plan.get("args") or {},
    }
    return state


def tool_node(state: AgentState):
    print("CHIAMO TOOL NODE")
    tool_name = state["tool_plan"]["tool"]
    query = state["query"]
    messages = state.get("messages", [])
    result = execute_tool(
        tool_name,
        query,
        selected_doc=state.get("selected_doc"),
        messages=messages,
        context=state.get("context", ""),
        current_file=state.get("current_file", "")
    )
    
    if result and "pdf" in result: # Se abbiamo generato un PDF NON serve far parlare l'LLM
            return {
                "final_answer": "Ho creato il report PDF! Scaricalo qui sotto 👇",
                "pdf_path": result["pdf"]
            }

    return {"tool_result": result, "current_file": result.get("current_file", "")}


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

    context = state.get('context', '').strip()

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

Contesto temporaneo:
{context if context else "(nessun file temporaneo caricato)"}

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
3. Se il risultato del tool contiene il codice di un file, usalo come base per analisi, refactoring o spiegazioni — non dire che non hai accesso al file
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




def genera_prompt_finanziario(crypto_name, dati_tecnici, notizie_web):
    prompt = f"""
    Sei un Analista Finanziario Esperto specializzato nel mercato delle Criptovalute.
    Il tuo compito è analizzare la seguente criptovaluta e fornire una raccomandazione motivata: COMPRARE, VENDERE o ATTENDERE (HOLD).

    NOTIZIE DI MERCATO RECENTI:
    {notizie_web}

    DATI TECNICI IN TEMPO REALE (Valuta: EUR):
    - Prezzo Attuale: {dati_tecnici.get('prezzo_attuale')} €
    - Variazione nelle ultime 24 ore: {dati_tecnici.get('variazione_24h')}%
    - Volume di Trading 24h: {dati_tecnici.get('volume_24h')} €

    LINEE GUIDA PER IL VERDETTO:
    1. Sii oggettivo e prudente. Non promettere guadagni certi.
    2. Se le notizie sono estremamente negative (FUD, problemi legali), prediligi un verdetto di ATTENDERE o VENDERE.
    3. Se il volume è in forte crescita e la variazione a 24h è positiva unita a news rialziste, valuta il COMPRARE impostando un livello di stop-loss logico.
    4. Includi SEMPRE un Disclaimer Finanziario alla fine del report.

    Fornisci l'analisi strutturata in capitoli Markdown:
    ### 📊 Analisi di Mercato per {crypto_name}
    - **Panoramica Tecnica**
    - **Analisi del Sentiment (News)**
    - **Verdetto Finale ed Esecuzione**
    """
    return prompt