import streamlit as st

from mcp_client import mcp_tool_node
from rag_engine import add_documents, reset_database, collection, get_file_hash
from utils.general import get_db_stats, convert_to_langchain_messages, load_file_text
import os
import asyncio
from nodes import *
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from streamlit_mic_recorder import mic_recorder
from dto.managers.radio_manager import radio_manager
from utils.speech_to_text import transcribe_audio
from utils.text_to_speech import generate_tts

# TODO: FAR GENERARE DOCUMENTI, FAR RISPONDERE IN MARKDOWN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# LangGraph in una frase
# È una macchina a stati (state machine) dove:
# - ogni nodo = una funzione Python
# - lo stato = un dizionario che viaggia tra i nodi (AgentState)
# - gli edge = decidono chi viene eseguito dopo(le relazioni tra i nodi)

# Istanzio l'oggetto Grafo che ha come parametro lo stato il quale verrà utilizzato dai vari nodi / edge
graph = StateGraph(AgentState)

# Qui registro tutti i nodi che saranno presenti sul grafo con id e funzione associata
graph.add_node("router", router_node)
graph.add_node("tool", tool_node)
graph.add_node("direct_llm_answer", direct_llm_answer)
graph.add_node("llm", llm_node)
graph.add_node("mcp_tool", mcp_tool_node)

def decide_path(state: AgentState):
    """
    Funzione per decidere quale sarà la funzione chiamata successivamente al nodo eseguito
    """
    tool = state["tool_plan"]["tool"].strip()

    if tool == "none":
        return "direct_llm_answer"

    if tool.startswith("mcp_"):
        return "mcp_tool"

    return "tool"

# Qui diciamo che deve iniziare da questo nodo con id "router"
graph.set_entry_point("router")
# Qui stiamo praticamente dicendo:
# - Dopo aver eseguito il nodo router, chiama decide_path(state).
# - La funzione restituisce una stringa.
# - Usa quella stringa per decidere il prossimo nodo.
#              router
#         /             \
# direct_llm_answer     tool
graph.add_conditional_edges(
    "router",
    decide_path,
    {
        "tool": "tool",
        "mcp_tool": "mcp_tool",
        "direct_llm_answer": "direct_llm_answer"
    }
)

graph.add_edge("tool", "llm")
graph.add_edge("mcp_tool", "llm") #END)
graph.add_edge("llm", END)
graph.add_edge("direct_llm_answer", END)


# Diagramma aggiornato:
#               router
#          /      |      \
#       tool  mcp_tool   direct_llm_answer
#         \     /               |
#           llm                END
#            |
#           END

app = graph.compile()

## FE
st.set_page_config(page_title="Local RAG Chat", layout="wide")  # Titolo del tab

# TODO: INTEGRARE QUESTA PARTE PER CONVERSARE DIRETTAMENTE CON LLM
#if "assistant_speaking" not in st.session_state:
#    st.session_state.assistant_speaking = False

#if "voice_mode" not in st.session_state:
#    st.session_state.voice_mode = False

#st.session_state.voice_mode = st.toggle(
#    "🎙️ Modalità conversazione vocale continua",
#    value=st.session_state.voice_mode
#)

st.sidebar.header("🗄️ Database")  # Titolo sidebar a sinistra

# --- STATS ---
doc_count = get_db_stats(collection)
st.sidebar.write(f"Chunk salvati: **{doc_count}**")

# --- DOC FILTER ---
st.sidebar.subheader("📄 Filtra per documento")

docs = collection.get()["metadatas"]
doc_names = list(set([m["file"] for m in docs])) if docs else []

selected_doc = st.sidebar.selectbox(
    "Scegli documento",
    ["Tutti"] + doc_names
)

if selected_doc == "Tutti":
    selected_doc = None


# --- RESET DB ---
if st.sidebar.button("🗑️ Reset database"):
    reset_database()
    st.sidebar.success("Database cancellato!")
    st.rerun()

st.title("🤖 Chat con i tuoi PDF") #  (Ollama + ChromaDB)

st.write("🎙️ Usa il microfono per dettare la tua domanda.")
audio = mic_recorder(
    start_prompt="🎤 Inizia a parlare",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
    key="mic_header"
)

# --- Sidebar Upload PDF ---
st.sidebar.header("📄 Carica documento")

uploaded_files = st.sidebar.file_uploader("Carica PDF", type="pdf", accept_multiple_files=True)

# memoria sessione per evitare doppia indicizzazione
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if uploaded_files:
    new_files = []
    file_hash_map = {}

    for f in uploaded_files:
        file_hash = get_file_hash(f)

        if file_hash not in st.session_state.processed_files:
            new_files.append(f)
            file_hash_map[f.name] = file_hash

    if new_files:
        with st.spinner("Indicizzazione automatica..."):
            add_documents(new_files, file_hash_map)

        for f in new_files:
            st.session_state.processed_files.add(f.name)

        st.sidebar.success("Documenti indicizzati!")

# --- Temporary file upload in sidebar above radio ---
st.sidebar.header("📂 Carica file temporaneo")
if "temp_file" not in st.session_state:
    st.session_state.temp_file = None

temp_file = st.sidebar.file_uploader(
    "Carica PDF, TXT, CSV o Excel per lettura temporanea",
    type=["pdf", "txt", "md", "csv", "xls", "xlsx"],
    accept_multiple_files=False,
    key="temp_upload_left"
)

if temp_file:
    temp_text = load_file_text(temp_file)
    st.session_state.temp_file = {
        "name": temp_file.name,
        "content": temp_text,
        "type": temp_file.type,
    }
    st.sidebar.success(f"File temporaneo caricato: {temp_file.name}")
    file_content_limit = 300
    if temp_text:
        st.sidebar.write(temp_text[:file_content_limit] + ("..." if len(temp_text) > file_content_limit else ""))
    else:
        st.sidebar.warning("Impossibile leggere il file temporaneo.")

if st.session_state.get("temp_file"):
    if st.sidebar.button("🗑️ Rimuovi file temporaneo", key="remove_temp_left"):
        st.session_state.temp_file = None
        st.experimental_rerun()


# --- Controllo radio ---
radio_on = radio_manager.is_playing()
if radio_on:
    st.sidebar.header("🎧 Controllo Radio")
    st.sidebar.success("Radio in riproduzione")
    if radio_manager.current_audio:
        st.sidebar.write(f"Stazione attuale: **{radio_manager.current_audio}**")
    if st.sidebar.button("🔴 Ferma radio"):
        try:
            stopped = radio_manager.stop_radio()
            if stopped:
                st.sidebar.success("Radio spenta.")
            else:
                st.sidebar.info("Non c'era alcuna radio in riproduzione.")
        except RuntimeError as exc:
            st.sidebar.error(str(exc))

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra messaggi precedenti
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Input utente
query = st.chat_input("Fai una domanda sui documenti o usa il file uploader temporaneo qui sopra", key="chat_input")

if audio:
    with st.spinner("Trascrizione vocale..."):
        query = f"🎤 {transcribe_audio(audio['bytes'])}"

if query:
    st.chat_message("user").write(query)

    temp_file_context = st.session_state.get("temp_file", {})
    use_temp_context = st.session_state.get("use_temp_context", True)
    context_text = ""

    if temp_file_context:
        # Se l'utente parla esplicitamente del file, includo comunque il contesto.
        query_lower = query.lower()
        file_keywords = ["file", "caricato", "questo file", "file temporaneo", "contenuto del file", "nel file", "documento", "allegato"]
        if any(keyword in query_lower for keyword in file_keywords):
            use_temp_context = True
            st.info(f"Contesto temporaneo caricato: {temp_file_context['name']}")
        elif use_temp_context:
            st.info(f"Contesto temporaneo attivo: {temp_file_context['name']}")
        else:
            st.info(f"File temporaneo caricato ma non incluso in questa richiesta.")

        if use_temp_context:
            context_text = (
                f"File temporaneo: {temp_file_context['name']}\n"
                + temp_file_context.get("content", "")
            )

    # Prepariamo i messaggi per LangGraph
    # Includiamo i messaggi precedenti + la query attuale convertita in HumanMessage, distinguendoli tra messaggi scritti dall'utente con quelli scritti dal sistema
    history = convert_to_langchain_messages(st.session_state.messages)
    current_query_msg = HumanMessage(content=query)

    current_messages = list(st.session_state.messages)

    with st.expander("Ragionamenti in corso", expanded=False) as reasoning_expander:
        reasoning_panel = reasoning_expander.empty()
    reasoning_lines = ["🧠 Inizio elaborazione della richiesta..."]
    reasoning_panel.markdown(
        "<div style='background:rgba(0,0,0,0.05); padding:12px; border-radius:10px; color:#111;'>"
        + "<strong>Ragionamenti in corso:</strong><br>"
        + "<br>".join([f"- {line}" for line in reasoning_lines])
        + "</div>",
        unsafe_allow_html=True,
    )

    async def get_streaming_response():
        full_response = ""
        final_state = None

        def append_reasoning(line: str):
            reasoning_lines.append(line)
            reasoning_panel.markdown(
                "<div style='background:rgba(0,0,0,0.05); padding:12px; border-radius:10px; color:#111;'>"
                + "<strong>Ragionamenti in corso:</strong><br>"
                + "<br>".join([f"- {item}" for item in reasoning_lines])
                + "</div>",
                unsafe_allow_html=True,
            )

        with st.spinner("Sto pensando..."), st.chat_message("assistant"):
            # 1. Creiamo i segnaposto grafici DENTRO il messaggio dell'assistente
            mcp_log_placeholder = st.empty()
            mcp_progress_placeholder = st.empty()
            container = st.empty()

            # Usiamo astream_events per intercettare i singoli token e i nostri eventi custom
            async for event in app.astream_events({
                "query": query,
                "messages": history + [current_query_msg],
                "context": context_text,
                "selected_doc": selected_doc,
                "tool_result": {},
                "final_answer": ""
            }, version="v2"):

                #print("event: ", event["event"])
                #print(dir(event))
                #print(event)
                # ── 🤖 INTERCETTAZIONE EVENTI PERSONALIZZATI DA MCP ──
                if event["event"] == "on_custom_event":
                    # Se il server MCP ha inviato un log di testo
                    if event["name"] == "mcp_log":
                        log_text = event["data"].get("text", "")
                        mcp_log_placeholder.caption(f"⚙️ **Status Tool:** {log_text}")
                        append_reasoning(f"Tool: {log_text}")

                    # Se il server MCP ha inviato una percentuale di avanzamento
                    elif event["name"] == "mcp_progress":
                        pct = event["data"].get("percentage", 0)
                        mcp_progress_placeholder.progress(pct, text=f"Elaborazione Tool: {pct}%")
                        append_reasoning(f"Avanzamento tool: {pct}%")

                # ── 🔄 EVENTI DI CATENA ──
                if event["event"] == "on_chain_start":
                    name = event.get("name", "")
                    if name:
                        append_reasoning(f"Avvio nodo/chain: {name}")
                    else:
                        append_reasoning("Avvio elaborazione catena")

                if event["event"] == "on_chain_end":
                    name = event.get("name", "")
                    if name:
                        append_reasoning(f"Nodo/chain completato: {name}")
                    else:
                        append_reasoning("Elaborazione catena completata")
                    if event.get("name") == "llm":
                        final_state = event["data"].get("output") if isinstance(event["data"], dict) else None

                # ── ✍️ STREAMING DEL TESTO DELL'LLM ──
                if event["event"] == "on_chat_model_stream":
                    # Appena l'LLM comincia a rispondere, puliamo i widget dell'MCP per non sporcare la chat
                    mcp_log_placeholder.empty()
                    mcp_progress_placeholder.empty()

                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        container.markdown(full_response + "▌")  # Cursore effetto scrittura
                        if len(reasoning_lines) == 1:
                            append_reasoning("Generazione della risposta in corso...")

            # Pulizia finale di sicurezza per i widget e il testo
            mcp_log_placeholder.empty()
            mcp_progress_placeholder.empty()
            container.markdown(full_response)
            append_reasoning("Elaborazione completata.")
            print("final_state: ", final_state)

            # 🎁 SE ESISTE PDF → MOSTRA DOWNLOAD BUTTON
            if final_state and "pdf_path" in final_state:
                with open(final_state["pdf_path"], "rb") as file:
                    st.download_button(
                        label="📄 Scarica il report PDF",
                        data=file,
                        file_name="report.pdf",
                        mime="application/pdf"
                    )
        tts_file = await generate_tts(full_response)
        st.audio(tts_file)
        return full_response

    # Esegue il loop asincrono
    final_answer = asyncio.run(get_streaming_response())

    # Salviamo in session_state nel formato Streamlit per la visualizzazione al prossimo rerun
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": final_answer})


    # --- Aggiorna il pannello radio se lo stato è cambiato ---
    if radio_manager.is_playing() != radio_on:
        st.rerun()