import streamlit as st
from rag_engine import add_documents, reset_database, collection, get_file_hash
from utils.general import get_db_stats
import os
import asyncio
from nodes import *
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from streamlit_mic_recorder import mic_recorder
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

def decide_path(state: AgentState):
    """
    Funzione per decidere quale sarà la funzione chiamata successivamente al nodo eseguito
    """
    if state["tool_plan"]["tool"].strip() == "none":
        return "direct_llm_answer"

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
        "direct_llm_answer": "direct_llm_answer"
    }
)

# Qui registro i vari tipi di edge, a differenza del flusso senza LangGraph sto dicendo quando finisci solo con "tool" vai sempre al nodo "llm" mentre se va
# al nodo "direct_llm_answer" finisce direttamente
graph.add_edge("tool", "llm")
graph.add_edge("llm", END)  # Dopo aver eseguito il nodo "llm" finisce il flusso chiamando END di LangGraph che significa: “Il workflow finisce qui”
graph.add_edge("direct_llm_answer", END)

# Diagramma aggiornato:
#           router
#          /     \
#       tool     direct_llm_answer
#         |            |
#        llm          END
#         |
#        END

app = graph.compile()

## FE
st.set_page_config(page_title="Local RAG Chat", layout="wide")  # Titolo del tab

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

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra messaggi precedenti
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def convert_to_langchain_messages(streamlit_messages):
    lc_messages = []
    for m in streamlit_messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages


# Input utente
query = st.chat_input("Fai una domanda sui documenti")
# Input utente vocale
audio = mic_recorder(
    start_prompt="🎤 Inizia a parlare",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
    key="mic_footer"
)

if audio:
    with st.spinner("Trascrizione vocale..."):
        query = f"🎤 {transcribe_audio(audio['bytes'])}"

if query:
    st.chat_message("user").write(query)

    # Prepariamo i messaggi per LangGraph
    # Includiamo i messaggi precedenti + la query attuale convertita in HumanMessage, distinguendoli tra messaggi scritti dall'utente con quelli scritti dal sistema
    history = convert_to_langchain_messages(st.session_state.messages)
    current_query_msg = HumanMessage(content=query)

    current_messages = list(st.session_state.messages)

    async def get_streaming_response():
        full_response = ""
        final_state = None
        with st.spinner("Sto pensando..."), st.chat_message("assistant"):
            container = st.empty()

            # Usiamo astream_events per intercettare i singoli token
            async for event in app.astream_events({
                "query": query,
                "messages": history + [current_query_msg],
                "context": "",
                "selected_doc": selected_doc,
                "tool_result": {},
                "final_answer": ""
            }, version="v2"):

                # Cerchiamo l'evento di streaming del modello chat
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        container.markdown(full_response + "▌")  # Cursore effetto scrittura

                # 🧠 STATO FINALE DEL GRAFO
                if event["event"] == "on_chain_end" and event["name"] == "llm":
                    final_state = event["data"]["output"]

            container.markdown(full_response)  # Pulizia finale senza cursore
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

