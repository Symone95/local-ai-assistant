import streamlit as st

from mcp_integration.client import mcp_tool_node
from rag_engine import add_documents, reset_database, collection, get_file_hash
from tools import image_analyser_stream
from utils.general import clean_code_content, clean_post_content, get_db_stats, convert_to_langchain_messages, load_file_text, extract_code_block
import os
import asyncio
import json
import uuid
from nodes import *
from agent_graph import build_graph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from dto.managers.radio_manager import radio_manager
from utils.text_to_speech import generate_tts
from voice.agente import AgenteVocale
from voice.engine import ottieni_motore

# TODO: FAR GENERARE DOCUMENTI, FAR RISPONDERE IN MARKDOWN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# LangGraph in una frase
# È una macchina a stati (state machine) dove:
# - ogni nodo = una funzione Python
# - lo stato = un dizionario che viaggia tra i nodi (AgentState)
# - gli edge = decidono chi viene eseguito dopo(le relazioni tra i nodi)

graph = build_graph()


@st.cache_resource
def compile_app():
    """
    Il checkpointer serve all'human-in-the-loop: conserva lo stato del grafo mentre l'utente
    decide se approvare. Streamlit rilancia lo script a ogni interazione, quindi la compilazione
    va messa in cache: altrimenti ogni click creerebbe un checkpointer nuovo e vuoto.
    """
    return graph.compile(checkpointer=InMemorySaver())


app = compile_app()

## FE
st.set_page_config(page_title="Local AI Assistant", layout="wide")  # Titolo del tab

# --- 🎙️ Modalità vocale continua ---------------------------------------------------------
# Il motore gira in un thread suo (vedi voice/engine.py) e non tocca Streamlit: la UI si limita
# a leggerne lo stato. Va in cache perché Streamlit rilancia lo script a ogni interazione e il
# thread non deve ripartire da zero ogni volta.
@st.cache_resource
def get_voice_engine():
    # ottieni_motore() tiene l'istanza a livello di modulo: anche se Streamlit invalidasse
    # questa cache, non nascerebbe un secondo motore che parla sopra il primo.
    agente = AgenteVocale()
    return ottieni_motore(agente.rispondi)


voice_engine = get_voice_engine()


@st.fragment(run_every=0.5)
def pannello_vocale():
    """Si aggiorna da solo mezzo secondo alla volta, senza rieseguire tutta la pagina."""
    stato = voice_engine.snapshot()
    if not stato.attivo:
        return

    icone = {"preparo": "⏳", "in ascolto": "🎙️", "trascrivo": "🔤", "penso": "🤔", "parlo": "🗣️"}
    with st.container(border=True):
        st.markdown(f"### {icone.get(stato.fase, '•')} {stato.fase.capitalize()}")
        if stato.fase == "preparo":
            st.caption("Carico i modelli e misuro il rumore di fondo: aspetta il microfono prima di parlare.")

        # Livello del microfono: rende visibile se l'audio arriva e se supera la soglia.
        if stato.fase == "in ascolto":
            if stato.microfono_muto:
                st.error(
                    "🔇 Dal microfono non arriva alcun segnale. Controlla che il terminale da cui "
                    "hai avviato l'app abbia il permesso di usare il microfono "
                    "(Impostazioni di Sistema → Privacy e sicurezza → Microfono) e che sia "
                    "selezionato il dispositivo giusto."
                )
            else:
                scala = max(stato.soglia * 3, 0.02)
                st.progress(min(stato.livello / scala, 1.0))
                stato_voce = "parlato rilevato" if stato.livello > stato.soglia else "silenzio"
                st.caption(
                    f"Livello microfono {stato.livello:.4f} · soglia {stato.soglia:.4f} → {stato_voce}. "
                    "Se parli e il livello resta sotto soglia, abbassa VOICE_VAD_FACTOR nel .env."
                )
        if stato.trascrizione:
            st.markdown(f"**Tu:** {stato.trascrizione}")
        if stato.risposta:
            st.markdown(f"**Assistente:** {stato.risposta}")
        if stato.errore:
            st.error(stato.errore)
        if stato.turni:
            with st.expander(f"Turni precedenti ({len(stato.turni)})"):
                for domanda, risposta in reversed(stato.turni[:-1] or ()):
                    st.markdown(f"- **Tu:** {domanda}\n- **Assistente:** {risposta}")


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

st.title("🤖 Local AI Assistant") #  (Ollama + ChromaDB + LangGraph)

# Un solo comando per la voce: premere il microfono avvia la conversazione (non la chiude,
# come faceva il vecchio push-to-talk, che trascriveva solo allo stop).
stato_vocale = voice_engine.snapshot()

if stato_vocale.attivo:
    if st.button("⏹️ Chiudi la conversazione vocale", use_container_width=True):
        voice_engine.stop()
        st.rerun()
else:
    if st.button("🎤 Parla con l'assistente", use_container_width=True, type="primary"):
        voice_engine.start()
        st.rerun()
    st.caption(
        "Premi e parla liberamente: non serve tenere premuto nulla. Il turno si chiude da solo "
        "dopo un attimo di silenzio e l'assistente risponde a voce. Oppure scrivi qui sotto."
    )

pannello_vocale()

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


# -- Upload immagini da elaborare
st.sidebar.header("👁️ Analizzatore di immagini")

# Widget per il caricamento dell'immagine
uploaded_image = st.sidebar.file_uploader("Carica un'immagine (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Mostra l'immagine caricata sulla UI
    # st.image(uploaded_image, caption="Immagine caricata", use_container_width=True)
    
    # Crea un input di testo per fare una domanda specifica sull'immagine
    domanda = st.text_input("Cosa vuoi sapere su questa immagine?", "Descrivi cosa vedi in questa immagine.")
    
    if st.button("Analizza"):
        mcp_log_placeholder = st.empty()
        with st.spinner("Sto guardando l'immagine..."):
            
            # 💡 Salva temporaneamente il file su disco perché Ollama legge i percorsi file
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_image.name)
            mcp_log_placeholder.caption(f"⚙️ **Image Tool:** Immagine caricata: {uploaded_image.name}")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            try:
                # Chiamata a Ollama con il modello multimodale
                mcp_log_placeholder.caption(f"⚙️ **Analizzo immagine**")
                st.subheader("🤖 Analisi dell'assistente:")
                st.write_stream(image_analyser_stream(temp_path, domanda, mcp_log_placeholder))
                mcp_log_placeholder.caption(f"⚙️ **Immagine analizzata**")
                
            except Exception as e:
                st.error(f"Errore: {e}")
                
            finally:
                # Pulizia: rimuove il file temporaneo
                if os.path.exists(temp_path):
                    os.remove(temp_path)


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

if "current_file" not in st.session_state:
    st.session_state.current_file = ""

# Mostra messaggi precedenti
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Identifica la conversazione per il checkpointer: serve a riprendere il grafo dopo l'approvazione
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}


# --- Human-in-the-loop: richiesta di approvazione in sospeso ---
# Il grafo si e' fermato dentro mcp_tool prima di eseguire un'azione che tocca l'esterno.
# La seconda condizione evita di ridisegnare il pannello nel rerun in cui la decisione e' gia' presa
richiesta = st.session_state.get("approvazione_richiesta")
if richiesta and "hitl_decisione" not in st.session_state:
    with st.chat_message("assistant"):
        st.warning(f"🛑 **Serve il tuo via libera.** {richiesta.get('descrizione', '')}")
        st.caption(f"Tool: `{richiesta.get('tool')}` — argomenti esatti che verranno usati:")
        st.code(json.dumps(richiesta.get("args", {}), indent=2, ensure_ascii=False), language="json")

        col_ok, col_no = st.columns(2)
        if col_ok.button("✅ Esegui", key="hitl_ok", type="primary"):
            # Si rimandano indietro gli stessi argomenti mostrati: cio' che l'utente approva
            # e' esattamente cio' che verra' eseguito.
            st.session_state.hitl_decisione = {"approvato": True, "args": richiesta.get("args", {})}
            st.rerun()
        if col_no.button("❌ Annulla", key="hitl_no"):
            st.session_state.hitl_decisione = {"approvato": False}
            st.rerun()


# Input utente
query = st.chat_input("Fai una domanda sui documenti o usa il file uploader temporaneo qui sopra", key="chat_input")

# Una decisione presa sul pannello di approvazione riprende il grafo invece di iniziare un giro nuovo
decisione_hitl = st.session_state.pop("hitl_decisione", None)
in_ripresa = decisione_hitl is not None
if in_ripresa:
    query = st.session_state.get("query_in_sospeso", "")
    st.session_state.pop("approvazione_richiesta", None)

if query or in_ripresa:
    if not in_ripresa:
        # Ogni nuova domanda gira su un thread pulito: lo stato `messages` usa il reducer
        # add_messages, quindi riusare il thread accumulerebbe la storia a ogni turno.
        # Se c'era un'approvazione in sospeso e l'utente ha chiesto altro, viene abbandonata.
        st.session_state.pop("approvazione_richiesta", None)
        st.session_state.pop("query_in_sospeso", None)
        st.session_state.thread_id = str(uuid.uuid4())
        graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}

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

    async def get_streaming_response(graph_input):
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
            header_container = st.empty()   # testo prima del code block
            code_container = st.empty()     # codice con numerazione righe
            footer_container = st.empty()   # testo dopo il code block (fonti, ecc.)
            container = st.empty()          # fallback per risposte testuali normali
            code_detected = False

            # Usiamo astream_events per intercettare i singoli token e i nostri eventi custom
            async for event in app.astream_events(graph_input, version="v2", config=graph_config):

                # print("event: ", event["event"])
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
                    if event.get("name") == "tool":
                        output = event["data"].get("output") if isinstance(event["data"], dict) else None
                        if output and output.get("current_file"):
                            st.session_state.current_file = output["current_file"]
                    if event.get("name") == "llm":
                        final_state = event["data"].get("output") if isinstance(event["data"], dict) else None

                # ── ✍️ STREAMING DEL TESTO DELL'LLM ──
                if event["event"] == "on_chat_model_stream":
                    # Solo i nodi che producono la risposta finale: gli altri usano l'LLM per
                    # lavoro interno (estrazione citta', riscrittura query, reranking) e i loro
                    # token non devono finire nella chat.
                    if event["metadata"].get("langgraph_node") not in ("llm", "direct_llm_answer"):
                        continue

                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content

                        current_file = st.session_state.get("current_file", "")
                        if any(keyword in full_response.lower() for keyword in ["file", "contenuto"]) or current_file:
                            try:
                                pre, lang, code, post, _ = extract_code_block(full_response)

                                code_detected = True
                                mcp_log_placeholder.caption(f"📄 Sto leggendo il file: {current_file}")
                                header_container.markdown(pre.strip())
                                code_container.code(clean_code_content(code or "") + "▌", language=lang, line_numbers=True)
                                
                                cleaned_post = clean_post_content(post)
                                footer_container.markdown(cleaned_post.strip())
                                
                                container.empty()
                            except Exception as e:
                                print(f"Errore rendering codice: {e}")
                                code_detected = False
                                container.markdown(full_response + "▌")
                        else:
                            # Streaming normale per messaggi testuali
                            container.markdown(full_response + "▌")

                        if len(reasoning_lines) == 1:
                            append_reasoning("Generazione della risposta in corso...")

            # Rendering finale: manteniamo il code block se rilevato, altrimenti markdown
            if code_detected:
                pre, lang, code, post, _ = extract_code_block(full_response)
                header_container.markdown(pre.strip())
                code_container.code(clean_code_content(code or ""), language=lang, line_numbers=True)
                cleaned_post = clean_post_content(post)
                footer_container.markdown(cleaned_post.strip())
                container.empty()
            else:
                container.markdown(full_response)

            # Pulizia finale dei widget temporanei
            mcp_log_placeholder.empty()
            mcp_progress_placeholder.empty()

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
        if full_response.strip():
            tts_file = await generate_tts(full_response)
            st.audio(tts_file)
        return full_response

    # Ripresa dopo una decisione dell'utente, oppure giro nuovo a partire dalla domanda
    if in_ripresa:
        graph_input = Command(resume=decisione_hitl)
    else:
        graph_input = {
            "query": query,
            "messages": history + [current_query_msg],
            "context": context_text,
            "selected_doc": selected_doc,
            "tool_result": {},
            "final_answer": ""
        }

    # Esegue il loop asincrono
    final_answer = asyncio.run(get_streaming_response(graph_input))

    # Il grafo si e' fermato su un interrupt? Allora c'e' un'approvazione da chiedere.
    snapshot = app.get_state(graph_config)
    interruzioni = [i for task in snapshot.tasks for i in task.interrupts]

    if interruzioni:
        st.session_state.approvazione_richiesta = interruzioni[0].value
        st.session_state.query_in_sospeso = query
        if not in_ripresa:
            st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

    # Salviamo in session_state nel formato Streamlit per la visualizzazione al prossimo rerun
    st.session_state.pop("query_in_sospeso", None)
    if not in_ripresa:
        st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": final_answer})


    # --- Aggiorna il pannello radio se lo stato è cambiato ---
    if radio_manager.is_playing() != radio_on:
        st.rerun()