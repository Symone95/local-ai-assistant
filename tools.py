import json
import ollama
import re
from rag_engine import get_files_with_upload_date_tool, get_files_in_db_tool, conversational_search_tool, \
    summarize_document_tool, extract_filename_from_query, generate_report_content
from utils.general import generate_pdf_report, extract_city_from_query, extract_path_with_llm
from dto.managers.radio_manager import radio_manager
from dto.managers.meteo_manager import meteo_manager
from dto.managers.terminal_manager import terminal_manager


TOOLS = [
    ## Strumenti per gestione documenti e RAG
    {
        "name": "search_documents",
        "description": "Cerca informazioni nei documenti caricati",
        "input": "query"
    },
    {
        "name": "list_documents",
        "description": "Restituisce la lista dei file presenti nel database",
        "input": "none"
    },
    {
        "name": "get_upload_dates",
        "description": "Restituisce quando sono stati caricati i documenti",
        "input": "none"
    },
    {
        "name": "summarize_document",
        "description": "Crea un riassunto completo di un documento",
        "input": "file_name"
    },
    {
        "name": "generate_pdf_report",
        "description": "Genera un report PDF professionale su qualsiasi argomento",
        "input": "title, content"
    },
    ## Strumenti MCP per Ansible e gestione infrastruttura
    {
        "name": "mcp_list_playbooks_tool",
        "description": "Lista playbook disponibili",
        "input": ""
    },
    {
        "name": "mcp_run_ansible_playbook",
        "description": "Esegue un playbook Ansible in locale o su inventory remoto",
        "input": "playbook_path, inventory_path"
    },
    {
        "name": "mcp_generate_ansible_playbook_tool",
        "description": "Genera un playbook Ansible YAML da una richiesta testuale",
        "input": "query"
    },
    {
        "name": "mcp_save_playbook_tool",
        "description": "Salva playbook ansible",
        "input": "name, content"
    },
    ## Strumento per gestione radio
    {
        "name": "radio_tool",
        "description": "Gestisce le stazioni radio e riproduce stazioni conosciute",
        "input": "query"
    },
    ## Strumento per gestione del meteo
    {
        "name": "meteo_tool",
        "description": "Fornisce informazioni sul meteo",
        "input": "query"
    },
    ## Strumento per esecuzione comandi terminale e gestione filesystem
    {
        "name": "terminal_tool",
        "description": "Legge, analizza, rifattorizza o mostra il contenuto di file e cartelle del progetto. Usalo quando l'utente menziona un file specifico (.py, .js, .ts, ecc.) o vuole analizzare/leggere/refactoring del codice.",
        "input": "query"
    }
]

# TODO: SPOSTARE TUTTO SU MCP
def execute_tool(tool_name: str, query: str = None, selected_doc=None, messages=None, context="", current_file=""):

    print("tool_name: ", tool_name)

    if tool_name == "search_documents":
        context_result, structured_sources = conversational_search_tool(query, messages, selected_doc, temp_context=context)
        return {
            "context": context_result,
            "sources": structured_sources
        }

    if tool_name == "list_documents":
        files = get_files_in_db_tool()
        return {"files": files}

    if tool_name == "get_upload_dates":
        dates = get_files_with_upload_date_tool()
        return {"dates": dates}

    if tool_name == "summarize_document":
        # l'agente deve aver passato il nome del file nella query
        file_name = extract_filename_from_query(query)
        return summarize_document_tool(file_name)

    if tool_name == "generate_pdf_report":
        print("✍️ Genero contenuto report con LLM...")
        report_markdown = generate_report_content(query, messages)

        print("📄 Creo PDF...")
        pdf_result = generate_pdf_report(title="AI Generated Report", content=report_markdown)

        return {
            "report_text": report_markdown,
            "pdf": pdf_result
        }
    
    if tool_name == "radio_tool":
        query_text = query or ""
        query_lower = query_text.lower()

        # Prova a trovare una stazione dalla domanda dell'utente
        stations = radio_manager.get_station_list()
        station = None

        stop_pattern = re.compile(r"\b(stop|ferma|spegni|spegnila|spengila|arresta|chiudi|disattiva|metti giù|muto)\b", re.IGNORECASE)
        if stop_pattern.search(query_lower):
            try:
                stopped = radio_manager.stop_radio()
                if stopped:
                    return {
                        "status": "stopped",
                        "message": "La radio è stata spenta.",
                        "station": radio_manager.current_audio
                    }
                return {
                    "status": "stopped",
                    "message": "Non c'era alcuna radio in riproduzione.",
                    "station": None
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        # Controllo URL diretto nella query
        url_match = re.search(r"https?://\S+", query_text)
        if url_match:
            station_url = url_match.group(0).strip()
            try:
                radio_manager.play_radio(station_url)
                return {
                    "status": "playing",
                    "station": station_url,
                    "message": f"Riproduco l'URL radio: {station_url}",
                    "stations": [s["nome"] for s in stations]
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        # Match del nome stazione
        for radio in sorted(stations, key=lambda x: -len(x["nome"])):
            nome_lower = radio["nome"].lower()
            if nome_lower in query_lower:
                station = radio
                break
            if nome_lower.replace("radio", "").strip() in query_lower:
                station = radio
                break

        if station:
            try:
                radio_manager.play_radio(station["url"])
                return {
                    "status": "playing",
                    "station": station["nome"],
                    "url": station["url"],
                    "message": f"Sto riproducendo {station['nome']}.",
                    "current_audio": station["nome"]
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        random_play_pattern = re.compile(r"\b(accendi|accendila|metti|mettila|apri|attiva|avvia|riproduci|ascolta)\b.*\bradio\b|\bradio\b.*\b(accendi|accendila|metti|mettila|apri|attiva|avvia|riproduci|ascolta)\b", re.IGNORECASE)
        if random_play_pattern.search(query_text) and not station:
            try:
                radio_manager.random_radio()
                return {
                    "status": "playing",
                    "message": "Accendo una radio casuale.",
                    "current_audio": radio_manager.current_audio
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        return {
            "status": "list",
            "message": "Non ho identificato una stazione precisa. Ecco le radio disponibili:",
            "stations": [s["nome"] for s in stations]
        }

    if tool_name == "meteo_tool":
        city = extract_city_from_query(query or "")
        if not city:
            return {
                "status": "error",
                "message": "Non ho riconosciuto una città nell'input. Per favore specifica una città per il meteo."
            }

        meteo_result = meteo_manager.current_weather(q=city)
        return {
            "status": "success",
            "city": city,
            "data": meteo_result
        }
    
    if tool_name == "terminal_tool":
        path = extract_path_with_llm(query or "", current_file=current_file)
        try:
            result = terminal_manager.list_and_read_files(path)
            return {"status": "success", "path": path, "content": result, "current_file": path}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


    return {"error": "Tool non trovato"}


def tool_planner(query, messages=None, context=""):
    
    if query:
        query_lower = query.lower()

        if re.search(r"\b(meteo|tempo|previsioni|pioggia|sole|neve|vento|temperatura|umidità)\b", query_lower):
            return json.dumps({"tool": "meteo_tool", "query": query})

        # Terminal/Filesystem requests: elenca o leggi file/cartelle
        if re.search(r"\b(lista|elenca|mostra|apri|leggi|leggere|leggerlo|visualizza)\b.*\b(file|cartella|cartelle|folder|dir|directory)\b", query_lower):
            return json.dumps({"tool": "terminal_tool", "query": query})

        # File con estensione esplicita: analisi, refactoring, lettura codice
        if re.search(r"\b\w[\w\-]*\.(py|js|ts|jsx|tsx|java|go|cpp|c|cs|rb|php|sh|yaml|yml|json|md)\b", query_lower):
            return json.dumps({"tool": "terminal_tool", "query": query})

        if re.search(r"\b(radio|stazione|stazioni|ascolta|ascoltiamo|ascoltare|riproduci|metti la radio|radio105|rtl 102\.5|rtl|deejay)\b", query_lower):
            return json.dumps({"tool": "radio_tool", "query": query})

        stop_pattern = re.compile(r"\b(stop|ferma|spegni|spegni la radio|spegni radio|spegnila|arresta|chiudi|disattiva|metti giù|muto)\b", re.IGNORECASE)
        prior_text = "".join(
            [msg.get("content", "").lower() if isinstance(msg, dict) else getattr(msg, "content", "").lower() for msg in (messages or [])]
        )
        if stop_pattern.search(query_lower) and (
            "radio" in query_lower
            or radio_manager.is_playing()
            or "radio" in prior_text
            or any(station["nome"].lower() in prior_text for station in radio_manager.get_station_list())
        ):
            return json.dumps({"tool": "radio_tool", "query": query})

    tools_description = "\n".join(
        [f"{t['name']}: {t['description']}" for t in TOOLS]
    )

    # Costruisci il contesto conversazionale se disponibile
    context_section = ""
    if messages:
        context_section = """
CONTESTO CONVERSAZIONALE PRECEDENTE:
"""
        # Prendi gli ultimi 2-3 messaggi per il contesto
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        for msg in recent_messages:
            # Gestisci sia dizionari che oggetti LangChain (HumanMessage, AIMessage, ecc.)
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                # Oggetti LangChain come HumanMessage, AIMessage
                role = getattr(msg, "type", "user")
                content = getattr(msg, "content", "")
            
            content = content[:200] if content else ""  # Limita per brevità
            context_section += f"- {role}: {content}\n"
    else:
        context_section = "Nessun contesto conversazionale precedente."

    file_section = ""
    if context:
        file_section = f"\nFILE TEMPORANEO CARICATO:\n{context[:3000]}\n"  # Limita la lunghezza del prompt
    else:
        file_section = "\nNessun file temporaneo caricato.\n"

    prompt = f"""
Sei un AI che decide quale tool usare. Analizza la DOMANDA ATTUALE considerando il contesto e il contenuto del file temporaneo, se presente.

{context_section}

{file_section}

Se è presente un file temporaneo caricato e la domanda riguarda quel file, usa preferibilmente lo strumento "search_documents" per interrogare il contenuto del file temporaneo.
Non rispondere direttamente con DLA quando la richiesta chiede di analizzare o spiegare il file caricato.
Parole chiave da considerare: "file caricato", "questo file", "file temporaneo", "file allegato", "nel file", "contenuto del file", "cosa fa il file", "descrivi il file", "cosa contiene il file", "analizza il file".

Tools disponibili:
{tools_description}

STRATEGIA DI DECISIONE:

1. VALUTA SE LA DOMANDA È CORRELATA AL CONTESTO:
   - Se la domanda continua/approfondisce l'argomento precedente → usa il contesto per interpretarla
   - Se la domanda è SU UN ARGOMENTO COMPLETAMENTE DIVERSO → ignora il contesto e analizza solo la nuova domanda
   - Indicatori di cambio di argomento: parole come "invece", "adesso", "passiamo a", "dimenticati di", o richieste su temi totalmente diversi

2. CRITERI DI DECISIONE (usa SOLO uno):

   a) ANSIBLE/INFRASTRUCTURE (→ tool MCP):
      - Keyword: "playbook", "ansible", "server", "installazione", "configurazione", "deploy", "infra", "docker", "DevOps"
      - Usa: mcp_generate_ansible_playbook_tool, mcp_list_playbooks_tool, mcp_run_ansible_playbook, mcp_save_playbook_tool

   b) DOCUMENT SEARCH (→ search_documents):
      - Keyword: "cerca", "trova", "ricerca", "dimmi", "informazioni", "cos'è", "spiegami", "trovo in", "cerca nel", "quali documenti", "parlano di"
      - Anche pronomi come "questo", "quello" che si riferiscono al contesto precedente
      - Quando: l'utente chiede informazioni contenute nei documenti

   c) DOCUMENT LISTING (→ list_documents):
      - Keyword: "quali documenti", "elenco", "lista", "quanti file", "cosa c'è", "quali file"
      - Quando: chiede quali documenti sono disponibili nel database

   d) DOCUMENT DATES (→ get_upload_dates):
      - Keyword: "quando", "data", "caricato", "upload", "timestamp"
      - Quando: chiede date di caricamento

   e) DOCUMENT SUMMARY (→ summarize_document):
      - Keyword: "riassunto", "riassunti", "summary", "summarize", "resume di"
      - Usare SOLO se ESPLICITAMENTE richiesto

   f) PDF REPORT GENERATION (→ generate_pdf_report):
      - Keyword: "genera report", "report pdf", "crea report", "genera documento", "report su"
      - Quando: chiede un report PDF

   g) RADIO (→ radio_tool):
      - Keyword: "radio", "stazione", "ascolta", "ascoltare", "metti la radio", "riproduci radio", "Radio105", "RTL 102.5"
      - Quando: l'utente vuole ascoltare una radio o gestire le stazioni radio
    
   h) METEO (→ meteo_tool):
    - Keyword: "meteo", "tempo", "previsioni", "pioggia", "sole", "neve"
    - Quando: l'utente vuole conoscere le informazioni meteorologiche

   i) FILESYSTEM / CODICE (→ terminal_tool):
      - Keyword: nomi di file con estensione (.py, .js, .ts, .yaml, ecc.), "mostrami il codice", "leggi il file", "analizza il file", "rifattorizza", "refactor", "cosa fa", "spiega questo file", "apri", "visualizza"
      - Quando: l'utente menziona un file specifico del progetto o vuole leggerne/analizzarne il codice

   j) NESSUN TOOL (→ "none"):
      - Conversazioni, saluti, domande generiche
      - Domande che richiedono di descrivere, spiegare, riassumere o analizzare il file temporaneo caricato

REGOLE IMPORTANTI:
- Se è presente un file temporaneo e la domanda parla esplicitamente del file caricato, preferisci "none" e rispondi direttamente usando il contenuto del file
- Se la domanda è AMBIGUA o correlata al contesto → preferisci "search_documents"
- Se la domanda è su argomento diverso → decidi indipendentemente dal contesto
- Rispondi SOLO in JSON valido, senza spiegazioni extra

OUTPUT JSON:
- Se serve tool: {{"tool": "nome_tool", "query": "query_da_passare", "params": {{}}}}
- Se nessun tool: {{"tool": "none"}}

Domanda attuale: {query}
"""

    response = ollama.chat(
        model="qwen2.5-coder:3b", #"llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
