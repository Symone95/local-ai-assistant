import ollama
from rag_engine import get_files_with_upload_date_tool, get_files_in_db_tool, conversational_search_tool, \
    summarize_document_tool, extract_filename_from_query, generate_report_content
from utils.general import generate_pdf_report

TOOLS = [
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
    }
]

# TODO: SPOSTARE TUTTO SU MCP
def execute_tool(tool_name: str, query: str = None, selected_doc=None, messages=None):

    print("tool_name", tool_name)

    if tool_name == "search_documents":
        context, structured_sources = conversational_search_tool(query, messages, selected_doc)
        return {
            "context": context,
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

    return {"error": "Tool non trovato"}


def tool_planner(query, messages=None):
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

    prompt = f"""
Sei un AI che decide quale tool usare. Analizza la DOMANDA ATTUALE considerando il contesto.

{context_section}

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

   g) NESSUN TOOL (→ "none"):
      - Conversazioni, saluti, domande generiche

REGOLE IMPORTANTI:
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
