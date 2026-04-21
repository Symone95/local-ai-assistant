import ollama
from rag_engine import get_files_with_upload_date, get_files_in_db, conversational_search

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
    }
]

def execute_tool(tool_name: str, query: str = None, selected_doc=None, messages=None):
    print("tool_name", tool_name)

    if tool_name == "search_documents":
        context, structured_sources = conversational_search(query, messages, selected_doc)
        return {
            "context": context,
            "sources": structured_sources
        }

    if tool_name == "list_documents":
        files = get_files_in_db()
        return {"files": files}

    if tool_name == "get_upload_dates":
        dates = get_files_with_upload_date()
        return {"dates": dates}

    return {"error": "Tool non trovato"}


def tool_planner(query):
    tools_description = "\n".join(
        [f"{t['name']}: {t['description']}" for t in TOOLS]
    )

    prompt = f"""
Sei un AI che decide quale tool usare.

Tools disponibili:
{tools_description}

Regole:
- Rispondi SOLO in JSON
- Se serve un tool → {{ "tool": "nome_tool", "query": "..." }}
- Se NON serve tool → {{ "tool": "none" }}

Domanda: {query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
