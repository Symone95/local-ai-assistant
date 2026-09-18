import os
import re
import sys
from pathlib import Path

import ollama

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage

from dto.agent_state import AgentState
from utils.general import extract_code_block, clean_code_content, loads_json_loose

# Path assoluti: cosi' l'MCP funziona anche se Streamlit viene lanciato da un'altra directory.
# Entrambi sono sovrascrivibili da .env per puntare a un server MCP diverso.
SERVER_DIR = Path(os.getenv("MCP_SERVER_DIR", Path(__file__).parent / "servers" / "devops_ansible")).resolve()
SERVER_PATH = str(SERVER_DIR / "server.py")
# Di default il server gira con l'interprete di questa app; se ha un venv suo, indicalo in MCP_PYTHON.
SERVER_PYTHON = os.getenv("MCP_PYTHON", sys.executable)

_stdio_cm = None
_session_cm = None

@asynccontextmanager
async def mcp_session():
    # cwd=SERVER_DIR: il server usa path relativi (ansible_memory/, inventory.ini),
    # che altrimenti si risolverebbero nella directory di questa app.
    server_params = StdioServerParameters(
        command=SERVER_PYTHON,
        args=[SERVER_PATH],
        cwd=str(SERVER_DIR),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# Nomi di parametro che si accontentano della domanda dell'utente cosi' com'e'...
FREE_TEXT_FIELDS = {"query", "input", "prompt", "request", "task", "description"}
# ...e nomi che vogliono invece un blocco di codice/contenuto prodotto in precedenza.
CONTENT_FIELDS = {"content", "body", "code", "playbook", "yaml", "text", "file_content"}


def _last_code_block(messages) -> str:
    """Ultimo blocco di codice presente nelle risposte dell'assistente, dal piu' recente."""
    for msg in reversed(messages or []):
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if not content:
            continue
        _, _, code, _, _ = extract_code_block(content)
        if code and code.strip():
            return clean_code_content(code).strip()
    return ""


def _sanitize_name(value: str) -> str:
    """
    Normalizza un identificatore proposto dall'LLM: niente estensione, niente path.
    Il server compone `ansible_memory/<name>.yml`, quindi uno slash o un '..' scriverebbe altrove.
    """
    name = str(value).strip().strip("'\"")
    name = re.sub(r"\.(ya?ml)$", "", name, flags=re.IGNORECASE)
    name = os.path.basename(name).replace("..", "")
    name = re.sub(r"[^\w\-]+", "_", name).strip("_")
    return name


def _extract_fields_with_llm(fields, schema, tool_name, query, messages) -> dict:
    """
    Chiede al modello del router di ricavare i campi mancanti dalla richiesta e dalla conversazione.
    Restituisce solo i campi che ha saputo valorizzare.
    """
    props = schema.get("properties", {})
    campi = "\n".join(
        f'- {f}: {props.get(f, {}).get("description") or props.get(f, {}).get("type", "string")}'
        for f in fields
    )

    storia = ""
    for msg in (messages or [])[-3:]:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        ruolo = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "user")
        storia += f"- {ruolo}: {str(content)[:400]}\n"

    prompt = f"""Estrai i parametri per il tool "{tool_name}" dalla richiesta dell'utente.

CONVERSAZIONE RECENTE:
{storia or "(nessuna)"}

RICHIESTA ATTUALE: {query}

PARAMETRI DA RICAVARE:
{campi}

Regole:
- Rispondi SOLO con un oggetto JSON contenente esattamente queste chiavi: {", ".join(fields)}
- Per un nome di file usa un identificatore breve, senza estensione e senza percorsi
- Se un parametro non e' ricavabile, usa la stringa vuota
"""

    try:
        response = ollama.chat(model="qwen2.5-coder:3b", messages=[{"role": "user", "content": prompt}])
        estratti = loads_json_loose(response["message"]["content"]) or {}
    except Exception as e:
        print(f"⚠️  Estrazione argomenti fallita: {e}")
        return {}

    return {f: estratti[f] for f in fields if estratti.get(f)}


async def resolve_tool_args(session, tool_name, state, args) -> tuple:
    """
    Completa gli argomenti richiesti dal tool leggendo il suo inputSchema.

    Il router produce quasi sempre solo la query testuale: i tool con parametri strutturati
    (es. name, content) fallirebbero la validazione lato server. Qui i campi mancanti vengono
    ricavati, nell'ordine, da: argomenti gia' presenti, query dell'utente, blocchi di codice
    della conversazione, estrazione con l'LLM.

    Ritorna (args, mancanti): se `mancanti` non e' vuoto la chiamata non va tentata.
    """
    query = state.get("query", "")
    messages = state.get("messages", [])

    tools = await session.list_tools()
    schema = next((t.inputSchema or {} for t in tools.tools if t.name == tool_name), {})
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    if not props:  # tool senza parametri: non passare nulla
        return {}, []

    # Tiene solo gli argomenti che il tool conosce davvero
    risolti = {k: v for k, v in (args or {}).items() if k in props and v not in (None, "")}

    da_ricavare = []
    for campo in required:
        if campo in risolti:
            continue
        if campo in FREE_TEXT_FIELDS:
            risolti[campo] = query
        elif campo in CONTENT_FIELDS:
            blocco = _last_code_block(messages)
            if blocco:
                risolti[campo] = blocco
            else:
                da_ricavare.append(campo)
        else:
            da_ricavare.append(campo)

    if da_ricavare:
        print(f"DEBUG: ricavo {da_ricavare} dalla conversazione...")
        risolti.update(_extract_fields_with_llm(da_ricavare, schema, tool_name, query, messages))

    if "name" in risolti:
        risolti["name"] = _sanitize_name(risolti["name"])

    mancanti = [c for c in required if not str(risolti.get(c, "")).strip()]
    return risolti, mancanti


async def mcp_tool_node(state):
    print("CHIAMO MCP TOOL")
    original_tool_name = state["tool_plan"]["tool"]
    tool_name = original_tool_name.replace("mcp_", "")
    args = dict(state["tool_plan"].get("args", {}) or {})

    # Alias comuni prodotti dall'LLM
    for alias in ("input", "prompt"):
        if alias in args:
            args["query"] = args.pop(alias)

    # Log critico per debuggare nel terminale di Streamlit
    print(f"DEBUG: Cerco di invocare {tool_name} sul server...")

    try:
        async with mcp_session() as session:

            args, mancanti = await resolve_tool_args(session, tool_name, state, args)
            if mancanti:
                messaggio = (
                    f"Per usare {original_tool_name} mi manca: {', '.join(mancanti)}. "
                    "Puoi indicarmelo esplicitamente nella richiesta?"
                )
                print(f"⚠️  {messaggio}")
                return {"final_answer": messaggio, "tool_result": messaggio}

            print(f"DEBUG: argomenti risolti -> { {k: str(v)[:60] for k, v in args.items()} }")

            # Esegui la chiamata al tool
            result = await session.call_tool(tool_name, args)

            # Estrazione sicura del testo
            if hasattr(result, "content"):
                output = "".join([c.text for c in result.content if hasattr(c, 'text')])
            else:
                output = str(result)

            print(f"✅ Risultato MCP ricevuto: {output[:350]}...")

            return {
                "final_answer": output,
                "tool_result": output,
                # Fondamentale: aggiungi un messaggio altrimenti il grafo non sa cosa è successo
                "messages": [HumanMessage(content=output, name=original_tool_name)]
            }
    except Exception as e:
        print(f"❌ ERRORE MCP NODE: {str(e)}")
        return {"final_answer": f"Errore nel tool MCP: {str(e)}"}
