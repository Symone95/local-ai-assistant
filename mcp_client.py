from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage

from dto.agent_state import AgentState

SERVER_PATH = "../ai_mcp/server.py"
_stdio_cm = None
_session_cm = None

@asynccontextmanager
async def mcp_session():
    # Passa i parametri correttamente
    server_params = StdioServerParameters(command="python", args=[SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def mcp_tool_node(state):
    print("CHIAMO MCP TOOL")
    original_tool_name = state["tool_plan"]["tool"]
    tool_name = original_tool_name.replace("mcp_", "")
    args = state["tool_plan"].get("args", {})

    if args:
        # Se l'LLM ha chiamato il parametro 'input' o 'prompt', rinominalo in 'query'
        if "input" in args:
            args["query"] = args.pop("input")
        elif "prompt" in args:
            args["query"] = args.pop("prompt")
    else:
        # Se args è vuoto {}, Pydantic fallisce. Mettiamo un valore di fallback
        args = {"query": state.get("query", "Genera playbook")}

    # Log critico per debuggare nel terminale di Streamlit
    print(f"DEBUG: Cerco di invocare {tool_name} sul server...")

    try:
        async with mcp_session() as session:

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
