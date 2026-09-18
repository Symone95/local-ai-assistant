from mcp.server.fastmcp import FastMCP, Context
from ansible_tools import *
import logging
import os
import sys
import traceback
import subprocess

logger = logging.getLogger("mcp_server")

mcp = FastMCP("DevOps-Ansible")

@mcp.tool(name="generate_ansible_playbook_tool", description="Genera un playbook Ansible YAML da una richiesta testuale")
async def generate_ansible_playbook_tool(query: str, *, context: Context):
    """Genera un playbook Ansible YAML da una richiesta testuale"""
    try:
        logger.info(f"Ricevuta richiesta di generazione per: {query}")

        # Notifica iniziale
        await context.info("Inizializzazione del modello Ollama e preparazione del prompt...")
        await context.report_progress(10, 100)

        # Chiamiamo la funzione asincrona passando anche il 'context' MCP per i progressi
        risultato = await generate_ansible_playbook_async(query, context)

        await context.info("Playbook generato con successo!")
        await context.report_progress(100, 100)
        return risultato

    except Exception as e:
        sys.stderr.write(f"❌ ERRORE NEL TOOL: {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        return f"Errore interno al server MCP: {str(e)}"


async def generate_ansible_playbook_tool_old(query: str, *, context: Context):
    """Genera un playbook Ansible YAML da una richiesta testuale"""
    try:
        await context.info("Sto facendo la ricerca")
        await context.report_progress(20, 100)
        logger.debug("ESEGUO: mcp_generate_ansible_playbook")
        return generate_ansible_playbook(query)
    except Exception as e:
        sys.stderr.write(f"❌ ERRORE NEL TOOL: {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        return f"Errore interno al server MCP: {str(e)}"

@mcp.tool()
def run_playbook_tool(name: str):
    """Esegue un playbook ansible"""
    return run_playbook(name)

@mcp.tool(name="save_playbook_tool", description="Salva playbook ansible")
def save_playbook_tool(name: str, content: str):
    """Salva playbook ansible"""
    return save_playbook(name, content)

@mcp.tool(name="list_playbooks_tool", description="Lista playbook disponibili")
async def list_playbooks_tool(*, context: Context):
    """Lista playbook disponibili"""
    logger.info("CHIAMATO")
    await context.info("Sto facendo la ricerca")
    await context.report_progress(20, 100)

    await context.info("Quasi terminato la ricerca")
    await context.report_progress(70, 100)
    risultato = list_playbooks()

    ris = {"msg": f"Ricerca completata. Trovati {len(risultato)} elementi.",
           "items": risultato}
    logger.info(f"Ricerca completata.", ris)
    return ris


# bash_tool esegue comandi arbitrari sulla macchina che ospita il server: e' registrato
# solo se lo si abilita esplicitamente con MCP_ENABLE_BASH=1.
if os.getenv("MCP_ENABLE_BASH") == "1":
    @mcp.tool()
    def bash_tool(command: str):
        """Execute bash commands"""
        return run_bash(command)

    logger.warning("bash_tool ABILITATO: il server accetta comandi bash arbitrari.")


def run_bash(cmd: str):
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    return result.stdout + result.stderr

if __name__ == "__main__":
    logger.info("RUN")
    mcp.run()