import subprocess
import os
import ollama
from ollama import AsyncClient
from mcp.server.fastmcp import Context

from server import logger

ANSIBLE_DIR = "ansible_memory"


def run_playbook(playbook_name: str):
    path = f"{ANSIBLE_DIR}/{playbook_name}.yml"

    result = subprocess.run(
        ["ansible-playbook", "-i", "inventory.ini", path],
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def get_ansible_playbooks():
    import os

    if not os.path.exists("ansible_memory"):
        return []

    return os.listdir("ansible_memory")


def save_playbook(name: str, content: str):
    os.makedirs(ANSIBLE_DIR, exist_ok=True)
    path = f"{ANSIBLE_DIR}/{name}.yml"

    with open(path, "w") as f:
        f.write(content)

    return {"saved": path}


def list_playbooks():
    if not os.path.exists(ANSIBLE_DIR):
        return []
    return os.listdir(ANSIBLE_DIR)


async def generate_ansible_playbook_async(query: str, context: Context):
    prompt = f"""
Sei un esperto DevOps.

Genera un playbook Ansible valido.

Regole:
- SOLO YAML
- Niente spiegazioni
- Compatibile Ansible 2.14+
- Usa hosts: local se non specificato

Task:
{query}
"""

    await context.info("Avvio la generazione del codice con Llama3 (Streaming attivo)...")
    await context.report_progress(30, 100)

    # Inizializziamo il client asincrono di Ollama
    client = AsyncClient()

    # Avviamo la chat in modalità STREAMING
    response_stream = await client.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    full_content = ""
    chunks_count = 0

    # Leggiamo i blocchi di testo mentre Ollama li genera
    async for chunk in response_stream:
        text_piece = chunk["message"]["content"]
        full_content += text_piece
        chunks_count += 1

        # Ogni 15 blocchi di testo (token), aggiorniamo il progresso simulando l'avanzamento (fino a max 90%)
        if chunks_count % 15 == 0:
            progresso_stimato = min(30 + (chunks_count // 5), 90)
            await context.report_progress(progresso_stimato, 100)
            # Log leggero per vedere che si muove nel terminale del server
            logger.debug(f"Generazione in corso... Progresso: {progresso_stimato}%")

    await context.info("Compilazione dello YAML completata.")
    await context.report_progress(95, 100)

    return {
        "playbook": full_content
    }


def generate_ansible_playbook_old(query: str):
    prompt = f"""
Sei un esperto DevOps.

Genera un playbook Ansible valido.

Regole:
- SOLO YAML
- Niente spiegazioni
- Compatibile Ansible 2.14+
- Usa hosts: local se non specificato

Task:
{query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "playbook": response["message"]["content"]
    }