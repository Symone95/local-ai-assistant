# DevOps-Ansible — server MCP

Server MCP che espone tool per generare, salvare, elencare ed eseguire playbook Ansible usando un
LLM locale (`llama3` via Ollama). È il server usato dai tool `mcp_*` dell'app: il client è
[../../client.py](../../client.py).

## Tool esposti

| Tool | Firma | Cosa fa |
|---|---|---|
| `generate_ansible_playbook_tool` | `(query)` | Genera un playbook YAML da una richiesta in linguaggio naturale, in streaming da `llama3` |
| `save_playbook_tool` | `(name, content)` | Salva il playbook in `ansible_memory/<name>.yml` |
| `list_playbooks_tool` | `()` | Elenca i playbook in `ansible_memory/` |
| `run_playbook_tool` 🛑 | `(name)` | Esegue `ansible-playbook -i inventory.ini ansible_memory/<name>.yml` |
| `bash_tool` ⚠️🛑 | `(command)` | Esegue un comando bash arbitrario. **Registrato solo con `MCP_ENABLE_BASH=1`** |

🛑 = richiede approvazione umana nella chat prima di essere eseguito (vedi [../../README.md](../../README.md#approvazione-umana-human-in-the-loop)).

`bash_tool` non ha alcun filtro sui comandi: chiunque parli con questo server ottiene esecuzione di
codice sulla macchina che lo ospita. Abilitalo solo consapevolmente, su una macchina di sviluppo.

## Prerequisiti

- [Ollama](https://ollama.com) in esecuzione con `llama3` (`ollama pull llama3`)
- `ansible` / `ansible-playbook` nel `PATH` (serve solo a `run_playbook_tool`)
- `mcp` e `ollama` installati nell'interprete che lancia il server

## Avvio

**Non serve avviarlo a mano per usare l'app**: il client lo lancia come sottoprocesso a ogni
chiamata. Per provarlo da solo:

```bash
# dalla root del progetto
venv/bin/python mcp_integration/servers/devops_ansible/server.py     # modalità stdio, resta in attesa

# oppure con l'Inspector, che dà una UI per invocare i tool a mano
npx @modelcontextprotocol/inspector venv/bin/python mcp_integration/servers/devops_ansible/server.py
```

## Uso da Claude Desktop

```json
{
  "mcpServers": {
    "devops-ansible": {
      "command": "/percorso/assoluto/local-ai-assistant/venv/bin/python",
      "args": ["/percorso/assoluto/local-ai-assistant/mcp_integration/servers/devops_ansible/server.py"]
    }
  }
}
```

## Note

- I path sono relativi alla working directory del processo. Il client passa `cwd` = questa
  cartella, quindi i playbook finiscono in `mcp_integration/servers/devops_ansible/ansible_memory/`.
- `run_playbook_tool` si aspetta un `inventory.ini` in questa cartella: non è versionato, perché
  contiene gli host del tuo ambiente.
