# Integrazione MCP

Tutto ciò che riguarda il Model Context Protocol vive qui.

```
mcp_integration/
├── client.py                  # sessione stdio + nodo `mcp_tool` del grafo LangGraph
└── servers/
    └── devops_ansible/        # server MCP per Ansible
```

| Componente | Ruolo |
|---|---|
| [client.py](client.py) | Avvia il server come sottoprocesso, invoca il tool e riporta l'output nello stato del grafo |
| [servers/devops_ansible/](servers/devops_ansible/) | Genera, salva, elenca ed esegue playbook Ansible |

## Come il client sceglie il server

```python
SERVER_DIR    = os.getenv("MCP_SERVER_DIR", "mcp_integration/servers/devops_ansible")
SERVER_PYTHON = os.getenv("MCP_PYTHON", sys.executable)
```

Il sottoprocesso viene lanciato con `cwd = SERVER_DIR`, così i path relativi usati dal server
(`ansible_memory/`, `inventory.ini`) si risolvono nella sua cartella e non in quella dell'app.

## Approvazione umana

I tool in `APPROVAL_REQUIRED_TOOLS` ([client.py](client.py)) sospendono il grafo con `interrupt()`
prima di essere eseguiti: la UI mostra tool e argomenti, e il nodo riparte solo con un
`Command(resume={"approvato": True, "args": {...}})`.

```python
APPROVAL_REQUIRED_TOOLS = os.getenv("MCP_APPROVAL_TOOLS", "run_playbook_tool,bash_tool")
```

Aggiungi qui qualsiasi tool che scriva, esegua o mandi qualcosa fuori dall'app. Gli argomenti
approvati vengono riutilizzati così come sono: alla ripresa LangGraph riesegue il nodo da capo, e
senza questo accorgimento una nuova risoluzione potrebbe cambiarli dopo l'approvazione.

## Aggiungere un server

1. Crea `servers/<nome>/server.py` con i tuoi tool FastMCP.
2. Dichiara i tool nel registro in [../tools.py](../tools.py) col prefisso `mcp_` davanti al nome
   reale del tool: il client lo rimuove prima di invocarlo (`mcp_foo_tool` → `foo_tool`).
3. Punta il client al nuovo server con `MCP_SERVER_DIR` nel `.env`.

Il client parla con **un solo server per volta**: quello indicato da `MCP_SERVER_DIR`.
