# 🤖 Local AI Assistant — RAG su PDF e agente multi-tool con LangGraph + Ollama

Assistente conversazionale **completamente locale** (nessuna API OpenAI): i modelli girano su
[Ollama](https://ollama.com), gli embedding sono calcolati in locale con `sentence-transformers`
e i documenti sono indicizzati in un database vettoriale **ChromaDB** persistente su disco.

L'interfaccia è una web app **Streamlit**; il cervello è un grafo **LangGraph** che, ad ogni
domanda, sceglie se rispondere direttamente, interrogare i documenti o invocare uno dei tool
disponibili (meteo, radio, ricerca web, lettura del codice sorgente, generazione di report PDF,
playbook Ansible via MCP…).

---

## Cosa fa

### 📄 RAG sui PDF
- Upload multiplo di PDF dalla sidebar, con deduplica tramite hash MD5 del file.
- Estrazione paginata del testo (`pypdf`), pulizia, chunking con overlap (1200/200 caratteri).
- Embedding con `all-MiniLM-L6-v2` e persistenza in `chroma_db/`.
- Retrieval **document-centric**: i chunk recuperati vengono raggruppati per file, riordinati per
  pagina e ricomposti in documenti interi, poi **riordinati da un LLM** (reranking con voto 0–10).
- **Conversational RAG**: prima del retrieval la domanda viene riscritta in forma *standalone*
  usando la storia della chat, così i pronomi e i riferimenti impliciti non rompono la ricerca.
- Filtro per singolo documento e reset del database dalla sidebar.

### 💬 Chat con memoria e risposte in streaming
- La conversazione è mantenuta in `st.session_state` e riconvertita in messaggi LangChain.
- I token dell'LLM vengono mostrati man mano che arrivano (`astream_events`).
- Pannello **"Ragionamenti in corso"**: mostra in tempo reale quale nodo del grafo è attivo e quale
  tool è stato scelto. L'UI è già predisposta anche per log e barra di avanzamento dei tool MCP
  ([app_with_mcp.py:363](app_with_mcp.py#L363)), ma oggi nessuno emette quegli eventi: le notifiche
  `context.info` / `report_progress` del server MCP non vengono ancora propagate dal client.
- Se la risposta contiene codice, viene renderizzata in un blocco con syntax highlighting e
  numerazione di riga anziché come testo semplice.

### 🛠️ Tool disponibili
| Tool | Cosa fa |
|---|---|
| `search_documents` | Ricerca semantica nei PDF indicizzati (o nel file temporaneo caricato) |
| `list_documents` / `get_upload_dates` | Elenca i file nel DB e le rispettive date di caricamento |
| `summarize_document` | Riassunto strutturato di un intero documento |
| `generate_pdf_report` | Scrive un report in markdown con l'LLM e lo impagina in PDF (`reportlab`) — scaricabile dalla chat |
| `terminal_tool` | Legge, mostra e analizza file e cartelle del progetto (utile per spiegare o rifattorizzare codice); l'accesso è confinato alla directory di lavoro |
| `internet_search_tool` | Ricerca web in tempo reale via DuckDuckGo per notizie, prezzi, eventi |
| `meteo_tool` | Condizioni meteo correnti di una città (OpenWeatherMap) |
| `radio_tool` | Avvia/ferma 12 stazioni radio italiane in streaming via VLC |
| `crypto_analyzer_tool` | Dati di mercato di una criptovaluta + notizie recenti, per un'analisi finanziaria |
| `mcp_*` | Genera, salva, elenca ed esegue playbook **Ansible** tramite il server MCP incluso in [mcp_integration/](mcp_integration/) |

### 🎙️ Voce, 👁️ immagini, 📂 file al volo
- **Speech-to-text**: registrazione dal microfono e trascrizione locale con Whisper.
- **Text-to-speech**: la risposta viene letta ad alta voce con `edge-tts` (voce italiana).
- **Analisi immagini**: modello multimodale LLaVA in streaming, con doppio fallback OCR —
  Tesseract per il testo stampato e TrOCR (`trocr-base-handwritten`) per il manoscritto, attivati
  automaticamente quando l'output di LLaVA risulta incoerente o ripetitivo.
- **File temporaneo**: PDF, TXT, MD, CSV o Excel caricato come contesto "usa e getta", senza
  finire nel database vettoriale.

---

## Come funziona il grafo

```
                     router
              /        |         \
           tool    mcp_tool   direct_llm_answer
              \      /               |
                 llm                END
                  |
                 END
```

- **`router`** — un `tool_planner` decide quale strumento usare. Prima applica regole veloci basate
  su espressioni regolari (Ansible, meteo, radio, file con estensione, segnali temporali…), poi, se
  nessuna scatta, delega la scelta a un LLM che risponde in JSON. La risposta viene interpretata con
  `parse_tool_plan()` ([nodes.py:17](nodes.py#L17)), che tollera il JSON incapsulato in un blocco
  markdown e scarta i tool inesistenti; il piano conserva anche `query` e `params`.
- **`tool`** — esegue lo strumento scelto e mette il risultato nello stato.
- **`mcp_tool`** — instrada le richieste `mcp_*` al server MCP su stdio, completando gli argomenti
  mancanti in base all'`inputSchema` del tool.
- **`llm`** — compone la risposta finale a partire da risultato del tool, contesto e storia della
  chat, chiudendo sempre con le fonti citate.
- **`direct_llm_answer`** — scorciatoia per small talk e domande generiche, senza tool né RAG.

Lo stato condiviso tra i nodi è `AgentState` ([dto/agent_state.py](dto/agent_state.py)): query,
messaggi, contesto, fonti, piano del tool, risultato, ultimo file aperto e risposta finale.

---

## 🔌 Integrazione MCP

I tool `mcp_*` non sono implementati nel corpo dell'app: sono **delegati a un server MCP**, che
l'app avvia da sé come sottoprocesso e con cui dialoga su **stdio**. Tutto ciò che riguarda MCP vive
in [mcp_integration/](mcp_integration/): il client da un lato, i server dall'altro.

```
mcp_integration/
├── client.py                  # sessione stdio + nodo `mcp_tool` del grafo
└── servers/
    └── devops_ansible/        # server MCP per Ansible (generazione ed esecuzione playbook)
```

Il server incluso è `DevOps-Ansible`, ma il meccanismo è generico: qualsiasi server MCP che esponga
tool con i nomi attesi può prenderne il posto.

### Come avviene una chiamata

1. Il `tool_planner` sceglie un tool il cui nome inizia per `mcp_`. Le query che contengono
   *ansible* o *playbook* sono instradate da una **regola deterministica** ([tools.py:277](tools.py#L277)),
   che distingue anche il verbo: *elenca* → `list`, *esegui* → `run`, *salva* → `save`, altrimenti
   `generate`. Senza quella regola la scelta resterebbe in mano al modello del router, che a volte
   rispondeva `none` mandando la richiesta al ramo `direct_llm_answer`.
2. L'edge condizionale in [app_with_mcp.py:49](app_with_mcp.py#L49) instrada al nodo `mcp_tool`.
3. [mcp_integration/client.py](mcp_integration/client.py) apre una sessione stdio, **rimuove il
   prefisso `mcp_`** e invoca il tool omonimo sul server.
4. L'output torna nello stato e il nodo `llm` compone la risposta finale.

Il prefisso è quindi solo una convenzione di routing: `mcp_list_playbooks_tool` lato client
corrisponde a `list_playbooks_tool` lato server.

> **Non serve avviare il server a mano.** `mcp_session()` apre una sessione nuova a ogni chiamata:
> il server viene lanciato, risponde e si chiude. Comodo in sviluppo, ma significa anche che il
> server non mantiene stato tra una richiesta e l'altra, e che paghi l'avvio del processo ogni volta.

### Tool esposti dal server DevOps-Ansible

| Nome lato client | Tool sul server | Cosa fa |
|---|---|---|
| `mcp_generate_ansible_playbook_tool` | `generate_ansible_playbook_tool(query)` | Genera un playbook YAML da una richiesta in linguaggio naturale, con `llama3` via Ollama |
| `mcp_save_playbook_tool` | `save_playbook_tool(name, content)` | Salva il playbook in `ansible_memory/` |
| `mcp_list_playbooks_tool` | `list_playbooks_tool()` | Elenca i playbook disponibili |
| `mcp_run_playbook_tool` | `run_playbook_tool(name)` | Esegue un playbook salvato con `ansible-playbook -i inventory.ini` |

Il server espone anche `bash_tool`, che esegue comandi bash arbitrari: **non è registrato** a meno di
avviarlo con `MCP_ENABLE_BASH=1`, e non è dichiarato nel registro dei tool dell'app.

### Come vengono ricavati gli argomenti

Il router produce quasi sempre solo la query testuale, mentre i tool hanno parametri strutturati
(`save_playbook_tool` vuole `name` e `content`). Prima di ogni chiamata il client legge
l'`inputSchema` del tool e completa i campi obbligatori nell'ordine
([client.py:107](mcp_integration/client.py#L107)):

1. argomenti già presenti nel piano del router;
2. campi di testo libero (`query`, `input`, `prompt`…) → la domanda dell'utente;
3. campi di contenuto (`content`, `code`, `playbook`…) → l'ultimo blocco di codice presente nella
   conversazione, così "salvalo come nginx" riusa il playbook appena generato;
4. quel che resta → estrazione con l'LLM del router da richiesta e storia recente.

Se dopo questi passaggi un campo obbligatorio è ancora vuoto, la chiamata **non viene tentata**:
l'utente riceve "mi manca: `name`" invece di un errore di validazione. I valori destinati a `name`
sono normalizzati (niente estensione, niente percorsi) perché il server li usa per comporre un path.

Il meccanismo è guidato dallo schema, quindi vale per qualsiasi server MCP, non solo per Ansible.

### Prerequisiti del server

- [Ollama](https://ollama.com) con il modello `llama3` (`ollama pull llama3`).
- `ansible` / `ansible-playbook` nel `PATH` — servono solo a `run_playbook_tool`.
- `mcp` e `ollama` nell'interprete che lancia il server: di default è **lo stesso venv dell'app**
  (`sys.executable`), quindi non serve un secondo ambiente.

Due variabili opzionali nel `.env` permettono di cambiare tutto senza toccare il codice:

| Variabile | Default | A cosa serve |
|---|---|---|
| `MCP_SERVER_DIR` | `mcp_integration/servers/devops_ansible` | Cartella del server da lanciare |
| `MCP_PYTHON` | l'interprete corrente | Interprete con cui avviare il server (utile se ha un venv suo) |

Il client passa al sottoprocesso `cwd = MCP_SERVER_DIR`: il server usa path relativi, quindi i
playbook finiscono in `mcp_integration/servers/devops_ansible/ansible_memory/` (ignorata da git,
come `inventory.ini`).

### Usare un altro server MCP

Crea la cartella del server sotto [mcp_integration/servers/](mcp_integration/servers/), puntala con
`MCP_SERVER_DIR` e dichiara i suoi tool nel registro in [tools.py:40](tools.py#L40), col prefisso
`mcp_` davanti al nome reale del tool. Il nodo `mcp_tool` non ha nulla di specifico su Ansible.
Il client parla con **un server per volta**.

### Provare il server da solo

```bash
npx @modelcontextprotocol/inspector venv/bin/python mcp_integration/servers/devops_ansible/server.py
```

L'Inspector elenca i tool esposti e permette di invocarli a mano: è il modo più rapido per capire se
un errore viene dal server o dal routing dell'agente.

---

## Struttura del progetto

| Percorso | Contenuto |
|---|---|
| [app_with_mcp.py](app_with_mcp.py) | App Streamlit completa (RAG + MCP + voce + immagini) — **entrypoint** |
| [app.py](app.py) | Versione ridotta senza MCP né analisi immagini |
| [nodes.py](nodes.py) | Nodi del grafo LangGraph e prompt di sistema |
| [tools.py](tools.py) | Registro dei tool, planner di routing, OCR e analisi immagini |
| [rag_engine.py](rag_engine.py) | Indicizzazione, retrieval, reranking, riscrittura query, riassunti |
| [pdf_loader.py](pdf_loader.py) | Estrazione, pulizia e chunking del testo dai PDF |
| [mcp_integration/client.py](mcp_integration/client.py) | Client MCP su stdio e nodo `mcp_tool` |
| [mcp_integration/servers/](mcp_integration/servers/) | Server MCP inclusi nel progetto (oggi: `devops_ansible`) |
| [dto/managers/](dto/managers/) | Manager per radio, meteo, terminale e ricerca internet |
| [utils/general.py](utils/general.py) | PDF report, parsing dei blocchi di codice, utility varie |
| [utils/speech_to_text.py](utils/speech_to_text.py), [utils/text_to_speech.py](utils/text_to_speech.py) | Whisper e edge-tts |
| [.env.example](.env.example) | Variabili d'ambiente attese (da copiare in `.env`) |
| `chroma_db/` | Database vettoriale persistente — locale, non versionato |
| `reports/` | Report PDF generati — locali, non versionati |

---

## Requisiti

- Python 3.11+
- [Ollama](https://ollama.com) in esecuzione in locale, con i modelli usati dal progetto:
  ```bash
  ollama pull qwen2.5-coder:3b   # router e risposte
  ollama pull llama3             # riassunti, reranking, report
  ollama pull llava-llama3       # analisi immagini (in alternativa: llava)
  ```
- **VLC** installato (per lo streaming radio) e **Tesseract** (`ita`+`eng`) se si vuole il fallback OCR.
- Per i tool `mcp_*` il server è incluso in [mcp_integration/servers/](mcp_integration/servers/) e
  viene avviato dall'app: servono solo `ansible` nel `PATH` e il modello `llama3` su Ollama.
  Dettagli in [Integrazione MCP](#-integrazione-mcp).

Installazione:

```bash
git clone git@github.com:Symone95/local-ai-assistant.git
cd local-ai-assistant
python -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env   # poi inserisci le tue chiavi
```

> `requirements.txt` copre tutto ciò che [app_with_mcp.py](app_with_mcp.py) importa all'avvio,
> MCP e voce compresi. Restano fuori solo le dipendenze caricate on demand: `pytesseract` (fallback
> OCR su testo stampato) e i pesi di `trocr-base-handwritten`, scaricati alla prima immagine
> manoscritta.

## Avvio

```bash
venv/bin/streamlit run app_with_mcp.py
```

## Note

- Le chiavi API vivono in un file `.env` (ignorato da git), letto con `python-dotenv`.
  [.env.example](.env.example) elenca le variabili attese: oggi solo `OPENWEATHER_API_KEY`,
  necessaria al `meteo_tool` (chiave gratuita su
  [openweathermap.org](https://home.openweathermap.org/api_keys)). Senza la variabile l'app parte
  lo stesso: è solo il tool meteo a rispondere con un messaggio di configurazione mancante.
- Tutto gira in locale: nessun dato dei documenti lascia la macchina, tranne le query esplicite
  verso i servizi web (ricerca internet, meteo, TTS).
- Restano fuori da git, oltre a `.env`: il database vettoriale (`chroma_db/`), i report PDF
  (`reports/`), i playbook e l'inventory dei server MCP (`ansible_memory/`, `inventory.ini`), le
  immagini caricate (`temp_images/`) e gli `.mp3` prodotti dal text-to-speech.
