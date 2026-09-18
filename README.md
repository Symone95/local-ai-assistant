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
- Pannello **"Ragionamenti in corso"**: mostra in tempo reale quale nodo del grafo è attivo, quale
  tool è stato scelto e l'avanzamento delle operazioni MCP.
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
| `mcp_*` | Genera, salva, elenca ed esegue playbook **Ansible** tramite un server MCP esterno |

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
  su espressioni regolari (meteo, radio, file con estensione, segnali temporali…), poi, se nessuna
  scatta, delega la scelta a un LLM che risponde in JSON.
- **`tool`** — esegue lo strumento scelto e mette il risultato nello stato.
- **`mcp_tool`** — instrada le richieste `mcp_*` al server MCP esterno su stdio.
- **`llm`** — compone la risposta finale a partire da risultato del tool, contesto e storia della
  chat, chiudendo sempre con le fonti citate.
- **`direct_llm_answer`** — scorciatoia per small talk e domande generiche, senza tool né RAG.

Lo stato condiviso tra i nodi è `AgentState` ([dto/agent_state.py](dto/agent_state.py)): query,
messaggi, contesto, fonti, piano del tool, risultato, ultimo file aperto e risposta finale.

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
| [mcp_client.py](mcp_client.py) | Client MCP su stdio e nodo `mcp_tool` |
| [dto/managers/](dto/managers/) | Manager per radio, meteo, terminale e ricerca internet |
| [utils/general.py](utils/general.py) | PDF report, parsing dei blocchi di codice, utility varie |
| [utils/speech_to_text.py](utils/speech_to_text.py), [utils/text_to_speech.py](utils/text_to_speech.py) | Whisper e edge-tts |
| `chroma_db/` | Database vettoriale persistente |
| `reports/` | Report PDF generati |

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
- Per i tool `mcp_*` serve un server MCP separato, atteso in `../ai_mcp/server.py`
  (configurabile in [mcp_client.py](mcp_client.py#L8)).

Installazione:

```bash
git clone git@github.com:Symone95/local-ai-assistant.git
cd local-ai-assistant
python -m venv venv
venv/bin/pip install -r requirements.txt
```

> `requirements.txt` copre il nucleo dell'app. Le funzionalità opzionali richiedono anche
> `mcp`, `openai-whisper`, `edge-tts`, `streamlit-mic-recorder`, `langchain-community`,
> `duckduckgo-search` e `pytesseract`.

## Avvio

```bash
venv/bin/streamlit run app_with_mcp.py
```

## Note

- La chiave OpenWeatherMap è attualmente hardcoded in
  [dto/managers/meteo_manager.py](dto/managers/meteo_manager.py) — conviene spostarla in una
  variabile d'ambiente prima di pubblicare il progetto.
- Tutto gira in locale: nessun dato dei documenti lascia la macchina, tranne le query esplicite
  verso i servizi web (ricerca internet, meteo, TTS).
