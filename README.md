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

### 🛑 Human-in-the-loop sulle azioni che toccano l'esterno
- I tool che modificano qualcosa fuori dall'app (di serie `run_playbook_tool` e `bash_tool`) non
  partono da soli: il grafo si **sospende** e la chat mostra tool, argomenti esatti e due pulsanti.
- È il meccanismo nativo di LangGraph — `interrupt()` più un checkpointer — non un finto prompt di
  conferma: lo stato resta congelato sul checkpoint finché non arriva la decisione.
- Viene eseguito **esattamente ciò che è stato approvato**: gli argomenti mostrati tornano indietro
  con la conferma, così una nuova risoluzione non può sostituirli.
- La lista dei tool da approvare si cambia da `.env` con `MCP_APPROVAL_TOOLS`.

### 🗣️ Conversazione vocale continua
- **Un pulsante solo, all'inizio**: premi il microfono e la conversazione parte. Da lì in poi si
  parla e basta: il turno si chiude da solo dopo ~0,6s di silenzio (rilevamento a energia, con
  soglia calibrata sul rumore di fondo della stanza).
- **Risposte brevi**: il grafo accetta istruzioni di stile e la modalità vocale chiede due o tre
  frasi colloquiali, senza elenchi né simboli. A voce una risposta strutturata come quella della
  chat scritta durerebbe un minuto.
- **Risponde mentre sta ancora pensando**: la risposta viene pronunciata *frase per frase* man
  mano che l'LLM la genera, invece di attendere il testo completo. Sulla macchina di sviluppo,
  dalla fine del parlato alla prima parola pronunciata passa **~1,5–2s**.
- **Barge-in** (spento di default): parlando sopra l'assistente lo si zittisce. Richiede le
  cuffie — vedi i limiti più sotto.
- **Approvazione a voce**: se il grafo chiede il via libera per un'azione sensibile, la domanda
  arriva parlata e si risponde "sì" o "no".
- Due interfacce sullo stesso motore: pannello nell'app Streamlit e CLI (`python -m voice.cli`).

### 🎙️ Voce a turni, 👁️ immagini, 📂 file al volo
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
  mancanti in base all'`inputSchema` del tool e sospendendo il grafo se il tool richiede
  approvazione umana.
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

### Approvazione umana (human-in-the-loop)

Generare un playbook è innocuo; eseguirlo no. I tool elencati in `APPROVAL_REQUIRED_TOOLS`
([client.py:46](mcp_integration/client.py#L46), configurabili con `MCP_APPROVAL_TOOLS`) non vengono
invocati finché una persona non dà il via libera.

```
utente: "esegui il playbook nginx"
   │
   ├─ router → mcp_tool: risolve gli argomenti → {"name": "nginx"}
   │
   ├─ interrupt()  ⏸  il grafo si ferma, lo stato va sul checkpointer
   │
   ├─ la chat mostra tool, argomenti e i pulsanti ✅ Esegui / ❌ Annulla
   │
   └─ Command(resume={"approvato": True, "args": {...}}) ▶ il nodo riparte ed esegue
```

Tre dettagli che rendono il meccanismo affidabile:

- **L'app è compilata con un checkpointer** (`InMemorySaver`) dentro `@st.cache_resource`
  ([app_with_mcp.py:88](app_with_mcp.py#L88)). Streamlit rilancia lo script a ogni click: senza la
  cache, ogni interazione creerebbe un checkpointer nuovo e vuoto, perdendo la sospensione.
- **Si esegue ciò che è stato approvato.** Alla ripresa LangGraph riesegue il nodo da capo, quindi
  una nuova risoluzione degli argomenti potrebbe produrre valori diversi da quelli mostrati. Gli
  argomenti approvati tornano dentro il `resume` e hanno la precedenza.
- **`interrupt()` sta fuori dalla sessione MCP.** Sospende sollevando un'eccezione di controllo: se
  attraversasse il task group `anyio` del client stdio verrebbe incapsulata in un `ExceptionGroup` e
  LangGraph non la riconoscerebbe più. Per questo gli schemi dei tool sono letti una volta e messi
  in cache, così la risoluzione degli argomenti non richiede una sessione aperta.

Ogni domanda nuova gira su un `thread_id` nuovo (lo stato `messages` usa il reducer `add_messages`,
riusare il thread duplicherebbe la storia); il thread viene conservato solo per la ripresa. Se
l'utente ignora la richiesta e chiede altro, l'approvazione in sospeso viene abbandonata.

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

## 🗣️ Modalità vocale

Il pacchetto [voice/](voice/) implementa una conversazione parlata continua. Non dipende da
Streamlit: riceve una funzione che, dato il testo dell'utente, produce i token della risposta.

```
in ascolto ──(silenzio > 800ms)──▶ trascrivo ──▶ penso ──▶ parlo ──┐
     ▲                                                             │
     └──────────────(fine risposta, oppure barge-in)───────────────┘
```

| Modulo | Ruolo |
|---|---|
| [voice/engine.py](voice/engine.py) | Macchina a stati, rilevamento del turno, taglio in frasi, barge-in |
| [voice/sources.py](voice/sources.py) | Sorgente audio: microfono reale o array in memoria (per i test) |
| [voice/stt.py](voice/stt.py) | Trascrizione con `faster-whisper` |
| [voice/tts.py](voice/tts.py) | Sintesi `edge-tts` decodificata in memoria |
| [voice/voce_sistema.py](voice/voce_sistema.py) | Sintesi con la voce di sistema macOS (`say`) — predefinita |
| [voice/player.py](voice/player.py) | Riproduzione interrompibile |
| [voice/agente.py](voice/agente.py) | Ponte verso il grafo, storia e approvazioni parlate |
| [voice/cli.py](voice/cli.py) | Modalità da terminale |

### Come si usa

```bash
# da terminale — la via più reattiva
venv/bin/python -m voice.cli

# oppure dall'app: premi 🎤 "Parla con l'assistente" sotto il titolo
venv/bin/streamlit run app_with_mcp.py
```

Nell'app c'è **un solo comando vocale**: il pulsante del microfono avvia la conversazione e la
chiude. Non è un push-to-talk — non va tenuto premuto, e non è lo stop a far partire la richiesta:
si preme una volta, si parla quando compare 🎙️ *In ascolto*, e si continua a parlare finché serve.
Mentre il pulsante mostra ⏳ *Preparo*, l'assistente sta caricando i modelli e misurando il rumore
di fondo: parlare in quel momento non verrebbe sentito.

### Perché sembra reattiva

Il tempo percepito non è quello della risposta intera, ma quello della **prima frase**. La
risposta viene quindi tagliata appena c'è una frase conclusa e mandata subito in sintesi, mentre
il resto si sta ancora generando; la frase successiva si sintetizza mentre la precedente suona.

Il taglio non spezza i numeri decimali (`24.79°C` resta una cosa sola) e i blocchi di codice non
vengono letti: a voce diventano "ti ho scritto il codice nella chat".

Perché le frasi suonino come un discorso unico e non a scatti servono due accorgimenti:

- **I silenzi ai bordi vengono tagliati.** edge-tts consegna ogni frase con ~190ms di silenzio in
  testa e ~870ms in coda: pronunciando una frase alla volta si sommano in pause di oltre un secondo
  fra una e l'altra. [voice/tts.py](voice/tts.py) li rimuove lasciando 30ms di margine, per non
  tagliare l'attacco delle consonanti.
- **Lo stream di uscita resta aperto.** Aprire e chiudere il dispositivo audio a ogni frase
  introduceva un'altra pausa fissa; [voice/player.py](voice/player.py) tiene lo stream vivo e lo
  alimenta con un buffer, che serve anche al barge-in per zittire l'assistente all'istante.

Su tre frasi consecutive il silenzio aggiunto passa da ~3,9s a **0,22s**.

Arrivarci ha richiesto di togliere di mezzo quattro attese che nessuno vedeva, tutte misurate:

| Causa | Prima | Dopo |
|---|---|---|
| **Prompt del router** da ~1150 token, rivalutato a ogni turno perché la cache di Ollama conserva solo l'ultimo contesto | 12,2s | ~1,0s |
| **Convenevoli** che passavano dal router LLM: ora "ciao", "grazie", "chi sei" hanno una regola diretta | 13,4s | 0,00s |
| **Nodi che usavano `llm.invoke`**: la risposta usciva tutta insieme a generazione finita, invece di token per token | 2,4s al primo token | 0,34s |
| **Due client Ollama con opzioni diverse** (`ChatOllama(num_ctx=4096)` e `ollama.chat` con i default): il modello veniva ricaricato a ogni alternanza | ~1,8s per turno | 0s |

Il quarto punto è il più insidioso: Ollama tiene in memoria **una** configurazione per modello, e
chiamarlo con opzioni diverse da due punti del codice lo fa scaricare e ricaricare in continuazione.
Per questo esiste [llm_client.py](llm_client.py): un solo client, condiviso da router, nodi e
riscrittura delle query. La modalità vocale lo scalda all'avvio insieme al prompt del router,
così il primo "ciao" non paga il caricamento.

Misure sulla macchina di sviluppo (M-series, Ollama locale), a modelli caldi:

| Fase | Tempo |
|---|---|
| Chiusura del turno dopo che smetti di parlare | ~0,3s |
| Trascrizione (`faster-whisper small`) | 1,7–2,6s |
| Router su convenevoli e comandi riconosciuti da regola | 0,00s |
| Router quando decide l'LLM | ~1,0s |
| Primo token della risposta | 0,3–2,9s (secondo il tool coinvolto) |
| Sintesi della prima frase (`edge-tts`) | ~0,6s |

Sommati, dalla fine del parlato alla prima parola pronunciata: **~3s** per una battuta di
conversazione, **~8,7s** per una domanda che passa da un tool.

> **Sulla scelta del modello di trascrizione.** Su audio sintetico pulito `base` sembra quattro
> volte più veloce di `small`. Su microfono reale il confronto si ribalta: `base` produce testo
> sbagliato e ci mette *di più* a convergere (2,8–6,0s contro 1,7–2,7s), perché fatica sul parlato
> rumoroso. Il default è `small` — misurato su registrazioni vere, non su TTS.

Due accorgimenti pesano sulla qualità di quel che viene capito:

- **Pre-roll di 300ms.** L'audio veniva raccolto solo da quando l'energia supera la soglia, ma
  l'attacco di una parola è più debole del resto: `"Ciao, che tempo fa Milano"` diventava
  `"che tempo fa me lano"`. Ora i frame immediatamente precedenti vengono conservati e anteposti.
- **Taglio della coda muta.** Davanti a mezzo secondo di silenzio Whisper inventa parole
  (`"Eh... no... Poi mi marzi"`), che poi mandano fuori strada il router. L'audio viene chiuso
  sull'ultimo frame con voce vera, più 250ms.

### Due motori di sintesi

| Backend | Quando conviene |
|---|---|
| **`say`** (predefinito su macOS) | Voce di sistema. Parla un processo del sistema operativo: niente rete, niente decodifica mp3, nessun codice Python nel percorso audio. È la scelta robusta: non può andare a scatti. |
| `edge` | Voci neurali di edge-tts, più naturali. Passa dal player Python, che sotto carico può restare indietro e far ripetere l'ultimo blocco alla scheda audio. |

```bash
VOICE_TTS_BACKEND=say      # oppure "edge"
VOICE_SAY_VOICE=Alice      # voci italiane: Alice, Eddy, Flo, Grandma, Grandpa…
```

Con `say` si risparmiano anche ~1,9s all'avvio, perché non serve aprire in anticipo il
dispositivo audio.

### Se non ti sente

Mentre è in ascolto il pannello mostra il **livello del microfono** e la soglia calcolata, perché
"In ascolto" da solo non distingue un microfono muto da una soglia troppo alta:

```
🎙️ In ascolto
▓▓▓▓▓░░░░░░░░░░
Livello microfono 0.0008 · soglia 0.0060 → silenzio
```

| Cosa vedi | Causa | Rimedio |
|---|---|---|
| 🔇 "Dal microfono non arriva alcun segnale" | Il processo non ha accesso al microfono: su macOS `sounddevice` apre lo stream senza errori e riceve zeri | Impostazioni di Sistema → Privacy e sicurezza → Microfono: autorizza il terminale (o l'IDE) da cui lanci l'app, poi riavviala |
| Il livello si muove ma resta sotto soglia | Voce lontana o ambiente rumoroso: la soglia è il rumore di fondo × `VOICE_VAD_FACTOR` | Abbassa `VOICE_VAD_FACTOR` (da 3.0 a 2.0) nel `.env` |
| Risponde a se stesso | Sente la propria voce dagli altoparlanti (nessuna cancellazione d'eco) | Usa le cuffie, o alza `VOICE_COOLDOWN_MS` |

### Limiti da conoscere

- **L'audio è catturato lato server** con `sounddevice`, quindi funziona quando l'app gira sulla
  stessa macchina che ascolta — che è il caso d'uso di questo progetto. Per un uso da remoto
  servirebbe l'audio del browser (WebRTC).
- **Il barge-in è spento di default**, perché non c'è cancellazione d'eco. Misurando quanto il
  microfono sente la voce dell'assistente dagli altoparlanti, il 13% dei frame supera già la soglia
  di interruzione: basta poco (volume più alto, microfono più vicino) perché si auto-inneschi,
  spezzando la risposta a metà e facendola sembrare ripetuta. Con le cuffie si riattiva con
  `VOICE_BARGE_IN=1`.
- **Con il backend `edge` lo stream di uscita usa blocchi grandi** (2048 frame, 131ms di latenza). Il callback che
  alimenta la scheda audio è codice Python: mentre il modello genera e la sintesi decodifica, può
  restare indietro di oltre 200ms, e il dispositivo a secco ripete l'ultimo blocco — si sente come
  un'eco. Con il buffer ampio gli stalli misurati sotto carico passano da 7 a zero.
- Il pannello Streamlit si aggiorna con un `st.fragment(run_every=0.5)`: lo stato mostrato può
  essere indietro di mezzo secondo rispetto a quello che si sente.

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
| [agent_graph.py](agent_graph.py) | Costruzione del grafo, condivisa fra app Streamlit e modalità vocale |
| [llm_client.py](llm_client.py) | Client Ollama condiviso: una sola configurazione, niente ricaricamenti |
| [voice/](voice/) | Conversazione vocale continua (motore, STT, TTS, CLI) |
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
  ollama pull llama3.2:3b        # router e risposte
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
