import json
import ollama

from llm_client import completa
import re
from rag_engine import get_files_with_upload_date_tool, get_files_in_db_tool, conversational_search_tool, \
    summarize_document_tool, extract_filename_from_query, generate_report_content
from utils.general import generate_pdf_report, extract_city_from_query, extract_path_with_llm, ottieni_dati_crypto
from dto.managers.radio_manager import radio_manager
from dto.managers.meteo_manager import meteo_manager
from dto.managers.terminal_manager import terminal_manager
from dto.managers.internet_search_manager import internet_search_tool


TOOLS = [
    ## Strumenti per gestione documenti e RAG
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
    },
    {
        "name": "summarize_document",
        "description": "Crea un riassunto completo di un documento",
        "input": "file_name"
    },
    {
        "name": "generate_pdf_report",
        "description": "Genera un report PDF professionale su qualsiasi argomento",
        "input": "title, content"
    },
    ## Strumenti MCP per Ansible e gestione infrastruttura
    {
        "name": "mcp_list_playbooks_tool",
        "description": "Lista playbook disponibili",
        "input": ""
    },
    {
        "name": "mcp_run_playbook_tool",
        "description": "Esegue un playbook Ansible gia' salvato, indicandone il nome (senza estensione)",
        "input": "name"
    },
    {
        "name": "mcp_generate_ansible_playbook_tool",
        "description": "Genera un playbook Ansible YAML da una richiesta testuale",
        "input": "query"
    },
    {
        "name": "mcp_save_playbook_tool",
        "description": "Salva playbook ansible",
        "input": "name, content"
    },
    {
        "name": "crypto_analyzer_tool",
        "description": "Analisi finanziaria di una criptovaluta: dati di mercato in tempo reale piu' notizie recenti",
        "input": "nome della criptovaluta"
    },
    ## Strumento per gestione radio
    {
        "name": "radio_tool",
        "description": "Gestisce le stazioni radio e riproduce stazioni conosciute",
        "input": "query"
    },
    ## Strumento per gestione del meteo
    {
        "name": "meteo_tool",
        "description": "Fornisce informazioni sul meteo",
        "input": "query"
    },
    ## Strumento per esecuzione comandi terminale e gestione filesystem
    {
        "name": "terminal_tool",
        "description": "Legge, analizza, rifattorizza o mostra il contenuto di file e cartelle del progetto. Usalo quando l'utente menziona un file specifico (.py, .js, .ts, ecc.) o vuole analizzare/leggere/refactoring del codice.",
        "input": "query"
    },
    ## Strumento per ricerche su internet
    {
        "name": "internet_search_tool",
        "description": "Cerca informazioni in tempo reale su internet. Usalo per fatti che cambiano nel tempo: prezzi, notizie, risultati sportivi, versioni software, chi è attualmente in carica, eventi recenti. NON usarlo per conoscenza generale stabile.",
        "input": "query"
    }
]

# TODO: SPOSTARE TUTTO SU MCP
def execute_tool(tool_name: str, query: str = None, selected_doc=None, messages=None, context="", current_file=""):

    print("tool_name: ", tool_name)

    if tool_name == "search_documents":
        context_result, structured_sources = conversational_search_tool(query, messages, selected_doc, temp_context=context)
        return {
            "context": context_result,
            "sources": structured_sources
        }

    if tool_name == "list_documents":
        files = get_files_in_db_tool()
        return {"files": files}

    if tool_name == "get_upload_dates":
        dates = get_files_with_upload_date_tool()
        return {"dates": dates}

    if tool_name == "summarize_document":
        # l'agente deve aver passato il nome del file nella query
        file_name = extract_filename_from_query(query)
        return summarize_document_tool(file_name)

    if tool_name == "generate_pdf_report":
        print("✍️ Genero contenuto report con LLM...")
        report_markdown = generate_report_content(query, messages)

        print("📄 Creo PDF...")
        pdf_result = generate_pdf_report(title="AI Generated Report", content=report_markdown)

        return {
            "report_text": report_markdown,
            "pdf": pdf_result.get("file_path", "")
        }
    
    if tool_name == "radio_tool":
        query_text = query or ""
        query_lower = query_text.lower()

        # Prova a trovare una stazione dalla domanda dell'utente
        stations = radio_manager.get_station_list()
        station = None

        stop_pattern = re.compile(r"\b(stop|ferma|spegni|spegnila|spengila|arresta|chiudi|disattiva|metti giù|muto)\b", re.IGNORECASE)
        if stop_pattern.search(query_lower):
            try:
                stopped = radio_manager.stop_radio()
                if stopped:
                    return {
                        "status": "stopped",
                        "message": "La radio è stata spenta.",
                        "station": radio_manager.current_audio
                    }
                return {
                    "status": "stopped",
                    "message": "Non c'era alcuna radio in riproduzione.",
                    "station": None
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        # Controllo URL diretto nella query
        url_match = re.search(r"https?://\S+", query_text)
        if url_match:
            station_url = url_match.group(0).strip()
            try:
                radio_manager.play_radio(station_url)
                return {
                    "status": "playing",
                    "station": station_url,
                    "message": f"Riproduco l'URL radio: {station_url}",
                    "stations": [s["nome"] for s in stations]
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        # Match del nome stazione
        for radio in sorted(stations, key=lambda x: -len(x["nome"])):
            nome_lower = radio["nome"].lower()
            if nome_lower in query_lower:
                station = radio
                break
            if nome_lower.replace("radio", "").strip() in query_lower:
                station = radio
                break

        if station:
            try:
                radio_manager.play_radio(station["url"])
                return {
                    "status": "playing",
                    "station": station["nome"],
                    "url": station["url"],
                    "message": f"Sto riproducendo {station['nome']}.",
                    "current_audio": station["nome"]
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        random_play_pattern = re.compile(r"\b(accendi|accendila|metti|mettila|apri|attiva|avvia|riproduci|ascolta)\b.*\bradio\b|\bradio\b.*\b(accendi|accendila|metti|mettila|apri|attiva|avvia|riproduci|ascolta)\b", re.IGNORECASE)
        if random_play_pattern.search(query_text) and not station:
            try:
                radio_manager.random_radio()
                return {
                    "status": "playing",
                    "message": "Accendo una radio casuale.",
                    "current_audio": radio_manager.current_audio
                }
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "stations": [s["nome"] for s in stations]
                }

        return {
            "status": "list",
            "message": "Non ho identificato una stazione precisa. Ecco le radio disponibili:",
            "stations": [s["nome"] for s in stations]
        }

    if tool_name == "meteo_tool":
        city = extract_city_from_query(query or "")
        if not city:
            return {
                "status": "error",
                "message": "Non ho riconosciuto una città nell'input. Per favore specifica una città per il meteo."
            }

        meteo_result = meteo_manager.current_weather(q=city)
        return {
            "status": "success",
            "city": city,
            "data": meteo_result
        }
    
    if tool_name == "terminal_tool":
        path = extract_path_with_llm(query or "", current_file=current_file)
        try:
            result = terminal_manager.list_and_read_files(path)
            return {"status": "success", "path": path, "content": result, "current_file": path}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    if tool_name == "internet_search_tool":
        # Esegui la ricerca web
        risultato_web = internet_search_tool(query)
        
        # Salvi il risultato nel contesto o nel tool_result così l'LLM lo legge nel nodo successivo
        return {
            "tool_result": {"content": risultato_web},
            "context": context + f"\n\nRisultati di ricerca internet:\n{risultato_web}"
        }

    if tool_name == "crypto_analyzer_tool":
        # 1. Estrai i dati tecnici
        dati = ottieni_dati_crypto(query)  # Il router deve estrarre l'ID della crypto (es: 'bitcoin')

        # 2. Sfrutta il tool Google che hai già implementato per cercare le news
        notizie = internet_search_tool(f"ultime notizie {query} mercato criptovalute")

        # 3. Costruisci il contesto arricchito per il nodo LLM successivo
        contesto_finanziario = f"DATI_CRYPTO:{dati}\nNOTIZIE_CRYPTO:{notizie}"

        return {
            "tool_result": {"content": contesto_finanziario},
            "context": context + f"\n\nContext Finanziario:\n{contesto_finanziario}"
        }

    return {"error": "Tool non trovato"}


# Parole che, da sole, compongono un convenevolo: se la frase contiene solo queste, non serve
# alcun tool. Basta una parola fuori elenco perche' la domanda segua il percorso normale.
PAROLE_CONVENEVOLI = {
    "ciao", "ehi", "hey", "salve", "buongiorno", "buonasera", "buonanotte", "arrivederci",
    "come", "stai", "va", "tutto", "bene", "grazie", "mille", "prego", "perfetto", "ottimo",
    "ok", "okay", "d", "accordo", "a", "dopo", "presto", "ci", "vediamo", "chi", "sei", "ti",
    "chiami", "cosa", "che", "sai", "fare", "mi", "senti", "puoi", "aiutarmi", "e", "tu",
}


def tool_planner(query, messages=None, context=""):
    
    if query:
        query_lower = query.lower().strip()

        # Convenevoli e domande sull'assistente: nessun tool serve, e soprattutto non serve
        # interpellare il router LLM, che su questo prompt costa qualche secondo.
        # La scorciatoia scatta solo se la frase e' FATTA SOLO di convenevoli: "ciao" passa di
        # qui, "ciao, che tempo fa a Milano?" no, altrimenti il saluto iniziale nasconderebbe
        # la richiesta vera.
        parole = re.findall(r"\w+", query_lower)
        if parole and all(parola in PAROLE_CONVENEVOLI for parola in parole):
            return json.dumps({"tool": "none"})

        # Richieste creative o di spiegazione generale: nessun tool. Si escludono le query che
        # nominano un file, che vanno invece al terminal_tool.
        if not re.search(r"\.\w{1,4}\b", query_lower) and re.match(
            r"^(raccontami|racconta|inventa|inventami|scrivimi una (poesia|storia|canzone|filastrocca)|"
            r"fammi una battuta|dimmi una barzelletta|indovinello)\b",
            query_lower,
        ):
            return json.dumps({"tool": "none"})

        # Ansible/MCP: prima delle altre regole, perche' "genera un playbook" verrebbe
        # altrimenti intercettato da generate_pdf_report e "playbook.yml" da terminal_tool.
        if re.search(r"\b(ansible|playbook|playbooks)\b", query_lower):
            if re.search(r"\b(elenca|elencami|lista|listami|quali|quanti|mostra|mostrami|disponibili|salvati)\b", query_lower):
                return json.dumps({"tool": "mcp_list_playbooks_tool", "query": query})
            if re.search(r"\b(esegui|eseguilo|lancia|lancialo|avvia|avvialo|run|applica)\b", query_lower):
                return json.dumps({"tool": "mcp_run_playbook_tool", "query": query})
            if re.search(r"\b(salva|salvalo|memorizza|conserva)\b", query_lower):
                return json.dumps({"tool": "mcp_save_playbook_tool", "query": query})
            return json.dumps({"tool": "mcp_generate_ansible_playbook_tool", "query": query})

        if re.search(r"\b(cripto|analizza|btc|eth|criptovalute)\b", query_lower):
            return json.dumps({"tool": "crypto_analyzer_tool", "query": query})

        if re.search(r"\b(meteo|tempo|previsioni|pioggia|sole|neve|vento|temperatura|umidità)\b", query_lower):
            return json.dumps({"tool": "meteo_tool", "query": query})

        # Terminal/Filesystem requests: elenca o leggi file/cartelle
        if re.search(r"\b(lista|elenca|mostra|apri|leggi|leggere|leggerlo|visualizza)\b.*\b(file|cartella|cartelle|folder|dir|directory)\b", query_lower):
            return json.dumps({"tool": "terminal_tool", "query": query})

        # File con estensione esplicita: analisi, refactoring, lettura codice
        if re.search(r"\b\w[\w\-]*\.(py|js|ts|jsx|tsx|java|go|cpp|c|cs|rb|php|sh|yaml|yml|json|md)\b", query_lower):
            return json.dumps({"tool": "terminal_tool", "query": query})

        if re.search(r"\b(radio|stazione|stazioni|ascolta|ascoltiamo|ascoltare|riproduci|metti la radio|radio105|rtl 102\.5|rtl|deejay)\b", query_lower):
            return json.dumps({"tool": "radio_tool", "query": query})
        
        if re.search(r"\b(pdf|report|genera|crea|fornisci|scrivi)\b", query_lower):
            return json.dumps({"tool": "generate_pdf_report", "query": query})

        # Internet search: segnali temporali espliciti + argomenti del mondo reale
        _temporal = re.search(
            r"\b(questo weekend|questa settimana|questo mese|oggi|domani|dopodomani|stanotte|stasera|adesso|attualmente|di recente|ultimamente|ultime notizie|ora|in questo momento)\b",
            query_lower
        )
        _realworld = re.search(
            r"\b(eventi|evento|concerti|concerto|mostre|mostra|spettacoli|spettacolo|notizie|news|prezzi|prezzo|quotazione|risultati|partite|partita|classifica|manifestazioni)\b",
            query_lower
        )
        if _temporal and _realworld:
            return json.dumps({"tool": "internet_search_tool", "query": query})

        # Internet search: richiesta esplicita di informazioni recenti/web anche senza marcatori temporali
        if re.search(
            r"\b(ultime notizie|breaking news|notizie di oggi|cosa è successo|chi ha vinto|chi vincerà|versione più recente|ultimo aggiornamento|prezzo attuale|quotazione attuale)\b",
            query_lower
        ):
            return json.dumps({"tool": "internet_search_tool", "query": query})

        stop_pattern = re.compile(r"\b(stop|ferma|spegni|spegni la radio|spegni radio|spegnila|arresta|chiudi|disattiva|metti giù|muto)\b", re.IGNORECASE)
        prior_text = "".join(
            [msg.get("content", "").lower() if isinstance(msg, dict) else getattr(msg, "content", "").lower() for msg in (messages or [])]
        )
        if stop_pattern.search(query_lower) and (
            "radio" in query_lower
            or radio_manager.is_playing()
            or "radio" in prior_text
            or any(station["nome"].lower() in prior_text for station in radio_manager.get_station_list())
        ):
            return json.dumps({"tool": "radio_tool", "query": query})

    tools_description = "\n".join(
        [f"{t['name']}: {t['description']}" for t in TOOLS]
    )

    # Costruisci il contesto conversazionale se disponibile
    context_section = ""
    if messages:
        context_section = """
CONTESTO CONVERSAZIONALE PRECEDENTE:
"""
        # Prendi gli ultimi 2-3 messaggi per il contesto
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        for msg in recent_messages:
            # Gestisci sia dizionari che oggetti LangChain (HumanMessage, AIMessage, ecc.)
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                # Oggetti LangChain come HumanMessage, AIMessage
                role = getattr(msg, "type", "user")
                content = getattr(msg, "content", "")
            
            content = content[:200] if content else ""  # Limita per brevità
            context_section += f"- {role}: {content}\n"
    else:
        context_section = "Nessun contesto conversazionale precedente."

    file_section = ""
    if context:
        file_section = f"\nFILE TEMPORANEO CARICATO:\n{context[:3000]}\n"  # Limita la lunghezza del prompt
    else:
        file_section = "\nNessun file temporaneo caricato.\n"

    # Prompt volutamente compatto: viene rivalutato a ogni turno (la cache di Ollama conserva
    # solo l'ultimo contesto, che nel frattempo e' quello della risposta). Ogni migliaio di token
    # in piu' qui sono secondi di attesa prima che l'assistente cominci a rispondere.
    prompt = f"""Scegli UN tool per la domanda dell'utente, o "none" se non ne serve nessuno.

TOOL:
{tools_description}

REGOLE:
1. Informazioni che cambiano nel tempo (notizie, prezzi, risultati, eventi, "oggi/adesso/ultime")
   → internet_search_tool
2. Domanda sui documenti caricati o sul file temporaneo → search_documents;
   quali documenti ci sono → list_documents; quando sono stati caricati → get_upload_dates;
   riassunto esplicito di un documento → summarize_document
3. Richiesta esplicita di un report o di un PDF → generate_pdf_report
4. Ansible, playbook, deploy, infrastruttura → il corrispondente tool mcp_*
5. File del progetto o codice da leggere/spiegare/modificare → terminal_tool
6. Meteo → meteo_tool; radio → radio_tool; criptovalute → crypto_analyzer_tool
7. Conversazione, spiegazioni di concetti stabili, domande sul file gia' caricato,
   creativita' (storie, poesie, battute) → "none"

{context_section}
{file_section}

Rispondi SOLO con JSON, senza testo attorno:
{{"tool": "nome_tool", "query": "{query}"}}  oppure  {{"tool": "none"}}

Domanda attuale: {query}
"""

    # Client condiviso: vedi llm_client.py — usare qui un client diverso farebbe ricaricare
    # il modello a ogni alternanza fra router e nodi di risposta.
    return completa(prompt)


def _ocr_with_trocr(image_path: str, mcp_log_placeholder=None) -> str:
    """
    TrOCR (microsoft/trocr-base-handwritten) — modello specifico per testo manoscritto.
    Molto più affidabile di LLaVA per corsivo e handwriting.
    Processa l'immagine segmentandola in bande orizzontali (righe di testo).
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch

        print("🔤 TrOCR: carico modello handwritten...")
        if mcp_log_placeholder:
            mcp_log_placeholder.caption(f"⚙️ Carico modello TrOCR per testo manoscritto...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Segmenta in bande orizzontali per simulare le righe di testo
        # TrOCR è addestrato su singole righe — più bande = migliore accuratezza
        num_bands = max(4, h // 80)
        band_h = h // num_bands
        lines = []

        for i in range(num_bands):
            mcp_log_placeholder.caption(f"⚙️ *Analizzando l'immagine con TrOCR — riga {i+1}/{num_bands}*")
            y0 = i * band_h
            y1 = min(y0 + band_h, h)
            band = img.crop((0, y0, w, y1))

            pixel_values = processor(images=band, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if text:
                lines.append(text)

        return "\n".join(lines)

    except Exception as e:
        print(f"⚠️ TrOCR fallito: {e}")
        return ""


def _ocr_with_tesseract(image_path: str, mcp_log_placeholder=None) -> str:
    """
    Tesseract OCR con preprocessing — affidabile solo per testo stampato/digitale.
    Non riconosce corsivo o scrittura a mano.
    """
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if max(w, h) < 1500:
            scale = max(2, 1500 // max(w, h))
            img = img.resize((w * scale, h * scale), Image.LANCZOS)

        gray = img.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(2.5)
        gray = gray.filter(ImageFilter.SHARPEN)
        gray = gray.filter(ImageFilter.SHARPEN)
        if mcp_log_placeholder:
            mcp_log_placeholder.caption(f"⚙️ **Analizzando l'immagine con Tesseract OCR**")

        text = pytesseract.image_to_string(gray, lang="ita+eng", config="--oem 3 --psm 3").strip()
        if len(text.split()) < 5:
            sparse = pytesseract.image_to_string(gray, lang="ita+eng", config="--oem 3 --psm 11").strip()
            if len(sparse) > len(text):
                text = sparse

        lines = [l for l in text.splitlines() if len(l.strip()) > 1]
        return "\n".join(lines)
    except Exception as e:
        print(f"⚠️ Tesseract OCR fallito: {e}")
        return ""


import subprocess as _sp
_LLAVA_MODEL = "llava-llama3" if "llava-llama3" in _sp.run(
    ["ollama", "list"], capture_output=True, text=True
).stdout else "llava"


def _llava_output_is_valid(text: str) -> bool:
    """Controlla se l'output di LLaVA è un risultato sensato."""
    words = text.split()
    if len(words) < 5:
        return False
    # Rileva loop di ripetizioni: se una sequenza di 4 parole si ripete più di 2 volte
    joined = " ".join(words)
    for i in range(len(words) - 4):
        phrase = " ".join(words[i:i+4])
        if joined.count(phrase) > 2:
            return False
    return True


def image_analyser_stream(image_path: str, domanda: str = "Descrivi cosa vedi in questa immagine.", mcp_log_placeholder=None):
    """
    Generatore per st.write_stream():
    - Streamma i token LLaVA in real-time (visibili subito in UI)
    - Dopo che LLaVA finisce, valida il risultato
    - Se non valido, appende in streaming il fallback (Tesseract → TrOCR)
    """
    _text_request = re.search(
        r"\b(testo|scritto|scrivi|leggi|trascrivi|parole|corsivo|lettera|cosa c['\s]è scritto)\b",
        domanda.lower()
    )
    _base = (
        f"{domanda}\n\nSe nell'immagine c'è testo scritto, leggilo e riportalo nella risposta."
        if _text_request else domanda
    )
    prompt = f"{_base}\n\nRispondi esclusivamente in italiano."

    chunks = []
    try:
        stream = ollama.chat(
            model=_LLAVA_MODEL,
            messages=[
                {"role": "system", "content": "Sei un assistente visivo. Rispondi SEMPRE in italiano."},
                {"role": "user", "content": prompt, "images": [image_path]}
            ],
            stream=True,
            options={
                "num_predict": 400,
                "temperature": 0.35,
                "repeat_penalty": 1.2,
                "top_k": 40,
                "top_p": 0.85,
            }
        )
        for chunk in stream:
            mcp_log_placeholder.caption(f"⚙️ **Analizzando l'immagine**")
            token = chunk["message"]["content"]
            yield token
            chunks.append(token)
    except Exception as e:
        yield f"Errore LLaVA: {e}"
        return

    llava_result = "".join(chunks).strip()

    if _llava_output_is_valid(llava_result):
        return  # output valido, streaming già completato

    # Fallback 1: Tesseract
    yield "\n\n⚠️ Testo non leggibile, provo OCR...\n"
    tesseract_text = _ocr_with_tesseract(image_path)
    if tesseract_text:
        yield tesseract_text
        return

    # Fallback 2: TrOCR
    yield "\n⚠️ Provo TrOCR per testo manoscritto...\n"
    trocr_text = _ocr_with_trocr(image_path, mcp_log_placeholder=mcp_log_placeholder)
    if trocr_text:
        yield trocr_text
        return

    yield "\nImpossibile analizzare l'immagine."