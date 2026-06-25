import hashlib
import re
from chromadb.api.models.Collection import Collection
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import subprocess

import pandas as pd
from pdf_loader import load_pdf

import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5-coder:3b",  # llama3"
                 num_ctx=4096,              # con questo dico di non andare oltre i 4k di token
                 format="json",
                 temperature=0.0,
                 num_predict=-1
                )

def convert_to_langchain_messages(streamlit_messages):
    lc_messages = []
    for m in streamlit_messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages

def group_by_file(structured_sources):
    files = {}

    for s in structured_sources:
        if s["file"] not in files:
            files[s["file"]] = s

    return list(files.values())

def get_file_hash(file):
    return hashlib.md5(file.getvalue() if not isinstance(file, bytes) else file).hexdigest()

def make_source_link(file, page):
    # return f"[📄 {file} - pag.{page}](#)"
    return f'<a href="docs/{file}#page={page}" target="_blank">📄 {file} - pag.{page}</a>'

def extract_between(text, start, end):
    try:
        return text.split(start)[1].split(end)[0]
    except:
        return ""

def extract_keywords(query):
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if len(w) > 4]

def get_db_stats(collection: Collection):
    data = collection.get()
    return len(data["ids"])

def get_full_document(file_name: str, collection: Collection):
    """
    Recupera tutti i chunk di un documento e li unisce in ordine di pagina
    """
    data = collection.get(where={"file": file_name})

    if not data["metadatas"]:
        return None

    # ordina per pagina
    sorted_chunks = sorted(
        zip(data["documents"], data["metadatas"]),
        key=lambda x: x[1]["page"]
    )

    full_text = "\n\n".join([doc for doc, _ in sorted_chunks])
    return full_text

def load_file_text(uploaded_file, max_chars: int = 20000):
    """
    Legge un file temporaneo da usare come contesto di lettura.
    Supporta PDF, TXT, MD, CSV, XLS e XLSX.
    """
    uploaded_file.seek(0)
    filename = getattr(uploaded_file, "name", "")
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        text = load_pdf(uploaded_file)
    elif ext in {".txt", ".md"}:
        raw = uploaded_file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
    elif ext == ".csv":
        try:
            df = pd.read_csv(uploaded_file)
            text = df.to_string(index=False)
        except Exception:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
    elif ext in {".xls", ".xlsx"}:
        try:
            df = pd.read_excel(uploaded_file)
            text = df.to_string(index=False)
        except Exception:
            text = ""
    else:
        text = ""

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[troncato]"
    return text

def _md_inline_to_rl(text: str) -> str:
    """Converte formattazione inline markdown in tag ReportLab (XML-safe)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def generate_pdf_report(title: str, content: str):
    """
    Genera un PDF nella cartella /reports convertendo markdown in stili ReportLab.
    Ritorna un dict con file_path.
    """
    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/report_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        filename,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm
    )
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name="ReportH1", parent=styles["Title"], fontSize=20, spaceAfter=14, spaceBefore=0))
    styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontSize=14, spaceAfter=8, spaceBefore=16))
    styles.add(ParagraphStyle(name="ReportH3", parent=styles["Heading3"], fontSize=12, spaceAfter=6, spaceBefore=10))
    styles.add(ParagraphStyle(name="ReportBullet", parent=styles["BodyText"], leftIndent=18, spaceBefore=2, spaceAfter=2))

    story = []

    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph(f"Generato il: {date_str}", styles["Italic"]))
    story.append(Spacer(1, 16))

    for line in content.split("\n"):
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 8))
        elif stripped.startswith("### "):
            story.append(Paragraph(_md_inline_to_rl(stripped[4:]), styles["ReportH3"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(_md_inline_to_rl(stripped[3:]), styles["ReportH2"]))
        elif stripped.startswith("# "):
            story.append(Paragraph(_md_inline_to_rl(stripped[2:]), styles["ReportH1"]))
        elif stripped.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {_md_inline_to_rl(stripped[2:])}", styles["ReportBullet"]))
        elif re.match(r"^\d+\.\s", stripped):
            text = re.sub(r"^\d+\.\s+", "", stripped)
            story.append(Paragraph(f"• {_md_inline_to_rl(text)}", styles["ReportBullet"]))
        else:
            story.append(Paragraph(_md_inline_to_rl(stripped), styles["BodyText"]))
            story.append(Spacer(1, 4))

    doc.build(story)

    return {
        "status": "success",
        "file_path": filename,
        "message": f"PDF creato: {filename}"
    }

def run_ansible_playbook(playbook_path: str, inventory_path: str = "inventory.ini"):
    cmd = [
        "ansible-playbook --check",
        "-i", inventory_path,
        playbook_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode
    }

def save_ansible_playbook(name: str, content: str):
    os.makedirs("ansible_memory", exist_ok=True)

    path = f"ansible_memory/{name}.yml"

    with open(path, "w") as f:
        f.write(content)

    return {"saved_path": path}

def get_ansible_playbooks():
    import os

    if not os.path.exists("ansible_memory"):
        return []

    return os.listdir("ansible_memory")

def extract_city_from_query(query: str) -> str:
    """Estrai una città dalla richiesta dell'utente per meteo_tool."""
    if not query:
        return ""

    query = query.lower().strip()
    query = re.sub(r"[\?\.!,:]", "", query)

    # Cerca pattern come "a Milano", "a Roma", "in Firenze", "per Napoli", "di Torino"
    match = re.search(
        r"\b(?:a|in|per|di)\s+([a-zàèéìíòóùúçœæ]+(?:[\s\-][a-zàèéìíòóùúçœæ]+)*)",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: prendi l'ultima parte utile della frase se ci sono parole chiave del meteo.
    weather_keywords = r"\b(meteo|tempo|previsioni|pioggia|sole|neve|vento|temperatura|umidità)\b"
    if re.search(weather_keywords, query):
        tokens = query.split()
        stopwords = {
            "oggi", "domani", "stasera", "mattina", "sera", "notte",
            "nel", "nella", "nelle", "sul", "sulla", "sulle",
            "del", "della", "dello", "dei", "degli", "delle",
            "per", "a", "in", "di", "che", "come",
            "è", "c", "e", "ma", "o", "se"
        }
        while tokens and tokens[-1] in stopwords:
            tokens.pop()
        if tokens:
            return tokens[-1].strip()

    return ""

def extract_path_with_llm(query: str, current_file: str = "") -> str:
    """Usa Llama per capire il file richiesto, sfruttando la memoria del grafo."""
    
    prompt_template = """
    Sei un assistente software esperto. Il tuo compito è capire quale FILE o CARTELLA l'utente vuole analizzare, leggere o rifattorizzare.

    Contesto attuale:
    L'ultimo file su cui l'utente ha lavorato o che ha aperto è: "{current_file}"

    Regole rigide:
    1. Rispondi SOLO con un oggetto JSON valido: {{"path": "nome_del_file_o_cartella"}}
    2. Se l'utente si riferisce a "quel file", "questo file", "rifattorizzalo" o formule simili, e il Contesto attuale NON è vuoto, restituisci il nome del file presente nel contesto attuale.
    3. Se l'utente non specifica alcun file e non si riferisce a quello precedente, restituisci "".
    4. Non aggiungere altro testo.

    Richiesta utente: "{query}"

    Risposta JSON:"""

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    try:
        raw_response = chain.invoke({
            "query": query, 
            "current_file": current_file
        }).strip()
        
        # Pulizia blocchi markdown ```json
        if "```" in raw_response:
            raw_response = raw_response.split("```")[1]
            if raw_response.startswith("json"):
                raw_response = raw_response[4:]
                
        data = json.loads(raw_response.strip())
        candidate_path = data.get("path", "").strip()
        
        if candidate_path.startswith("/") or ".." in candidate_path:
            return ""
            
        return candidate_path
    except Exception as e:
        print(f"Errore nell'estrazione LLM: {e}")
        return ""


def extract_code_block(text):
    """
    Parsa una risposta markdown ed estrae le parti attorno al code block.
    Ritorna (pre, lang, code, post, chiuso):
        - pre:    testo prima del ```
        - lang:   linguaggio dichiarato (es. "python")
        - code:   contenuto del blocco senza backtick
        - post:   testo dopo il blocco chiuso
        - chiuso: True se il blocco ``` è già chiuso
    """
    idx_open = text.find("```")
    if idx_open == -1:
        return text, None, None, "", False

    pre = text[:idx_open]
    rest = text[idx_open + 3:]

    nl = rest.find("\n")
    if nl == -1:
        return pre, rest.strip() or "plaintext", "", "", False

    lang = rest[:nl].strip() or "plaintext"
    rest = rest[nl + 1:]

    idx_close = rest.find("```")
    if idx_close == -1:
        return pre, lang, rest, "", False

    code = rest[:idx_close]
    post = rest[idx_close + 3:]
    if post.startswith("\n"):
        post = post[1:]
    return pre, lang, code, post, True

def clean_code_content(code: str) -> str:
    """Rimuove dal codice estratto le righe di metadati generate dall'LLM."""
    import re
    cleaned = []
    for line in code.split("\n"):
        s = line.strip()
        if re.match(r"^-{3,}", s):
            continue
        if re.match(r"^📚\s*Fonti:", s):
            continue
        if re.match(r"^Contenuto di\s+\S+", s):
            continue
        if s.startswith("|"):                          # riga tabella markdown
            continue
        if re.match(r"^[-|: ]+$", s) and len(s) > 2: # separatore tabella |---|
            continue
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)

def clean_post_content(text: str) -> str:
    """
    Filtra il testo dopo il code block: mantiene solo righe utili
    (fonti, testo normale) e scarta tabelle markdown e separatori.
    """
    import re
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|"):           # riga di tabella markdown
            continue
        if re.match(r"^[-|: ]+$", s) and len(s) > 2:  # separatore tabella |---|
            continue
        if re.match(r"^-{3,}", s):      # separatori ---
            continue
        if re.match(r"^Contenuto di\s+\S+", s):
            continue
        kept.append(line)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)