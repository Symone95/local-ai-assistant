import hashlib
import re
from chromadb.api.models.Collection import Collection
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import pandas as pd
from pdf_loader import load_pdf

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


def generate_pdf_report(title: str, content: str):
    """
    Genera un PDF nella cartella /reports e ritorna il path del file.
    """

    # crea cartella se non esiste
    os.makedirs("reports", exist_ok=True)

    # nome file con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/report_{timestamp}.pdf"

    # crea documento
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    story = []

    # Titolo
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))

    # Data generazione
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph(f"Generato il: {date_str}", styles["Italic"]))
    story.append(Spacer(1, 20))

    # Contenuto: splittiamo per paragrafi
    paragraphs = content.split("\n")

    for p in paragraphs:
        if p.strip() == "":
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 12))

    # build pdf
    doc.build(story)

    return {
        "status": "success",
        "file_path": filename,
        "message": f"PDF creato: {filename}"
    }


import subprocess

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