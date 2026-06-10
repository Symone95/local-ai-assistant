import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import logging
from datetime import datetime

from pdf_loader import chunk_text, load_pdf_paginated
from utils.general import get_file_hash, get_full_document
from langchain_core.messages import HumanMessage, AIMessage

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Embedding model (caricato una volta sola)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma persistente
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

# TOOLS
def get_files_with_upload_date_tool():
    data = collection.get()

    if not data["metadatas"]:
        return {}

    file_dates = {}

    for m in data["metadatas"]:
        file = m["file"]
        date = m.get("uploaded_at")

        # prendiamo la prima occorrenza (tutti i chunk hanno la stessa)
        if file not in file_dates and date:
            file_dates[file] = date

    return file_dates

def get_files_in_db_tool():
    data = collection.get(include=["metadatas"])

    files = {}
    for meta in data["metadatas"]:
        files[meta["file_hash"]] = meta["file"]

    # lista pulita
    return list(files.values())


def reset_database():
    global collection
    client.delete_collection("docs")
    collection = client.get_or_create_collection("docs")

def add_documents(uploaded_files, file_hash_map):
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_hash = file_hash_map[file_name]

        # 1. estrazione + cleaning globale
        full_text = load_pdf_paginated(uploaded_file)

        # 2. chunking centralizzato
        for page_num, text in full_text:
            chunks = chunk_text(text, chunk_size=1200, overlap=200)

            # 3. embeddings
            embeddings = embed_model.encode(chunks)

            for i, chunk in enumerate(chunks):
                # chunk_id = str(uuid.uuid4())
                chunk_id = get_file_hash((file_hash + str(page_num) + chunk).encode())

                all_chunks.append(chunk)
                all_embeddings.append(embeddings[i])
                all_ids.append(chunk_id)

                all_metadata.append({
                    "id": chunk_id,
                    "file": file_name,
                    "file_hash": file_hash,
                    "page": page_num,
                    "chunk": chunk,
                    "uploaded_at": datetime.now().isoformat()
                })

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

def merge_chunks_by_file(documents, metadatas):
    """
    Unisce i chunk dello stesso file,
    rimuove duplicati e ordina per pagina.
    """

    files = {}

    for doc, meta in zip(documents, metadatas):
        file = meta["file"]
        page = meta["page"]

        if file not in files:
            files[file] = {}

        # uso dict per evitare chunk duplicati
        files[file][(page, doc)] = doc

    merged_docs = []

    for file, chunks_dict in files.items():
        # (page, doc)
        chunks = list(chunks_dict.keys())

        # ordina per pagina
        chunks_sorted = sorted(chunks, key=lambda x: x[0])

        full_text = "\n\n".join([c[1] for c in chunks_sorted])

        merged_docs.append({
            "file": file,
            "text": full_text
        })

    return merged_docs

def search_context(query, selected_doc=None, k_chunks=20):
    """
    Retrieval document-centric:
    - cerca chunk semanticamente simili
    - raggruppa per file
    - ricostruisce i documenti unendo i chunk
    - restituisce CONTEXT numerato stile Perplexity
    """

    # 1️⃣ Embedding query
    query_embedding = embed_model.encode([query])[0]

    # 2️⃣ Query Chroma
    if selected_doc:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_chunks,
            where={"file": selected_doc}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k_chunks
        )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 🔴 Se DB vuoto
    if not documents:
        return "", []

    # 3️⃣ GROUP + SORT + MERGE CHUNKS → DOCUMENTI
    merged_docs = merge_chunks_by_file(documents, metadatas)
    texts = [d["text"] for d in merged_docs]

    # 🔥 RERANKING
    texts = rerank_chunks(query, texts)

    merged_docs = [{"file": "merged", "text": t} for t in texts]

    # 4️⃣ COSTRUZIONE CONTEXT + SOURCES
    context_blocks = []
    structured_sources = []

    for i, doc in enumerate(merged_docs):
        doc_index = i + 1
        file_name = doc["file"]
        full_text = doc["text"]

        # snippet per preview fonti
        snippet = full_text[:300].replace("\n", " ")

        structured_sources.append({
            "index": doc_index,
            "file": file_name,
            "page": "multi",
            "text": snippet
        })

        context_blocks.append(
            f"[{doc_index}] DOCUMENTO: {file_name}\n{full_text}"
        )

    context = "\n\n".join(context_blocks)

    return context, structured_sources

def build_chat_history(messages, max_turns=6):
    """
    Costruisce history compatta per il retriever usando oggetti LangChain.
    """
    history = []

    # Prendiamo gli ultimi N messaggi
    recent = messages[-max_turns:] if messages else []

    for msg in recent:
        # Identifichiamo il ruolo dall'oggetto LangChain
        if isinstance(msg, HumanMessage) or msg.type == "human":
            role = "Utente"
        elif isinstance(msg, AIMessage) or msg.type == "ai":
            role = "Assistente"
        else:
            role = "Sistema"

        history.append(f"{role}: {msg.content.strip()}")

    return "\n".join(history)

def query_mentions_temp_file(query: str) -> bool:
    keywords = [
        "file caricato", "questo file", "file temporaneo", "file allegato",
        "nel file", "contenuto del file", "cosa fa il file", "descrivi il file",
        "cosa contiene il file", "analizza il file", "documento"
    ]
    lower_query = query.lower()
    return any(keyword in lower_query for keyword in keywords)


def conversational_search_tool(query, messages, selected_doc=None, temp_context=None):
    """
    Pipeline completa conversational RAG
    """

    # 1. build memory
    chat_history = build_chat_history(messages)

    # 2. rewrite query 🔥
    standalone_query = rewrite_query_with_memory(query, chat_history)

    print("🔵 Original query:", query)
    print("🟢 Rewritten query:", standalone_query)

    # 3. temporary file search
    if temp_context and query_mentions_temp_file(query):
        if isinstance(temp_context, dict):
            temp_name = temp_context.get("name", "file_temporaneo")
            temp_text = temp_context.get("content", "")
        else:
            temp_name = "file_temporaneo"
            temp_text = str(temp_context)

        if temp_text:
            context = f"[1] DOCUMENTO: {temp_name}\n{temp_text}"
            sources = [{
                "index": 1,
                "file": temp_name,
                "page": "n/a",
                "text": temp_text[:300].replace("\n", " ")
            }]
            return context, sources

    # 4. retrieval normale
    context, sources = search_context(standalone_query, selected_doc)

    return context, sources


def summarize_document_tool(file_name: str):
    document_text = get_full_document(file_name, collection)

    if not document_text:
        return "Documento non trovato nel database."

    prompt = f"""
Sei un assistente esperto nell'analisi di documenti.

Fai un riassunto chiaro e strutturato del documento.

Regole:
- Scrivi in italiano
- Usa bullet points
- Evidenzia le informazioni importanti
- Non inventare nulla

Documento:
{document_text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def extract_filename_from_query(query: str):
    files = collection.get()["metadatas"]
    file_names = list(set([m["file"] for m in files]))

    query_lower = query.lower()

    for name in file_names:
        if name.lower() in query_lower:
            return name

    # fallback → primo documento
    return file_names[0] if file_names else None

# --- FUNZIONI LLM ---
def ask_llm(query, context):
    """
    Funzione per chiamare llm one shot e avere una risposta caricata tutta insieme, se vuoi una risposta caricata parola per parola utilizza la funzione `stream_llm_answer`
    :param query:
    :param context:
    :return:
    """
    prompt = f"""
Usa il contesto per rispondere.

Contesto:
{context}

Domanda: {query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def rewrite_query_with_memory(query, chat_history):
    """
    Riscrive la domanda rendendola standalone.
    Fondamentale per conversational retrieval.
    """

    if not chat_history:
        return query

    prompt = f"""
Sei un sistema che riscrive domande per un motore di ricerca.

Obiettivo:
Trasforma la domanda in una domanda standalone completa, usando il contesto della conversazione se necessario e mantenendo la stessa nazionalità della lingua usata dall'utente nella risposta.

NON rispondere alla domanda.
Produci SOLO la domanda riscritta mantenendo la stessa nazionalità della lingua usata dall'utente.

Conversazione:
{chat_history}

Domanda attuale:
{query}

Domanda standalone:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    rewritten = response["message"]["content"].strip()

    # fallback sicurezza
    if len(rewritten) < 5:
        return query

    return rewritten


def generate_report_content(query: str, messages=None):
    """
    Fa scrivere all'LLM il contenuto del report in modo strutturato.
    """

    prompt = f"""
Sei un assistente che scrive report professionali.

Scrivi un report ben strutturato in italiano con:
- Titolo
- Introduzione
- Sezioni con sottotitoli
- Conclusione

Argomento del report:
{query}

Produci SOLO il testo del report in markdown.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    report_text = response["message"]["content"]
    return report_text


def rerank_chunks(query, chunks):
    scored = []

    for c in chunks:
        prompt = f"""
Valuta quanto questo testo risponde alla domanda.

Domanda: {query}

Testo:
{c}

Rispondi SOLO con un numero da 0 a 10.
"""

        res = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            score = float(res["message"]["content"].strip())
        except:
            score = 0

        scored.append((score, c))

    # ordina per rilevanza
    scored.sort(key=lambda x: x[0], reverse=True)
    print("RERANK FATTO")

    return [c for score, c in scored[:5]]


def generate_ansible_playbook(query: str):
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