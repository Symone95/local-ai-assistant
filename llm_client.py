"""
Client LLM condiviso da tutti i componenti.

Ollama tiene in memoria una configurazione per modello: se due parti del progetto lo chiamano
con opzioni diverse (per esempio `ChatOllama(num_ctx=4096)` da una parte e `ollama.chat` con i
default dall'altra), il modello viene scaricato e ricaricato a ogni alternanza, con 1-2 secondi
di attesa ogni volta. Passando tutti da qui la configurazione e' una sola e il modello resta caldo.
"""
import os

from langchain_ollama import ChatOllama

# llama3.2:3b, misurato contro qwen2.5-coder:3b sugli stessi casi: italiano corretto
# ("Sto bene, grazie! Come te?" invece di "Come possoi aiutarti"), risposte piu' adatte alla
# voce (arrotonda "circa 20 gradi" invece di scandire 19.86°C) e routing piu' accurato
# (7/8 contro 6/8) a parita' di dimensione e con latenza inferiore.
MODELLO = os.getenv("LLM_MODEL", "llama3.2:3b")
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))
KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "30m")

llm = ChatOllama(model=MODELLO, num_ctx=NUM_CTX, keep_alive=KEEP_ALIVE)


def completa(prompt: str) -> str:
    """Risposta completa come stringa, per chi non ha bisogno dello streaming."""
    return llm.invoke(prompt).content
