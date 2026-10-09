"""
Sintesi vocale con la voce di sistema di macOS (`say`).

Alternativa a edge-tts, pensata per quando la riproduzione risulta a scatti: `say` parla in un
processo separato, quindi non passa dal callback audio in Python (che sotto carico puo' restare
indietro), non decodifica mp3 e non usa la rete. In cambio la voce e' meno naturale.
"""
import shutil
import subprocess
import threading

from voice import config
from voice.tts import pulisci_per_voce

_processo = None
_lock = threading.Lock()


def disponibile() -> bool:
    return shutil.which("say") is not None


def parla(testo: str) -> bool:
    """Pronuncia il testo e attende la fine. Ritorna False se e' stato interrotto."""
    testo = pulisci_per_voce(testo)
    if not testo:
        return True

    global _processo
    with _lock:
        _processo = subprocess.Popen(
            ["say", "-v", config.VOCE_SISTEMA, testo],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        corrente = _processo

    codice = corrente.wait()
    with _lock:
        if _processo is corrente:
            _processo = None

    return codice == 0


def zittisci():
    """Interrompe subito la voce in corso (serve a barge-in e stop)."""
    with _lock:
        if _processo is not None and _processo.poll() is None:
            _processo.terminate()
