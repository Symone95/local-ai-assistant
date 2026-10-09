"""Sintesi vocale con edge-tts, decodificata in memoria per poterla riprodurre e fermare subito."""
import io
import re

import edge_tts
import numpy as np
import soundfile as sf

from voice import config

# Il parlato non deve leggere il codice: in chat resta visibile, a voce si annuncia e basta.
_BLOCCO_CODICE = re.compile(r"```.*?```", re.DOTALL)
_BLOCCO_CODICE_APERTO = re.compile(r"```.*", re.DOTALL)
_MARKDOWN = re.compile(r"[*_`#>|]+")
_VOCE_ELENCO = re.compile(r"\n\s*[-*•]\s*|\n+")
# La coda "Fonti: [file.pdf] (pagina 3)" ha senso scritta, non letta ad alta voce: il nodo di
# risposta la impone sempre, quindi qui si toglie invece di sperare che il modello ubbidisca.
_FONTI = re.compile(r"\n*\s*(fonti|fonte|sources?)\s*:.*$", re.IGNORECASE | re.DOTALL)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
_SPAZI = re.compile(r"\s+")
_PUNTEGGIATURA_DOPPIA = re.compile(r"([:;.!?])\s*,")


def pulisci_per_voce(testo: str) -> str:
    """Toglie da un testo cio' che non ha senso pronunciare."""
    if not testo:
        return ""

    testo = _BLOCCO_CODICE.sub(" Ti ho scritto il codice nella chat. ", testo)
    testo = _BLOCCO_CODICE_APERTO.sub(" Ti ho scritto il codice nella chat. ", testo)

    # Gli elenchi puntati, letti ad alta voce, diventano una sequenza di frammenti staccati.
    # Il modello ogni tanto li produce comunque, nonostante le istruzioni: qui si trasformano
    # in un discorso continuo, senza dipendere da quanto il modello sia ubbidiente.
    testo = _FONTI.sub(" ", testo)
    testo = _VOCE_ELENCO.sub(", ", testo)
    testo = _MARKDOWN.sub(" ", testo)
    testo = _EMOJI.sub(" ", testo)
    testo = _SPAZI.sub(" ", testo)
    testo = _PUNTEGGIATURA_DOPPIA.sub(r"\1", testo)   # evita "dettagli:, Umidità"
    return testo.strip()


async def sintetizza(testo: str):
    """
    Genera l'audio di una frase e lo restituisce come (array float32, sample rate).
    Niente file temporanei: l'mp3 di edge-tts viene decodificato in memoria.
    """
    testo = pulisci_per_voce(testo)
    if not testo:
        return None, None

    buffer = bytearray()
    async for chunk in edge_tts.Communicate(testo, voice=config.VOCE_TTS).stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])

    if not buffer:
        return None, None

    audio, sr = sf.read(io.BytesIO(bytes(buffer)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return taglia_silenzio(audio, sr), sr


def taglia_silenzio(audio, sr, soglia_relativa=0.01, margine_testa_ms=None, margine_coda_ms=None):
    """
    Rimuove il silenzio a inizio e fine frase.

    edge-tts restituisce ogni frase con ~190ms di silenzio davanti e ~870ms dietro: pronunciando
    una frase alla volta quei silenzi si sommano in pause di oltre un secondo. Non vanno pero'
    azzerati: in coda ne serve quanto basta a sentire lo stacco fra due frasi, altrimenti il
    parlato suona appiccicato. In testa basta un margine per non tagliare le consonanti iniziali.
    """
    margine_testa_ms = config.MARGINE_TESTA_MS if margine_testa_ms is None else margine_testa_ms
    margine_coda_ms = config.MARGINE_CODA_MS if margine_coda_ms is None else margine_coda_ms
    if audio is None or len(audio) == 0:
        return audio

    picco = float(np.max(np.abs(audio)))
    if picco <= 0:
        return audio

    sopra = np.where(np.abs(audio) > soglia_relativa * picco)[0]
    if len(sopra) == 0:
        return audio

    inizio = max(0, int(sopra[0]) - int(sr * margine_testa_ms / 1000))
    fine = min(len(audio), int(sopra[-1]) + int(sr * margine_coda_ms / 1000))
    return audio[inizio:fine]
