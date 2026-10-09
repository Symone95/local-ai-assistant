"""Trascrizione locale con faster-whisper (piu' rapido di openai-whisper a parita' di modello)."""
import numpy as np

from voice import config

_modello = None


def get_modello():
    """Carica il modello una volta sola: il primo caricamento costa qualche secondo."""
    global _modello
    if _modello is None:
        from faster_whisper import WhisperModel
        print(f"🔤 Carico faster-whisper '{config.MODELLO_STT}' ({config.COMPUTE_TYPE})...")
        _modello = WhisperModel(config.MODELLO_STT, compute_type=config.COMPUTE_TYPE)
        print("🔤 Modello pronto.")
    return _modello


def trascrivi(audio: np.ndarray) -> str:
    """Trascrive un array float32 mono a 16 kHz. Ritorna stringa vuota se non c'e' parlato."""
    if audio is None or len(audio) == 0:
        return ""

    segmenti, _ = get_modello().transcribe(
        audio.astype(np.float32),
        language=config.LINGUA,
        vad_filter=True,        # scarta i silenzi residui
        beam_size=1,            # beam=1: piu' veloce, sufficiente per frasi parlate
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segmenti).strip()
