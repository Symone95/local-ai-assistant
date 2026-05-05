from faster_whisper import WhisperModel
import tempfile
import os

# modello leggero per realtime CPU
model = WhisperModel(
    "base",          # puoi usare "small" se hai CPU potente
    device="cpu",
    compute_type="int8"
)

def transcribe_chunk(audio_bytes):
    """
    Trascrive piccoli chunk audio velocemente.
    """

    # salva chunk temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    segments, _ = model.transcribe(
        tmp_path,
        beam_size=1,
        vad_filter=True,          # 🔥 voice activity detection
        vad_parameters=dict(min_silence_duration_ms=300)
    )

    text = " ".join([seg.text for seg in segments])
    os.remove(tmp_path)
    return text