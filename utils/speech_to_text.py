import whisper
import tempfile
import os

model = whisper.load_model("base")  # puoi usare tiny/base/small

def transcribe_audio(audio_bytes):
    # Salviamo temporaneamente il file audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    os.remove(tmp_path)

    return result["text"]