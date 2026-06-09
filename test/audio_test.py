import sounddevice as sd
import numpy as np
import queue
import time
from faster_whisper import WhisperModel

# ===== CONFIG =====
SAMPLE_RATE = 16000
CHUNK_SECONDS = .3          # ogni quanti secondi trascrivere
MODEL_SIZE = "tiny"        # tiny / base / small

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, compute_type="int8")
print("Model loaded!")

audio_queue = queue.Queue()
recording = True


def audio_callback(indata, frames, time_info, status):
    """Viene chiamata continuamente dal microfono"""
    if status:
        print(status)
    audio_queue.put(indata.copy())


print("🎤 Sto ascoltando... premi CTRL+C per fermare")

# file transcript
transcript_file = open("transcript.txt", "w", encoding="utf-8")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback,
):
    audio_buffer = np.empty((0, 1), dtype="float32")
    last_process_time = time.time()

    try:
        while True:
            # raccogli audio dalla coda
            while not audio_queue.empty():
                data = audio_queue.get()
                audio_buffer = np.concatenate((audio_buffer, data))

            # ogni X secondi → trascrivi
            if time.time() - last_process_time > CHUNK_SECONDS:

                if len(audio_buffer) > SAMPLE_RATE:  # almeno 1 sec audio
                    audio_np = audio_buffer.flatten()

                    segments, _ = model.transcribe(audio_np, language="it")

                    text_chunk = ""
                    for segment in segments:
                        text_chunk += segment.text + " "

                    text_chunk = text_chunk.strip()

                    if text_chunk:
                        print("📝", text_chunk)

                        transcript_file.write(text_chunk + " ")
                        transcript_file.flush()

                    # svuota buffer dopo trascrizione
                    audio_buffer = np.empty((0, 1), dtype="float32")

                last_process_time = time.time()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stop registrazione")

transcript_file.close()