import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from datetime import datetime

MODEL_SIZE = "medium"
SAMPLE_RATE = 16000
CHANNELS = 1
DEVICE_INDEX = 0  # microfono MacBook Air

audio_queue = queue.Queue()

print("Carico modello Whisper...")
model = WhisperModel(MODEL_SIZE, compute_type="int8")
print("Modello pronto!")

file = open("trascrizione.txt", "a")

def audio_callback(indata, frames, time, status):
    if status:
        print("Status:", status)

    # 🔴 conversione fondamentale per Whisper
    audio = indata[:, 0]              # mono
    audio = audio.astype(np.float32) # float32
    audio_queue.put(audio)

def trascrivi():
    audio_buffer = np.array([], dtype=np.float32)

    while True:
        data = audio_queue.get()
        audio_buffer = np.concatenate((audio_buffer, data))

        # trascrive ogni 2 secondi (più reattivo)
        if len(audio_buffer) > SAMPLE_RATE * 2:
            chunk = audio_buffer.copy()
            audio_buffer = np.array([], dtype=np.float32)

            print("🟡 Sto trascrivendo...")

            segments, _ = model.transcribe(
                chunk,
                language="it",
                vad_filter=True,
                task="transcribe",
                beam_size=5,
            )

            for segment in segments:
                testo = segment.text.strip()
                if testo:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    riga = f"[{timestamp}] {testo}"
                    print("📝", riga)
                    file.write(riga + "\n")
                    file.flush()

threading.Thread(target=trascrivi, daemon=True).start()

print("🎤 Sto ascoltando... parla forte e chiaro!")

with sd.InputStream(
        device=DEVICE_INDEX,
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="float32",
        channels=CHANNELS,
        callback=audio_callback):
    while True:
        pass