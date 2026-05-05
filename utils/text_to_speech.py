import edge_tts
import uuid

async def generate_tts(text):
    filename = f"{uuid.uuid4()}.mp3"
    communicate = edge_tts.Communicate(text, voice="it-IT-DiegoNeural")
    await communicate.save(filename)
    return filename