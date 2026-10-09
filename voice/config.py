"""Parametri della modalita' vocale, tutti sovrascrivibili da .env."""
import os
import sys

# --- Audio ---
SAMPLE_RATE = 16000          # richiesto da Whisper
FRAME_MS = 30                # granularita' dell'analisi di energia
CANALI = 1

# --- Rilevamento del parlato (VAD a energia) ---
# Soglia = rumore di fondo misurato all'avvio x questo fattore.
VAD_FATTORE = float(os.getenv("VOICE_VAD_FACTOR", "2.0"))
# Silenzio che chiude il turno: sotto questa soglia si smette di ascoltare e si trascrive.
SILENZIO_FINE_TURNO_MS = int(os.getenv("VOICE_SILENCE_MS", "600"))
# Sotto questa durata si considera un rumore, non una frase.
DURATA_MINIMA_MS = int(os.getenv("VOICE_MIN_SPEECH_MS", "400"))
# Oltre questa durata si chiude comunque il turno, per non restare in ascolto all'infinito.
DURATA_MASSIMA_MS = int(os.getenv("VOICE_MAX_SPEECH_MS", "20000"))

# --- Trascrizione ---
# Su audio reale (microfono in stanza) "small" e' sia piu' accurato sia piu' veloce di "base":
# base su voce rumorosa produce testo sbagliato e ci mette di piu' a convergere.
MODELLO_STT = os.getenv("VOICE_STT_MODEL", "small")   # tiny/base/small/medium
COMPUTE_TYPE = os.getenv("VOICE_STT_COMPUTE", "int8")
LINGUA = os.getenv("VOICE_LANG", "it")

# --- Sintesi ---
VOCE_TTS = os.getenv("VOICE_TTS_VOICE", "it-IT-DiegoNeural")
# Lunghezza minima di un blocco prima di mandarlo alla sintesi. Solo la prima frase e' corta
# (serve a partire subito): dalla seconda in poi si accumula, perche' ogni blocco e' una
# sintesi a se' con una sua intonazione, e spezzettare troppo fa suonare il parlato a scatti.
FRASE_MINIMA_CARATTERI = int(os.getenv("VOICE_MIN_SENTENCE", "140"))

# --- Barge-in: interrompere l'assistente parlandogli sopra ---
# Spento di default: senza cancellazione d'eco la voce dell'assistente rientra dal microfono
# e si interrompe da sola, spezzando la risposta. Con le cuffie si puo' accendere.
BARGE_IN = os.getenv("VOICE_BARGE_IN", "0") == "1"
# Soglia piu' alta mentre l'assistente parla: senza cancellazione d'eco, la sua stessa voce
# rientra dal microfono. Con le cuffie si puo' abbassare, con gli altoparlanti va alzata.
BARGE_IN_FATTORE = float(os.getenv("VOICE_BARGE_IN_FACTOR", "8.0"))
BARGE_IN_MS = int(os.getenv("VOICE_BARGE_IN_MS", "300"))

# Pausa fra la fine della risposta e il nuovo ascolto: evita che l'assistente senta
# la coda della propria voce dagli altoparlanti e si risponda da solo.
PAUSA_DOPO_RISPOSTA_MS = int(os.getenv("VOICE_COOLDOWN_MS", "400"))

# Silenzio lasciato ai bordi di ogni blocco sintetizzato. In testa poco, in coda quanto basta a
# sentire lo stacco fra due frasi: azzerarlo le fa suonare appiccicate e innaturali.
MARGINE_TESTA_MS = int(os.getenv("VOICE_HEAD_MS", "40"))
MARGINE_CODA_MS = int(os.getenv("VOICE_TAIL_MS", "200"))

# Audio conservato PRIMA che la voce superi la soglia: senza, l'attacco della prima parola
# viene perso e la trascrizione ne risente parecchio.
PREROLL_MS = int(os.getenv("VOICE_PREROLL_MS", "300"))

# Dimensione del blocco e latenza dello stream di uscita. Valori alti danno al callback Python
# il margine per non restare indietro quando CPU e GIL sono occupati da modello e sintesi.
BLOCCO_USCITA = int(os.getenv("VOICE_OUT_BLOCKSIZE", "2048"))
LATENZA_USCITA = os.getenv("VOICE_OUT_LATENCY", "high")

# Quanto testo al massimo viene pronunciato. Il modello ignora spesso la richiesta di brevita'
# e produce risposte da venti secondi di parlato: qui si taglia a voce, mentre il testo completo
# resta visibile nella chat.
MAX_CARATTERI_PARLATI = int(os.getenv("VOICE_MAX_SPOKEN_CHARS", "240"))

# Motore di sintesi: "edge" (edge-tts, voce naturale, passa dal player Python) oppure
# "say" (voce di sistema macOS, meno naturale ma riprodotta da un processo esterno: immune
# agli scatti dovuti al callback audio affamato).
TTS_BACKEND = os.getenv("VOICE_TTS_BACKEND", "say" if sys.platform == "darwin" else "edge")
VOCE_SISTEMA = os.getenv("VOICE_SAY_VOICE", "Alice")

# Pavimento assoluto della soglia: sotto questo valore si rischia di scambiare il respiro o il
# fruscio per parlato. Abbassarlo rende l'ascolto piu' sensibile a voce bassa o microfono lontano.
SOGLIA_MINIMA = float(os.getenv("VOICE_MIN_THRESHOLD", "0.0035"))

# Ogni quanto ricalcolare il rumore di fondo mentre si ascolta. La stanza cambia (ventola che
# parte, finestra aperta) e una soglia fissa misurata all'avvio smette di andare bene.
RICALIBRA_OGNI_S = float(os.getenv("VOICE_RECALIBRATE_S", "4"))

# Se per questo tempo non arriva un solo frame dal microfono, lo stream viene riaperto: su macOS
# puo' morire in silenzio quando un altro processo riconfigura il dispositivo audio.
WATCHDOG_MIC_S = float(os.getenv("VOICE_MIC_WATCHDOG_S", "4"))
