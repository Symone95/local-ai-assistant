"""
Riproduzione audio continua e interrompibile.

Le frasi arrivano una alla volta, ma devono suonare come un discorso unico. Per questo lo stream
di uscita resta **aperto** fra una frase e l'altra e viene alimentato da un buffer: aprire e
chiudere il dispositivo a ogni frase introduceva pause e scatti. Il buffer serve anche al
barge-in, che deve poter zittire l'assistente all'istante svuotandolo.
"""
import threading
import time

import numpy as np
import sounddevice as sd

from voice import config


class Player:
    def __init__(self):
        self._stream = None
        self._sample_rate = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._in_riproduzione = threading.Event()

    @property
    def sta_parlando(self) -> bool:
        return self._in_riproduzione.is_set()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"⚠️  uscita audio: {status}")

        with self._lock:
            disponibili = min(frames, len(self._buffer))
            if disponibili:
                outdata[:disponibili, 0] = self._buffer[:disponibili]
                self._buffer = self._buffer[disponibili:]

        if disponibili < frames:      # buffer esaurito: silenzio, senza chiudere lo stream
            outdata[disponibili:, 0] = 0

    def _assicura_stream(self, sample_rate):
        if self._stream is not None and self._sample_rate == sample_rate:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

        self._sample_rate = sample_rate
        # Blocchi grandi e latenza alta: il callback e' codice Python e, mentre il modello
        # genera e la sintesi decodifica, puo' non essere schedulato per oltre 200ms. Con il
        # buffer di default (46ms) il dispositivo resta a secco e l'audio singhiozza; con
        # 2048 frame (131ms) gli stalli misurati sotto carico passano da 7 a zero.
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=config.BLOCCO_USCITA,
            latency=config.LATENZA_USCITA,
            callback=self._callback,
        )
        self._stream.start()

    def prepara(self, sample_rate=24000):
        """
        Apre il dispositivo audio in anticipo.

        L'apertura costa quasi due secondi su macOS: farla alla prima frase ritarderebbe proprio
        la risposta che deve sembrare immediata. 24 kHz e' il campionamento di edge-tts.
        """
        try:
            self._assicura_stream(sample_rate)
        except Exception as e:
            print(f"⚠️  impossibile preparare l'uscita audio: {e}")

    def riproduci(self, audio, sample_rate, controllo_interruzione=None) -> bool:
        """
        Accoda una frase e attende che sia stata riprodotta.

        `controllo_interruzione` viene interrogata durante l'attesa: se ritorna True il buffer
        viene svuotato e l'audio si interrompe subito. Ritorna True se la frase è finita da sola.
        """
        if audio is None or len(audio) == 0:
            return True

        self._stop.clear()
        self._in_riproduzione.set()
        try:
            self._assicura_stream(sample_rate)
            with self._lock:
                self._buffer = np.concatenate([self._buffer, np.asarray(audio, dtype=np.float32)])

            while True:
                with self._lock:
                    rimasto = len(self._buffer)

                if rimasto == 0:
                    # Il buffer è vuoto ma l'ultimo blocco è ancora nelle mani del dispositivo
                    time.sleep(self._stream.latency or 0.05)
                    return True

                if self._stop.is_set() or (controllo_interruzione and controllo_interruzione()):
                    with self._lock:
                        self._buffer = np.zeros(0, dtype=np.float32)
                    return False

                time.sleep(0.01)
        finally:
            self._in_riproduzione.clear()

    def stop(self):
        self._stop.set()
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)

    def chiudi(self):
        self.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
