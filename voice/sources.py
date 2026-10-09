"""Sorgenti audio: il microfono in esercizio, un array in memoria nei test."""
import queue
import time

import numpy as np
import sounddevice as sd

from voice import config

DIMENSIONE_FRAME = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)


class MicSource:
    """Microfono di sistema: i frame arrivano in una coda dal callback di sounddevice."""

    def __init__(self, device=None):
        self._coda = queue.Queue()
        self._stream = None
        self._device = device

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️  audio: {status}")
        self._coda.put(indata[:, 0].astype(np.float32).copy())

    def __enter__(self):
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CANALI,
            dtype="float32",
            blocksize=DIMENSIONE_FRAME,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def leggi_frame(self, timeout=0.5):
        """Ritorna il prossimo frame, o None se entro il timeout non arriva nulla."""
        try:
            return self._coda.get(timeout=timeout)
        except queue.Empty:
            return None

    def riapri(self):
        """Chiude e riapre lo stream: rimedio quando il microfono smette di consegnare audio."""
        try:
            self.__exit__()
        except Exception as e:
            print(f"⚠️  chiusura del microfono: {e}")
        self.__enter__()

    def svuota(self):
        while not self._coda.empty():
            try:
                self._coda.get_nowait()
            except queue.Empty:
                break


class ArraySource:
    """
    Sorgente finta alimentata da un array: permette di provare il motore senza microfono.

    Scorre a tempo reale (un frame ogni FRAME_MS) come farebbe una scheda audio: senza questa
    attesa il motore consumerebbe l'intero audio durante la calibrazione del rumore di fondo.
    """

    def __init__(self, audio: np.ndarray, coda_infinita=True, tempo_reale=True):
        self._frames = [
            audio[i:i + DIMENSIONE_FRAME]
            for i in range(0, len(audio), DIMENSIONE_FRAME)
        ]
        self._i = 0
        self._coda_infinita = coda_infinita
        self._tempo_reale = tempo_reale

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def leggi_frame(self, timeout=0.5):
        if self._tempo_reale:
            time.sleep(config.FRAME_MS / 1000)

        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            if len(frame) < DIMENSIONE_FRAME:
                frame = np.pad(frame, (0, DIMENSIONE_FRAME - len(frame)))
            return frame
        # Esaurito l'audio si continua con silenzio, cosi' il motore chiude il turno
        return np.zeros(DIMENSIONE_FRAME, dtype=np.float32) if self._coda_infinita else None

    def riapri(self):
        pass

    def svuota(self):
        pass
