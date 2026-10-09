"""
Motore della conversazione vocale continua.

Il ciclo e' una macchina a stati che gira in un thread proprio:

    in ascolto ──(fine turno per silenzio)──▶ trascrivo ──▶ penso ──▶ parlo ──┐
         ▲                                                                    │
         └──────────────(fine risposta, oppure barge-in)──────────────────────┘

Due scelte fanno la differenza sulla reattivita' percepita:
  * il turno si chiude da solo dopo un po' di silenzio, senza premere nulla;
  * la risposta viene pronunciata **frase per frase** mentre l'LLM la sta ancora generando,
    quindi l'assistente inizia a parlare dopo la prima frase e non a risposta finita.

Il motore non sa nulla di Streamlit: riceve una funzione `rispondi(testo) -> AsyncIterator[str]`
che produce i token della risposta, e notifica i cambi di stato con `on_update`.
"""
import asyncio
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from voice import config, stt, tts, voce_sistema
from voice.player import Player
from voice.sources import DIMENSIONE_FRAME, MicSource

FRAMES_PER_SECONDO = 1000 / config.FRAME_MS
# 60ms di voce continua: sotto questa soglia si tratta di un rumore isolato, non di parlato.
FRAMES_CONSECUTIVI_PER_VOCE = 2


_LUCCHETTO_AVVIO = threading.Lock()
_MOTORE_ATTIVO = None       # unico motore vivo nel processo
_MOTORE_CONDIVISO = None    # istanza riusata dalle interfacce


def ottieni_motore(rispondi, on_update=None):
    """
    Restituisce il motore del processo, creandolo solo la prima volta.

    Le interfacce (app Streamlit, CLI) devono passare da qui invece di costruirsi un
    VoiceEngine: con Streamlit, che riesegue lo script di continuo, e' l'unico modo per essere
    certi che non ne nascano due che parlano insieme.
    """
    global _MOTORE_CONDIVISO

    if _MOTORE_CONDIVISO is None:
        _MOTORE_CONDIVISO = VoiceEngine(rispondi=rispondi, on_update=on_update)
    elif on_update is not None:
        _MOTORE_CONDIVISO._on_update = on_update

    return _MOTORE_CONDIVISO


def scalda_ollama():
    """
    Manda una richiesta minima al modello del grafo, con le *sue* opzioni.

    Ollama carica il modello alla prima chiamata e lo ricarica se cambiano opzioni come num_ctx:
    scaldarlo con parametri diversi da quelli dei nodi non servirebbe a nulla. Per questo si usa
    direttamente l'oggetto ChatOllama di nodes.py.
    """
    try:
        from nodes import llm
        llm.invoke("ok")

        # Scalda anche il prompt del router: e' lungo (~2500 token) e la prima valutazione
        # costa una decina di secondi. Le chiamate successive riusano la cache del prefisso e
        # scendono sotto il secondo, quindi conviene pagarlo qui invece che al primo "ciao".
        from tools import tool_planner
        tool_planner("riscaldamento del router")

        print("🔥 modelli e prompt del router pronti in memoria")
    except Exception as e:
        print(f"⚠️  preriscaldamento non riuscito: {e}")


# Fine frase "sicura": punteggiatura seguita da spazio o fine testo, mai fra due cifre
# (altrimenti "24.79°C" verrebbe letto come due frasi).
_FINE_FRASE = re.compile(r"(?<!\d)[.!?…:;](?=\s|$)|\n")


def entro_budget(testo: str, budget: int) -> str:
    """
    Riduce il testo all'ultima frase intera che rientra nel budget di caratteri.

    Serve a non sforare il tetto del parlato con l'ultimo blocco, e a non troncare a meta'
    di una parola: meglio dire una frase in meno che lasciare la voce sospesa.
    """
    if budget <= 0:
        return ""
    if len(testo) <= budget:
        return testo

    taglio = 0
    for match in _FINE_FRASE.finditer(testo[:budget]):
        taglio = match.end()

    return testo[:taglio] if taglio else ""


def taglia_frase(buffer: str):
    """
    Se nel buffer c'e' una frase conclusa la restituisce insieme al resto, altrimenti (None, buffer).

    Il taglio richiede almeno un carattere dopo la punteggiatura: aspettare il token successivo
    costa nulla ed evita di spezzare numeri e abbreviazioni a meta'.
    """
    ultimo = None
    for match in _FINE_FRASE.finditer(buffer):
        if match.end() < len(buffer):
            ultimo = match

    if ultimo is None:
        return None, buffer

    return buffer[:ultimo.end()], buffer[ultimo.end():]


@dataclass(frozen=True)
class StatoVocale:
    attivo: bool = False
    fase: str = "spento"          # spento | preparo | in ascolto | trascrivo | penso | parlo
    trascrizione: str = ""        # ultima frase capita
    risposta: str = ""            # risposta in corso
    turni: tuple = field(default_factory=tuple)   # ((utente, assistente), ...)
    errore: str = ""
    livello: float = 0.0          # volume istantaneo del microfono
    soglia: float = 0.0           # sopra questo valore si considera parlato
    microfono_muto: bool = False  # nessun segnale da qualche secondo: permessi o device sbagliato


class VoiceEngine:
    def __init__(self, rispondi, on_update=None, source=None):
        """
        rispondi:  funzione async che, dato il testo dell'utente, produce i token della risposta
        on_update: callback sincrona invocata a ogni cambio di stato (facoltativa)
        source:    sorgente audio; di default il microfono di sistema
        """
        self._rispondi = rispondi
        self._on_update = on_update
        self._source = source or MicSource()

        self._player = Player()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._stato = StatoVocale()
        self._soglia = None
        self._barge_in_richiesto = False
        self._riapri_sorgente = False

    # ── stato condiviso con la UI ────────────────────────────────────────────────

    def snapshot(self) -> StatoVocale:
        with self._lock:
            return self._stato

    def _aggiorna(self, **campi):
        with self._lock:
            self._stato = replace(self._stato, **campi)
            corrente = self._stato
        if self._on_update:
            try:
                self._on_update(corrente)
            except Exception as e:      # la UI non deve poter fermare il motore
                print(f"⚠️  on_update: {e}")

    # ── ciclo di vita ───────────────────────────────────────────────────────────

    def start(self):
        global _MOTORE_ATTIVO

        # Streamlit puo' rieseguire lo script due volte a distanza di pochi millisecondi (due
        # schede aperte, un doppio clic): senza lucchetto due chiamate passerebbero entrambe il
        # controllo sul thread e partirebbero due cicli sullo stesso microfono.
        with _LUCCHETTO_AVVIO:
            if self._thread and self._thread.is_alive():
                return

            # Il microfono e gli altoparlanti sono uno solo: due motori nello stesso processo si
            # sovrappongono, e si sentono due voci che dicono la stessa cosa sfasate, come un'eco.
            # Succede per esempio se Streamlit invalida la cache e ricrea l'oggetto mentre il
            # thread precedente e' ancora vivo: qui il vecchio viene chiuso prima di partire.
            if _MOTORE_ATTIVO is not None and _MOTORE_ATTIVO is not self:
                print("⚠️  un altro motore vocale era attivo: lo chiudo prima di avviare questo")
                _MOTORE_ATTIVO.stop()

            _MOTORE_ATTIVO = self
            self._stop.clear()
            self._thread = threading.Thread(target=self._esegui, name="voice-engine", daemon=True)
            self._thread.start()

    def stop(self, timeout=5.0):
        global _MOTORE_ATTIVO

        if _MOTORE_ATTIVO is self:
            _MOTORE_ATTIVO = None

        self._stop.set()
        self._player.stop()
        voce_sistema.zittisci()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._aggiorna(attivo=False, fase="spento")

    def _esegui(self):
        try:
            asyncio.run(self._ciclo())
        except Exception as e:
            print(f"❌ motore vocale: {e}")
            self._aggiorna(attivo=False, fase="spento", errore=str(e))

    # ── ascolto e rilevamento del turno ─────────────────────────────────────────

    def _calibra(self, source) -> float:
        """Misura il rumore di fondo per qualche decimo di secondo e ne ricava la soglia."""
        livelli = []
        scadenza = time.monotonic() + 0.6
        while time.monotonic() < scadenza:
            frame = source.leggi_frame(timeout=0.3)
            if frame is not None:
                livelli.append(float(np.sqrt(np.mean(frame ** 2))))

        fondo = float(np.median(livelli)) if livelli else 0.0
        soglia = max(fondo * config.VAD_FATTORE, config.SOGLIA_MINIMA)
        print(f"🎚️  rumore di fondo {fondo:.4f} → soglia {soglia:.4f}")
        return soglia

    def _ascolta_turno(self, source):
        """
        Accumula audio finche' l'utente parla, e lo restituisce quando il silenzio
        supera SILENZIO_FINE_TURNO_MS. Ritorna None se non c'e' stato parlato.
        """
        frames_silenzio_fine = int(config.SILENZIO_FINE_TURNO_MS / config.FRAME_MS)
        frames_minimi = int(config.DURATA_MINIMA_MS / config.FRAME_MS)
        frames_massimi = int(config.DURATA_MASSIMA_MS / config.FRAME_MS)

        raccolti = []
        frames_parlato = 0
        frames_silenzio = 0
        parlato_iniziato = False
        ultimo_report = 0.0
        frames_silenziosi = 0
        frames_sopra = 0
        ultimo_frame = time.monotonic()
        ultima_ricalibrazione = time.monotonic()
        rumore_recente = deque(maxlen=300)

        # Anello di frame precedenti al rilevamento: l'attacco di una parola e' piu' debole del
        # resto e sta sotto soglia, quindi senza questo il primo fonema andrebbe perso — ed e' un
        # danno serio per la trascrizione ("Ciao, che tempo fa" diventava "che tempo fa").
        preroll = deque(maxlen=max(1, int(config.PREROLL_MS / config.FRAME_MS)))

        while not self._stop.is_set():
            frame = source.leggi_frame(timeout=0.5)
            if frame is None:
                # Su macOS lo stream di ingresso puo' morire in silenzio quando un altro
                # processo riconfigura il dispositivo: senza questo controllo il motore
                # resterebbe "in ascolto" per sempre senza sentire piu' nulla.
                if time.monotonic() - ultimo_frame > config.WATCHDOG_MIC_S:
                    print("⚠️  il microfono non consegna piu' audio: riapro lo stream")
                    self._riapri_sorgente = True
                    return None
                continue

            ultimo_frame = time.monotonic()
            energia = float(np.sqrt(np.mean(frame ** 2)))

            # Soglia adattiva. Si raccoglie l'energia di OGNI frame e si stima il rumore di
            # fondo con il 20° percentile: anche mentre qualcuno parla, un quinto dei frame
            # sono pause fra le parole. Usare solo i frame sotto soglia non funzionerebbe,
            # perche' quando il rumore sale sopra la soglia non se ne raccoglierebbe nessuno.
            rumore_recente.append(energia)
            if (not parlato_iniziato and len(rumore_recente) > 60
                    and time.monotonic() - ultima_ricalibrazione > config.RICALIBRA_OGNI_S):
                ultima_ricalibrazione = time.monotonic()
                nuova = max(float(np.percentile(rumore_recente, 20)) * config.VAD_FATTORE,
                            config.SOGLIA_MINIMA)
                if abs(nuova - self._soglia) > self._soglia * 0.2:
                    print(f"🎚️  soglia aggiornata: {self._soglia:.4f} → {nuova:.4f}")
                    self._soglia = nuova

            # Servono piu' frame consecutivi sopra soglia perche' sia parlato: un singolo picco
            # (una porta, un clic, la ventola) altrimenti azzera il contatore del silenzio e il
            # turno non si chiude piu', arrivando fino al tetto dei secondi massimi.
            frames_sopra = frames_sopra + 1 if energia > self._soglia else 0
            e_parlato = frames_sopra >= FRAMES_CONSECUTIVI_PER_VOCE

            # Il livello va mostrato nell'interfaccia: senza, "in ascolto" e' una scatola chiusa
            # e non si distingue un microfono muto da una soglia troppo alta.
            frames_silenziosi = frames_silenziosi + 1 if energia < 1e-5 else 0
            adesso = time.monotonic()
            if adesso - ultimo_report > 0.15:
                ultimo_report = adesso
                self._aggiorna(
                    livello=energia,
                    soglia=self._soglia,
                    microfono_muto=frames_silenziosi > int(3000 / config.FRAME_MS),
                )

            if e_parlato:
                if not parlato_iniziato:
                    parlato_iniziato = True
                    raccolti.extend(preroll)      # recupera l'attacco della prima parola
                    self._aggiorna(fase="in ascolto", trascrizione="")
                raccolti.append(frame)
                frames_parlato += 1
                frames_silenzio = 0
            elif parlato_iniziato:
                raccolti.append(frame)        # un po' di coda aiuta la trascrizione
                frames_silenzio += 1
                if frames_silenzio >= frames_silenzio_fine:
                    break                     # il turno si chiude qui: l'utente ha finito
            else:
                preroll.append(frame)

            if parlato_iniziato and len(raccolti) > frames_massimi:
                break

        if not parlato_iniziato or frames_parlato < frames_minimi:
            return None

        # Si taglia il silenzio finale: Whisper, davanti a mezzo secondo di quiete, tende a
        # inventare parole ("Eh... no... Poi mi marzi") che poi confondono il router.
        audio = np.concatenate(raccolti)
        coda = int(config.SAMPLE_RATE * 0.25)
        finestra = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)
        for i in range(len(audio) - finestra, 0, -finestra):
            if float(np.sqrt(np.mean(audio[i:i + finestra] ** 2))) > self._soglia:
                return audio[:min(len(audio), i + finestra + coda)]

        return audio

    # ── barge-in ────────────────────────────────────────────────────────────────

    def _controlla_barge_in(self, source):
        """
        Ritorna una funzione che dice se l'utente ha ripreso a parlare mentre l'assistente parla.

        Senza cancellazione d'eco la voce dell'assistente rientra dal microfono, percio' qui la
        soglia e' molto piu' alta che in ascolto normale (BARGE_IN_FATTORE).
        """
        soglia = self._soglia * (config.BARGE_IN_FATTORE / config.VAD_FATTORE)
        richiesti = max(1, int(config.BARGE_IN_MS / config.FRAME_MS))
        contatore = {"n": 0}

        def interrotto():
            if self._stop.is_set():
                return True

            # Si continua a leggere anche con il barge-in spento: altrimenti la coda del
            # microfono accumula tutta la risposta appena pronunciata, che rientra dagli
            # altoparlanti e al turno successivo verrebbe scambiata per una nuova domanda.
            frame = source.leggi_frame(timeout=0.0) if hasattr(source, "leggi_frame") else None
            if frame is None or not config.BARGE_IN:
                return False

            energia = float(np.sqrt(np.mean(frame ** 2)))
            contatore["n"] = contatore["n"] + 1 if energia > soglia else 0
            if contatore["n"] >= richiesti:
                print("🙋 barge-in: l'utente ha ripreso la parola")
                self._barge_in_richiesto = True
                return True
            return False

        return interrotto

    # ── ciclo principale ────────────────────────────────────────────────────────

    async def _ciclo(self):
        with self._source as source:
            # Fase esplicita: fra il click e il primo ascolto ci sono il caricamento del
            # modello di trascrizione e il preriscaldamento di Ollama. Dire "in ascolto" qui
            # farebbe parlare l'utente a vuoto.
            self._aggiorna(attivo=True, fase="preparo", errore="")
            self._soglia = self._calibra(source)
            await asyncio.to_thread(stt.get_modello)     # precarica: la prima frase non deve attendere
            await asyncio.to_thread(scalda_ollama)       # e nemmeno la prima risposta
            if config.TTS_BACKEND != "say":
                self._player.prepara()                   # apre il dispositivo audio: costa ~1,9s

            while not self._stop.is_set():
                # Breve pausa prima di riascoltare: senza cancellazione d'eco, la coda della
                # propria voce dagli altoparlanti verrebbe scambiata per un nuovo turno.
                if self.snapshot().fase == "parlo":
                    await asyncio.sleep(config.PAUSA_DOPO_RISPOSTA_MS / 1000)

                source.svuota()
                self._aggiorna(fase="in ascolto")

                audio = await asyncio.to_thread(self._ascolta_turno, source)

                if self._riapri_sorgente:
                    self._riapri_sorgente = False
                    source.riapri()
                    self._soglia = self._calibra(source)
                    continue

                if audio is None or self._stop.is_set():
                    continue

                self._aggiorna(fase="trascrivo")
                testo = await asyncio.to_thread(stt.trascrivi, audio)
                if not testo or len(testo) < 2:
                    continue

                print(f"📝 utente: {testo}")
                self._aggiorna(fase="penso", trascrizione=testo, risposta="")

                risposta = await self._rispondi_e_parla(testo, source)

                turni = self.snapshot().turni + ((testo, risposta),)
                self._aggiorna(turni=turni, risposta=risposta)

    async def _rispondi_e_parla(self, testo, source):
        """
        Consuma i token della risposta e manda in sintesi una frase alla volta.

        Sintesi e riproduzione lavorano in parallelo: appena una frase e' completa parte la sua
        sintesi, mentre la precedente e' ancora in riproduzione. La coda conserva l'ordine, cosi'
        l'assistente parla senza pause fra una frase e l'altra.
        """
        self._barge_in_richiesto = False
        interrotto = self._controlla_barge_in(source)

        coda = asyncio.Queue()
        riproduttore = asyncio.create_task(self._riproduci_in_ordine(coda, interrotto))

        completa = ""
        da_pronunciare = ""
        prima_frase = True
        pronunciati = 0

        try:
            async for token in self._rispondi(testo):
                if self._stop.is_set() or self._barge_in_richiesto:
                    break

                completa += token
                da_pronunciare += token
                self._aggiorna(risposta=completa)

                # La prima frase parte con una soglia piu' bassa: e' quella che determina quanto
                # l'assistente sembra reattivo. Le successive si accumulano un po' di piu', per
                # non spezzettare il parlato.
                soglia = 4 if prima_frase else config.FRASE_MINIMA_CARATTERI

                if pronunciati >= config.MAX_CARATTERI_PARLATI:
                    continue     # il resto resta scritto nella chat, non viene letto

                frase, resto = taglia_frase(da_pronunciare)
                if frase and len(frase.strip()) >= soglia:
                    da_pronunciare = resto
                    frase = entro_budget(frase, config.MAX_CARATTERI_PARLATI - pronunciati)
                    if not frase.strip():
                        continue
                    pronunciati += len(frase)
                    if prima_frase:
                        print("🗣️  inizio a parlare (il resto si sta ancora generando)")
                        prima_frase = False
                    await coda.put(self._prepara_voce(frase))

            coda_finale = entro_budget(da_pronunciare, config.MAX_CARATTERI_PARLATI - pronunciati)
            if coda_finale.strip() and not self._barge_in_richiesto and not self._stop.is_set():
                await coda.put(self._prepara_voce(coda_finale))
        finally:
            await coda.put(None)          # segnale di fine
            await riproduttore

        return completa

    def _prepara_voce(self, frase):
        """
        Mette in coda la frase da pronunciare.

        Con la voce di sistema basta il testo (ci pensa `say`, in un processo suo); con edge-tts
        si avvia subito la sintesi, cosi' procede mentre la frase precedente e' ancora in corso.
        """
        if config.TTS_BACKEND == "say":
            return frase
        return asyncio.ensure_future(tts.sintetizza(frase))

    def _parla_con_sistema(self, testo, interrotto):
        """Pronuncia con `say`, restando pronta a zittirlo se arriva uno stop o un barge-in."""
        finito = threading.Event()

        def esegui():
            try:
                voce_sistema.parla(testo)
            finally:
                finito.set()

        threading.Thread(target=esegui, name="voce-say", daemon=True).start()

        while not finito.wait(0.05):
            if self._stop.is_set() or (interrotto and interrotto()):
                voce_sistema.zittisci()
                return False
        return True

    async def _riproduci_in_ordine(self, coda, interrotto):
        """Pronuncia le frasi nell'ordine in cui sono state accodate."""
        while True:
            elemento = await coda.get()
            if elemento is None:
                break

            if self._stop.is_set() or self._barge_in_richiesto:
                if not isinstance(elemento, str):
                    elemento.cancel()
                continue

            self._aggiorna(fase="parlo")

            if isinstance(elemento, str):
                await asyncio.to_thread(self._parla_con_sistema, elemento, interrotto)
                continue

            audio, sr = await elemento
            if audio is None:
                continue

            await asyncio.to_thread(self._player.riproduci, audio, sr, interrotto)

