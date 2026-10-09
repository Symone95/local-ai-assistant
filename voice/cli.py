"""
Modalita' vocale da terminale: la via piu' reattiva, senza Streamlit di mezzo.

    venv/bin/python -m voice.cli

Parla quando vuoi: il turno si chiude da solo dopo un attimo di silenzio, e l'assistente
comincia a rispondere appena ha pronta la prima frase. Ctrl-C per uscire.
"""
import signal
import sys
import time

from voice.agente import AgenteVocale
from voice.engine import ottieni_motore

ICONE = {
    "preparo": "⏳",
    "in ascolto": "🎙️ ",
    "trascrivo": "🔤",
    "penso": "🤔",
    "parlo": "🗣️ ",
    "spento": "⏹️ ",
}


def main():
    agente = AgenteVocale()
    ultima_fase = {"valore": None}

    def su_aggiornamento(stato):
        if stato.fase == "in ascolto" and stato.microfono_muto:
            print("\r🔇 nessun segnale dal microfono: controlla i permessi del terminale".ljust(70),
                  end="", flush=True)
            return

        if stato.fase != ultima_fase["valore"]:
            ultima_fase["valore"] = stato.fase
            print(f"\r{ICONE.get(stato.fase, '•')} {stato.fase}...".ljust(60), end="", flush=True)

    motore = ottieni_motore(agente.rispondi, on_update=su_aggiornamento)

    def chiudi(*_):
        print("\n⏹️  chiudo...")
        motore.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, chiudi)

    print("🎙️  Modalita' vocale attiva. Parla pure — Ctrl-C per uscire.\n")
    motore.start()

    turni_mostrati = 0
    while True:
        time.sleep(0.3)
        stato = motore.snapshot()
        if len(stato.turni) > turni_mostrati:
            for domanda, risposta in stato.turni[turni_mostrati:]:
                print(f"\r📝 tu: {domanda}".ljust(80))
                print(f"🤖 assistente: {risposta}\n")
            turni_mostrati = len(stato.turni)
        if not stato.attivo and stato.errore:
            print(f"\n❌ {stato.errore}")
            break


if __name__ == "__main__":
    main()
