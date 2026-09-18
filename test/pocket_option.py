import asyncio
from pocketoptionapi_async import AsyncPocketOptionClient


async def main():
    # 1. Incolla qui l'intero messaggio SSID copiato dal browser (inizia con 42["auth"...)
    # Esempio: '42["auth",{"session":"abcd123...", "isDemo":1, ...}]'
    SSID = '42["auth",{"session":"c607a6c8030fa7c442badf03690d6edb","isDemo":1,"uid":127718602,"platform":1}]'

    # 2. Inizializza il client (imposta is_demo=False se usi il conto reale)
    client = AsyncPocketOptionClient(SSID, is_demo=True)

    try:
        print("Tentativo di connessione ai server di Pocket Option...")
        # Stabilisce la connessione WebSocket in background
        await client.connect()
        print("Connessione riuscita!")

        # 3. Recupera e mostra il bilancio del conto
        bilancio = await client.get_balance()
        print(f"Bilancio attuale del conto: {bilancio}")

        # 4. Sottoscrizione a un asset per ricevere i prezzi (es. EUR/USD OTC o normale)
        asset_selezionato = "EURUSD_OTC"
        print(f"Sottoscrizione ai dati in tempo reale per: {asset_selezionato}")

        # Avvia lo stream delle candele (es. candele da 60 secondi)
        await client.start_candles_stream(asset_selezionato, size=60)

        # Ciclo continuo per leggere i dati aggiornati
        while True:
            # Recupera le candele più recenti ricevute dallo stream
            candele = await client.get_candles(asset_selezionato, size=60)
            if candele:
                # Mostra l'ultima candela disponibile (il prezzo attuale)
                ultima_candela = candele[-1]
                print(f"Prezzo attuale {asset_selezionato}: {ultima_candela}")

            # Attende un secondo prima del controllo successivo per non sovraccaricare la CPU
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    finally:
        # Chiude la connessione in modo pulito in caso di interruzione
        print("Chiusura della connessione...")
        await client.disconnect()


# Avvia l'applicazione asincrona
if __name__ == "__main__":
    asyncio.run(main())
