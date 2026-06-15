from langchain_community.tools import DuckDuckGoSearchRun

# Inizializza l'istanza del motore di ricerca
ddg_search = DuckDuckGoSearchRun()

def internet_search_tool(query: str) -> str:
    """
    Cerca informazioni in tempo reale su internet tramite DuckDuckGo.
    Utile per rispondere a domande su eventi recenti, news o dati non presenti nel DB.
    """
    try:
        # Esegui la ricerca sul web
        risultati = ddg_search.run(query)
        print("risultati: ", risultati)
        return risultati
    except Exception as e:
        # Fallback di sicurezza in caso di temporaneo blocco delle API
        return f"Errore durante la ricerca web: {str(e)}. Prova a riformulare la domanda."