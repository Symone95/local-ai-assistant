"""
Ponte fra il motore vocale e il grafo LangGraph.

Espone un oggetto con un metodo `rispondi(testo)` che produce i token della risposta: e' cio'
che VoiceEngine si aspetta. Si occupa anche dell'human-in-the-loop, che a voce diventa una
domanda parlata a cui si risponde "si'" o "no".
"""
import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_graph import build_graph

# A voce una risposta lunga e' insopportabile: quella di prova durava quasi un minuto.
STILE_PARLATO = (
    "\nIMPORTANTE: questa risposta verra' letta ad alta voce. Sii breve e colloquiale, "
    "al massimo due o tre frasi. Niente elenchi puntati, niente simboli, niente fonti: "
    "solo la sostanza, come la diresti a voce. Apri con una frase molto breve, cosi' chi "
    "ascolta sente subito la risposta."
)

_AFFERMATIVO = re.compile(r"\b(s[iì]+|certo|va bene|conferm[oa]|procedi|vai|esegui|ok|okay|d'accordo)\b", re.I)
_NEGATIVO = re.compile(r"\b(no|annulla|ferma|lascia stare|niente|non farlo|meglio di no)\b", re.I)


class AgenteVocale:
    """Mantiene storia e checkpointer fra un turno e l'altro della conversazione parlata."""

    def __init__(self, app=None):
        self.app = app or build_graph().compile(checkpointer=InMemorySaver())
        self.storia = []
        self._config_in_attesa = None      # config del thread sospeso su un'approvazione
        self._richiesta_in_attesa = None   # payload dell'interrupt

    @property
    def in_attesa_di_approvazione(self) -> bool:
        return self._config_in_attesa is not None

    async def rispondi(self, testo: str):
        """Async generator dei token della risposta. Gestisce anche il turno di approvazione."""
        if self.in_attesa_di_approvazione:
            async for token in self._gestisci_approvazione(testo):
                yield token
            return

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        ingresso = {
            "query": testo,
            "messages": self.storia + [HumanMessage(content=testo)],
            "context": "",
            "tool_result": {},
            "final_answer": "",
            "stile": STILE_PARLATO,
        }

        risposta = ""
        async for token in self._streama(ingresso, config):
            risposta += token
            yield token

        # Il grafo si e' fermato per chiedere il permesso? A voce lo si chiede parlando.
        richiesta = self._interrupt_in_sospeso(config)
        if richiesta:
            self._config_in_attesa = config
            self._richiesta_in_attesa = richiesta
            domanda = self._formula_domanda(richiesta)
            risposta += domanda
            yield domanda
        else:
            self._memorizza(testo, risposta)

    async def _gestisci_approvazione(self, testo: str):
        """Interpreta la risposta parlata dell'utente alla richiesta di approvazione."""
        config, richiesta = self._config_in_attesa, self._richiesta_in_attesa

        if _AFFERMATIVO.search(testo):
            decisione = {"approvato": True, "args": richiesta.get("args", {})}
            apertura = "Va bene, procedo. "
        elif _NEGATIVO.search(testo):
            decisione = {"approvato": False}
            apertura = "D'accordo, non eseguo nulla. "
        else:
            yield "Non ho capito: vuoi che proceda? Rispondi si' oppure no."
            return

        self._config_in_attesa = None
        self._richiesta_in_attesa = None

        yield apertura
        risposta = apertura
        async for token in self._streama(Command(resume=decisione), config):
            risposta += token
            yield token

        self._memorizza(testo, risposta)

    # Nodi la cui uscita e' la risposta destinata all'utente. Gli altri nodi usano l'LLM per
    # lavoro interno (estrazione citta', riscrittura query, reranking): quei token non vanno
    # ne' letti ad alta voce ne' mostrati.
    NODI_DI_RISPOSTA = {"llm", "direct_llm_answer"}

    async def _streama(self, ingresso, config):
        async for event in self.app.astream_events(ingresso, version="v2", config=config):
            if event["event"] != "on_chat_model_stream":
                continue
            if event["metadata"].get("langgraph_node") not in self.NODI_DI_RISPOSTA:
                continue
            contenuto = event["data"]["chunk"].content
            if contenuto:
                yield contenuto

    def _interrupt_in_sospeso(self, config):
        snapshot = self.app.get_state(config)
        interruzioni = [i for task in snapshot.tasks for i in task.interrupts]
        return interruzioni[0].value if interruzioni else None

    @staticmethod
    def _formula_domanda(richiesta) -> str:
        argomenti = ", ".join(f"{k}: {v}" for k, v in (richiesta.get("args") or {}).items())
        return (
            f" Attenzione: sto per usare {richiesta.get('tool')}"
            + (f" con {argomenti}" if argomenti else "")
            + ". Vuoi che proceda? Rispondi si' oppure no."
        )

    def _memorizza(self, domanda, risposta):
        self.storia = (self.storia + [HumanMessage(content=domanda), AIMessage(content=risposta)])[-10:]
