import os

class TerminalManager:

    def __init__(self):
        self.workspace_dir = os.path.abspath("./")


    def _find_by_name(self, name: str) -> str | None:
        """Cerca ricorsivamente un file o cartella per nome nel workspace."""
        for root, dirs, files in os.walk(self.workspace_dir):
            if name in files or name in dirs:
                return os.path.join(root, name)
        return None

    def list_and_read_files(self, sotto_cartella: str = "") -> str:
        """
        Usa questo tool per vedere quali cartelle e file sono disponibili
        e leggerne il codice. Puoi passare un path relativo, un nome di file
        o una sottocartella — se non viene trovato direttamente, viene cercato
        automaticamente in tutto il workspace.
        """
        target_path = os.path.abspath(os.path.join(self.workspace_dir, sotto_cartella))

        if not target_path.startswith(self.workspace_dir):
            return "Errore: Accesso negato. Non puoi uscire dalla cartella del progetto."

        # Se il path diretto non esiste, cerca per nome nel workspace
        if not os.path.exists(target_path) and sotto_cartella:
            found = self._find_by_name(os.path.basename(sotto_cartella))
            if found:
                target_path = found
            else:
                return f"Nessun file o cartella trovato con il nome '{sotto_cartella}'."

        if os.path.isdir(target_path):
            files = os.listdir(target_path)
            rel = os.path.relpath(target_path, self.workspace_dir)
            return f"Contenuto della cartella '{rel}': {files}"

        if os.path.isfile(target_path):
            rel = os.path.relpath(target_path, self.workspace_dir)
            with open(target_path, "r", encoding="utf-8") as f:
                return f"--- Contenuto di {rel} ---\n{f.read()}"
            

terminal_manager = TerminalManager()