class BaseModelAdapter:
    model_name = ""
    category = ""
    description = ""

    def __init__(self):
        self._pipe = None  # common convention

    def load(self):
        raise NotImplementedError

    def run(self, payload):
        raise NotImplementedError

    def info(self):
        # Return a dict so InfoFrame can format it nicely
        return {
            "Model": self.model_name or self.__class__.__name__,
            "Category": self.category or "Unknown",
            "Description": self.description or "",
        }
