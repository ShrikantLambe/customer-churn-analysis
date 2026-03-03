class ChatMemory:
    def __init__(self):
        self.history = []
        self.state = {}

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_history(self):
        return self.history

    def set_state(self, key, value):
        self.state[key] = value

    def get_state(self, key, default=None):
        return self.state.get(key, default)

    def clear(self):
        self.history = []
        self.state = {}
