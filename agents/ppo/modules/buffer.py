class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.obs_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obs_values[:]
        del self.is_terminals[:]