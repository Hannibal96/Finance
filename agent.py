from enviorment import *
import random


class Agent:
    def __init__(self):
        pass

    def act(self, state: State) -> Action:
        pass

    def update(self, state: State):
        pass

    def __str__(self):
        pass


class RandomAgent(Agent):
    def __init__(self, weights=(1/3, 1/3, 1/3)):
        super().__init__()
        self.weights = weights

    def act(self, state: State) -> Action:
        choice = random.choices(list(Action), weights=self.weights)
        return choice[0]


class MinMaxAgent(Agent):
    def __init__(self, decline=0.1):
        super().__init__()
        self.max = 0
        self.decline = decline

    def act(self, state: State) -> Action:
        if state.value < (1-self.decline) * self.max:
            return Action.BUY
        elif state.value >= self.max:
            return Action.SELL
        return Action.STAY

    def update(self, state: State):
        self.max = max(state.value, self.max)


