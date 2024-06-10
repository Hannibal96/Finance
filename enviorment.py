from data import *
import numpy as np
from enum import Enum
import pandas


class Action(Enum):
    STAY = 0
    BUY = 1
    SELL = 2


class State:
    def __init__(self, money: float, stocks: float, date=None, stock_value=None):
        self.money = money
        self.stocks = stocks
        self.value = stock_value
        self.date = date
        self.deposit_flag = True

    def update(self, date, value):
        self.value = value
        self.deposit_flag = self.date is None or self.date.day > date.day
        self.date = date

    def sell(self, fraction: float = 1.0):
        self.money += self.stocks * fraction * self.value
        self.stocks -= self.stocks * fraction

    def buy(self, fraction: float = 1.0):
        self.stocks += fraction * self.money / self.value
        self.money -= fraction * self.money

    def _deposit_rule(self):
        return self.deposit_flag

    def deposit(self, deposit):
        if self._deposit_rule():
            self.money += deposit

    def inflation(self, gamma):
        if self._deposit_rule():
            self.money *= gamma

    def evaluate(self):
        return self.money + self.value * self.stocks

    def __str__(self):
        s = f"{self.money}$ #{self.stocks:.3f} stocks at {self.value:.3f} in {self.date}"
        return s


class Env:
    def __init__(self, stock_data: DataStock, deposit: float, gamma: float = 1.0, initial: float = 0.0):
        self.data_values = stock_data.stock_values
        self.data_dates = stock_data.stock_dates
        self.alpha_month = stock_data.alpha_month
        self.gamma = gamma
        self.deposit = deposit
        self.x0 = initial
        self.idx = 0
        self.curr_state = State(money=self.x0, stocks=0)
        self._update_state()

    def reset(self):
        self.idx = 0
        self.curr_state = State(money=self.x0, stocks=0)
        self._update_state()

    def _update_state(self) -> bool:
        self.curr_state.update(date=self.data_dates[self.idx],
                               value=self.data_values[self.idx])
        self.idx += 1
        self.curr_state.deposit(deposit=self.deposit)
        return self.idx == len(self.data_values)

    def step(self, action: Action):
        assert action in Action
        done = self._update_state()
        if action == Action.BUY or done:
            self.curr_state.buy()
        elif action == Action.SELL:
            self.curr_state.sell()
        elif action == Action.STAY:
            pass
        else:
            raise NotImplemented
        return self.curr_state, done

    def ode_theory(self):
        prev_day = 0
        xn = self.x0 + self.deposit
        xn_list = []
        for date in self.data_dates:
            if date.day < prev_day:
                xn *= self.alpha_month
                xn += self.deposit
            prev_day = date.day
            xn_list.append(xn)
        print(xn_list[-1])
        return xn_list




