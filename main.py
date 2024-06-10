import random
import numpy as np
import numpy.random
from enviorment import *
from agent import *


def eval_agent(env: Env, agent: Agent):
    env.reset()
    money_list = [env.curr_state.evaluate()]
    while True:
        agent.update(env.curr_state)
        action = agent.act(env.curr_state)
        state, done = env.step(action=action)
        money_list.append(env.curr_state.evaluate())
        if done:
            break
    return money_list


if __name__ == "__main__":
    stock_data = DataStock(symbol='^GSPC', start_date='2014-01-01')
    alpha = stock_data.alpha_month
    env = Env(stock_data=stock_data, deposit=1_000)
    dates = env.data_dates

    agent = RandomAgent(weights=[0, 1, 0])
    money_list = eval_agent(env=env, agent=agent)
    plt.plot(dates, money_list, label="BUY")

    agent = MinMaxAgent(decline=0.05)
    money_list = eval_agent(env=env, agent=agent)
    plt.plot(dates, money_list, label="MinMax")

    # plt.plot(dates, env.ode_theory(), label="Theory")

    plt.legend()
    plt.grid()
    plt.show()







