import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LinearModel(nn.Module):
    def __init__(self, num_features, num_outputs):
        super(LinearModel, self).__init__()
        self.linear_layer = nn.Linear(num_features, num_outputs)

    def forward(self, x):
        return self.linear_layer(x)


class DQNAgent:
    def __init__(self, epsilon=0.5, model=LinearModel, num_features=4, num_outputs=2, update_cycle=100, optimizer=optim.Adam, lr=1e-3, ):
        self.epsilon = epsilon
        self.q_net = model(num_features=num_features, num_outputs=num_outputs)
        self.target_q_net = model(num_features=num_features, num_outputs=num_outputs)  # only for calculating the next_Q values updated every update_cycle
        self.optimizer = optimizer(self.q_net.parameters(), lr=lr)

        self.memory = []    # Tuples of <State, Action, Reward, Next State>
        self.stock_value_history = []
        self.money_history = []
        self.stock_holding_history = []
        self.losses = []    # Losses over the epochs
        self.global_max = 0
        self.state = None

        self.update_cycle = update_cycle
        self.batch_size = 32

        self.gamma = 0.999774402

    def _copy_params_to_target_q_net(self, epoch):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

    def _augment_state(self, state):
        money, stocks, current_stock_price = state
        self.global_max = max(self.global_max, current_stock_price)
        return torch.tensor([money, stocks, current_stock_price, self.global_max], dtype=torch.float32).unsqueeze(0)

    def act(self, state):
        money, stocks, current_stock_price = state
        self.stock_value_history.append(current_stock_price)
        self.money_history.append(money)
        self.stock_holding_history.append(stocks)
        self.state = self._augment_state(state)

        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, 2, (1,)).item()
        with torch.no_grad():
            return torch.argmax(self.q_net(state)).item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        samples = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = np.array(self.memory, dtype=object)[samples]
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Compute the Q values
        self.q_net.train()
        self.target_q_net.eval()
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1).values
            target_q_values = rewards + self.gamma * next_q_values

        # Compute the loss
        loss = F.mse_loss(q_values, target_q_values)
        self.losses.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



