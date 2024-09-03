import numpy as np
import random

class Agent:
    def __init__(self, environment, learning_rate=0.01, gamma=0.9, epsilon_decay=0.99, epsilon_min=0.1):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space.n
        self.Q = np.random.rand(self.observation_space, self.n_actions)

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = 1
        self.eps_min = epsilon_min
        self.epc_decay = epsilon_decay

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.epc_decay)

    def predict_action(self, state):
        return np.argmax(self.Q[state, :])

    def training_action(self, state):
        # Take random action
        if random.random() <= self.eps:
            return random.randint(0, self.n_actions-1)
        # Predict best action
        else:
            return self.predict_action(state)

    def fit(self, state, action, reward, terminated, truncated, new_state):
        max_future_q = np.max(self.Q[new_state, :]) if not terminated else 0
        q_current = self.Q[state][action]

        new_q = (1 - self.lr) * q_current + self.gamma * self.lr * (reward + max_future_q)
        self.Q[state, action] = new_q