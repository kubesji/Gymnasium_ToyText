import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from agent import Agent
from utils import *


# Create environment
env = gym.make("Taxi-v3", render_mode="ansi")
agent = Agent(env, epsilon_decay=0.995, learning_rate=0.025)

# Initialise stuff needed for exploration and exploitation
# High number of episodes is needed to fully explore all possible combinations
EPISODES, TESTS, AVG_SAMPLES = 50000, 50, 100
history = []

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]

    terminated, truncated = False, False
    steps = 0
    while not terminated and not truncated:
        action = agent.training_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)

        agent.fit(state, action, int(reward), terminated, truncated, new_state)

        state = new_state
        steps += 1

    history.append(steps)
    if (e + 1) % AVG_SAMPLES == 0:
        print(f"Episode {e+1:5d}: "
              f"Average number of steps to reach goal is {sum(history[-AVG_SAMPLES:])/AVG_SAMPLES:6.2f}. "
              f"Current epsilon = {agent.eps:.3f}")
        agent.decay_epsilon()

plt.plot(history)
plt.show()

print("-------------- TEST --------------")
history.clear()
found = 0
for t in range(TESTS):
    state = env.reset()[0]
    steps, terminated, truncated = 0, False, False
    while not terminated and not truncated:
        action = agent.predict_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)

        state = new_state
        steps += 1

    if int(reward) > 0:
        found += 1
        history.append(steps)
        print(f"Test {t+1:4d}: {success('found')} the goal after {steps:3d}. ", end="")
        print(f"Average number of steps to reach goal is {sum(history)/len(history):6.2f}. ")
    else:
        print(f"Test {t+1:4d}: {fail('did not find')} goal.")

print(f"-----------------\n"
      f"Passenger reached destination {found} times out of {TESTS} rides. "
      f"Average steps taken: {sum(history)/len(history):6.2f}"
      f"\n-----------------")
