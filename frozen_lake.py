import gymnasium as gym
import numpy as np
import random
from utils import *
from agent import Agent
import matplotlib.pyplot as plt


# Create environment and agent
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
agent = Agent(env, epsilon_decay=0.9975, learning_rate=0.025, epsilon_min=0.1)

EPISODES, MAX_STEPS = 100000, 200
TESTS = 50
AVG_SAMPLES = 100
history = []

for e in range(EPISODES):
    state = env.reset()[0]
    reward, steps = 0, 0
    terminated, truncated = False, False

    while True:
        steps += 1
        action = agent.training_action(state)

        new_state, reward, terminated, truncated, info = env.step(action)
        agent.fit(state, action, reward, terminated, truncated, new_state)
        state = new_state

        if truncated or terminated:
            break

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
    steps, reward = 0, 0
    while True:
        action = agent.predict_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        if terminated or truncated:
            break

    if int(reward) > 0:
        found += 1
        history.append(steps)
        print(f"Test {t+1:4d}: {success('found')} the goal after {steps:3d}. "
              f"Average number of steps to reach goal is {sum(history)/len(history):6.2f}. ")
    else:
        print(f"Test {t+1:4d}: {fail('did not find')} goal.")

print(f"-----------------\n"
      f"Agent reached destination {found} times out of {TESTS} tries. "
      f"Average steps taken: {sum(history)/len(history):6.2f}"
      f"\n-----------------")
