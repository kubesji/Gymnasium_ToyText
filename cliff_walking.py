import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from agent import Agent


# Create environment
env = gym.make("CliffWalking-v0")
agent = Agent(env, epsilon_decay=0.995, learning_rate=0.025, epsilon_min=0.1)

# Initialise stuff needed for exploration and exploitation
TESTS, EPISODES = 50, 1000
history = []

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]
    drops, steps = 0, 0
    while True:
        action = agent.training_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        agent.fit(state, action, reward, terminated, truncated, new_state)

        state = new_state
        drops += 1 if reward == -100 else 0
        steps += 1

        if truncated:
            print("WTF")
            exit(-1)

        if terminated:
            break

    history.append(steps)
    print(f"Episode {e+1:5d}: Found the goal after {steps}. Agent experienced {drops:3d} drops to the void since. "
          f"Average number of steps to reach goal is {sum(history[-10:])/10:6.2f}. "
          f"Current epsilon = {agent.eps:.3f}")
    agent.decay_epsilon()

plt.plot(history)
plt.show()

print("-------------- TEST --------------")
history.clear()
for t in range(TESTS):
    state = env.reset()[0]
    steps, drops, terminated = 0, 0, False
    while not terminated:
        action = agent.predict_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state
        drops += 1 if reward == -100 else 0
        steps += 1

    history.append(steps)
    print(f"Found the goal after {steps}. Agent experienced {drops:3d} drops to the void since. "
          f"Average number of steps to reach goal is {sum(history) / len(history):6.2f}.")

