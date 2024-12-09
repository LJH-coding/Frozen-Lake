import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        print(f"State {state}, Action {action}: {env.P[state][action]}")

def value_iteration(env, gamma=0.99, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    PI = np.zeros(n_states, dtype=int)

    # Start Code Here
    # End Code Here

    for s in range(n_states):
        q_values = [
            sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward, _ in env.P[s][a])
            for a in range(n_actions)
        ]
        PI[s] = np.argmax(q_values)
    return V, PI

V, PI = value_iteration(env)
print("Optimal Value Function:\n", V.reshape((4, 4)))
print("Optimal Policy:\n", PI.reshape((4, 4)))

(obs, info), total_reward = env.reset(), 0
while True:
    obs, reward, termination, truncated, info = env.step(PI[obs])
    total_reward += reward
    env.render()
    if termination or truncated:
        break
print("return: ", total_reward)
