import gym
import time
import numpy as np


def main():
    env = gym.make('FrozenLake-v1')  # utworzenie środowiska
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Q_table = np.zeros((n_observations, n_actions))
    lr = 0.1
    discount_factor = 0.95
    exploration_proba = 1.0
    min_exploration_proba = 0.1

    # Uczenie
    for _ in range(10000):
        current_state = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < exploration_proba:  # eksploracja
                action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
            else:  # eksploatacja
                action = np.argmax(Q_table[current_state, :])
            next_state, reward, done, _ = env.step(action)
            Q_table[current_state, action] = (1 - lr) * Q_table[current_state, action] + \
                                             lr * (reward + discount_factor * max(Q_table[next_state, :]))
            current_state = next_state
        exploration_proba = max(min_exploration_proba, exploration_proba - 0.001)
    print(Q_table)

    # Testowanie
    total_episodes = 500
    total_rewards = []
    for i in range(total_episodes):
        done = False
        total_e_reward = 0
        current_state = env.reset()
        while not done:
            action = np.argmax(Q_table[current_state, :])
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            total_e_reward += reward
        total_rewards.append(total_e_reward)
        env.render()

    env.close()  # zamknięcie środowiska

    print("\nSkuteczność:", np.sum(total_rewards) / total_episodes)


if __name__ == '__main__':
    main()
