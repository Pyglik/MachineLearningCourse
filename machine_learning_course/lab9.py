import gym


def main():
    env = gym.make('CartPole-v1')  # utworzenie środowiska
    env.reset()  # reset środowiska do stanu początkowego
    observation = [0, 0, 0, 0]
    for _ in range(1000):  # kolejne kroki symulacji
        env.render()  # renderowanie obrazu
        # action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
        if observation[2] > 0:  # proste sterowanie
            action = 1
        else:
            action = 0

        observation, reward, done, info = env.step(action)  # wykonanie akcji
        print(observation)
        print(reward)
        print(done)
        print(info)
        print('-'*30)
        # if done:
        #     env.reset()
    env.close()  # zamknięcie środowiska


if __name__ == '__main__':
    main()
