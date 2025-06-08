import argparse
import random

class ExperienceReplay:
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.experience = [None]*capacity
        self.n = 0

    def append(self, s0, a0, r0, s1):
        self.experience[self.n % self.capacity] = [s0, a0, r0, s1]
        self.n += 1

    def sample(self, k):
        if self.n < self.capacity:
            return random.choices(self.experience[:self.n], k=k)
        return random.choices(self.experience, k=k)




def main():
    import gymnasium as gym
    from time import sleep
    env = gym.make("LunarLander-v3", render_mode=None)

    replay_capacity = 4096

    experience_replay = ExperienceReplay(capacity=replay_capacity)

    episodes = 0

    while 1:
        observation, info = env.reset()
        print(f"{episodes=}")

        episode_over = False
        while not episode_over:
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
        episodes += 1

    env.close()

if __name__ == "__main__":
    main()
