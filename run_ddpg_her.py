# DDPG with HER

import numpy as np
import gym
from ddpg_her import DDPG_HER
import matplotlib.pyplot as plt
import pickle


def get_reward(state, goal):
    value = np.abs(state[0] - goal)
    if np.abs(value) == 0:
        return 0
    else:
        return -1


k = 4
env = gym.make('MountainCarContinuous-v0')
agent = DDPG_HER(1, 2, 1, 1)

total_reward = []
for episode in range(300):
    state = env.reset()
    cum_reward = 0
    trajectory = []
    goal = 0.45
    var = 1
    for step in range(1000):  # execution
        action = np.clip(np.random.normal(agent.choose_action(state, goal), var), -1, 1)
        next_state, reward, done, _ = env.step([action])
        cum_reward += reward
        trajectory.append([state, action, next_state])
        state = next_state
        if done:
            print('Episode', episode, ' Complete at reward ', cum_reward, '!!!')
            break
        if step == 1000 - 1:
            print('Episode', episode, ' finished at reward ', cum_reward)
    for element in trajectory:
        replay_state = element[0]
        replay_action = element[1]
        replay_next_state = element[2]
        replay_reward = get_reward(replay_state, goal)
        agent.store_transition(replay_state, replay_action, replay_reward, replay_next_state, goal)
        indices = np.random.choice(len(trajectory), k)
        for index in indices:
            sampled_element = trajectory[index]
            sampled_goal = sampled_element[2]
            replay_reward = get_reward(replay_next_state, sampled_goal[0])
            agent.store_transition(replay_state, replay_action, replay_reward, replay_next_state, sampled_goal[0])
    for step in range(1000):
        agent.learn()
    total_reward.append(cum_reward)
    var -= 0.01

plt.plot(total_reward)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('MountainCar continuous')
plt.savefig('ddpg_her')
pickle.dump(total_reward, open('DDPG_HER'))
