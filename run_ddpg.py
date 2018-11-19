# normal DDPG

import numpy as np
import gym
from ddpg import DDPG
import matplotlib.pyplot as plt
import pickle

env = gym.make('MountainCarContinuous-v0')

agent = DDPG(a_dim=1, s_dim=2, a_bound=1)

total_reward = []
for episode in range(300):
    state = env.reset()
    var = 1
    cum_reward = 0
    for step in range(1000):
        action = np.clip(np.random.normal(agent.choose_action(state), var), -1, 1)
        next_state, reward, done, _ = env.step([action])
        # print(action)
        cum_reward += reward
        agent.store_transition(state, action, reward, next_state)
        state = next_state
        if done:
            print('Episode', episode, ' Complete at reward ', cum_reward, '!!!')
            break
        if step == 1000 - 1:
            print('Episode', episode, ' finished at reward ', cum_reward)
    total_reward.append(cum_reward)
    if var > 0.1:
        var -= 0.01
    for step in range(1000):
        agent.learn()

plt.plot(total_reward)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('MountainCar continuous')
plt.savefig('ddpg')
pickle.dump(total_reward,open('DDPG'))
