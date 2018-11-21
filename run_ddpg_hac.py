# hierarchical Actor-critic
# 参照Hierachical Actor-critic 这篇论文

import numpy as np
from ddpg_her import DDPG_HER
import gym
import matplotlib.pyplot as plt
import pickle

k = 4
test = 0.5
total_reward = []


def get_reward(state, goal):  # only for the continous mountain car problem.
    value = np.abs(state[0] - goal)
    if np.abs(value) == 0:
        return 0
    else:
        return -1


controller = DDPG_HER(1, 2, 1, 1, name_scope='controller')
meta_controller = DDPG_HER(1, 2, 1, 1,
                           name_scope='meta_controller')  # meta_controller的bound要设下 [-1.2,0.6] 0.9*output-0.3即可
env = gym.make('MountainCarContinuous-v0')
# meta-controller is used to find a goal

for episode in range(300):
    cum_reward = 0
    state = env.reset()
    meta_trajectory = []
    goal = 0.45
    var = 1
    goal_achieved = False
    testing_signal = False
    for step1 in range(20):  # for meta-controller
        meta_state = state
        meta_action = meta_controller.choose_action(meta_state, goal)
        meta_action = meta_action * 0.9 - 0.3  # scale to [-1.2,0.6]
        if np.random.uniform() > test:  # 加噪音执行
            meta_action = np.clip(np.random.normal(meta_action, var), -1.2, 0.6)
            # this goal is used to select subgoals
        else:
            testing_signal = True
        primitive_state = meta_state
        primitive_trajectory = []
        for step2 in range(50):  # controller select a suitable goal
            primitive_action = controller.choose_action(primitive_state, meta_action)
            if not testing_signal:
                primitive_action = np.clip(np.random.normal(primitive_action, var), -1, 1)
            primitive_next_state, reward, done, _ = env.step([primitive_action])
            cum_reward += reward
            primitive_trajectory.append([primitive_state, primitive_action, primitive_next_state])
            primitive_state = primitive_next_state
            if get_reward(primitive_state, meta_action) is 0 or done:
                break
        # perform level 0 HER
        for element in primitive_trajectory:
            cp_state = element[0]
            cp_action = element[1]
            cp_next_state = element[2]
            cp_reward = get_reward(cp_next_state, meta_action)
            controller.store_transition(cp_state, cp_action, cp_reward, cp_next_state, meta_action)
            indices = np.random.choice(len(element), k)
            for indices in indices:
                sample_element = primitive_trajectory[indices]
                sample_sate = sample_element[2]
                sample_goal = sample_sate[0]
                cp_reward = get_reward(cp_next_state, sample_goal)
                controller.store_transition(cp_state, cp_action, cp_reward, cp_next_state, sample_goal)
        meta_next_state = primitive_state
        if np.abs(primitive_state[0] - meta_action) is not 0 and testing_signal:
            meta_controller.store_transition(meta_state, meta_action, -100, meta_next_state, goal)
        meta_trajectory.append([meta_state, primitive_action, meta_next_state])
        if done:  # if achieve the goal then print
            print('Episode ', episode, 'complete at reward ', cum_reward, '!!!')
            break
    total_reward.append(cum_reward)
    # perform level 1 HER
    if step1 is 20 - 1 and step2 is 50 - 1:
        print('Episode ', episode, 'finished at reward ', cum_reward)
    for element in meta_trajectory:
        cp_state = element[0]
        cp_action = element[1]
        cp_next_state = element[2]
        cp_reward = get_reward(cp_next_state, meta_action)
        meta_controller.store_transition(cp_state, cp_action, cp_reward, cp_next_state, meta_action)
        indices = np.random.choice(len(element), k)
        for indices in indices:
            sample_element = meta_trajectory[indices]
            sample_sate = sample_element[2]
            sample_goal = sample_sate[0]
            cp_reward = get_reward(cp_next_state, sample_goal)
            meta_controller.store_transition(cp_state, cp_action, cp_reward, cp_next_state, sample_goal)
    for step in range(1000):
        meta_controller.learn()
        controller.learn()
    if var > 0.1:
        var -= 0.01

plt.plot(total_reward)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('MountainCar continuous')
plt.savefig('ddpg_her')
pickle.dump(total_reward, open('DDPG_HER'))
