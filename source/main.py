import gym
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

import time
import numpy as np

from modules.input_preprocessing_utils import *
from modules.generic_python_tools import *

parameters = {
    "number_of_episodes": 10,
}

env = UnityToGymWrapper(
    UnityEnvironment(),
    allow_multiple_obs=True, # not exactly sure what this does,
)

try:
    """
    Observation: [Camera Input (20, 20, 1), (angle_difference, LidarRay FrontRight, LidarRay FrontMiddle, LidarRay FrontLeft, LidarRayBackMiddle)
    Action Space: [angularSpeed? -1(left):1(right), speed? -1(back):1(forward)]
    """
    # Reset the environment
    initial_observations = env.reset()
    print('initial_observations is:')
    print('action space range:')
    print('High: ',env.action_space.high)
    print('Low: ',env.action_space.low)

    for episode_index in range(parameters["number_of_episodes"]):
        env.reset()
        done = False
        episode_rewards = 0
        while not done:
            # take a random action
            actions = env.action_space.sample()  # rotation, acceleration
            actions = np.array([0,0.1])
            obs, reward, done, _ = env.step(actions)
            episode_rewards += reward
        print(f"Total reward this episode: {episode_rewards}")
finally:
    env.close()
