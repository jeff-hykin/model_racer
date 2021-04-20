import include
import gym
UnityToGymWrapper = include.file("/Users/jeffhykin/repos/ml-agents/gym-unity/gym_unity/envs/__init__.py", { "__file__": __file__ }).UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

parameters = {
    "number_of_episodes": 10,
}

env = UnityToGymWrapper(
    UnityEnvironment(),
    allow_multiple_obs=True, # not exactly sure what this does,
)

try:
    # Reset the environment
    initial_observations = env.reset()
    print('initial_observations is:')
    print(initial_observations)

    for episode_index in range(parameters["number_of_episodes"]):
        env.reset()
        done = False
        episode_rewards = 0
        while not done:
            # take a random action
            actions = env.action_space.sample()
            obs, reward, done, _ = env.step(actions)
            episode_rewards += reward
        print(f"Total reward this episode: {episode_rewards}")
finally:
    env.close()