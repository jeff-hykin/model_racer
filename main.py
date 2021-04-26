import numpy as np
import time
import gym
from stable_baselines.common import set_global_seeds
from os.path import join, isdir, isfile

# local imports 
from utils import utils
from argument_defaults import Arguments, parse_arguments

# 
# entrypoint
# 
def main(arguments: Arguments):
    # 
    # environment setup
    # 
    set_global_seeds(arguments.random_generator_seed)
    hyperparams, stats_path = utils.get_saved_hyperparams(arguments.stats_path, norm_reward=arguments.reward_should_be_normalized)
    hyperparams['vae_path'] = arguments.vae_path
    env = utils.create_test_env(stats_path=stats_path, seed=arguments.random_generator_seed, log_dir=arguments.reward_log_path, hyperparams=hyperparams)
    model = utils.ALGOS[arguments.main_algorithm_name].load(arguments.model_path)
    
    # 
    # start the processing loop
    # 
    return run_agent(arguments, env)

# 
# agent's learning loop
# 
def run_agent(arguments: Arguments, env):
    observation = env.reset()
    running_reward = 0.0
    episode_length = 0
    for each_timestep in timesteps:
        
        # 
        # ask the model what it is going to do
        # 
        action, _ = model.predict(observation, deterministic=arguments.is_deterministic)
        
        # clip the action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box): action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # 
        # do what the model asked
        # 
        observation, reward, done, infos = env.step(action)
        
        # [stats/logging]
        if arguments.should_render: env.render('human')
        running_reward += reward[0]
        episode_length += 1
        
        # exit condition
        if done and arguments.logging_detail >= 1:
            # NOTE: for env using VecNormalize, the mean reward
            # is a normalized reward when `--norm_reward` flag is passed
            print("Episode Reward: {:.2f}".format(running_reward))
            print("Episode Length:", episode_length)
            running_reward = 0.0
            episode_length = 0
    
    # 
    # clean up the environment
    # 
    env.reset()
    env.envs[0].env.exit_scene()
    try:
        # looks like there was an issue here before
        env.envs[0].env.close_connection()
    except Exception as error:
        print('trouble with env.close_connection() in '+str(__file__))
        time.sleep(0.5)
