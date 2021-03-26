# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin

import argparse
import os
import time
import gym
import numpy as np
from stable_baselines.common import set_global_seeds
from os.path import join, isdir, isfile

# local imports 
import config
from utils import utils

# 
# 
# arguments + argument processing
# 
# 
if True:
    parser = argparse.ArgumentParser()
    parser.add_argument(    '-f',        '--folder' , type=str            , default='logs' , required=False , choices=list(utils.ALGOS.keys()) , help='Log folder'                                                , )
    parser.add_argument(                   '--algo' , type=str            , default='sac'  ,                ,                                  , help='RL Algorithm'                                              , )
    parser.add_argument(    '-n',   '--n-timesteps' , type=int            , default=1000   ,                ,                                  , help='number of timesteps'                                       , )
    parser.add_argument(                 '--exp-id' , type=int            , default=0      ,                ,                                  , help='Experiment ID (-1: no exp folder, 0: latest)'              , )
    parser.add_argument(                '--verbose' , type=int            , default=1      ,                ,                                  , help='Verbose mode (0: no output, 1: INFO)'                      , )
    parser.add_argument(              '--no-render' , action='store_true' , default=False  ,                ,                                  , help='Do not render the environment (useful for tests)'          , )
    parser.add_argument(          '--deterministic' , action='store_true' , default=False  ,                ,                                  , help='Use deterministic actions'                                 , )
    parser.add_argument(            '--norm-reward' , action='store_true' , default=False  ,                ,                                  , help='Normalize reward if applicable (trained with VecNormalize)', )
    parser.add_argument(                   '--seed' , type=int            , default=0      ,                ,                                  , help='Random generator seed'                                     , )
    parser.add_argument(             '--reward-log' , type=str            , default=''     ,                ,                                  , help='Where to log reward'                                       , )
    parser.add_argument(  '-vae',      '--vae-path' , type=str            , default=''     ,                ,                                  , help='Path to saved VAE'                                         , )
    parser.add_argument( '-best',    '--best-model' , action='store_true' , default=False  ,                ,                                  , help='Use best saved model of that experiment (if it exists)'    , )
    args = parser.parse_args()

    # create full name local variables of args
    main_algorithm_name, main_log_folder, reward_log_folder, experiment_number, reward_should_be_normalized, logging_detail, random_generator_seed, vae_path, number_of_timesteps, should_render = (args.algo, args.folder, args.reward_log, args.exp_id, args.norm_reward, args.verbose, args.seed, args.vae_path, args.n_timesteps, not args.no_render, )

    # 
    # defaults
    # 
    default_folder    = join(main_log_folder, main_algorithm_name)
    experiment_number = utils.get_latest_run_id(default_folder, config.ENV_ID)       if experiment_number == 0 else experiment_number
    log_path          = join(default_folder, f"{config.ENV_ID}_{experiment_number}") if experiment_number > 0  else default_folder
    best_path         = '_best'                                                      if args.best_model        else ''
    model_path        = join(log_path, f"{config.ENV_ID}{best_path}.pkl")
    stats_path        = join(log_path, config.ENV_ID)
    reward_log_path   = reward_log_folder if reward_log_folder != '' else None
    deterministic     = args.deterministic or (main_algorithm_name in ['ddpg', 'sac']) # Force deterministic for SAC and DDPG
    timesteps         = range(number_of_timesteps)

    # 
    # sanity checks
    # 
    assert isdir(log_path), f"The {log_path} folder was not found"
    assert isfile(model_path), f"No model found for {main_algorithm_name} on {config.ENV_ID}, path: {model_path}"

    # 
    # logging
    # 
    if logging_detail >= 1: print(f"Deterministic actions: {deterministic}")




# 
# 
# actual logic
# 
# 
if True:
    # 
    # setup
    # 
    set_global_seeds(random_generator_seed)
    hyperparams, stats_path = utils.get_saved_hyperparams(stats_path, norm_reward=reward_should_be_normalized)
    hyperparams['vae_path'] = vae_path
    env = utils.create_test_env(stats_path=stats_path, seed=args.seed, log_dir=reward_log_path, hyperparams=hyperparams)
    model = utils.ALGOS[main_algorithm_name].load(model_path)

    # 
    # init loop
    # 
    observation = env.reset()
    running_reward = 0.0
    episode_length = 0
    for each_timestep in timesteps:
        
        # 
        # ask the model what it is going to do
        # 
        action, _ = model.predict(observation, deterministic=deterministic)
        
        # clip the action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box): action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # 
        # do what the model asked
        # 
        observation, reward, done, infos = env.step(action)
        
        # [stats/logging]
        if should_render: env.render('human')
        running_reward += reward[0]
        episode_length += 1
        
        # exit condition
        if done and logging_detail >= 1:
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
