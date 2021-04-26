from dataclasses import dataclass
import time
from os.path import join, isdir, isfile
import numpy as np
from stable_baselines.common import set_global_seeds
import gym
import argparse

# local imports 
import config
from utils import utils

# 
# entrypoint
# 
if __name__ == '__main__':
    from main import Arguments, parse_arguments, run_agent, setup_default_environment 
    # ^ its weird but it lets __name__ be at the top
    
    arguments = Arguments(**parse_arguments())
    
    # 
    # environment setup
    # 
    env = setup_default_environment(arguments)
    
    # 
    # start the main processing loop
    # 
    run_agent(arguments, env)

# 
# customization args
# 
@dataclass
class Arguments:
    main_algorithm_name         : str  = 'sac'
    should_render               : bool = True
    number_of_timesteps         : int  = 1000
    experiment_number           : int  = 0
    logging_detail              : int  = 1
    random_generator_seed       : int  = 0
    use_best_model              : bool = False
    is_deterministic            : bool = False
    reward_should_be_normalized : bool = False
    main_log_folder             : str  = 'logs'
    reward_log_folder           : str  = ''
    default_folder              : str  = ''
    vae_path                    : str  = ''
    default_folder              : str  = ''
    log_path                    : str  = ''
    model_path                  : str  = 'jetcar_baseline_weights.pkl'
    stats_path                  : str  = ''
    reward_log_path             : str  = ''
    
    # 
    # some defaults need to be computed
    # 
    def __init__(self, **kwargs):
        # just copy all the args onto self
        for each_key, each_value in kwargs.items():
            setattr(self, each_key, each_value)
        
        # 
        # defaults
        # 
        self.default_folder      = join(self.main_log_folder, self.main_algorithm_name)
        self.experiment_number   = utils.get_latest_run_id(self.default_folder, config.ENV_ID)            if self.experiment_number == 0 else self.experiment_number
        self.log_path            = join(self.default_folder, f"{config.ENV_ID}_{self.experiment_number}") if self.experiment_number > 0  else self.default_folder
        self.best_name_extension = '_best' if self.use_best_model else ''
        self.model_path          = join(self.log_path, f"{config.ENV_ID}{self.best_name_extension}.pkl")
        self.stats_path          = join(self.log_path, config.ENV_ID)
        self.reward_log_path     = self.reward_log_folder if self.reward_log_folder != '' else self.log_path
        self.is_deterministic    = self.is_deterministic or (self.main_algorithm_name in ['ddpg', 'sac']) # Force deterministic for SAC and DDPG
        self.timesteps           = range(self.number_of_timesteps)

        # 
        # sanity checks
        # 
        import os
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.stats_path, exist_ok=True)
        os.makedirs(self.reward_log_path, exist_ok=True)
        assert isfile(self.model_path), f"No model found for {self.main_algorithm_name} on {config.ENV_ID}, path: {self.model_path}"

        # 
        # logging
        # 
        if self.logging_detail >= 1: print(f"Deterministic actions: {self.is_deterministic}")


# 
# if getting arguments from commandline
# 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(    '-f',        '--folder' , type=str            , default='logs' , required=False , choices=list(utils.ALGOS.keys()) , help='Log folder'                                                , )
    parser.add_argument(                   '--algo' , type=str            , default='sac'  ,                                                     help='RL Algorithm'                                              , )
    parser.add_argument(    '-n',   '--n-timesteps' , type=int            , default=1000   ,                                                     help='number of timesteps'                                       , )
    parser.add_argument(                 '--exp-id' , type=int            , default=0      ,                                                     help='Experiment ID (-1: no exp folder, 0: latest)'              , )
    parser.add_argument(                '--verbose' , type=int            , default=1      ,                                                     help='Verbose mode (0: no output, 1: INFO)'                      , )
    parser.add_argument(              '--no-render' , action='store_true' , default=False  ,                                                     help='Do not render the environment (useful for tests)'          , )
    parser.add_argument(          '--deterministic' , action='store_true' , default=False  ,                                                     help='Use deterministic actions'                                 , )
    parser.add_argument(            '--norm-reward' , action='store_true' , default=False  ,                                                     help='Normalize reward if applicable (trained with VecNormalize)', )
    parser.add_argument(                   '--seed' , type=int            , default=0      ,                                                     help='Random generator seed'                                     , )
    parser.add_argument(             '--reward-log' , type=str            , default=''     ,                                                     help='Where to log reward'                                       , )
    parser.add_argument(  '-vae',      '--vae-path' , type=str            , default=''     ,                                                     help='Path to saved VAE'                                         , )
    parser.add_argument( '-best',    '--best-model' , action='store_true' , default=False  ,                                                     help='Use best saved model of that experiment (if it exists)'    , )
    args = parser.parse_args()
    return {
        "main_algorithm_name": args.algo,
        "main_log_folder": args.folder,
        "reward_log_folder": args.reward_log,
        "experiment_number": args.exp_id,
        "reward_should_be_normalized": args.norm_reward,
        "logging_detail": args.verbose,
        "random_generator_seed": args.seed,
        "vae_path": args.vae_path,
        "number_of_timesteps": args.n_timesteps,
        "should_render": not args.no_render,
        "use_best_model": args.best_model,
        "is_deterministic": args.deterministic,
    }

# 
# agent's learning loop
# 
def run_agent(arguments: Arguments, env):
    # get the agent
    model = utils.ALGOS[arguments.main_algorithm_name].load(arguments.model_path)
    
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

# 
# env setup
# 
def setup_default_environment(arguments):
    """
    from stable_baselines.common import set_global_seeds
    Args:
        arguments.random_generator_seed
        arguments.reward_log_path
        arguments.reward_should_be_normalized
        arguments.stats_path
        arguments.vae_path
    """
    set_global_seeds(arguments.random_generator_seed)
    hyperparams, stats_path = utils.get_saved_hyperparams(arguments.stats_path, norm_reward=arguments.reward_should_be_normalized)
    hyperparams['vae_path'] = arguments.vae_path
    env = utils.create_test_env(stats_path=stats_path, seed=arguments.random_generator_seed, log_dir=arguments.reward_log_path, hyperparams=hyperparams)
    return env