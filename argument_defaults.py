# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
# adapted again by Jeff Hykin

from dataclasses import dataclass
import argparse
from os.path import join, isdir, isfile

# local imports 
import config
from utils import utils


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