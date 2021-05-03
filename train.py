# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
# Edited by Sheelabhadra Dey
# Edited again by Jeff Hykin

# builtin
import argparse
import os
from collections import OrderedDict
from pprint import pprint
import tensorflow as tf

# external
import numpy as np
import gym
from stable_baselines.common import set_global_seeds
from stable_baselines.ppo2.ppo2 import constfn

# internal
from config import Z_SIZE, BASE_ENV, ENV_ID, WHICH_ENV
from utils.utils import (
    make_env,
    ALGOS,
    linear_schedule,
    get_latest_run_id,
    create_callback,
    JoyStick,
)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 
# entrypoint
# 
if __name__ == '__main__':
    # import from this file (yeah its a little bit werid/recursive)
    from train import (
        Arguments,
        misc_setup,
        setup_env,
        ddpg_initilzation,
        fine_tuning_setup,
        training,
        save_info,
    )
    args = Arguments().get_args_from_cli()    
    vae, hyperparameters, tensorboard_log, normalize, save_path, n_timesteps, params_path, saved_hyperparameters = misc_setup(args)
    env, vae, n_stack, hyperparameters, joystick = setup_env(vae, hyperparameters, which_env=WHICH_ENV, joystick=None) # optional: joystick = JoyStick()
    hyperparameters = ddpg_initilzation(args, hyperparameters)
    model = fine_tuning_setup(args, env, hyperparameters, tensorboard_log, normalize)
    model = training(args, model, save_path, n_timesteps, joystick=None)
    save_info(args, model, vae, env, saved_hyperparameters, normalize, save_path, params_path)

# 
# Arguments
# 
import dataclasses
Arguments = None
@dataclasses.dataclass
class Arguments:
    algo                  : str  = "sac"
    n_timesteps           : int  = -1
    log_interval          : int  = -1
    tensorboard_log       : str  = ""
    log_folder            : str  = "logs"
    vae_path              : str  = ""
    save_vae              : bool = False
    seed                  : int  = 0
    expert_guidance_steps : int  = 50000
    base_policy_path      : str  = ""
    trained_agent         : str  = ""    

    def get_args_from_cli(self) -> Arguments:
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo"                 ,                  type=str,            default="sac",      required=False,       choices=list(ALGOS.keys()),     help="RL Algorithm",)
        parser.add_argument("--n-timesteps"          , "-n"           , type=int,            default=-1,                                                               help="Overwrite the number of timesteps"              ,)
        parser.add_argument("--log-interval"         ,                  type=int,            default=-1,                                                               help="Override log interval (default: -1, no change)" ,)
        parser.add_argument("--tensorboard-log"      , "-tb"          , type=str,            default="",                                                               help="Tensorboard log dir"                            ,)
        parser.add_argument("--log-folder"           , "-f"           , type=str,            default="logs",                                                           help="Log folder"                                     ,)
        parser.add_argument("--vae-path"             , "-vae"         , type=str,            default="",                                                               help="Path to saved VAE"                              ,)
        parser.add_argument("--save-vae"             ,                  action="store_true", default=False,                                                            help="Save VAE"                                       ,)
        parser.add_argument("--seed"                 ,                  type=int,            default=0,                                                                help="Random generator seed"                          ,)
        parser.add_argument("--expert-guidance-steps", "-expert-steps", type=int,            default=50000,                                                            help="Number of steps of expert guidance"             ,)
        parser.add_argument("--base-policy-path"     , "-base"        , type=str,            default="",                                                               help="Path to saved model for the base policy"        ,)
        parser.add_argument("--trained-agent"        , "-i"           , type=str,            default="",                                                               help="Path to a pretrained agent to continue training",)
        return parser.parse_args()

# 
# environment
# 
def setup_env(vae, hyperparameters, which_env="unity_racer_env", joystick=None):
    from envs.jet_racer import JetRacerEnv
    from envs.unity_racer_env import UnityRacerEnv
    from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
    
    # 
    # env choice
    # 
    if which_env == 'dummy':
        env = DummyVecEnv([make_env(args.seed, vae=vae)])
    elif which_env == 'jetracer':
        env = JetRacerEnv(vae=vae)
    elif which_env == 'unityracer':
        env = UnityRacerEnv(vae=vae)

    # 
    # Optional Frame-stacking
    # 
    n_stack = 1
    if hyperparameters.get("frame_stack", False):
        n_stack = hyperparameters["frame_stack"]
        if not args.teleop:
            env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparameters["frame_stack"]
    
    return env, vae, n_stack, hyperparameters, joystick


# 
# supporting setup
# 
def misc_setup(args):
    from config import  Z_SIZE, BASE_ENV, ENV_ID
    
    # 
    # init misc
    # 
    set_global_seeds(args.seed)
    tensorboard_log = (None if args.tensorboard_log == "" else args.tensorboard_log + "/" + ENV_ID)
    print("=" * 10, ENV_ID, args.algo, "=" * 10)
    
    # 
    # setup VAE
    # 
    from vae.controller import VAEController
    from os.path import isfile
    vae_path = args.vae_path
    if type(vae_path) == str and vae_path != "":
        print("VAE: Loading pretrained")
        assert not isfile(vae_path), f"VAE: trying to load, but no file found at {vae_path}"
        vae = VAEController(z_size=None)
        vae.load(vae_path)
        print("VAE: Loaded")
    else:
        print(f"VAE: Randomly initializing with size {Z_SIZE}")
        vae = VAEController(z_size=Z_SIZE)
        # Save network if randomly initilizing
        args.save_vae = True
    print(f"VAE: number of latent variables (z_size): {vae.z_size}")

    # 
    # setup hyperparameters
    # 
    import yaml
    import os
    hyperparameters_path = f"hyperparams/{args.algo}.yml"
    with open(hyperparameters_path, "r") as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)[BASE_ENV]
    # Sort
    saved_hyperparameters = OrderedDict([(key, hyperparameters[key]) for key in sorted(hyperparameters.keys())])
    saved_hyperparameters["vae_path"] = args.vae_path
    if vae is not None: saved_hyperparameters["z_size"] = vae.z_size
    
    # 
    # setup paths
    # 
    log_path = os.path.join(args.log_folder, args.algo)
    save_path = os.path.join(log_path, "{}_{}".format(ENV_ID, get_latest_run_id(log_path, ENV_ID) + 1))
    params_path = os.path.join(save_path, ENV_ID)
    os.makedirs(params_path, exist_ok=True)
    
    # 
    # learning rate
    # 
    # Create learning rate schedules for ppo2 and sac
    if args.algo in ["ppo2", "sac"]:
        for key in ["learning_rate", "cliprange"]:
            if key not in hyperparameters:
                continue
            if isinstance(hyperparameters[key], str):
                schedule, initial_value = hyperparameters[key].split("_")
                initial_value = float(initial_value)
                hyperparameters[key] = linear_schedule(initial_value)
            elif isinstance(hyperparameters[key], float):
                hyperparameters[key] = constfn(hyperparameters[key])
            else:
                raise ValueError("Invalid valid for {}: {}".format(key, hyperparameters[key]))
    # 
    # timesteps
    # 
    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparameters["n_timesteps"])
    del hyperparameters["n_timesteps"]
    
    # 
    # normalization
    # 
    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparameters.keys():
        normalize = hyperparameters["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparameters["normalize"]
    return vae, hyperparameters, tensorboard_log, normalize, save_path, n_timesteps, params_path, saved_hyperparameters

#
# ddpg (basically optional)
# 
def ddpg_initilzation(args, hyperparameters):
    from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
    # Parse noise string for DDPG
    if args.algo == "ddpg" and hyperparameters.get("noise_type") is not None:
        noise_type = hyperparameters["noise_type"].strip()
        noise_std = hyperparameters["noise_std"]
        n_actions = env.action_space.shape[0]
        if "adaptive-param" in noise_type:
            hyperparameters["param_noise"] = AdaptiveParamNoiseSpec(
                initial_stddev=noise_std, desired_action_stddev=noise_std
            )
        elif "normal" in noise_type:
            hyperparameters["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
            )
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparameters["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
            )
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparameters["noise_type"]
        del hyperparameters["noise_std"]
    
    return hyperparameters

# 
# load pretrained
# 
def fine_tuning_setup(args, env, hyperparameters, tensorboard_log, normalize):
    import os
    # Check if this does RL fine-tuning
    if args.trained_agent.endswith(".pkl") and os.path.isfile(args.trained_agent):
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparameters["policy"]

        model = ALGOS[args.algo].load(
            args.trained_agent,
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=1,
            **hyperparameters
        )

        exp_folder = args.trained_agent.split(".pkl")[0]
        if normalize:
            print("Loading saved running average")
            env.load_running_average(exp_folder)
    else:
        # Train an agent from scratch
        algorithm = ALGOS[args.algo]
        model = algorithm(
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=1,
            **hyperparameters,
        )
    return model

# 
# configure argumentss
#
def training(args, model, save_path, n_timesteps, joystick=None):
    import os
    import keras
    from config import ENV_ID
    should_use_base_policy = args.base_policy_path != ""
    
    # 
    # default learning arguments
    # 
    kwargs = {"save_path": save_path}
    if args.log_interval > -1 : kwargs.update({"log_interval": args.log_interval})
    if args.algo == "sac"     : kwargs.update({"callback"    : create_callback(args.algo, os.path.join(save_path, ENV_ID + "_best"), verbose=1)} )
    if not should_use_base_policy: 
        # Train agent from scratch
        model.learn(n_timesteps, **kwargs) 
    # 
    # based policy learning
    # 
    else: # Train agent using JIRL
        print("Loading Base Policy for JIRL ...")
        model.learn_jirl(
            n_timesteps,
            **kwargs,
            base_policy=keras.models.load_model(args.base_policy_path),
            joystick=joystick,
            expert_guidance_steps=args.expert_guidance_steps,
        )
    
    return model

# 
# save model, hyperparameters, vae, env-stats
# 
def save_info(args, model, vae, env, saved_hyperparameters, normalize, save_path, params_path):
    import os
    import yaml
    from stable_baselines.common.vec_env import VecFrameStack
    from config import ENV_ID
    
    # 
    # model 
    # 
    model.save(os.path.join(save_path, ENV_ID))
    
    # 
    # hyperparameters
    # 
    with open(os.path.join(params_path, "config.yml"), "w") as f:
        yaml.dump(saved_hyperparameters, f)
    
    # 
    # vae
    # 
    if args.save_vae and vae is not None:
        print("Saving VAE")
        vae.save(os.path.join(params_path, "vae"))
    
    # 
    # env
    # 
    if normalize:
        # Unwrap
        if isinstance(env, VecFrameStack):
            env = env.venv
        # Important: save the running average, for testing the agent we need that normalization
        env.save_running_average(params_path)
