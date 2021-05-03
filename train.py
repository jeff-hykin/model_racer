# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
# Edited by Sheelabhadra Dey
# Edited again by Jeff Hykin
import argparse
import os
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
import yaml
import gym
import keras
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import (
    AdaptiveParamNoiseSpec,
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines.ppo2.ppo2 import constfn

from config import (
    Z_SIZE,
    BASE_ENV,
    ENV_ID,
)
from utils.utils import (
    make_env,
    ALGOS,
    linear_schedule,
    get_latest_run_id,
    load_vae,
    create_callback,
    JoyStick,
)

from envs.vae_env import JetVAEEnv

# 
# entrypoint
# 
if __name__ == '__main__':
    from train import Arguments, misc_setup,
    
    args = Arguments().get_args_from_cli()
    
    vae, hyperparams, tensorboard_log, normalize, save_path, n_timesteps, params_path, saved_hyperparams = misc_setup(args)


import dataclasses
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
    random_features       : bool = False
    expert_guidance_steps : int  = 50000
    base_policy_path      : str  = "logs/sac/DonkeyVae-v0-level-0_2/DonkeyVae-v0-level-0_best.pkl"
    trained_agent         : str  = ""    

    def get_args_from_cli(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo"                 ,                  type=str,            default="sac",      required=False,       choices=list(ALGOS.keys()),     help="RL Algorithm",)
        parser.add_argument("--n-timesteps"          , "-n"           , type=int,            default=-1,                                                               help="Overwrite the number of timesteps"              ,)
        parser.add_argument("--log-interval"         ,                  type=int,            default=-1,                                                               help="Override log interval (default: -1, no change)" ,)
        parser.add_argument("--tensorboard-log"      , "-tb"          , type=str,            default="",                                                               help="Tensorboard log dir"                            ,)
        parser.add_argument("--log-folder"           , "-f"           , type=str,            default="logs",                                                           help="Log folder"                                     ,)
        parser.add_argument("--vae-path"             , "-vae"         , type=str,            default="",                                                               help="Path to saved VAE"                              ,)
        parser.add_argument("--save-vae"             ,                  action="store_true", default=False,                                                            help="Save VAE"                                       ,)
        parser.add_argument("--seed"                 ,                  type=int,            default=0,                                                                help="Random generator seed"                          ,)
        parser.add_argument("--random-features"      ,                  action="store_true", default=False,                                                            help="Use random features"                            ,)
        parser.add_argument("--expert-guidance-steps", "-expert-steps", type=int,            default=50000,                                                            help="Number of steps of expert guidance"             ,)
        parser.add_argument("--base-policy-path"     , "-base"        , type=str,            default="logs/sac/DonkeyVae-v0-level-0_2/DonkeyVae-v0-level-0_best.pkl",  help="Path to saved model for the base policy"        ,)
        parser.add_argument("--trained-agent"        , "-i"           , type=str,            default="",                                                               help="Path to a pretrained agent to continue training",)
        return parser.get_args_from_cli()


def misc_setup(arg):
    # 
    # init misc
    # 
    set_global_seeds(args.seed)
    tensorboard_log = (None if args.tensorboard_log == "" else args.tensorboard_log + "/" + ENV_ID)
    print("=" * 10, ENV_ID, args.algo, "=" * 10)
    
    
    # 
    # setup VAE
    # 
    vae = None
    if args.vae_path != "":
        print("Loading VAE ...")
        vae = load_vae(args.vae_path)
    elif args.random_features:
        print("Randomly initialized VAE")
        vae = load_vae(z_size=Z_SIZE)
        # Save network
        args.save_vae = True
    else:
        print("Learning from pixels...")

    # 
    # setup hyperparameters
    # 
    # load from yaml
    with open("hyperparams/{}.yml".format(args.algo), "r") as f:
        hyperparams = yaml.load(f)[BASE_ENV]
    # Sort
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    saved_hyperparams["vae_path"] = args.vae_path
    if vae is not None: saved_hyperparams["z_size"] = vae.z_size
    
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
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], float):
                hyperparams[key] = constfn(hyperparams[key])
            else:
                raise ValueError("Invalid valid for {}: {}".format(key, hyperparams[key]))
    # 
    # timesteps
    # 
    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams["n_timesteps"])
    del hyperparams["n_timesteps"]
    
    # 
    # normalization
    # 
    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams["normalize"]
    return vae, hyperparams, tensorboard_log, normalize, save_path, n_timesteps, params_path, saved_hyperpara

# Create the environment
# env = DummyVecEnv([make_env(args.seed, vae=vae)])
env = JetVAEEnv(vae=vae)

# Create the joystick object
js = None  # JoyStick()

# Optional Frame-stacking
n_stack = 1
if hyperparams.get("frame_stack", False):
    n_stack = hyperparams["frame_stack"]
    if not args.teleop:
        env = VecFrameStack(env, n_stack)
    print("Stacking {} frames".format(n_stack))
    del hyperparams["frame_stack"]

# Parse noise string for DDPG
if args.algo == "ddpg" and hyperparams.get("noise_type") is not None:
    noise_type = hyperparams["noise_type"].strip()
    noise_std = hyperparams["noise_std"]
    n_actions = env.action_space.shape[0]
    if "adaptive-param" in noise_type:
        hyperparams["param_noise"] = AdaptiveParamNoiseSpec(
            initial_stddev=noise_std, desired_action_stddev=noise_std
        )
    elif "normal" in noise_type:
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
    elif "ornstein-uhlenbeck" in noise_type:
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
    else:
        raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
    print("Applying {} noise with std {}".format(noise_type, noise_std))
    del hyperparams["noise_type"]
    del hyperparams["noise_std"]

# Check if this does RL fine-tuning
if args.trained_agent.endswith(".pkl") and os.path.isfile(args.trained_agent):
    # Continue training
    print("Loading pretrained agent")
    # Policy should not be changed
    del hyperparams["policy"]

    model = ALGOS[args.algo].load(
        args.trained_agent,
        env=env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        **hyperparams
    )

    exp_folder = args.trained_agent.split(".pkl")[0]
    if normalize:
        print("Loading saved running average")
        env.load_running_average(exp_folder)
else:
    # Train an agent from scratch
    model = ALGOS[args.algo](
        env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams
    )

agent = None
should_use_base_policy = args.base_policy_path != ""
if should_use_base_policy:
    print("Loading Base Policy for JIRL ...")
    agent = keras.models.load_model(args.base_policy_path)

# 
# configure argumentss
# 
kwargs = {
    "save_path": save_path
}
if args.log_interval > -1 : kwargs.update({"log_interval": args.log_interval})
if args.algo == "sac"     : kwargs.update({ "callback": create_callback(args.algo, os.path.join(save_path, ENV_ID + "_best"), verbose=1)} )
if should_use_base_policy : kwargs.update({"base_policy": agent})
if should_use_base_policy : kwargs.update({"expert_guidance_steps": args.expert_guidance_steps})
if should_use_base_policy : kwargs.update({"joystick": js})

# 
# training
# 
if not should_use_base_policy:
    # Train agent from scratch
    model.learn(n_timesteps, **kwargs) 
else:
    # Train agent using JIRL
    model.learn_jirl(n_timesteps, **kwargs)

# 
# Save info
# 
# model 
model.save(os.path.join(save_path, ENV_ID))
# hyperparams
with open(os.path.join(params_path, "config.yml"), "w") as f:
    yaml.dump(saved_hyperparams, f)
# vae
if args.save_vae and vae is not None:
    print("Saving VAE")
    vae.save(os.path.join(params_path, "vae"))

if normalize:
    # Unwrap
    if isinstance(env, VecFrameStack):
        env = env.venv
    # Important: save the running average, for testing the agent we need that normalization
    env.save_running_average(params_path)
