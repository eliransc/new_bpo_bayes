from collections import deque
from subprocess import call
import gym
import os
import numpy as np
from bpo_env import BPOEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from gym.wrappers import normalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv



from callbacks import SaveOnBestTrainingRewardCallback
from callbacks import custom_scheduler


if __name__ == '__main__':
    #if true, load model for a new round of training
    
    running_time = 1000
    num_cpu = 1
    load_model = False
    model_name = "ppo_masked"
    config_type='slow_server' # slow_server, n_system, simple_linear
    # Create log dir
    log_dir = "./tmp/"

    os.makedirs(log_dir, exist_ok=True)


    # Create and wrap the environment
    env = BPOEnv(running_time=running_time, config_type=config_type, reward_function='AUC', write_to=log_dir)  # Initialize env
    env = Monitor(env, log_dir)  

    resource_str = ''
    for resource in env.simulator.resources:
        resource_str += resource + ','
    with open(f'{log_dir}results_{config_type}.txt', "w") as file:
        # Writing data to a file
        file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")
        
    #env = normalize.NormalizeReward(env) #rewards normalization
    #env = normalize.NormalizeObservation(env) #rewards normalization

    # Create the model
    model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=0.0001, n_steps=300, gamma=1, verbose=1)
    # custom_scheduler(0.001)
    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    #model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Create the callbacks list
    #log_callback = LogTraining(check_freq=3000, log_dir=log_dir)




    #callback = CallbackList([log_callback, checkpoint_callback])

    # Train the agent
    time_steps = 500000
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_name=f'{config_type}_{running_time}')
    model.learn(total_timesteps=int(time_steps), callback=callback)


    # For episode rewards, use env.get_episode_rewards()
    # env.get_episode_times() returns the wall clock time in seconds of each episode (since start)
    # env.rewards returns a list of ALL rewards. Of current episode?
    # env.episode_lengths returns the number of timesteps per episode
    # if num_cpu==1:
    print(env.get_episode_rewards())
    #     print(env.get_episode_times())


    model.save(model_name)

    import matplotlib.pyplot as plt
    plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, f"{model_name}")
    plt.show()