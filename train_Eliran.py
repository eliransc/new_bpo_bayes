from simulator import Simulator
from planners import GreedyPlanner, ShortestProcessingTime, DedicatedResourcePlanner, PPOPlanner, Bayes_planner
import pandas as pd

running_time = 5000
import numpy as np
import pickle as pkl


# You can build your bayesian optimization model around this framework:
# -Determine parameters for the planner
# -Run the simulation with the planner
# -Get the total_reward
def simulate_competition(A):

    simulator_fake = Simulator(running_time, ShortestProcessingTime(), config_type='high_utilization', reward_function='AUC')
    a1 = A[0]
    a2 = A[1]
    a3 = A[2]
    a4 = A[3]
    a5 = A[4]
    a6 = A[5]
    print(a1, a2, a3, a4, a5, a6)
    planner = Bayes_planner(a1, a2, a3, a4, a5, a6,
                            simulator_fake)  # ShortestProcessingTime() # Insert your planner here, input can be the parameters of your model
    planner1 = ShortestProcessingTime()

    # The config types dictates the system
    simulator = Simulator(running_time, planner, config_type='high_utilization', reward_function='AUC')
    # You can access some proporties from the simulation:
    # simulator.resource_pools: for each tasks 1) the resources that can process it and 2) the mean and variance of the processing time of that assignment
    # simulator.mean_interarrival_time
    # simulator.task_types
    # simulator.resources
    # simulator.initial_task_dist
    # simulator.resource_pools
    # simulator.transitions

    # You should want to optimize the total_reward, this is related to the mean cycle time, however the total reward also includes uncompleted casese
    # Total reward = total cycle time
    nr_uncompleted_cases, total_reward, CT_mean, CT_std, = simulator.run()
    print('stop')

    return CT_mean


def aggregate_sims(A):
    import time
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    model_num = np.random.randint(0, 1000)
    tot_res = []

    for ind in range(5):
        res = simulate_competition(A)
        print(res)
        tot_res.append(res)
        # pkl.dump(tot_res, open('run_500_res_simple_linear_high_utilisation3_' + str(model_num) + '.pkl', 'wb'))

    return np.array(tot_res).mean()


def main():
    A = [1, 1, 1, 1, 1, 5]
    simulate_competition(A)


if __name__ == "__main__":
    main()