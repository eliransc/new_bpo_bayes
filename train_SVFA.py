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

    simulator_fake = Simulator(running_time, ShortestProcessingTime(), config_type='complete', reward_function='AUC')
    a1 = A[0]
    a2 = A[1]
    a3 = A[2]
    a4 = A[3]
    a5 = A[4]
    a6 = A[5]
    a7 = A[6]
    # print(a1, a2, a3, a4, a5, a6, a7)
    planner = Bayes_planner(a1, a2, a3, a4, a5, a6,a7,simulator_fake)  # ShortestProcessingTime() # Insert your planner here, input can be the parameters of your model
    # planner1 = ShortestProcessingTime()

    # The config types dictates the system
    simulator = Simulator(running_time, planner, config_type='complete', reward_function='AUC')
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
    # print('stop')

    return CT_mean


def aggregate_sims(A): # a1, a2, a3, a4, a5, a6, a7
    import time
    # A = [ a1, a2, a3, a4, a5, a6, a7]
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    model_num = np.random.randint(0, 1000)
    tot_res = []

    for ind in range(1):
        res = simulate_competition(A)
        # print(res)
        tot_res.append(res)

    pkl.dump((A, tot_res), open('complete_opt1_' + str(model_num) + '.pkl', 'wb'))

    return -np.array(tot_res).mean()

 # open('single_bayes_' + 'high_utilization' + str(model_num) + '.pkl', 'wb'))

def main():

    high_utilization = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 4.620278449742129]

    low_utilization = [0.15, 0.15, 20.0, 0.15, 0.15, 0.15, 20.0]

    n_system = [0.015, 0.015, 20.0, 20.0, 6.173545342831394, 0.015, 10.0]

    down_stream = [0.15, 0.15, 20.0, 20.0, 18.911578855212127, 0.15, 20.0]

    # 3.5891129717483348, 0.015, 19.558505455930856, 7.3067908967685895, 2.4903986035491594, 10.125089778688992, 34.53567091112935
    complete = [0.09154961927740712, 2.912387229431585, 14.451013276379312, 29.36395671453757, 13.177016753562167, 12.861960177623466, 29.214770307928994]

    # 0.15, 7.96025975421663, 10.22775090867394, 20.0, 4.257145518125576, 0.15, 20.0
    slow_server = [0.015, 1.2484516706551265, 23.408393726166587, 25.290958991232003, 16.414990721120084, 5.583758355596822, 22.065706736839584]

    parallel = [0.15, 0.15, 20.0, 20.0, 20.0, 0.15, 8.28753107297809]

    complete_reversed = [ 14.655329405050026,
  2.2307821329123763,
  23.908040064026785,
  28.307739945329566,
  5.806455816634357,
  9.71514967582145,
  86.76148258110027]

    complete_parallel = [3.736149991524543,
  2.8272743051596576,
  22.42815830329431,
  8.581415287506237,
  22.296310430702498,
  7.678423005383678,
  69.5006129500131]

   # simulate_competition(A)

    A = complete

    get_results = aggregate_sims(A)

    import time
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    model_num = np.random.randint(0, 1000)


    pkl.dump(get_results, open(str(model_num) + '_complete_reversed.pkl', 'wb'))


if __name__ == "__main__":
    main()