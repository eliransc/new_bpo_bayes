from simulator import Simulator
from planners import GreedyPlanner, ShortestProcessingTime, DedicatedResourcePlanner, PPOPlanner
from time import time
import os


model_name = 'high_utlization'
model_names = os.listdir('./tmp/')

running_time = 5000
# Original main
def simulate_competition():

    for model_name in model_names:
        config_type=model_name[:model_name.find('1')-1]
        results = []
        times = []
        log_dir='./results/'
        for i in range(100):
            if i % 5 == 0:
                print(i, config_type)
            #planner = DedicatedResourcePlanner()
            #planner = ShortestProcessingTime()
            planner = PPOPlanner(f'{model_name}/{config_type}_5000')
            simulator = Simulator(running_time, planner, config_type=config_type, reward_function='AUC', write_to=log_dir)
            
            if type(planner) == PPOPlanner:
                planner.linkSimulator(simulator)
            
            if i == 0:
                resource_str = ''
                for resource in simulator.resources:
                    resource_str += resource + ','
                with open(f'{simulator.write_to}{planner}_results_{simulator.config_type}.txt', "w") as file:
                    # Writing data to a file
                    file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")



            t1 = time()
            result = simulator.run()
            #print(f'Simulation finished in {time()-t1} seconds')
            print(result)
            times.append(time()-t1)
            results.append(result)      

        # with open(f'{simulator.write_to}{planner}_results_{simulator.config_type}.txt', "w") as out_file:
        #     for i in range(len(results)):
        #         out_file.write(f'{times[i]},{results[i]}\n')

def main():
    simulate_competition()

if __name__ == "__main__":
    main()


"""
Characteristics of a business process (compared to other processes)
-Probabilistic routing
-Resource eligibility
-Shared resources in activities
-Resource availabiltiy (breaks, off-time, meetings, etc.)
-Variance in processing times between resources

"""

