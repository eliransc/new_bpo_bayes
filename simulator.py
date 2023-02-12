import numpy as np
from enum import Enum, auto
from collections import deque
import random
from planners import GreedyPlanner

class Event:
    def __init__(self, event_type, moment, task=None, resource=None):
        self.event_type = event_type
        self.moment = moment              
        self.task = task
        self.resource = resource
        self.cycle_time = 0

    def __lt__(self, other):
        return self.moment < other.moment


class EventType(Enum):
    CASE_ARRIVAL = auto()
    CASE_DEPARTURE = auto()
    TASK_START = auto()
    TASK_COMPLETE = auto()
    PLAN_TASKS = auto()
    RETURN_REWARD = auto()


class Case:
    _id = 0

    def __init__(self, moment):
        self.id = Case._id
        Case._id += 1
        self.arrival_time = moment
        self.departure_time = None
        self.tasks = []
        self.uncompleted_tasks = []
        self.completed_tasks = []        

    def add_task(self, task):
        self.uncompleted_tasks.append(task)

    def complete_task(self, task):
        self.uncompleted_tasks.remove(task)
        self.completed_tasks.append(task)


class Task:
    _id = 0

    def __init__(self, moment, case_id, task_type):
        self.id = Task._id
        Task._id += 1
        self.start_time = None
        self.case_id = case_id
        self.task_type = task_type


class Simulator:    
    def __init__(self, running_time, planner, config_type, reward_function=None, write_to=None):
        self.config_type = config_type
        if self.config_type == 'slow_server':
            config = {  'mean_interarrival_time'  : 2, # Mean time between two arrivals
                        'task_types'              : ['Task A', 'Task B'],
                        'resources'               : ['Resource 1', 'Resource 2'],
                        'initial_task_dist'       : {'Task A':1, 'Task B':0},
                        'resource_pools'          : {'Task A': {'Resource 1':(1.8, 0),
                                                                'Resource 2':(1.4, 0)},
                                                     'Task B': {'Resource 1':(1.4, 0),
                                                                'Resource 2':(25, 0)}},
                        'transitions'             : {'Task A': [0, 1, 0],
                                                     'Task B': [0, 0, 1]}}
        elif self.config_type == 'n_system':
            config = {  'mean_interarrival_time'  : 1,
                        'task_types'              : ['Task A', 'Task B'],
                        'resources'               : ['Resource 1', 'Resource 2'],
                        'initial_task_dist'       : {'Task A':0.5, 'Task B':0.5},
                        'resource_pools'          : {'Task A': {'Resource 1':(1.5, 0),
                                                                'Resource 2':(1.0, 0)},
                                                     'Task B': {'Resource 2':(1.2, 0)}},
                        'transitions'             : {'Task A': [0, 0, 1],
                                                     'Task B': [0, 0, 1]}}
            
        elif self.config_type == 'simple_linear_low_utilisation':
            config = {  'mean_interarrival_time'  : 2,
                        'task_types'              : ['Task A', 'Task B'],
                        'resources'               : ['Resource 1', 'Resource 2'],
                        'initial_task_dist'       : {'Task A':1, 'Task B':0},
                        'resource_pools'          : {'Task A': {'Resource 1':(1.6, 0),
                                                                'Resource 2':(1.4, 0)},
                                                     'Task B': {'Resource 1':(1.4, 0),
                                                                'Resource 2':(1.6, 0)}},
                        'transitions'             : {'Task A': [0, 1, 0],
                                                     'Task B': [0, 0, 1]}}

        elif self.config_type == 'simple_linear_high_utilisation3':
            config = {  'mean_interarrival_time'  : 2,
                        'task_types'              : ['Task A', 'Task B'],
                        'resources'               : ['Resource 1', 'Resource 2'],
                        'initial_task_dist'       : {'Task A':1, 'Task B':0},
                        'resource_pools'          : {'Task A': {'Resource 1':(2.0, 1),
                                                                'Resource 2':(1.9, 1)},
                                                     'Task B': {'Resource 1':(1.9, 1),
                                                                'Resource 2':(2.0, 1)}},
                        'transitions'             : {'Task A': [0, 1, 0],
                                                     'Task B': [0, 0, 1]}}
            
        self.running_time = running_time
        self.status = "RUNNING"
        self.debug = False

        self.now = 0
        self.events = []
        
        self.sumx = 0
        self.sumxx = 0
        self.sumw = 0

        self.available_tasks = []
        self.reserved_tasks = []
        self.completed_tasks = []

        self.available_resources = []        
        self.reserved_resources = []
        
        self.uncompleted_cases = {}
        self.completed_cases = {}

        self.mean_interarrival_time = config['mean_interarrival_time']
        self.task_types = config['task_types']
        self.resources = config['resources']       
        self.initial_task_dist = config['initial_task_dist']
        self.resource_pools = config['resource_pools']
        self.transitions = config['transitions']

        self.planner = planner 
        if self.planner != None: self.planner.resource_pools = self.resource_pools

        self.resource_total_busy_time = {resource:0 for resource in self.resources}
        self.resource_last_start = {resource:0 for resource in self.resources}

        # Reinforcement learning
        self.input = [resource + '_availability' for resource in self.resources] + \
                     [resource + '_busy_time' for resource in self.resources] + \
                     [resource + '_to_task' for resource in self.resources] + \
                     self.task_types # Should be lists of strings
        
        self.output = [(resource, task) for task in self.task_types for resource in self.resources if resource in self.resource_pools[task]] + ['Postpone']

        self.reward_function = reward_function
        self.write_to = write_to
        self.current_reward = 0
        self.total_reward = 0

        self.last_reward_moment = 0

        self.init_simulation()


    def generate_initial_task(self, case_id):
        rvs = random.random()
        prob = 0
        for tt, p in self.initial_task_dist.items():
            prob += p
            if rvs <= prob:
                task_type = tt
                break
        return Task(self.now, case_id, task_type)

    def generate_next_task(self, current_task):
        rvs = random.random()
        prob = 0
        for i, p in enumerate(self.transitions[current_task.task_type]):
            if i == len(self.transitions[current_task.task_type]) - 1: # If the next task is not another task, complete the case
                next_task_type = 'Complete'
                break
            prob += p
            if rvs <= prob:
                next_task_type = self.task_types[i]
                break
        return Task(self.now, current_task.case_id, next_task_type)

    def generate_case(self):        
        case = Case(self.now)
        self.uncompleted_cases[case.id] = case
        return case

    def process_assignment(self, assignment):
        #print(assignment, [task.task_type for task in self.available_tasks], [resource for resource in self.available_resources])        
        self.available_resources.remove(assignment[0])
        self.available_tasks.remove(assignment[1])
        # self.events.append(Event(EventType.TASK_START, self.now, assignment[0], assignment[1]))
        # self.events.sort()      

        self.resource_last_start[assignment[0]] = self.now
        assignment[1].start_time = self.now
        pt = self.sample_processing_time(assignment[0], assignment[1].task_type)
        self.events.append(Event(EventType.TASK_COMPLETE, self.now + pt, assignment[1], assignment[0]))
        self.events.sort()


    def sample_interarrival_time(self):
        return random.expovariate(1/self.mean_interarrival_time)

    def sample_processing_time(self, resource, task):
        (mu, sigma) = self.resource_pools[task][resource]
        pt = random.gauss(mu, sigma)
        while pt < 0:
            pt = random.gauss(mu, sigma)
        return pt

    def init_simulation(self):
        self.available_resources = [resource for resource in self.resources]
        self.events.append(Event(EventType.CASE_ARRIVAL, self.sample_interarrival_time()))

    def get_state(self):
        ### Resource binary, busy time, assigned to + nr ofo each task
        resources_available = [1 if x in self.available_resources else 0 for x in self.resources]
        resources_busy_time = [0 for _ in range(len(self.resources))]
        resources_assigned = [0 for _ in range(len(self.resources))]
        for event in self.events:
            if event.event_type == EventType.TASK_COMPLETE:
                resource_index = self.resources.index(event.resource)
                resources_busy_time[resource_index] = self.now - event.task.start_time
                resources_assigned[resource_index] = self.task_types.index(event.task.task_type) + 1

        if len(self.available_tasks) > 0:
            task_types_num = [sum([1 if task.task_type == el else 0 for task in self.available_tasks]) for el in self.task_types]
        else:
            task_types_num = [0 for el in self.task_types]

        return resources_available + resources_busy_time + resources_assigned + task_types_num


        ### Resource binary + proportion tasks + total task
        # av_resources_ones = [1 if x in self.available_resources else 0 for x in self.resources]

        # # Number of tasks
        # # task_types_num =  [np.sum(el in [o.task_type for o in self.available_tasks]) for el in self.task_types] 

        # # Normalized to max
        # if len(self.available_tasks) > 0:
        #     task_types_num = [sum([1 if task.task_type == el else 0 for task in self.available_tasks])/len(self.available_tasks) for el in self.task_types]
        # else:
        #     task_types_num = [0 for el in self.task_types]

        # return av_resources_ones + task_types_num + [len(self.available_tasks)]

    def run(self):
        while self.now <= self.running_time:
            #print(self.get_state())
            event = self.events.pop(0)            
            self.now = event.moment
            if self.now <= self.running_time: # To prevent next event time after running time
                if event.event_type == EventType.CASE_ARRIVAL:
                    current_reward = (self.now - self.last_reward_moment) * len(self.uncompleted_cases)
                    self.current_reward -= current_reward
                    self.total_reward -= current_reward
                    self.last_reward_moment = self.now

                    case = self.generate_case() # Automatically added to dict of uncompleted cases
                    self.available_tasks.append(self.generate_initial_task(case.id))
                    self.events.append(Event(EventType.CASE_ARRIVAL, self.now + self.sample_interarrival_time())) # Schedule new arrival
                    if len(self.available_tasks) > 0 and len(self.available_resources) > 0:
                        self.events.append(Event(EventType.PLAN_TASKS, self.now))
                    self.events.sort()

                if event.event_type == EventType.PLAN_TASKS:
                    # Calculate reward
                    current_reward = (self.now - self.last_reward_moment) * len(self.uncompleted_cases)
                    self.current_reward -= current_reward
                    self.total_reward -= current_reward
                    self.last_reward_moment = self.now

                    if self.planner == None: #in training mode, we do not provide a planner
                        # there only is an assignment if there are free resources and tasks
                        if len(self.available_tasks) > 0 and len(self.available_resources) > 0:
                            break # Return to gym environment
                    else: #at inference time, we call the plan function of the planner
                        assignments = self.planner.plan(self.available_tasks, self.available_resources, self.resource_pools)
                        for assignment in assignments:
                            self.process_assignment(assignment) # Reserves the task and resource, schedules TASK_START event

                # if event.event_type == EventType.TASK_START:
                #     event.task.start_time = self.now
                #     pt = self.sample_processing_time(event.task.task_type, event.resource)
                #     self.events.append(Event(EventType.TASK_COMPLETE, self.now + pt, event.task, event.resource))
                #     self.events.sort()

                if event.event_type == EventType.TASK_COMPLETE:
                    # Release resource
                    self.available_resources.append(event.resource)
                    self.resource_total_busy_time[event.resource] += self.now - self.resource_last_start[event.resource]
                    # Complete task
                    self.completed_tasks.append(event.task)
                    next_task = self.generate_next_task(event.task)
                    if next_task.task_type == 'Complete':
                        self.events.append(Event(EventType.CASE_DEPARTURE, self.now, event.task))
                    else:
                        self.available_tasks.append(next_task)
                        
                    if len(self.available_tasks) > 0 and len(self.available_resources) > 0:
                        self.events.append(Event(EventType.PLAN_TASKS, self.now))
                    self.events.sort()

                if event.event_type == EventType.CASE_DEPARTURE:
                    # Calculate reward
                    current_reward = (self.now - self.last_reward_moment) * len(self.uncompleted_cases)
                    self.current_reward -= current_reward
                    self.total_reward -= current_reward
                    self.last_reward_moment = self.now           

                    case = self.uncompleted_cases[event.task.case_id]
                    case.departure_time = self.now
                    del self.uncompleted_cases[case.id]
                    self.completed_cases[event.task.case_id] = case

                    cycle_time = case.departure_time - case.arrival_time
                    self.sumx += cycle_time
                    self.sumxx += cycle_time * cycle_time
                    self.sumw += 1

        if self.now > self.running_time:
            current_reward = (self.running_time - self.last_reward_moment) * len(self.uncompleted_cases)
            self.current_reward -= current_reward
            self.total_reward -= current_reward
            self.last_reward_moment = self.now

            self.status = "FINISHED"
            for event in self.events:
                if event.event_type == EventType.TASK_COMPLETE:
                    self.resource_total_busy_time[event.resource] += self.running_time - self.resource_last_start[event.resource]
            for case in self.uncompleted_cases.values():
                cycle_time = self.running_time - case.arrival_time
                self.sumx += cycle_time
                self.sumxx += cycle_time * cycle_time
                self.sumw += 1

            print(f'Uncompleted cases: {len(self.uncompleted_cases)}')
            print(f'Resource utilisation: {[(resource, busy_time/self.running_time) for resource, busy_time in self.resource_total_busy_time.items()]}')
            print(f'Total reward: {self.total_reward}. Total CT: {self.sumx}')
            print(f'Mean cycle time: {self.sumx/self.sumw}. Standard deviation: {np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}')
            
            utilisation = [busy_time/self.running_time for resource, busy_time in self.resource_total_busy_time.items()]
            resource_str = ''
            for i in range(len(self.resources)):
                resource_str += f'{utilisation[i]},'
            if self.planner != None:
                with open(f'{self.write_to}{self.planner}_results_{self.config_type}.txt', "a") as file:
                    file.write(f"{len(self.uncompleted_cases)},{resource_str}{self.total_reward},{self.sumx/self.sumw},{np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}\n")
            else:
                with open(f'{self.write_to}results_{self.config_type}.txt', "a") as file:
                    file.write(f"{len(self.uncompleted_cases)},{resource_str}{self.total_reward},{self.sumx/self.sumw},{np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}\n")

            return len(self.uncompleted_cases),self.total_reward,self.sumx/self.sumw,np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)
        


                