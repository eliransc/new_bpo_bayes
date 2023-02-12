import random
from abc import ABC, abstractmethod
import pandas as pd


class Bayes_planner:

    def __str__(self) -> str:
        return 'Bayesian'

    def __init__(self, a1, a2, a3, a4, a5, a6, simulator1):

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6

        df = pd.DataFrame([])

        tasks = [key for key in simulator1.resource_pools.keys()]
        for ind in range(len(tasks)):
            df.loc[ind, 'task'] = tasks[ind]
            df.loc[ind, 'prob_finish'] = simulator1.transitions[tasks[ind]][-1]
            for key in simulator1.resource_pools[tasks[ind]].keys():
                df.loc[ind, key] = simulator1.resource_pools[tasks[ind]][key][0]

        self.df = df

        task_ranking_dict = {}

        Tasks_names = ['Task A', 'Task B']

        for task in Tasks_names:

            if df.loc[df['task'] == task, 'Resource 1'].item() < df.loc[df['task'] == task, 'Resource 2'].item():
                task_ranking = ['Resource 1', 'Resource 2']
            else:
                task_ranking = ['Resource 2', 'Resource 1']

            task_ranking_dict[task] = task_ranking



        resourse_ranking_dict = {}

        res_names = ['Resource 1', 'Resource 2']

        for res in res_names:

            if df.loc[df['task'] == 'Task A', res].item() < df.loc[df['task'] == 'Task B', res].item():
                res_ranking = ['Task A', 'Task B']
            else:
                res_ranking = ['Task B', 'Task A']

            resourse_ranking_dict[res] = res_ranking



        resource_ranking_dict_score = {}

        for key in resourse_ranking_dict.keys():
            resource_ranking_dict_score[(key, resourse_ranking_dict[key][0])] = 1
            resource_ranking_dict_score[(key, resourse_ranking_dict[key][1])] = 2

        self.resource_ranking_dict_score = resource_ranking_dict_score

        task_ranking_dict_score = {}

        for key in task_ranking_dict.keys():
            task_ranking_dict_score[(key, task_ranking_dict[key][0])] = 1
            task_ranking_dict_score[(key, task_ranking_dict[key][1])] = 2

        self.task_ranking_dict_score = task_ranking_dict_score

    def give_queue_lenght(self, available_tasks):

        queue_len = {}
        keys_lens = [key for key in queue_len]
        for ind in range(len(available_tasks)):

            if available_tasks[ind].task_type in keys_lens:
                queue_len[available_tasks[ind].task_type] += 1
            else:
                queue_len[available_tasks[ind].task_type] = 1
                keys_lens = [key for key in queue_len]

        return queue_len

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task in available_tasks:
            for resource in available_resources:
                if resource in resource_pools[task.task_type]:
                    possible_assignments.append((resource, task))
        return list(set(possible_assignments))


    def plan(self, available_tasks, available_resources, resource_pools):

        available_resources = available_resources.copy()
        available_tasks = available_tasks.copy()

        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.

        assignments = []
        possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        while len(possible_assignments) > 0:

            queue_lens = self.give_queue_lenght(available_tasks)
            queue_len_keys = [key for key in queue_lens.keys()]
            df_scores = pd.DataFrame([])

            for assignment in possible_assignments:
                resource = assignment[0]
                task = assignment[1]

                mean_val = self.df.loc[self.df['task'] == task.task_type, resource].item()
                prob_fin = self.df.loc[self.df['task'] == task.task_type, 'prob_finish'].item()
                if task.task_type in queue_len_keys:
                    queue_lenght = queue_lens[task.task_type]
                else:
                    queue_lenght = 0

                score = self.a1 * mean_val - self.a2 * prob_fin - self.a3 * queue_lenght\
                        + self.a4*self.resource_ranking_dict_score[(resource, task.task_type)]+self.a5*self.task_ranking_dict_score[(task.task_type, resource)]
                curr_ind = df_scores.shape[0]

                df_scores.loc[curr_ind, 'task_type'] = task.task_type
                df_scores.loc[curr_ind, 'task'] = task
                df_scores.loc[curr_ind, 'resource'] = resource
                df_scores.loc[curr_ind, 'score'] = score

            df_scores = df_scores.sort_values(by=['score']).reset_index()
            best_score = df_scores.loc[0, 'score']
            if best_score < self.a6:
                available_resources.remove(df_scores.loc[0, 'resource'])
                available_tasks.remove(df_scores.loc[0, 'task'])
                assignments.append((df_scores.loc[0, 'resource'], df_scores.loc[0, 'task']))
                possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
            else:
                break


        return assignments




class Planner(ABC):
    """Abstract class that all planners must implement."""

    @abstractmethod
    def plan(self):
        """
        Assign tasks to resources from the simulation environment.

        :param environment: a :class:`.Simulator`
        :return: [(task, resource, moment)], where
            task is an instance of :class:`.Task`,
            resource is one of :attr:`.Problem.resources`, and
            moment is a number representing the moment in simulation time
            at which the resource must be assigned to the task (typically, this can also be :attr:`.Simulator.now`).
        """
        raise NotImplementedError


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""
    def __str__(self) -> str:
        return 'GreedyPlanner'

    def plan(self, available_tasks, available_resources):
        assignments = []
        available_resources = available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in available_tasks:
            for resource in available_resources:
                available_resources.remove(resource)
                assignments.append((task, resource))
                break
        return assignments

    


class ShortestProcessingTime(Planner):
    def __str__(self) -> str:
        return 'ShortestProcessingTime'

    def __init__(self):        
        self.resource_pools = None

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task in available_tasks:
            for resource in available_resources:
                if resource in resource_pools[task.task_type]:
                    possible_assignments.append((resource, task))
        return list(set(possible_assignments))
    
    def plan(self, available_tasks, available_resources, resource_pools):
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()        
        assignments = []

        possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        while len(possible_assignments) > 0:            
            spt = 999999
            for assignment in possible_assignments: #assignment[0] = task, assignment[1]= resource
                if self.resource_pools[assignment[1].task_type][assignment[0]][0] < spt:
                    spt = self.resource_pools[assignment[1].task_type][assignment[0]][0]
                    best_assignment = assignment
            
            available_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            assignments.append(best_assignment)
            possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        return assignments 


class DedicatedResourcePlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def __str__(self) -> str:
        return 'DedicatedResourcePlanner'

    def plan(self, available_tasks, available_resources):
        assignments = []
        available_resources = available_resources.copy()
        available_tasks = available_tasks.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in available_tasks:
            if task.task_type == 'Task B':
                if 'Resource 1' in available_resources:
                    assignments.append((task, 'Resource 1'))
                    available_resources.remove('Resource 1')
                    available_tasks.remove(task)
            if task.task_type == 'Task A':
                if 'Resource 2' in available_resources:
                    assignments.append((task, 'Resource 2'))
                    available_resources.remove('Resource 2')
                    available_tasks.remove(task)

        for assignment in assignments:
            if assignment[0].task_type == 'Task B' and assignment[1] != 'Resource 1':
                print('wrong 1')
            elif assignment[0].task_type == 'Task A' and assignment[1] != 'Resource 2':
                print('wrong 2')    
        
        return assignments


class PPOPlanner(Planner):
    def __str__(self) -> str:
        return 'DedicatedResourcePlanner'