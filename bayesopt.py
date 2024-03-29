import time


sys.path.append(r'C:\Users\user\workspace\scikit-optimize')
sys.path.append(r'C:\Users\user\workspace\scikit-optimize\skopt')


sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/scikit-optimize')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/scikit-optimize\skopt')
from bayes_opt import BayesianOptimization
# from bayesian_optimization import BayesianOptimization
# Supress NaN warnings
import warnings
warnings.filterwarnings("ignore", category =RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
import sys





import pickle as pkl

import os
import pandas as pd

from train_SVFA import aggregate_sims, simulate_competition
# from skopt.space import Real, Integer
# from skopt.utils import use_named_args
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
# from skopt.plots import plot_gaussian_process
# from skopt import gp_minimize
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs


def main():

    # Bounded region of parameter space

    sys_list =['low_utilization', 'high_utilization', 'slow_server', 'down_stream', 'n_system', 'parallel', 'complete', 'complete_reversed', 'complete_parallel']

    sys_str = sys_list[np.random.randint(9)]


    pbounds = {'a1': (0.015, 30),
               'a2': (0.015, 30),
               'a3': (0.015, 30),
               'a4': (0.015, 30),
               'a5': (0.015, 30),
               'a6': (0.015, 30),
               'a7': (15, 90)}

    import datetime

    now = datetime.datetime.now()
    rand_state = now.microsecond

    optimizer = BayesianOptimization(
        f=aggregate_sims,
        pbounds=pbounds,
        verbose=2,
        random_state=rand_state,
        allow_duplicate_points=True,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )



    optimizer.maximize(
        init_points=2,
        n_iter=20,
    )


    num = np.random.randint(1,100000000)
    
    vals = [res for i, res in enumerate(optimizer.res)]
    print(len(vals))


    pkl.dump(vals, open(r'slow_server'+str(num)+'.pkl', 'wb'))



if __name__ == "__main__":


    main()
