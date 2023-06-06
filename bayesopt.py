import time
from bayes_opt import BayesianOptimization
# from bayesian_optimization import BayesianOptimization
# Supress NaN warnings
import warnings
warnings.filterwarnings("ignore", category =RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\user\workspace\scikit-optimize')
sys.path.append(r'C:\Users\user\workspace\scikit-optimize\skopt')

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

    pbounds = {'a1': (0.15, 20),
               'a2': (0.15, 20),
               'a3': (0.15, 20),
               'a4': (0.15, 20),
               'a5': (0.15, 20),
               'a6': (0.15, 20),
               'a7': (0.15, 20)}


    optimizer = BayesianOptimization(
        f=aggregate_sims,
        pbounds=pbounds,
        verbose=2,
        random_state=10,
        allow_duplicate_points=True,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    if os.path.exists("./complete_parallel.json"):
        load_logs(optimizer, logs=["./high_utilization.json"]);

    optimizer.maximize(
        init_points=1,
        n_iter=1,
    )

    vals = [res for i, res in enumerate(optimizer.res)]
    print(len(vals))

    logger = JSONLogger(path="./high_utilization.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=0,
        n_iter=20,
    )
    
    num = np.random.randint(1,100000000)
    
    vals = [res for i, res in enumerate(optimizer.res)]
    print(len(vals))


    pkl.dump(vals, open(r'high_utilization_'+str(num)+'.pkl', 'wb'))



if __name__ == "__main__":


    main()
