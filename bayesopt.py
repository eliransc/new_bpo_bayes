import time
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

from train_Eliran import aggregate_sims, simulate_competition
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import numpy as np

import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize


def main():

    # Bounded region of parameter space

    space = [Real(0, 5, name='a1'),
             Real(0, 5, name='a2'),
             Real(0, 5, name='a3'),
             Real(0, 5, name='a4'),
             Real(0, 5, name='a5'),
             Real(7, 20, name='a6')]

    res = gp_minimize(aggregate_sims,  # the function to minimize
                      space,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=15,  # the number of evaluations of f
                      n_random_starts=3,  # the number of random initialization points
                      noise=0.1 ** 2,  # the noise level (optional)
                      random_state=1234)


    model_num = np.random.randint(0, 100000)

    pkl.dump((res.func_vals, res.x_iters), open(str(model_num) + '_res_slow_server.pkl', 'wb'))



if __name__ == "__main__":


    main()
