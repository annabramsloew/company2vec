import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer, Categorical
import pandas as pd
import glob 
import pickle
import os


### Script to run the hyperparameter search for RNN-based model.
### This script generates the values for the parameters
### You manually add the performance values.



N_TO_ASK = 5 # number of points to evaluate at the same time
N_TO_EVALUATE = 10 #number of points


def dump_results(optimizer, dir_path="..scikit_pretraining/", name="results_rnn.pkl"):
        if not(os.path.exists(dir_path)):
                os.mkdir(dir_path)
        with open(dir_path + name, "wb") as f:
                results = {"Xi": optimizer.Xi, 
                           "yi": optimizer.yi,
                           "optimizer": optimizer}
                pickle.dump(results, f)


params = [[370, 1, 0.1571789665157697, 0.00039],
          [64, 4, 0.05, 0.01],
          [128, 8, 0.35, 0.001],
          [256, 6, 0.0, 0.00001],
          [512, 2, 0.1, 0.005],
          #batch2
          [672, 8, 0.3039198585496867, 0.008819112584812413],
          [624, 1, 0.34652402607214117, 0.008704017609191114],
          [630, 1, 0.29027508595733326, 0.0003802510311963119],
          [758, 6, 0.3521947886968565, 0.0077009596201347355],
          [150, 3, 0.02535779578977603, 0.007037085712705645]
         ]
scores = [1.078,1.084,1.099,1.089,1.141,
          1.124, 1.122, 1.093, 1.136, 1.105]

search_space = [Integer(64, 768, name="hidden_size"),  
                Integer(1,8, name="n_layers"),
                Real(0.0, 0.5, name="dropout"), 
                Real(1e-5, 1e-2, name="lr")
                ]               



optimizer = Optimizer(dimensions = search_space,
                      base_estimator="GP",
                      acq_func="PI",
                      acq_optimizer="lbfgs",
                      random_state = 2021,
                      n_initial_points=1)

for i in range(len(scores)):
        optimizer.tell(params[i], scores[i])



print("Points evaluated:", len(optimizer.Xi))
query = optimizer.ask(n_points=N_TO_ASK)
query_queue = []
for q in query:
        query_queue.append(q)
        print("\t", query_queue[-1])

while len(optimizer.Xi) <= N_TO_EVALUATE:

        if len(query_queue) == 0:
                query = optimizer.ask(n_points=N_TO_ASK)
                for q in query:
                    query_queue.append( q)
                    print("\t", query_queue[-1])


        current_querry = query_queue.pop(0)
        try:
                metrics = np.log(1. - float(input("AUL for the %s: " %current_querry )))
        except:
                print("Wrong Format. Skipping the query...")
                continue


        print("\tMetric: %.3f" %metrics)
        optimizer.tell(current_querry, metrics)
        print("Points evaluated:", len(optimizer.Xi))
        dump_results(optimizer)


i_best = np.argmin(optimizer.yi)
print("Best metric: %.3f" %optimizer.yi[i_best])
print("\t", optimizer.Xi[i_best])