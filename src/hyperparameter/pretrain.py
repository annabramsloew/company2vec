import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer
import pandas as pd
import glob 
import pickle
import os

N_TO_ASK = 4
N_TO_EVALUATE = 25

def adjust_value(val:int,  d: int = 8, min_val: int = 0) -> int:
        remainder = val % d
        if val - remainder < min_val:
                val = val + d
                remainder = val % d
                return val - remainder
        return val - remainder

def dump_results(optimizer, dir_path="../", name="results2.pkl"):
        if not(os.path.exists(dir_path)):
                os.mkdir(dir_path)
        with open(dir_path + name, "wb") as f:
                results = {"Xi": optimizer.Xi, 
                           "yi": optimizer.yi,
                           "optimizer": optimizer}
                pickle.dump(results, f)

def adjust_ask(asked, keys):
        config = {}
        for i in range(len(asked)):
                config[keys[i]] = asked[i]
        config["hidden_size"] = adjust_value(config["hidden_size"], config["n_heads"], min_val=64)

        config["n_local"] = np.floor(config["n_heads"] * config["n_local"])
        if config["n_local"] == 0:
            config["local_window"] = 0

        
        return config

def adjust_tell(config):
        """Convert n_local back to the ratio"""
        config["n_local"] /= config["n_heads"]
        return [v for k,v in config.items()]


params = [280, 5, 10, 7/10, 2210,  436, 38]
batch1 = [[180, 4, 6, 0/6, 1174, 446, 0],
        [72, 5, 12, 0/12, 1285, 110, 0],
        [270, 8, 10, 5/10, 570, 200, 108],
        [128, 14, 8, 7/8, 1454, 75, 130]]

batch2 = [[286, 8, 11, 0/11, 1409, 103, 0],
          [192, 7, 12, 2/12, 1117, 283, 10],
          [324, 10, 4, 1/4, 1997, 429, 141],
          [252, 13, 3, 0/3, 2427, 185, 0]]

batch3 = [
    [182, 14, 7, 1/7, 1986, 381, 23],
    [112, 14, 7, 1/7, 2560, 512, 73]
]

batch4 = [
    [192, 4, 12, 9/12, 1886, 462, 10],
    [189, 6, 7, 1/7, 2117, 298, 9]
]

batch5 =[
        [209, 4, 11, 11/11, 1417, 512, 10],
        [99, 5, 11, 2/11, 2248, 512, 10],
        [120, 4, 10, 10/10, 2560, 512, 10],
        [168, 4, 14, 0/14, 852, 512, 0]
]

batch6 = [
    [192, 6, 12, 11.0, 1029, 209, 10],
    [220, 13, 10, 10.0, 512, 141, 10],
    [210, 5, 14, 14.0, 813, 152, 10],
    [207, 8, 9, 9.0, 808, 169, 11]
]

scores = [2.214]
batch1_scores = [3.105, 3.185, 2.635, 2.980]
batch2_scores = [3.064, 2.166, 2.952, 3.088]
batch3_scores = [2.687, 3.016]
batch4_scores = [2.144, 2.202]
batch5_scores = [2.126, 2.409, 2.280, 3.185]

scores = scores + batch1_scores + batch2_scores + batch3_scores + batch4_scores + batch5_scores
params = [params] + batch1 + batch2 + batch3 + batch4 + batch5
print(scores)
space_keys = ["hidden_size", "n_encoders", "n_heads", "n_local", "hidden_ff", "n_rand_features", "local_window"]

search_space = [Integer(64, 336, name="hidden_size"),  
                Integer(4,14, name="n_encoders"),
                Integer(3,14, name="n_heads"), 
                Real(0.0, 1.0, name="n_local"), 
                Integer(512,2560, name="hidden_ff"), 
                Integer(64,512, name="n_rand_features"),
                Integer(0,256, name="local_window"),
                ]               



optimizer = Optimizer(dimensions = search_space,
                      base_estimator="GP",
                      acq_func="PI",
                      acq_optimizer="lbfgs",
                      n_initial_points = 8,
                      random_state = 2021)

for i in range(len(scores)):
        optimizer.tell(params[i], scores[i])



print("Points evaluated:", len(optimizer.Xi))
query = optimizer.ask(n_points=N_TO_ASK)
query_adjusted = []

for i, q in enumerate(query):
        query_adjusted.append(adjust_ask(q, keys=space_keys))
        print("\t", query_adjusted[-1])

while len(optimizer.Xi) <= N_TO_EVALUATE:

        if len(query_adjusted) == 0:
                query = optimizer.ask(n_points=N_TO_ASK)
                for i, q in enumerate(query):
                        query_adjusted.append(adjust_ask(q, keys=space_keys))
                        print("\t", query_adjusted[-1])

        current_querry = query_adjusted.pop(0)
        try:
                metrics = float(input("Loss for the %s: " %current_querry ))
        except:
                print("Wrong Format. Skipping the query...")
                continue


        print("\tMetric: %.3f" %metrics)
        optimizer.tell(adjust_tell(current_querry), metrics)
        print("Points evaluated:", len(optimizer.Xi))
        dump_results(optimizer)


i_best = np.argmin(optimizer.yi)
print("Best metric: %.3f" %optimizer.yi[i_best])
print("\t", optimizer.Xi[i_best])