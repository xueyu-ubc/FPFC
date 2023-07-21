#!/usr/bin/env python
import numpy as np
import argparse
import os

from FLAlgorithms.FedAvg import FedAvg
from FLAlgorithms.PerFedAvg import PerFedAvg
from FLAlgorithms.LOCAL import LOCAL
from FLAlgorithms.IFCA import IFCA
from FLAlgorithms.CFL import CFL
from FLAlgorithms.FPFC import FPFC
from FLAlgorithms.FPFCL1 import FPFCL1
from FLAlgorithms.PACFL import PACFL
from FLAlgorithms.LG import LG
import torch


def main(dataset, algorithm, batch_size, learning_rate, e1, e2, lamda, a, xi, lamdaL1, threshold, thresholdL1, num_glob_iters,
         local_epochs, num_users, ratio, K, dimension, class_num, score, seed, n_basis, cluster_alpha, projectdir, datasetdir):

    np.random.seed(seed)  
    torch.manual_seed(seed)
    
    # Select algorithm
    if(algorithm == "FedAvg"):
        alg = FedAvg(batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
    if(algorithm == "LOCAL"):
        alg = LOCAL(batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
    
    if(algorithm == "PerFedavg"):
        alg = PerFedAvg(batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
 
    if(algorithm == "IFCA"):
        alg = IFCA(batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, K, dimension, class_num, score, seed, projectdir, datasetdir)

    if(algorithm == "CFL"):
        alg = CFL(batch_size, learning_rate, e1, e2, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

    if(algorithm == "FPFC"):
        alg = FPFC(batch_size, learning_rate, lamda, a, xi, threshold, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

    if(algorithm == "FPFCL1"):
        alg = FPFCL1(batch_size, learning_rate, lamdaL1, thresholdL1, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
    
    if(algorithm == "PACFL"):
        alg = PACFL(batch_size, learning_rate, n_basis, cluster_alpha, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
    
    if(algorithm == "LG"):
        alg = LG(batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

    alg.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Synthetic")
    parser.add_argument("--algorithm", type=str, default="FPFC", choices=["FedAvg", "PerFedavg", "LOCAL", "IFCA", "CFL", "FPFC", "FPFCL1","PACFL", "LG"]) 
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--e1", type=float, default=0.5, help="Parameter of CFL")
    parser.add_argument("--e2", type=float, default=1, help="Parameter of CFL")
    parser.add_argument("--lamda", type=float, default=0.6, help="Regularization parameter of FPFC")
    parser.add_argument("--a", type=float, default=3.7, help="Regularization parameter of FPFC")
    parser.add_argument("--xi", type=float, default=10e-5, help="smooth parameter of FPFC")
    parser.add_argument("--threshold", type=float, default=3, help="threshold parameter of FPFC for clustering")
    parser.add_argument("--thresholdL1", type=float, default=2, help="threshold parameter of FPFCL1 for clustering")
    parser.add_argument("--lamdaL1", type=float, default=0.6, help="Regularization parameter of FPFCL1")
    parser.add_argument("--num_global_iters", type=int, default = 2000, help="Communication rounds")
    parser.add_argument("--local_epochs", type=int, default=10, help="Local steps")
    parser.add_argument("--num_users", type=int, default=100, help="Number of devices")
    parser.add_argument("--K", type=int, default=4, help="Number of clusters")
    parser.add_argument("--dimension", type=int, default=60, help="dimension of model")
    parser.add_argument("--class_num", type=int, default=10, help="class number of model")
    parser.add_argument("--score", type=str, default="set", choices=["set", "em"], help="method to calculate scores in IFCA")
    parser.add_argument("--ratio", type=float, default=0.3, help="the proportion of active devices at each round")
    parser.add_argument("--seed", type=int, default=1, help="seed for randomness")
    parser.add_argument("--n_basis", type=int, default=5, help="basis for PACFL")
    parser.add_argument("--cluster_alpha", type=float, default=4, help="alpha for PACFL")
    parser.add_argument("--project_dir", type=str, default="output_synthetic")   
    parser.add_argument("--dataset_dir", type=str, default="output_synthetic")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate: {}".format(args.learning_rate))
    print("Number of devices: {}".format(args.num_users))
    print("Number of clusters: {}".format(args.K))
    print("Ratio of active devices: {}".format(args.ratio))
    print("Number of global rounds: {}".format(args.num_global_iters))
    print("Number of local rounds: {}".format(args.local_epochs))
    print("Dataset: {}".format(args.dataset))
    print("Seed: {}".format(args.seed))
    print("Project dir: {}".format(args.project_dir))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        e1 = args.e1,
        e2 = args.e2,
        lamda = args.lamda,
        a = args.a,
        xi = args.xi,
        threshold=args.threshold,
        thresholdL1=args.thresholdL1,
        lamdaL1 = args.lamdaL1,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        num_users = args.num_users,
        ratio = args.ratio,
        K=args.K,
        dimension= args.dimension,
        class_num = args.class_num,
        score = args.score,
        n_basis = args.n_basis,
        cluster_alpha = args.cluster_alpha,
        seed = args.seed,
        projectdir = args.project_dir,
        datasetdir = args.dataset_dir
        )