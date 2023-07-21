import os
import torch
from torch.utils.data import random_split
import torch.utils.data as Data
import numpy as np
from utils.util import *

class DatasetGenerate:
    def __init__(self, numusers, K, dimension, class_num, seed, projectdir, datasetdir):
        # Set up the main attributes
        self.num_users = numusers
        self.cluster = K
        self.dimension = dimension
        self.classnum = class_num
        self.seed = seed
        self.project_dir = projectdir
        self.dataset_dir = datasetdir
   
    def setup(self):

        self.dataset_dir = os.path.join(self.project_dir) 
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.dataset_fname = os.path.join(self.dataset_dir, 'dataset.pth') 
        print('seeding', self.seed)
        np.random.seed(self.seed)

    def generate_dataset(self):
        dataset = {}
        dataset['data'] = []
        dataset['test'] = []
        samples_per_user = ((np.random.lognormal(4, 2, (self.num_users)).astype(int) + 50) * 5).astype(int)

        dataset['samples_per_user'] = samples_per_user
        # generate dataset for each machine   
        cluster_assignment = [m_i//(self.num_users//self.cluster) for m_i in range(self.num_users)] # ex: [0,0,0,0, 1,1,1,1, 2,2,2,2] for m = 12, p = 3
        dataset['cluster_assignment'] = cluster_assignment  

         ## define some eprior ####
        mean_W = np.random.normal(0, 1, self.cluster)
        mean_b = mean_W
        print(mean_W)


        dataset['data'] = []
        dataset['test'] = []
        params_W = []
        params_b = []

        for p_i in range(self.cluster):
            W = torch.tensor(np.random.normal(mean_W[p_i], 1, (self.classnum, self.dimension)).astype(np.float32))
            b = torch.tensor(np.random.normal(mean_b[p_i], 1, (1, self.classnum)).astype(np.float32))
            params_W.append(W)
            params_b.append(b)


        for m_i in range(self.num_users):
            p_i = cluster_assignment[m_i]
            W = params_W[p_i]
            b = params_b[p_i]
            xx = np.random.random((samples_per_user[m_i], self.dimension))
            yy = np.zeros(samples_per_user[m_i])
            for j in range(samples_per_user[m_i]):
                tmp = np.dot(xx[j], W.t()).reshape(1,-1) + b.numpy() + np.random.normal(0, scale = 0.5, size = self.classnum).astype(np.float32)
                yy[j] = np.argmax(softmax(tmp))
               

            yy = torch.tensor(yy)
            xx = torch.tensor(xx)
            total_dataset = Data.TensorDataset(xx, yy)  
            train_data, test_data = random_split(total_dataset, [int(round(0.8*samples_per_user[m_i])), int(round(0.2*samples_per_user[m_i]))]) 
            
            train_X = train_data[:][0]
            train_y = train_data[:][1]

            test_X = test_data[:][0]
            test_y = test_data[:][1]

            dataset['data'].append((train_X, train_y))   
            dataset['test'].append((test_X, test_y))   
        
        dataset['params_W'] = params_W  
        dataset['params_b'] = params_b  

        self.dataset = dataset
        return dataset

    def save(self):
        torch.save(self.dataset, self.dataset_fname)  

        from pathlib import Path
        Path(os.path.join(self.project_dir,'result_data.txt')).touch()  

    def check_dataset(self):
        dataset = torch.load(self.dataset_fname)