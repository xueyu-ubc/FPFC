import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data

class MNISTGenerate:
    def __init__(self, numusers, K, class_num, seed, projectdir, datasetdir):
        # Set up the main attributes
        self.num_users = numusers
        self.cluster = K
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
        MNIST_TRAINSET_DATA_SIZE = 60000  #60000
        MNIST_TESTSET_DATA_SIZE = 10000   #10000
        
        dataset = {}
        dataset['data'] = []
        dataset['test'] = []
        samples_per_user = []
    
        (X_train, y_train) = self._load_MNIST(train=True)

        dataset['train_data_indices'], dataset['train_cluster_assign'] = \
            self._setup_dataset(MNIST_TRAINSET_DATA_SIZE, y_train, self.cluster, self.num_users, self.classnum)
        
        for m_i in range(self.num_users):
            indices = dataset['train_data_indices'][m_i]
            samples_per_user.append(len(indices))
            p_i = dataset['train_cluster_assign'][m_i]
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            label1 = np.argwhere(y_batch ==p_i)
            label2 = np.argwhere(y_batch ==8-p_i)
            y_batch[label1] = 8-p_i
            y_batch[label2] = p_i

            X_batch = X_batch.reshape(-1, 1, 28, 28)
            dataset['data'].append((X_batch, y_batch))  

            dataset['samples_per_user'] = samples_per_user
            dataset['cluster_assignment'] = dataset['train_cluster_assign'] 
            

        (X_test, y_test) = self._load_MNIST(train=False)
        dataset['test_data_indices'], dataset['test_cluster_assign'] = \
            self._setup_dataset(MNIST_TESTSET_DATA_SIZE, y_test, self.cluster, self.num_users, self.classnum, random=False)
        
        for m_i in range(self.num_users):
            indices = dataset['test_data_indices'][m_i]
            p_i = dataset['train_cluster_assign'][m_i]
            X_tbatch = X_test[indices]
            y_tbatch = y_test[indices]
            label1 = np.argwhere(y_tbatch==p_i)
            label2 = np.argwhere(y_tbatch==8-p_i)
            y_tbatch[label1] = 8-p_i
            y_tbatch[label2] = p_i

            X_tbatch = X_tbatch.reshape(-1, 1, 28, 28)
            dataset['test'].append((X_tbatch, y_tbatch)) 
            
        self.dataset = dataset
        return dataset

    def _setup_dataset(self, num_data, train_labels, p, m, c, random = True):
        alpha = 1.0
        data_indices = []
        cluster_assign = []
        train_idcs = np.random.permutation(num_data)
        n_classes = c
        n_clients = m
        label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

        class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

        client_idcs = [[] for _ in range(n_clients)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]

        data_indices = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

        cluster_assign = [m_i//(n_clients//p) for m_i in range(n_clients)]

        return data_indices, cluster_assign

    def _load_MNIST(self, train=True):
        if train:
            mnist_dataset = datasets.MNIST(root='.', train=True, transform=transforms.ToTensor(), download=True)
        else:
            mnist_dataset =datasets.MNIST(root='.', train=False, transform=transforms.ToTensor(), download=True)

        dl = DataLoader(mnist_dataset)

        X = dl.dataset.data # (60000,28, 28), ([10000, 28, 28])
        y = dl.dataset.targets #(60000)

        # normalize to have 0 ~ 1 range in each pixel

        X = X / 255.0  

        return X.float(), y    

    def save(self):
        torch.save(self.dataset, self.dataset_fname)  

        from pathlib import Path
        Path(os.path.join(self.project_dir,'result_data.txt')).touch() 

    def check_dataset(self):
        dataset = torch.load(self.dataset_fname)