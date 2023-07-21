import torch
import os
import numpy as np

class Base:
    def __init__(self, batch_size, learning_rate, local_epochs, numusers, ratio, dimension, class_num, seed, projectdir, datasetdir):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.num_users = numusers
        self.select_ratio = ratio
        self.classnum = class_num
        self.dimension = dimension
        self.seed = seed
        self.project_dir = projectdir
        self.dataset_dir = datasetdir

        self.dataset_fname = os.path.join(self.dataset_dir, 'dataset.pth')  
        self.dataset = torch.load(self.dataset_fname)   
        self.loss = torch.nn.CrossEntropyLoss()
    
    def initialize_weights(self):
        param_W = torch.tensor(np.random.normal(0, 1, (self.classnum, self.dimension)))
        param_b = torch.tensor(np.random.normal(0, 1,  (1, self.classnum)))
        num_models = len(self.models)

        for m_i in range(num_models):
            weight = self.models[m_i].weight()
            bias = self.models[m_i].bias()           
            weight.data = param_W.clone()
            bias.data = param_b.clone()

    def select_users(self):
        select_num = int(self.select_ratio * self.num_users)
        if(self.num_users == select_num):
            print("All users are selected")
            select_users = np.arange(0, self.num_users)
            return select_users
    
        select_num = min(self.num_users, select_num)
        return np.random.choice(list(range(self.num_users)), select_num, replace=False)

    def test(self, models):
        test_acc = []
        for m_i in range(self.num_users):
            (X, y) = self.dataset['test'][m_i]
            output = models[m_i](X)
            y = y.long()
            test_acc.append((torch.sum(torch.argmax(output, dim=1) == y)).item()/len(y))
        return np.mean(test_acc)

    def train_error(self, models):
        train_acc = 0
        for m_i in range(self.num_users):
            (X, y) = self.dataset['data'][m_i]
            output = models[m_i](X)
            y = y.long()
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()/len(y)

        return train_acc/self.num_users
