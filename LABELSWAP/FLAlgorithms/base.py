import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

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

    def eval(self, models, dat):
        batch_size = self.batch_size
        loss = []
        acc = []

        if(dat == "data"):
            data = self.dataset['data']
        elif(dat =="test"):
            data = self.dataset['test']
        for m_i in range(self.num_users):
            model = models[m_i]  
            (X, y) = data[m_i]
            dataset = TensorDataset(X.float(), y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            samples, correct = 0, 0
            l = 0

            with torch.no_grad():
                for i, (xx, yy) in enumerate(loader):
                    y_ = model(xx.float())
                    _, predicted = torch.max(y_.data, 1)

                    samples += yy.shape[0]
                    correct += (predicted == yy).sum().item()
                    los = torch.nn.CrossEntropyLoss()(y_, yy)
                    l += los.item()*yy.shape[0]
            
                loss.append(l/samples)
                acc.append(correct/samples)
            
        return np.mean(loss), np.mean(acc)
    

