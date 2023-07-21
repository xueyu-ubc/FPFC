import os
import time
import pickle
import torch
import numpy as np
import copy
from utils.util import *
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import ConvNet
from torch.utils.data import DataLoader, TensorDataset


class LG(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.output_dir = os.path.join(projectdir, 'results_LG_.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
        self.dataset = torch.load(self.dataset_fname)   
        self.models = [ConvNet() for m_i in range(self.num_users)]
        self.loss = torch.nn.CrossEntropyLoss()

    def run(self): 
        result_trainacc = []
        result_testacc = []
        result_trainloss = []
        result_testloss = []       
        result_iter = []
        results = {}
        select_num = int(self.select_ratio * self.num_users)         
        start_time = time.time()
        print('*'*50)
        for glob_iter in range(self.num_glob_iters):
            selected_users = np.random.choice(self.num_users, size = select_num, replace=False)
            for idx in selected_users:
                self.user_train(model_index = idx)  
               
            train_loss, train_acc = self.eval(self.models, dat = "data")
            test_loss, test_acc = self.eval(self.models, dat = "test") 

            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_trainloss.append(train_loss) 
            result_testloss.append(test_loss)
            result_iter.append(glob_iter)   

            self.aggregate_parameters(selected_users)

            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")

        duration = (time.time() - start_time)
        print("---train LG Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results["iter"] = result_iter
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results["trainloss"] = result_trainloss
        results["testloss"] = result_testloss

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')


    def aggregate_parameters(self, select_users):
        w_locals = []           
        weights = []
        n_samples = 0        
        for m_i in select_users:
            (X, y) = self.dataset['data'][m_i]
            w_local = copy.deepcopy(self.models[m_i].state_dict())
            w_locals.append(w_local)
            weights.append(len(y))
            n_samples += len(y)
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k]* weights[0]/n_samples
        
        for k in w_avg.keys():
            for i in range(1, len(select_users)):
                w_avg[k] = w_avg[k] + w_locals[i][k] * weights[i]/n_samples
        
        for m_i in range(self.num_users):
            self.models[m_i].load_state_dict(w_avg)


    def user_train(self, model_index): 
        (X, y) = self.dataset['data'][model_index]
        model = self.models[model_index]
        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate)

        data_train = TensorDataset(X, y)
        loader =  DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        for epoch in range(0, self.local_epochs):
            for x, y in loader: 
                optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(model(x.float()), y)
                loss.backward()
                optimizer.step()       


