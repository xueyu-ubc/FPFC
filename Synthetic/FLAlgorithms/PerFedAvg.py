import os
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import Mclr_Logistic

class PerFedAvg(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.output_dir = os.path.join(projectdir, 'results_Perfedavg.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
            
        self.dataset = torch.load(self.dataset_fname)   

        self.models = [Mclr_Logistic(dimension, class_num) for m_i in range(num_users)] 
        self.loss = torch.nn.CrossEntropyLoss()

    def run(self):    
        NUM_USER = self.num_users
        result_iter = []
        result_train = []
        result_test= []
        results_global = {}
        self.initialize_weights() 

        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            selected_users = self.select_users()
            for m_i in selected_users:
                self.user_train(model_index = m_i)  

            ## aggregate_parameters
            weight_avg, bias_avg = self.aggregate_parameters(selected_users)

            for m_i in range(NUM_USER):
                weight_i = self.models[m_i].weight()
                weight_i.data = weight_avg.data.clone()
                bias_i = self.models[m_i].bias()
                bias_i.data = bias_avg.data.clone()

            for m_i in range(NUM_USER):
                (X, y) = self.dataset['data'][m_i]
                optimizer = torch.optim.SGD(self.models[m_i].parameters(), lr = self.learning_rate)
                optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(self.models[m_i](X), y.long())
                loss.backward()
                optimizer.step()  

            # Evaluate model each interation
            train_acc = self.train_error(self.models) 
            test_acc = self.test(self.models) 

            result_iter.append(glob_iter)
            result_train.append(train_acc) 
            result_test.append(test_acc)

            print(f" iter {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")

        duration = (time.time() - start_time)
        print("---train FedAvg Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results_global["iter"] = result_iter
        results_global["train"] = result_train
        results_global["test"] = result_test

        with open(self.output_dir, 'wb') as outfile:  
            pickle.dump(results_global, outfile)  
            print(f'result written at {self.output_dir}')
        

    def aggregate_parameters(self, selected_users):
        weight_avg = torch.zeros((self.classnum, self.dimension), dtype=torch.float64)
        bias_avg = torch.zeros((1, self.classnum), dtype=torch.float64)
        sum_samples = 0
        for m_i in selected_users:
            (X, y) = self.dataset['data'][m_i]
            weight_i = self.models[m_i].weight()
            bias_i = self.models[m_i].bias()
           
            weight_avg += weight_i*len(y)
            bias_avg += bias_i*len(y)
            sum_samples += len(y)

        return weight_avg/sum_samples, bias_avg/sum_samples
    

    def user_train(self, model_index):    
        (X, y) = self.dataset['data'][model_index]
        optimizer = torch.optim.SGD(self.models[model_index].parameters(), lr = self.learning_rate)
        data_train = TensorDataset(X, y)
        loader =  DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        for epoch in range(0, self.local_epochs):
            for xx, yy in loader: 
                yy = yy.long()
                optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(self.models[model_index](xx), yy)
                loss.backward()
                optimizer.step()        

        weight = self.models[model_index].weight()
        bias = self.models[model_index].bias()
        return weight.data, bias.data

