import os
import time
import pickle
import torch
import numpy as np
from utils.util import *
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import Mclr_Logistic


class LG(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
    
        self.num_glob_iters = num_glob_iters        
        self.output_dir = os.path.join(projectdir, 'results_LG.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
        self.dataset = torch.load(self.dataset_fname)   
        self.models = [Mclr_Logistic(dimension, class_num) for m_i in range(num_users)] 
        self.loss = torch.nn.CrossEntropyLoss()


    def run(self): 
        NUM_USER = self.num_users
        result_trainacc = []
        result_testacc = []
        result_iter = []
        results = {}
        self.initialize_weights() 
        select_num = int(self.select_ratio * NUM_USER)


        start_time = time.time()
        print('*'*50)
        for glob_iter in range(self.num_glob_iters):
            selected_users = np.random.choice(NUM_USER, select_num, replace=False)
            for m_i in selected_users:
                self.user_train(model_index = m_i)   

            # Evaluate model each interation
            train_acc = self.train_error(self.models) 
            test_acc = self.test(self.models) 

            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_iter.append(glob_iter)

            weight_avg, bias_avg = self.aggregate_parameters(selected_users)

            for m_i in range(NUM_USER):
                weight_i = self.models[m_i].weight()
                weight_i.data = weight_avg.data.clone()
                bias_i = self.models[m_i].bias()
                bias_i.data = bias_avg.data.clone() 

            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")
        
        duration = (time.time() - start_time)
        print("---train LG Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results["iter"] = result_iter
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile)  
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
        model = self.models[model_index]
        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate)

        for epoch in range(0, self.local_epochs):
            # for x, y in loader: 
            optimizer.zero_grad()
            y = y.long()
            loss = torch.nn.CrossEntropyLoss()(model(X), y)
            loss.backward()
            optimizer.step()       
