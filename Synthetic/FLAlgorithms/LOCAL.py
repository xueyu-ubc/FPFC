import os
import time
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import Mclr_Logistic


class LOCAL(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension,class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.output_dir = os.path.join(projectdir, 'results_LOCAL.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
            
        self.dataset = torch.load(self.dataset_fname)   

        self.models = [Mclr_Logistic(dimension, class_num) for m_i in range(num_users)]
        self.loss = torch.nn.CrossEntropyLoss()

        self.epoch = None
        self.lr_decay_info = None


    def run(self):
        result_iter = []
        result_train = []
        result_test= []
        results_local = {}
        self.initialize_weights()  

        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            for m_i in range(self.num_users):
                (X, y) = self.dataset['data'][m_i]
                optimizer = torch.optim.SGD(self.models[m_i].parameters(), lr = self.learning_rate)
                data_train = TensorDataset(X, y)
                for iter in range(self.local_epochs):
                    optimizer.zero_grad()
                    y = y.long()
                    loss = self.loss(self.models[m_i](X), y)

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
        print("---train LOCAL Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results_local["iter"] = result_iter
        results_local["trainACC"] = result_train
        results_local["testACC"] = result_test

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results_local, outfile)  
            print(f'result written at {self.output_dir}')