import os
import time
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import ConvNet

class LOCAL(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension,class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.output_dir = os.path.join(projectdir, 'results_LOCAL.pickle')
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

        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
           
            for m_i in range(self.num_users):
                model = self.models[m_i]
                optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate)
                (X, Y) = self.dataset['data'][m_i]
                data_train = TensorDataset(X, Y)
                loader =  DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
                for epoch in range(0, self.local_epochs):
                    for x, y in loader: 
                        optimizer.zero_grad()
                        loss = torch.nn.CrossEntropyLoss()(model(x.float()), y)
                        loss.backward()
                        optimizer.step()

            train_loss, train_acc = self.eval(self.models, dat = "data")
            test_loss, test_acc = self.eval(self.models, dat = "test") 

            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_trainloss.append(train_loss) 
            result_testloss.append(test_loss)
            result_iter.append(glob_iter)   

            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")

        duration = (time.time() - start_time)
        print("---train LOCAL Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results["iter"] = result_iter
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results["trainloss"] = result_trainloss
        results["testloss"] = result_testloss

        
        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')