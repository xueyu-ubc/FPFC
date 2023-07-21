import os
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import ConvNet
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from utils.PerFedAvgClient import PerFedAvgClient


class PerFedavg(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.hf = 1
        self.output_dir = os.path.join(projectdir, 'results_Perfedavg.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')     
        self.dataset = torch.load(self.dataset_fname)  

        self.global_model = ConvNet()

        # init clients
        self.models = [
            PerFedAvgClient(
                client_id=client_id,
                alpha=self.learning_rate,
                beta=0.001,
                global_model=self.global_model,
                criterion=torch.nn.CrossEntropyLoss(),
                batch_size=self.batch_size,
                dataset=self.dataset,
                local_epochs=self.local_epochs)
                for client_id in range(self.num_users)
            ]


    def run(self):    
        select_num = int(self.select_ratio * self.num_users)
        result_trainacc = []
        result_testacc = []
        result_trainloss = []
        result_testloss = []
        result_iter = []
        results = {}

        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            participating_clients = np.random.choice(self.num_users, size = select_num, replace=False)
            model_params_cache = []
            # client local training
            for client_id in participating_clients:
                serialized_model_params, _ = self.models[client_id].train(global_model=self.global_model, hessian_free=self.hf)
                model_params_cache.append(serialized_model_params)

            # aggregate model parameters
            aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)
            SerializationTool.deserialize_model(self.global_model, aggregated_model_params)
            
            for client_id in range(self.num_users):
                serialized_model_params, _ = self.models[client_id].train(global_model=self.global_model,
                epochs=1,hessian_free=self.hf)
                SerializationTool.deserialize_model(self.models[client_id].model, serialized_model_params)

            train_loss, train_acc = self.eval(dat = "data")
            test_loss, test_acc = self.eval(dat = "test")
   
            result_iter.append(glob_iter)
            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_trainloss.append(train_loss)
            result_testloss.append(test_loss)
            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}") 

        duration = (time.time() - start_time)
        print("---train PerFedAvg Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 


        results["iter"] = result_iter
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results["trainloss"] = result_trainloss
        results["testloss"] = result_testloss

        with open(self.output_dir, 'wb') as outfile:  
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')
        
    def eval(self, dat):
        batch_size = self.batch_size
        if(dat == "data"):
            data = self.dataset['data']
        elif(dat =="test"):
            data = self.dataset['test']  
        loss = []
        acc = []
        for m_i in range(self.num_users):
            model = self.models[m_i].model  
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

