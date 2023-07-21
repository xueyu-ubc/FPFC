import os
import torch
import numpy as np
import time
import pickle
from utils.util import *
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import ConvNet
from utils.fl_devices import Server, Client
from torch.utils.data import DataLoader, TensorDataset

class CFL(Base):
    def __init__(self, batch_size, learning_rate, e1, e2, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.e1 = e1
        self.e2 = e2
        self.output_dir = os.path.join(projectdir, 'results_CFL_'+str(e1)+'_'+str(e2)+'.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
        self.dataset = torch.load(self.dataset_fname)  
        self.models = [Client(ConvNet, lambda x : torch.optim.SGD(x, lr=self.learning_rate), dat, idnum=i) 
            for i, dat in enumerate(self.dataset['data'])]
        self.server = Server(ConvNet, self.dataset['test'])
        self.loss = torch.nn.CrossEntropyLoss()
    

    def run(self):
        result_trainacc = []
        result_testacc = []
        result_trainloss = []
        result_testloss = []
        result_iter = []
        result_time = []
        results = {}
        NUM_USER = self.num_users
        select_num = int(self.select_ratio * NUM_USER)

        EPS_1 = self.e1
        EPS_2 = self.e2
        
        cluster_indices = [np.arange(self.num_users).astype("int")]
        client_clusters = [[self.models[i] for i in idcs] for idcs in cluster_indices]

        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            if glob_iter == 0:
                for m_i in range(self.num_users):
                    self.models[m_i].synchronize_with_server(self.server)

            selected_users = np.random.choice(NUM_USER, select_num, replace=False).tolist()

            participating_clients = [self.models[i] for i in selected_users]
            for client in participating_clients:
                train_stats = client.compute_weight_update(epochs=1)
                client.reset()
            
            update_weight = []
            for client in self.models:
                update_weight.append(client.dW)

            similarities = self.server.compute_pairwise_similarities(update_weight)

            cluster_indices_new = []
            for idc in cluster_indices:
                max_norm = self.server.compute_max_update_norm([update_weight[i] for i in idc])
                mean_norm = self.server.compute_mean_update_norm([update_weight[i] for i in idc])
                
                if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and glob_iter>20:
                    c1, c2 = self.server.cluster_clients(similarities[idc][:,idc]) 
                    # cluster_indices_new += [c1, c2]
                    cluster_indices_new += [idc[c1], idc[c2]]          
                else:
                    cluster_indices_new += [idc]

            cluster_indices = cluster_indices_new
            client_clusters = [[self.models[i] for i in idcs] for idcs in cluster_indices]
            
            self.server.aggregate_clusterwise(client_clusters, update_weight)

            indices = np.zeros(self.num_users)+100
            tmp = 0
            for idcs in cluster_indices:
                for i in idcs:
                    indices[i] = tmp
                tmp += 1

            iter_duration = (time.time() - start_time)
            train_loss, train_acc = self.eval_all(self.models, dat = "data")
            test_loss, test_acc = self.eval_all(self.models, dat = "test") 
            result_iter.append(glob_iter)
            result_time.append(iter_duration)
            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_trainloss.append(train_loss)
            result_testloss.append(test_loss)

            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")
            print(f"cluster {indices}")  

        duration = (time.time() - start_time)
        print("---train CFL Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 
       
        results["iter"] = result_iter
        results['time'] = result_time
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results["trainloss"] = result_trainloss
        results["testloss"] = result_testloss
        results['cluster_assign'] = indices
        results['e1'] = EPS_1
        results['e2'] = EPS_2

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile) 
            print(f'result written at {self.output_dir}')
 

    def eval_all(self, models, dat):
        batch_size = self.batch_size
        loss = []
        acc = []
        if(dat == "data"):
            data = self.dataset['data']
        elif(dat =="test"):
            data = self.dataset['test']
        for m_i in range(self.num_users):
            model = models[m_i].model 
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
    

