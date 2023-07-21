import os
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn.functional as F
import copy
from utils.util import *
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import Mclr_Logistic


class PACFL(Base):
    def __init__(self, batch_size, learning_rate, n_basis, cluster_alpha, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.n_basis = n_basis
        self.cluster_alpha = cluster_alpha
        self.output_dir = os.path.join(projectdir, 'results_PACFL_'+str(cluster_alpha)+'.pickle')
        self.dataset_fname = os.path.join(datasetdir,  'dataset.pth')  
        self.dataset = torch.load(self.dataset_fname)   

        self.models = [Mclr_Logistic(dimension, class_num) for m_i in range(num_users)] 
        self.loss = torch.nn.CrossEntropyLoss()
        self.linkage = "average"


    def run(self): 
        NUM_USER = self.num_users
        result_trainacc = []
        result_testacc = []
        result_iter = []
        results = {}

        select_num = int(self.select_ratio * NUM_USER)  
        self.initialize_weights() 

        start_time = time.time()
        print("-------------start traning with cluster alpha: ", self.cluster_alpha, " -------------")
        ######################   clustering  #####################################
        K = self.n_basis
        U_clients = []
        for m_i in range(NUM_USER):
            (X, y) = self.dataset['data'][m_i]
            uni_labels, cnt_labels = np.unique(y, return_counts=True)
            print(f'Labels: {uni_labels}, Counts: {cnt_labels}')
    
            U_temp = []
            for j in uni_labels:
                local_ds1 = X[y == j, :]
                local_ds1 = local_ds1.T   # n_features * n_samples
                if K > 0: 
                    u1_temp, sh1_temp, vh1_temp = np.linalg.svd(local_ds1, full_matrices=False)
                    u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0)
                    U_temp.append(u1_temp[:, 0:K])
                    
            U_clients.append(copy.deepcopy(np.hstack(U_temp)))    
            print(f'Shape of U: {U_clients[-1].shape}')
        
        clients_idxs = np.arange(NUM_USER)
        adj_mat = calculating_adjacency(clients_idxs, U_clients)    # NUM_USER * NUM_USER
        clusters = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=self.cluster_alpha, linkage=self.linkage)
        print('')
        print('Adjacency Matrix')
        print(adj_mat)
        print('')
        print('Clusters: ')
        print(clusters)
        print('')
        print(f'Number of Clusters {len(clusters)}')
        print('')

        clients_clust_id = []
        for i in range(NUM_USER):
            for j in range(len(clusters)):
                if i in clusters[j]:
                    clients_clust_id.append(j)
                    break
        print(f'Clients: Cluster_ID \n{clients_clust_id}')
        self.clustermodels = [copy.deepcopy(self.models[0].state_dict()) for _ in range(len(clusters))]
        print('*'*50)
        print('start training with cluster_alpha: ', self.cluster_alpha)
          
        for glob_iter in range(self.num_glob_iters):
            selected_users = np.random.choice(NUM_USER, size = select_num, replace=False)
            for m_i in selected_users:
                self.models[m_i].load_state_dict(copy.deepcopy(self.clustermodels[clients_clust_id[m_i]]))
                self.user_train(model_index = m_i)   

            ## aggregate_parameters
            for ci in range(len(clusters)): 
                cluster_users = []   
                for m_i in selected_users: 
                    if m_i in clusters[ci]:
                        cluster_users.append(m_i)
                if len(cluster_users) == 0:
                    continue
                else:           
                    self.aggregate_parameters(cluster_users, ci)

            # Evaluate model each interation
            train_acc = self.train_error(self.models) 
            test_acc = self.test(self.models) 

            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_iter.append(glob_iter)
            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")
        
        duration = (time.time() - start_time)
        print("---train PACFL Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results["iter"] = result_iter
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results['cluster_number'] = len(clusters)
        results["cluster_assignment"] = clients_clust_id

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')

    def aggregate_parameters(self, cluster_users, ci):
        w_locals = []           
        weights = []
        n_samples = 0
        for m_i in cluster_users:
            (X, y) = self.dataset['data'][m_i]
            w_local = copy.deepcopy(self.models[m_i].state_dict())
            w_locals.append(w_local)
            weights.append(len(y))
            n_samples += len(y)
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k]* weights[0]/n_samples
        
        for k in w_avg.keys():
            for i in range(1, len(cluster_users)):
                w_avg[k] = w_avg[k] + w_locals[i][k] * weights[i]/n_samples
        self.clustermodels[ci] = copy.deepcopy(w_avg)

    def user_train(self, model_index):    
        (X, y) = self.dataset['data'][model_index]
        optimizer = torch.optim.SGD(self.models[model_index].parameters(), lr = self.learning_rate)
        data_train = TensorDataset(X, y)
        for epoch in range(0, self.local_epochs):
            optimizer.zero_grad()      
            y = y.long()
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(self.models[model_index](X), y)
            loss.backward()
            optimizer.step() 
                
