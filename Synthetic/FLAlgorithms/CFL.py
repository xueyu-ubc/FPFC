import os
import torch
import numpy as np
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset
from utils.util import *
from sklearn.cluster import AgglomerativeClustering
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import Mclr_Logistic


class CFL(Base):
    def __init__(self, batch_size, learning_rate, e1, e2, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)

        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.e1 = e1
        self.e2 = e2
        self.class_num = class_num
        
        self.output_dir = os.path.join(projectdir, 'results_CFL_'+str(e1)+'_'+str(e2)+'.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
            
        self.dataset = torch.load(self.dataset_fname)   

        self.models = [Mclr_Logistic(dimension, class_num) for m_i in range(num_users)] 
        self.loss = torch.nn.CrossEntropyLoss()

    def run(self):
        result_train = []
        result_test = []
        result_iter = []
        results = {}

        EPS_1 = self.e1
        EPS_2 = self.e2
        user_sample = self.dataset['samples_per_user']  
        self.initialize_weights() 
        
        cluster_indices = [np.arange(self.num_users).astype("int")]

        start_time = time.time()
        print('*' * 50)
        print('start training CFL with e1 = ', EPS_1, ' e2 = ', EPS_2)
        for glob_iter in range(self.num_glob_iters):
            weight_old = torch.zeros([self.num_users, self.classnum, self.dimension], dtype=torch.float64)  
            bias_old = torch.zeros([self.num_users, self.classnum, 1], dtype=torch.float64)
            weight_updata = torch.zeros([self.num_users, self.classnum, self.dimension], dtype=torch.float64)  
            bias_updata = torch.zeros([self.num_users, self.classnum, 1], dtype=torch.float64)

            selected_users = self.select_users()

            for m_i in selected_users:
                weight_i = self.models[m_i].weight()
                weight_old[m_i] = weight_i.clone()
                bias_i = self.models[m_i].bias()
                bias_old[m_i] = bias_i.clone().reshape(-1,1)

            for m_i in selected_users:
                train_stats = self.compute_weight_new(model_index = m_i, epochs=1)
                weight_i = self.models[m_i].weight()
                bias_i = self.models[m_i].bias()
                temp_weight = weight_i.data.clone() - weight_old[m_i].data
                temp_bias = bias_i.data.clone().t() - bias_old[m_i].data
                weight_updata[m_i] = temp_weight
                bias_updata[m_i] = temp_bias
                
                weight_i.data = weight_old[m_i].data.clone()
                bias_i.data = bias_old[m_i].data.reshape(1,-1).clone()

            update_client = []
            for m_i in range(self.num_users):
                update_client.append(torch.cat((weight_updata[m_i], bias_updata[m_i].reshape(-1,1)),1))    
            similar = self.compute_pairwise_similarities(coef_updata = update_client)

            cluster_indices_new = []
            for idc in cluster_indices:
                max_norm = self.compute_max_update_norm([update_client[i] for i in idc])
                # mean_norm = self.compute_mean_update_norm([update_client[i] for i in idc])
                tmp = 0
                cluster_samples = 0
                for i in idc:
                    user_sample = self.dataset['data'][i][1].shape[0]
                    tmp += update_client[i] * user_sample
                    cluster_samples += user_sample
                mean_norm = torch.norm(tmp/cluster_samples, p='fro')
                
                if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and glob_iter>20:
                    c1, c2 = self.cluster_clients(similar[idc][:,idc]) 
                    cluster_indices_new += [idc[c1], idc[c2]]
                else:
                    cluster_indices_new += [idc]
                    
            cluster_indices = cluster_indices_new     
            
            for idcs in cluster_indices:
                tmp = 0
                cluster_samples = 0
                for i in idcs:
                    user_sample = self.dataset['data'][i][1].shape[0]
                    tmp += update_client[i] * user_sample
                    cluster_samples += user_sample
                tmp = tmp/cluster_samples

                for i in idcs:
                    weight_i = self.models[i].weight()
                    weight_i.data = weight_i.data.clone() + tmp[:,0:self.dimension]
                    bias_i = self.models[i].bias()
                    bias_i.data = bias_i.data.clone() + tmp[:,-1].reshape(1,-1)
                       
            train_acc = self.train_error(self.models) 
            test_acc= self.test(self.models) 

            result_iter.append(glob_iter)
            result_train.append(train_acc) 
            result_test.append(test_acc)
            
            print(f" iter {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")
            print(f"cluster {cluster_indices}")
        
        duration = (time.time() - start_time)
        print("---train CFL Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))    
       
        results["iter"] = result_iter
        results["train"] = result_train
        results["test"] = result_test
        results['cluster_assign'] = cluster_indices
        results['e1'] = EPS_1
        results['e2'] = EPS_2

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile) 
            print(f'result written at {self.output_dir}')
 

    def compute_weight_new(self, model_index, epochs=1):

        (X, y) = self.dataset['data'][model_index]
        dataset = TensorDataset(X, y)
        self.trainloader = DataLoader(dataset, batch_size=50, shuffle = True) 
        self.optimizer = torch.optim.SGD(self.models[model_index].parameters(), lr=self.learning_rate)
        train_stats = train_op(self.models[model_index], self.trainloader, self.optimizer, epochs)
        return train_stats


    def compute_pairwise_similarities(self, coef_updata):
        num_users = len(coef_updata)
        angles = torch.zeros([num_users, num_users])
        for m_i in range(num_users):
            s1 = coef_updata[m_i]
            for m_j in range(m_i, num_users):
                s2 = coef_updata[m_j]
                angles[m_i,m_j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)
                angles[m_j, m_i] = angles[m_i, m_j]

        return angles.numpy()


    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(update, p='fro') for update in cluster])


    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten() 
        return c1, c2
