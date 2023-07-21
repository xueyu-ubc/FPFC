import os
import time
import pickle
import torch
import numpy as np
from FLAlgorithms.base import Base
from utils.util import calculate_loss_grad, weight_update
from FLAlgorithms.trainmodel.models import Mclr_Logistic


class IFCA(Base):
    def __init__(self, batch_size, learning_rate, num_glob_iters, local_epochs, num_users, ratio, K, dimension, class_num, score, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
      
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.K = K
        self.score = score
        self.output_dir = os.path.join(projectdir, 'results_IFCA.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
            
        self.dataset = torch.load(self.dataset_fname)   

        self.models = [Mclr_Logistic(dimension, class_num) for k in range(K)]
        self.loss = torch.nn.CrossEntropyLoss()


    def run(self):
        self.initialize_weights() 
        for p_i in range(self.K):
            weight = self.models[p_i].weight()
            bias = self.models[p_i].bias()           
            weight.data = torch.tensor(np.random.normal(0, 1, (self.classnum, self.dimension))).clone()
            bias.data = torch.tensor(np.random.normal(0, 1,  (1, self.classnum))).clone()

        results = []

        start_time = time.time()
        cluster_assignment = [0 for m_i in range(self.num_users)]
        for glob_iter in range(self.num_glob_iters):
            selected_users = self.select_users()
            result, cluster_assignment = self.train(lr = self.learning_rate, selected_users = selected_users, cluster_assignment = cluster_assignment)
            result['epoch'] = glob_iter
            results.append(result)

            print(f"iter {glob_iter}  trainACC {result['train']:3f} testACC {result['test']:3f}")
            print(cluster_assignment)
        results.append(cluster_assignment)

        duration = (time.time() - start_time)
        print("---train IFCA Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        with open(self.output_dir, 'wb') as outfile:  
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')

    def train(self, lr, selected_users, cluster_assignment):
        result = {}
        # calc loss and grad
        losses = {}
        weight_sets = {}
        bias_sets = {}
        
        for m_i in selected_users:
            for p_i in range(self.K):
                (X, y) = self.dataset['data'][m_i]
                loss_val, weight_i, bias_i = calculate_loss_grad(self.models[p_i], lr, self.loss, X, y )
                losses[(m_i,p_i)] = loss_val                      
                weight_sets[(m_i,p_i)] = weight_i 
                bias_sets[(m_i,p_i)] = bias_i
        # calculate scores
        scores = {}
        for m_i in selected_users: 
            machine_losses = [ losses[(m_i,p_i)] for p_i in range(self.K) ]  

            if self.score == 'set':
                min_p_i = np.argmin(machine_losses)   
                for p_i in range(self.K):
                    if p_i == min_p_i:
                        scores[(m_i,p_i)] = 1
                    else:
                        scores[(m_i,p_i)] = 0

            elif self.score == 'em':

                from scipy.special import softmax
                softmaxed_loss = softmax(machine_losses)
                for p_i in range(self.K):
                    scores[(m_i,p_i)] = softmaxed_loss[p_i]

            else:
                assert self.score in ['set', 'em']

        # apply gradient update
        for p_i in range(self.K):
            cluster_scores = [ scores[(m_i,p_i)] for m_i in selected_users] 
            cluster_weights = [ weight_sets[(m_i,p_i)] for m_i in selected_users]  
            cluster_biass = [ bias_sets[(m_i,p_i)] for m_i in selected_users] 
            self.models[p_i].zero_grad()
            weight = self.models[p_i].weight()
            bias = self.models[p_i].bias()

            wtmp = weight_update(cluster_scores, cluster_weights)  
            btmp = weight_update(cluster_scores, cluster_biass)  
            
            weight.data = wtmp.clone()
            bias.data = btmp.clone()

        # evaluate min_losses 
        min_losses = []        
        for m_i in selected_users:
            machine_losses = [ losses[(m_i,p_i)] for p_i in range(self.K) ]
            min_loss = np.min(machine_losses)
            min_losses.append(min_loss)

            machine_scores = [ scores[(m_i,p_i)] for p_i in range(self.K) ]
            assign = np.argmax(machine_scores)
            cluster_assignment[m_i] = assign

         # Evaluate model each interation
        train_acc = self.train_error_and_loss(cluster_assignment) 
        test_acc = self.test(cluster_assignment) 

        result['train'] = train_acc
        result['test'] = test_acc
        result["min_loss"] = np.mean(min_losses)  
        result["min_losses"] = min_losses
        result["cluster_assignment"] = cluster_assignment       

        return result, cluster_assignment

    def test(self, cluster_assignment):
        test_acc = []
        for m_i in range(self.num_users):
            (X, y) = self.dataset['test'][m_i]
            output = self.models[cluster_assignment[m_i]](X)
            y = y.long()
            test_acc.append((torch.sum(torch.argmax(output, dim=1) == y)).item()/len(y))
        return np.mean(test_acc)


    def train_error_and_loss(self, cluster_assignment):
        train_acc = 0
        for m_i in range(self.num_users):
            (X, y) = self.dataset['data'][m_i]
            output = self.models[cluster_assignment[m_i]](X)
            y = y.long()
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()/len(y)
        return train_acc/self.num_users
    


