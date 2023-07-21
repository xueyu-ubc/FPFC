import os
import time
import pickle
import torch
import numpy as np
from FLAlgorithms.base import Base
from utils.util import *
from FLAlgorithms.trainmodel.models import Mclr_Logistic


class FPFCL1(Base):
    def __init__(self, batch_size, learning_rate, lamdaL1, thresholdL1, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.lamdal1 = lamdaL1
        self.threshold = thresholdL1
        self.output_dir = os.path.join(projectdir, 'results_FPFCL1_'+str(lamdaL1)+'.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')         
        self.dataset = torch.load(self.dataset_fname)   
        self.models = [Mclr_Logistic(dimension, class_num)  for m_i in range(num_users)] 
        self.loss = torch.nn.CrossEntropyLoss()

    def run(self):
        
        NUM_USER = self.num_users
        dimension =self.dimension
        lamda = self.lamdal1
        rho = 1 
        self.initialize_weights() 
       
        ## FL 
        result_test = []
        result_train = []      
        result_iter = []
        results = {}
       
        V = torch.zeros([NUM_USER, NUM_USER, self.classnum, (dimension+1)], dtype = torch.float)
        theta = torch.zeros([NUM_USER, NUM_USER, self.classnum, (dimension+1)], dtype = torch.float)

        for m_i in range(NUM_USER):
            weight_i = self.models[m_i].weight()
            bias_i = self.models[m_i].bias()
            coeff_i = torch.cat((weight_i.data, bias_i.data.reshape(-1,1)), 1)
            for m_j in range(NUM_USER):
                weight_j = self.models[m_j].weight()
                bias_j = self.models[m_j].bias()                 
                coeff_j =torch.cat((weight_j.data, bias_j.data.reshape(-1,1)),1) 
                theta[m_i][m_j] = coeff_i - coeff_j

        start_time = time.time()
        print('*'*50)
        print('start training FPFCL1 with lamda: ', lamda, ' and threshold: ', self.threshold)

        for glob_iter in range(self.num_glob_iters):

            selected_users = self.select_users()
            weight_old = torch.zeros([NUM_USER, self.classnum, dimension], dtype = torch.float)  
            bias_old = torch.zeros([NUM_USER, self.classnum, 1], dtype = torch.float)   
            for m_i in range(NUM_USER):
                weight_i = self.models[m_i].weight()
                weight_old[m_i] = weight_i.clone()
                bias_i = self.models[m_i].bias()
                bias_old[m_i] = bias_i.clone().t()

            for m_i in selected_users:       ## A_{k}
                weight_u = torch.zeros([self.classnum, dimension], dtype = torch.float)
                bias_u = torch.zeros([self.classnum, 1], dtype = torch.float)
                for m_j in range(NUM_USER):   ## U(1,m)
                    weight_u += (weight_old[m_j] + theta[m_i][m_j][:,:-1] - V[m_i][m_j][:,:-1]/rho)
                    bias_u += (bias_old[m_j] + theta[m_i][m_j][:,-1].reshape(-1,1) - V[m_i][m_j][:,-1].reshape(-1,1)/rho)
                weight_u = weight_u / NUM_USER
                bias_u = bias_u / NUM_USER

                self.client(model_index = m_i, weight_u = weight_u, bias_u = bias_u, rho = rho)
            delta = torch.zeros([self.classnum, dimension+1], dtype = torch.float) 

            ## update theta and v
            delta = torch.zeros([self.classnum, dimension+1], dtype = torch.float)
            for m_i in selected_users:
                weight_i = self.models[m_i].weight() 
                bias_i = self.models[m_i].bias()
                coef_i = torch.cat((weight_i.data, bias_i.data.reshape(-1,1)),1)
                for m_j in range(NUM_USER):
                    weight_j = self.models[m_j].weight() 
                    bias_j = self.models[m_j].bias()
                    coef_j = torch.cat((weight_j.data, bias_j.data.reshape(-1,1)),1)
                    delta = coef_i.data.float() - coef_j.data.float() + 1/(rho) * V[m_i][m_j]
                    theta[m_i][m_j] = softThresholding(z = delta, t = lamda/rho)
                    theta[m_j][m_i] = -theta[m_i][m_j]
            
                    V[m_i][m_j] = V[m_i][m_j] + rho * (coef_i.data.float() - coef_j.data.float() - theta[m_i][m_j])
                    V[m_j][m_i] = -V[m_i][m_j]

            if glob_iter == self.num_glob_iters-1:
                scad_thetanorm2 = np.zeros((self.num_users, self.num_users))
                coef_client = []
                for m_i in range(self.num_users):
                    for m_j in range(m_i+1, self.num_users):
                        scad_thetanorm2[m_i][m_j] =  torch.norm(theta[m_i][m_j], 2)
                        scad_thetanorm2[m_j][m_i] = scad_thetanorm2[m_i][m_j]
                       
                    weight_i = self.models[m_i].weight()  
                    bias_i = self.models[m_i].bias()
                    coef_i = torch.cat((weight_i.data, bias_i.data.reshape(-1,1)),1)
                    coef_client.append(coef_i.data)

                coef_avg, cluster_indices = threshold(self.num_users, self.dataset, scad_thetanorm2, coef_client, value = self.threshold)
                for m_i in range(self.num_users):
                    weight_i = self.models[m_i].weight()  
                    bias_i = self.models[m_i].bias()
                    weight_i.data = coef_avg[m_i][:,:-1].to(torch.float64)
                    bias_i.data = coef_avg[m_i][:,-1].to(torch.float64)
                
                print(f"cluster {cluster_indices}")

            # Evaluate model each interation
            train_acc = self.train_error(self.models) 
            test_acc = self.test(self.models) 
            result_iter.append(glob_iter)
            result_train.append(train_acc) 
            result_test.append(test_acc)
            print(f" iter {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")       

        duration = (time.time() - start_time)
        print("---train FPFCL1 Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results["iter"] = result_iter
        results["train"] = result_train
        results["test"] = result_test
        results['lam'] = lamda
        results['cluster_assign'] = cluster_indices

        
        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')


    def client(self, model_index, weight_u, bias_u, rho):    
        (X, y) = self.dataset['data'][model_index]
        model = self.models[model_index]      
        ## update weight and bias by GD OR SGD
        for epoch in range(self.local_epochs):
            yy_target = model(X)  
            yy = y.long()      
            loss = self.loss(yy_target, yy)
            model.zero_grad()
            loss.backward()        
            # self.optimizer.step()
            weight = model.weight()
            bias = model.bias()
            d_weight = weight.grad.clone()  
            d_bias = bias.grad.clone()             
            tmp_weight = weight.data - weight_u
            tmp_bias = bias.data - bias_u.t()
            weight.data -= self.learning_rate * (d_weight + rho * tmp_weight)   
            bias.data -= self.learning_rate * (d_bias + rho * tmp_bias)

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

 