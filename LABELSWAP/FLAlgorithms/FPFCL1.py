import os
import time
import pickle
import torch
import numpy as np
from FLAlgorithms.base import Base
from utils.util import *
from FLAlgorithms.trainmodel.models import ConvNet
from torch.utils.data import DataLoader, TensorDataset


class FPFCL1(Base):
    def __init__(self, batch_size, learning_rate, lamdaL1, num_glob_iters, local_epochs, num_users, ratio, dimension, class_num, seed,  projectdir, datasetdir):
        super().__init__(batch_size, learning_rate, local_epochs, num_users, ratio, dimension, class_num, seed, projectdir, datasetdir)
        
        np.random.seed(seed)  
        torch.manual_seed(seed)

        self.num_glob_iters = num_glob_iters
        self.lamdal1 = lamdaL1
        self.output_dir = os.path.join(projectdir, 'results_FPFCL1_'+str(lamdaL1)+'.pickle')
        self.dataset_fname = os.path.join(datasetdir, 'dataset.pth')  
            
        self.dataset = torch.load(self.dataset_fname)   

        self.models = [ConvNet() for m_i in range(self.num_users)]
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = learning_rate


    def run(self):
        
        NUM_USER = self.num_users
        dimension =self.dimension
        NUM_CLASS = self.classnum
        lamda = self.lamdal1
        rho = 1
        self.learning_rate = self.lr
        select_num = int(self.select_ratio * NUM_USER) 
       
        ## FL 
        result_trainacc = []
        result_testacc = []
        result_trainloss = []
        result_testloss = []       
        result_iter = []
        result_time = []

        results = {}
        server = ConvNet()
        sever_param = {key : value for key, value in server.named_parameters()}

        dimension = 16*5*5
        V = torch.zeros([NUM_USER, NUM_USER, NUM_CLASS, (dimension+1)], dtype = torch.float)
        theta = torch.zeros([NUM_USER, NUM_USER, NUM_CLASS, (dimension+1)], dtype = torch.float)

        start_time = time.time()
        print('*'*50)
        print('start training FPFCL1 with lamda: ', lamda)
        for glob_iter in range(self.num_glob_iters):
            self.epoch = glob_iter  
            if glob_iter == 0:
                for m_i in range(NUM_USER):
                    for name, param in self.models[m_i].named_parameters():
                        param.data = sever_param[name].data.clone()
                theta, V = self.server(V = V, lam = lamda, rho = 1, participating_clients = np.arange(NUM_USER)) 

            if glob_iter % 5 == 0:
                self.learning_rate *= 0.9

            participating_clients = np.random.choice(NUM_USER, size = select_num, replace=False)

            for m_i in range(NUM_USER):
                self.models[m_i].W_old = {key : value.clone() for key, value in self.models[m_i].named_parameters()}

            for m_i in participating_clients:  ## A_{k}
                weight_u = torch.zeros([NUM_CLASS, dimension], dtype = torch.float)
                bias_u = torch.zeros([NUM_CLASS, 1], dtype = torch.float)
                for m_j in range(NUM_USER):  
                    weight_u += (self.models[m_j].W_old['fc1.weight'].data + theta[m_i][m_j][:,:-1] - V[m_i][m_j][:,:-1]/rho)
                    bias_u += (self.models[m_j].W_old['fc1.bias'].data.reshape(-1,1) + theta[m_i][m_j][:,-1].reshape(-1,1) - V[m_i][m_j][:,-1].reshape(-1,1)/rho)
                weight_u = weight_u / NUM_USER
                bias_u = bias_u / NUM_USER

                self.client(model_index = m_i, weight_u = weight_u, bias_u = bias_u, rho = rho) 
            theta, V = self.server(V = V, lam = lamda, rho = 1, participating_clients = participating_clients) 
            iter_duration = time.time() - start_time

            train_loss, train_acc = self.eval(self.models, dat = "data")
            test_loss, test_acc  = self.eval(self.models, dat = "test")  
            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_trainloss.append(train_loss) 
            result_testloss.append(test_loss)
            result_iter.append(glob_iter)
            result_time.append(iter_duration)

            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")

        duration = (time.time() - start_time)
        print("---train FPFCL1 Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration)) 

        results["iter"] = result_iter
        results['time'] = result_time
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results["trainloss"] = result_trainloss
        results["testloss"] = result_testloss

        with open(self.output_dir, 'wb') as outfile:   
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')

    def server(self, V, lam, rho, participating_clients):
        NUM_USER = self.num_users
        NUM_CLASS = self.classnum
        dimension = self.dimension

        delta = torch.zeros([NUM_CLASS, dimension+1], dtype = torch.float)
        theta = torch.zeros([NUM_USER, NUM_USER, NUM_CLASS, (dimension+1)], dtype = torch.float)

        weights = {key : torch.zeros_like(value) for key, value in self.models[0].named_parameters()}
        for m_i in participating_clients: 
            model_i = self.models[m_i]
            for m_j in range(NUM_USER):
                model_j = self.models[m_j]
                client_j = {key : value for key, value in model_j.named_parameters()}
                for name, param in model_i.named_parameters():
                    if name == 'fc1.weight':
                        delta[:, :-1] = param.data.clone()-client_j[name].data.clone() +  1/rho * V[m_i][m_j][:,:-1].data.clone()
                    elif name == 'fc1.bias':
                        delta[:, -1] = param.data.clone()-client_j[name].data.clone() +  1/rho * V[m_i][m_j][:,-1].data.clone()

                theta[m_i][m_j] = softThresholding(z = delta, t = lam/rho) 
                theta[m_j][m_i] = -theta[m_i][m_j]

                for name, param in model_i.named_parameters():
                    if name == 'fc1.weight':
                        V[m_i][m_j][:, :-1] = V[m_i][m_j][:, :-1] + rho * (param.data.clone() - client_j[name].data.clone() - theta[m_i][m_j][:, :-1])
                    elif name == 'fc1.bias':
                        V[m_i][m_j][:, -1] = V[m_i][m_j][:,-1] + rho * (param.data.clone() - client_j[name].data.clone() - theta[m_i][m_j][:, -1])
                V[m_j][m_i] = -V[m_i][m_j]

            for name, param in model_i.named_parameters():
                weights[name] += param.data

        for name, param in self.models[0].named_parameters():
            weights[name] /= len(participating_clients)

        for m_i in participating_clients:
            for name, param in self.models[m_i].named_parameters():
                if name != 'fc1.weight' and name != 'fc1.bias':
                    param.data = weights[name]
        return theta, V

    def client(self, model_index, weight_u, bias_u, rho):
        batch_size = self.batch_size       
        (X, y) = self.dataset['data'][model_index]
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)
        model = self.models[model_index]

        ## update weight and bias by SGD  OR GD
        for epoch in range(0, self.local_epochs):
            for xx, yy in train_loader: 
                yy_target = model(xx)        
                loss = self.loss(yy_target, yy)
                model.zero_grad()
                loss.backward()  
                for name, param in model.named_parameters():
                    if name == 'fc1.weight':
                        tmp = param.data.clone() - weight_u
                        param.data -= self.learning_rate * (param.grad + rho*tmp)
                    elif name == 'fc1.bias':
                        tmp = param.data.clone() - torch.squeeze(bias_u)
                        param.data -= self.learning_rate * (param.grad + rho*tmp)         
                    else:
                        param.data -= self.learning_rate * param.grad
   

    def user_train(self, model_index): 
        (X, y) = self.dataset['data'][model_index]
        model = self.models[model_index]
        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate)
        data_train = TensorDataset(X, y)
        loader =  DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        for epoch in range(0, self.local_epochs):
            for x, y in loader: 
                optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(model(x.float()), y)
                loss.backward()
                optimizer.step()    