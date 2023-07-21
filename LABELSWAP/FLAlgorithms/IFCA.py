import os
import time
import pickle
import torch
import copy
import numpy as np
from FLAlgorithms.base import Base
from FLAlgorithms.trainmodel.models import ConvNet
from torch.utils.data import DataLoader, TensorDataset

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

        self.models = [ConvNet() for p_i in range(K)]
        self.criterion = torch.nn.CrossEntropyLoss()

    def run(self):
        num_epochs = self.num_glob_iters
        lr = self.learning_rate
        result_trainacc = []
        result_testacc = []
        result_trainloss = []
        result_testloss = []       
        result_iter = []
        results = {}
        # epoch 0
        cluster_assign = self.get_inference_clusters()
        print(cluster_assign)

        start_time = time.time()
        for glob_iter in range(num_epochs):
            self.epoch = glob_iter
            lr = self.lr_schedule(glob_iter)
            self.train(cluster_assign, lr = lr)

            train_loss, train_acc = self.eval_all(self.models, dat = "data", cluster_assign = cluster_assign)
            test_loss, test_acc = self.eval_all(self.models, dat = "test", cluster_assign = cluster_assign) 
            
            cluster_assign = self.get_inference_clusters()

            result_trainacc.append(train_acc) 
            result_testacc.append(test_acc)
            result_trainloss.append(train_loss) 
            result_testloss.append(test_loss)
            result_iter.append(glob_iter)
            print(f" epoch {glob_iter} trainACC {train_acc:3f} testACC {test_acc:3f}")
            print(cluster_assign)


        duration = (time.time() - start_time)
        print("---train IFCA Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))

        results["iter"] = result_iter
        results["trainacc"] = result_trainacc
        results["testacc"] = result_testacc
        results["trainloss"] = result_trainloss
        results["testloss"] = result_testloss

        with open(self.output_dir, 'wb') as outfile:  
            pickle.dump(results, outfile)  
            print(f'result written at {self.output_dir}')


    def lr_schedule(self, epoch):
        if self.learning_rate is None:
            self.learning_rate = self.learning_rate

        if epoch % 5 == 0 and epoch != 0:
            self.learning_rate = self.learning_rate * 0.9

        return self.learning_rate


    def train(self, cluster_assign, lr):
        p = self.K
        select_num = np.int32(self.select_ratio * self.num_users)
        selected_users = np.random.choice(self.num_users, size = select_num, replace=False)

        updated_models = []
        for m_i in selected_users:
            (X, y) = self.dataset['data'][m_i]
            p_i = cluster_assign[m_i] 
            model = copy.deepcopy(self.models[p_i])      
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle = True)
            for step_i in range(self.local_epochs):
                for xx, yy in loader:
                    y_logit = model(xx)
                    loss = self.criterion(y_logit, yy)
                    model.zero_grad()
                    loss.backward()
                    self.local_param_update(model, lr)
                model.zero_grad()
            
            updated_models.append(model)

        local_models = [[] for p_i in range(p)]
        for ii in range(len(selected_users)):
            p_i = cluster_assign[selected_users[ii]]
            local_models[p_i].append(updated_models[ii])

        for p_i, models in enumerate(local_models):
            if len(models) >0:
                self.global_param_update(models, self.models[p_i])


    def check_local_model_loss(self, local_models):
        # for debugging
        m = self.config['m']

        losses = []
        for m_i in range(m):
            (X, y) = self.dataset['data'][m_i]
            y_logit = local_models[m_i](X)
            loss = self.criterion(y_logit, y)

            losses.append(loss.item())

        return np.array(losses)


    def get_inference_clusters(self):
        m = self.num_users
        dataset = self.dataset['test']
        p = self.K
        losses = {}
        for m_i in range(m):
            (X, y) = dataset[m_i] 
            for p_i in range(p):
                model = self.models[p_i]
                y_logit = model(X)
                loss = self.criterion(y_logit, y) # loss of
                losses[(m_i,p_i)] = loss.item()

        # calculate loss and cluster the machines
        cluster_assign = []
        for m_i in range(m):
            machine_losses = [ losses[(m_i,p_i)] for p_i in range(p) ]
            min_p_i = np.argmin(machine_losses)
            cluster_assign.append(min_p_i)        
        return cluster_assign

    def local_param_update(self, model, lr):
        # gradient update manually
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad
        model.zero_grad()


    def global_param_update(self, local_models, global_model):
        # average of each weight
        weights = {}
        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)
                weights[name] += param.data

        for name, param in global_model.named_parameters():
            weights[name] /= len(local_models)
            param.data = weights[name].clone()

    def eval_all(self, models, dat, cluster_assign):
        batch_size = self.batch_size
        loss = []
        acc = []

        if(dat == "data"):
            data = self.dataset['data']
        elif(dat =="test"):
            data = self.dataset['test']
        for m_i in range(self.num_users):
            p_i = cluster_assign[m_i]
            model = models[p_i]  
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
    
