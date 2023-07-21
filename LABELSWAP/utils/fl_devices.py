import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader: 
            x, y = x, y
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x.float()), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  

    return running_loss / samples
      
def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0
    train_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x, y
            
            y_ = model(x.float())
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

            loss = torch.nn.CrossEntropyLoss()(y_, y)
            train_loss += loss.item()*y.shape[0]

    return correct/samples, train_loss/samples


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def reduce_add_average(targets, sources):
    for target in targets:
        for name in target.W:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target.W[name].data += tmp
        
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

    return angles.numpy()

class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn()
        self.data = data
        self.W = {key :value for key, value in self.model.named_parameters()}
        
    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)
  
  
class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, dat, idnum, batch_size=100, train_frac=1):
        dataset = TensorDataset(dat[0], dat[1])
        super().__init__(model_fn, dataset)  
        self.optimizer = optimizer_fn(self.model.parameters())
            
        self.data = dataset
        n_train = int(len(dataset)*train_frac)
        n_eval = len(dataset) - n_train 
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)
        
        self.id = idnum
        
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
    
    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"]*=0.99
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats  

    def reset(self): 
        copy(target=self.W, source=self.W_old)
    
    
class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, dat):
        super().__init__(model_fn,dat)
        self.model_cache = []
        
    
    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 
    
    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])
        
    def compute_pairwise_similarities(self, update_weight):
        return pairwise_angles(update_weight)
  
    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten() 
        return c1, c2
    
    def aggregate_clusterwise(self, client_clusters, update_weights):
        for cluster in client_clusters:
            reduce_add_average(targets=[client for client in cluster], 
                               sources=[update_weights[client.id] for client in cluster])
            
            
    def compute_max_update_norm(self, update_weights):
        return np.max([torch.norm(flatten(client)).item() for client in update_weights])

    
    def compute_mean_update_norm(self, update_weights):
        return torch.norm(torch.mean(torch.stack([flatten(client) for client in update_weights]), 
                                     dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs, 
                            {name : params[name].data.clone() for name in params}, 
                            [accuracies[i] for i in idcs])]


