import torch
import numpy as np
import copy
from typing import Iterable 


def calculate_loss_grad(model, criterion, X, y):
    y_target = model(X)
    y = y.long()
    loss = criterion(y_target, torch.squeeze(y))
    model.zero_grad()
    loss.backward()

    loss_value = loss.item()
    weight = model.weight()
    bias = model.bias()
    d_weight = weight.grad.clone()
    d_bias = bias.grad.clone()

    return loss_value, d_weight, d_bias

def gradient_update(scores, grads):
    m = len(grads)
    tmp = torch.zeros_like(grads[0])
    for m_i in range(m):
        tmp += scores[m_i] * grads[m_i]
    tmp /= m

    return tmp
    
def similarities(num_client, weight):
    sim = np.zeros((num_client, num_client))
    for m_i in range(num_client):
        for m_j in range(num_client):
            sim[m_i][m_j] =  torch.mean(torch.cosine_similarity(weight[m_i], weight[m_j]))
            sim[m_j][m_i] = sim[m_i][m_j]

    return sim

def norm2_diff(num_client, weight):
    diff = np.zeros((num_client, num_client))
    for m_i in range(num_client):
        for m_j in range(num_client):
            diff[m_i][m_j] =  torch.norm(weight[m_i] - weight[m_j], 2)
            diff[m_j][m_i] = diff[m_i][m_j]

    return diff

def softThresholding(z,t):
    s = 1 - t/torch.norm(z,  p = 'fro')
    if s>0:
        return s*z
    else:
        return torch.zeros_like(z)

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def random_normal_tensor(size, loc = 0, scale = 1):
    return torch.randn(size) * scale + loc

def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader: 
            x, y = x, y
            optimizer.zero_grad()
            y = y.long()
            y_target = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_target, y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  

    return running_loss / samples

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def flatten1(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten1(x):
                yield sub_x
        else:
            yield x
            
            
def calculating_adjacency(clients_idxs, U): 
        
    nclients = len(clients_idxs)
    
    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            #print(idx1)
            #print(U)
            #print(idx1)
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])
            
            #sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
            #sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
            #sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
            mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
            sim_mat[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi
           
    return sim_mat

def hierarchical_clustering(A, thresh=1.5, linkage='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix. 
    
    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)
    
    :return: clusters
    '''
    label_assg = {i: i for i in range(A.shape[0])}
    
    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A,-np.NINF)
        #print(f'step {step} \n {A}')
        step+=1
        ind=np.unravel_index(np.argmin(A, axis=None), A.shape)

        if A[ind[0],ind[1]]>thresh:
            print('Breaking HC')
            break
        else:
            np.fill_diagonal(A,0)
            if linkage == 'maximum':
                Z=np.maximum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'minimum':
                Z=np.minimum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'average':
                Z= (A[:,ind[0]] + A[:,ind[1]])/2
            
            A[:,ind[0]]=Z
            A[:,ind[1]]=Z
            A[ind[0],:]=Z
            A[ind[1],:]=Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            if type(label_assg[ind[0]]) == list: 
                label_assg[ind[0]].append(label_assg[ind[1]])
            else: 
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

            label_assg.pop(ind[1], None)

            temp = []
            for k,v in label_assg.items():
                if k > ind[1]: 
                    kk = k-1
                    vv = v
                else: 
                    kk = k 
                    vv = v
                temp.append((kk,vv))

            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten1(label_assg[k])))
        elif type(label_assg[k]) == int: 
            clusters.append([label_assg[k]])
            
    return clusters
