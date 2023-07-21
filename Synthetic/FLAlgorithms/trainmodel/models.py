import torch
import torch.nn.functional as F

class Mclr_Logistic(torch.nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
    
    def weight(self):
        return self.fc1.weight

    def bias(self):
        return self.fc1.bias

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

