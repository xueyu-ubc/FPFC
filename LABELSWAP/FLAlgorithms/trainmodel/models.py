import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.reshape = Reshape()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size = 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16,  kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        # x = self.reshape(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        return x


class CifarNet(torch.nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)