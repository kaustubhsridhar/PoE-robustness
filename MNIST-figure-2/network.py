import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

"""LeNet5"""
class LeNet5(nn.Module):
    """
    [1] https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
    [2] https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb 
    """
    def __init__(self, ACTIVATION, n_classes=10):
        super(LeNet5, self).__init__()
        n_acti = 5

        if "relu" in ACTIVATION:
            self.acti = [nn.ReLU() for i in range(n_acti)]
        elif "tanh" in ACTIVATION:
            self.acti = [nn.Tanh() for i in range(n_acti)]
        else:
            print("Specify activation function!")
            exit(0)
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.pool1(self.acti[0](self.conv1(x)))
        y = self.pool2(self.acti[1](self.conv2(y)))
        y = y.view(y.shape[0], -1)
        y = self.acti[2](self.fc1(y))
        y = self.acti[3](self.fc2(y))
        y = self.acti[4](self.fc3(y))
        return y

"""OneLayer"""
class Net(nn.Module):
    def __init__(self, IDENTIFIER):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)
        if IDENTIFIER in ["relu", "relu_noisy"]:
            self.acti = nn.ReLU()
        elif IDENTIFIER in ["tanh", "tanh_noisy"]:
            self.acti = nn.Tanh()
        elif IDENTIFIER in ["sigmoid", "sigmoid_noisy"]:
            self.acti = nn.Sigmoid()
        elif IDENTIFIER in ["leakyrelu", "leakyrelu_noisy"]:
            self.acti = nn.LeakyReLU()
        elif IDENTIFIER in ["hardtanh", "hardtanh_noisy"]:
            self.acti = nn.Hardtanh()
        elif IDENTIFIER in ["hardsigmoid", "hardsigmoid_noisy"]:
            self.acti = nn.Hardsigmoid()
        elif IDENTIFIER in ["swish", "swish_noisy", "silu"]: # SiLU same as swish but coined before.
            self.acti = nn.SiLU()
        elif IDENTIFIER in ["hardswish", "hardswish_noisy"]:
            self.acti = nn.Hardswish()
        else:
            print("Specify activation function!")
            exit(0)
        #self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        self.h1 = self.acti(self.fc(x))
        #self.h2 = self.fc2(self.h1)
        return self.h1