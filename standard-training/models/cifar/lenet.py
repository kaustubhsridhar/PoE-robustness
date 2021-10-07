import torch.nn as nn

class LeNet5(nn.Module):
    """
    [1] https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    [2] https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
    [3] formula for conv and pool layers: https://cs231n.github.io/convolutional-networks/#pool
    """
    def __init__(self, activation, n_classes=10):
        super(LeNet5, self).__init__()
        n_acti = 5

        if "relu" in activation:
            self.acti = [nn.ReLU() for i in range(n_acti)]
        elif "tanh" in activation:
            self.acti = [nn.Tanh() for i in range(n_acti)]
        else:
            print("Specify activation function!")
            exit(0)
        # W_{in} = 32
        self.conv1 = nn.Conv2d(3, 6, 5) # W_{out} = (32 - 5)/1 + 1 = 28
        self.pool1 = nn.MaxPool2d(2) # W_{out} = (28 - 2)/2 + 1 = 14 (since default stride = kernel size)
        self.conv2 = nn.Conv2d(6, 16, 5) # W_{out} = (14 - 5)/1 + 1 = 10
        self.pool2 = nn.MaxPool2d(2) #  # W_{out} = (10 - 2)/2 + 1 = 5

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        y = self.pool1(self.acti[0](self.conv1(x)))
        y = self.pool2(self.acti[1](self.conv2(y)))
        y = y.view(y.shape[0], -1)
        y = self.acti[2](self.fc1(y))
        y = self.acti[3](self.fc2(y))
        y = self.acti[4](self.fc3(y))
        return y

def lenet5(activation):
    """
    Constructs a LeNet5 model.
    """
    return LeNet5(activation)