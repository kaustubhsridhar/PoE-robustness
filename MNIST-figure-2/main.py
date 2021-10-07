# 1a. imports
import numpy as np
import argparse
import sys
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# 1b. custom imports
from utils import get_balanced_mnist784
from network import LeNet5, Net

# 2a. seed
np.random.seed(25)
torch.manual_seed(25) # torch.cuda.manual_seed_all(20)
# 3a. bash argument parsing
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parsing input options...") # ref: https://opensource.com/article/19/7/parse-arguments-python, scroll down to function definition in https://docs.python.org/3/library/argparse.html 
    parser.add_argument("-b", "--batch_size", type=int, default = 64, help="Batch size used in training.")
    parser.add_argument("-n", "--network", default = "LeNet5", help="Network: LeNet5 or OneLayer")
    parser.add_argument("-a", "--activation", default = "relu", help="Activation function used in network (all lowercase).")
    parser.add_argument("-l", "--learning_rate", type=float, default = 0.01, help="Learning rate (float).")
    parser.add_argument("-f", "--adv_folder", default="PGD/LeNet5_relu_0.01", help="destination from current directory to folder with adversarial data (also identifies location to save statistics within ./np_data folder) [given in both rounds]")
    parser.add_argument("-g", "--gpu", type=int, default = 0, help="GPU number (0-3)")
    parser.add_argument("-r2", "--second_round",dest='round2', default = False, action='store_true', help="Add this tag (no arg) in second round of training to also test on Adv data.")
    options = parser.parse_args(args)
    return options
def displayOptions(options):
    print("\n displaying options as <name> -> <type> : <value>")
    print("batch_size -> {} : {}, network -> {} : {}, activation -> {} : {}, learning_rate -> {} : {}, round -> {} : {}".format(type(options.batch_size), options.batch_size, type(options.network), options.network, type(options.activation), options.activation, type(options.learning_rate), options.learning_rate, type(options.round2+1), options.round2+1))

options = getOptions(sys.argv[1:])
displayOptions(options)

# 2b. device
device = torch.device("cuda:"+str(options.gpu) if torch.cuda.is_available() else "cpu")
print(device)

"""Data"""
# training and validation MNIST: 1x1x784 tensor with values in [0,1] for OneLayer
if "OneLayer" in options.network:
    full_train = datasets.MNIST('./', download=True, train=True)
    full_val = datasets.MNIST('./', download=True, train=False)
    trainset, trainloader = get_balanced_mnist784(full_train, 50000, data_normed = 1, batch_size = options.batch_size, shuffle = True)
    valset, valloader = get_balanced_mnist784(full_val, 8000, data_normed = 1, batch_size = 256, shuffle = False)
    # In second round, load adversarial data!
    if options.round2:
        t_scale = transforms.Lambda(lambda x: 1.0*x)
        adv_img_transforms = transforms.Compose([ transforms.Grayscale(), transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x)), t_scale])
        Advset = datasets.ImageFolder(options.adv_folder, transform=adv_img_transforms)
        Advloader = DataLoader(Advset, batch_size=256, shuffle=False, num_workers=3)
    # Analyze sets
    print("Trainset like {}x{}x{} with minimum {} and maximum {}".format(len(trainset), 1, trainset[0][0].shape, torch.min(trainset[0][0]), torch.max(trainset[0][0])))
    print("Valset like {}x{}x{} with minimum {} and maximum {}".format(len(valset), 1, valset[0][0].shape, torch.min(trainset[0][0]), torch.max(trainset[0][0])))
    # Analyze loader in training

# 1x28x28 (or 1x32x32 in [2]) tensor for LeNet5    
elif "LeNet5" in options.network:
    """
    [1] https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
    [2] https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb 
    """
    normal_transforms = transforms.Compose([transforms.ToTensor()]) # in [2], transforms.Resize((32, 32)) was added at start of transforms
    trainset = datasets.MNIST('./', download=True, train=True, transform=normal_transforms)
    valset = datasets.MNIST('./', download=True, train=False, transform=normal_transforms)
    trainloader = DataLoader(dataset=trainset, batch_size=options.batch_size, shuffle=True, num_workers=3)
    valloader = DataLoader(dataset=valset, batch_size=256, shuffle=False, num_workers=3)
    # In second round, load adversarial data (have to add Grayscale transform to make saved multi channel image -> single channel)!
    if options.round2:
        adv_img_transforms = transforms.Compose([ transforms.Grayscale(), transforms.ToTensor()]) # For [2], transforms.Resize((32, 32)) in the middle of this list
        Advset = datasets.ImageFolder(options.adv_folder, transform=adv_img_transforms)
        Advloader = DataLoader(Advset, batch_size=256, shuffle=False, num_workers=3)
    # Analyze sets # BELOW GIVES MNIST ORIGINAL SIZE
    print("Trainset (original MNIST) like {} with minimum {} and maximum {}".format(trainset.data.shape, torch.min(trainset[0][0]), torch.max(trainset[0][0])))
    print("Valset (original MNIST) like {} with minimum {} and maximum {}".format(valset.data.shape, torch.min(trainset[0][0]), torch.max(trainset[0][0])))
    # Analyze loader in training # REAL SIZE

"""Network and Activation Function"""
if "OneLayer" in options.network:
    net = Net(options.activation)
elif "LeNet5" in options.network:
    net = LeNet5(options.activation)
# transfer to device
net = net.to(device)
print(Net)
# add criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=options.learning_rate)

"""Training and Validation Loop"""
# validate function
def pytorch_validate(net, valloader):
    total_loss = 0; total = 0
    total_error = 0
    with torch.no_grad():
        for x, y in valloader:
            # continue validation
            x = x.to(device)
            y = y.to(device)
            #y_one_hot = F.one_hot(y, num_classes=10)
            #x = x.float(); y = y.float(); y_one_hot = y_one_hot.float()

            out = net(x)
            loss = criterion(out, y)
            _, preds = torch.max(out, 1)

            B = y.size(0); total += B
            total_error += torch.sum(preds != y.data)
            total_loss += loss.item()*B

    return total_loss/total, total_error/total

# train function
def train(net, trainloader, valloader, Advloader = None, epochs = 10):
    train_losses = []; val_losses = []; adv_losses = []
    train_errors = []; val_errors = []; adv_errors = []
    adv_loss, adv_error = -1, -1
    epoch = 0
    while epoch < epochs:
        epoch += 1
        for x, y in trainloader:
            # 1. move to device, get one_hot encoding
            x = x.to(device)
            y = y.to(device)
            #y_one_hot = F.one_hot(y, num_classes=10)
            #x = x.float(); y = y.float(); y_one_hot = y_one_hot.float()

            # 2. zero gradient buffer
            optimizer.zero_grad()      

            # 3. forward pass (with or without added noise; test for full rank)
            out = net(x)

            loss = criterion(out, y)
            _, preds = torch.max(out, 1)
            total = y.size(0)
            error = torch.sum(preds != y.data)/total

            # 4. backward pass
            loss.backward()

            # 5. Logging and debugging
            train_losses.append(loss.item())
            train_errors.append(error)

            # 6. One step of SGD
            optimizer.step()

        # 7. full batch validation every epoch
        val_loss, val_error = pytorch_validate(net, valloader)
        val_losses.append(val_loss)
        val_errors.append(val_error)
        if options.round2:
            adv_loss, adv_error = pytorch_validate(net, Advloader)
            adv_losses.append(adv_loss)
            adv_errors.append(adv_error)          
        # change below ---
        print("ep {}/{}: train loss {:.2f}, err {:.2f} | val loss {:.2f}, err {:.2f} | adv loss {:.2f}, err {:.2f}".format(epoch, epochs, loss.item(), error, val_loss, val_error, adv_loss, adv_error))

    if options.round2:
        lists_to_save = [train_losses, train_errors, val_losses, val_errors, adv_losses, adv_errors]
    else:
        lists_to_save = [train_losses, train_errors, val_losses, val_errors]
    # save data
    ct = 0
    os.system("mkdir ./np_data")
    os.system("mkdir ./np_data/PGD")
    os.system("mkdir ./np_data/CW")
    for list in lists_to_save:
        ct += 1
        np.save("./np_data/"+options.adv_folder+"_"+str(ct)+".npy", np.array(list))

    return net

"""Training and Validation"""
net = train(net, trainloader, valloader, Advloader if options.round2 else None)
os.system("mkdir ./model_statedicts")
torch.save(net.state_dict(), "./model_statedicts/mnist_"+options.network+"_"+options.activation+"_"+str(options.learning_rate)+".pth")