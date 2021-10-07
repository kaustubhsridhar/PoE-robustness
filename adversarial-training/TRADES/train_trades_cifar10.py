from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss

from more_utils.lipschitz import Weibull_Fitter # New: 
from more_utils.misc import AverageMeter # New: 
import pickle # New: 

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=75, metavar='N',
                    help='number of epochs to train') # previously, default=76
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8.0/255,
                    help='perturbation') # previously, default=0.031
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=25, type=int, metavar='N',
                    help='save frequency') # previously, default=1
parser.add_argument('--only-lipschitz', action='store_true', default=False,
                    help='Only perform calculation of LIpschitz constant') # New:   

args = parser.parse_args()

# ------------------------------ # New: ONLY LIPSCHITZ -------------------------------
def read_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        return obj

if args.only_lipschitz:
    print('Only calculating the lipschitz constant using saved Lipschitz object...')
    Lipschitz = read_object(os.path.join(args.model_dir,'lipschitz.pkl'))
    Lipschitz.fit()
    exit(0)

## Set visible devices
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# ------------------------------------------------------------------------------

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    grads = AverageMeter() # New: 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # New: 

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # New: collect gradients
        data_grad = data.grad.detach().clone()
        batch_gradients = data_grad.reshape(data.size(0), -1) # collapses [64, 3, 32, 32] to [64, 3*32*32]
        avg_batch_gradients = torch.sum(batch_gradients, dim=0) / data.size(0) # averages rows to give [3*32*32] tensor # Set: keepdim=True to get [1, 3*32*32] tensor
        grads.update(avg_batch_gradients, data.size(0)) # Note: avg_batch_gradient given. Reason: see self.sum in AverageMeter()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # New: After every epoch, collect params (weights / biases) # Ref: https://discuss.pytorch.org/t/access-all-weights-of-a-model/77672/10
    params = []
    for param in model.parameters():
        param = param.detach().clone() # Note: The value on left of '=' is assigned the name on right in Python! 
        params.append(param.view(-1))
    params = torch.cat(params)

    return grads.avg.cpu().numpy(), params.cpu().numpy() # New: grads and params returned # previously, no return statement


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    model = nn.DataParallel(WideResNet()).cuda() # previously, model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    Lipschitz = Weibull_Fitter(200, args.epochs) # New: 

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        gradient_vector, params_vector = train(args, model, device, train_loader, optimizer, epoch) # New: gradient_vector, params_vector arguments # previously, only the function was called and nothing was returned
        Lipschitz.update(gradient_vector, params_vector) # New: 

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
        # New: save Lipschitz object below
        if epoch == args.epochs:
            save_object(Lipschitz, os.path.join(model_dir,'lipschitz.pkl'))

    Lipschitz.fit() # New: calculate Lipschitz constant and print to stdout

# New: function below
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
