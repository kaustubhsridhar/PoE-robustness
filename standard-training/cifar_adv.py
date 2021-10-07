"""
    [1] https://github.com/BorealisAI/advertorch/blob/master/advertorch_examples/tutorial_attack_imagenet.ipynb 
    [2] ./cifar_plus.py 
"""

#%% imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
import sys
import os
from advertorch.attacks import L2PGDAttack, CarliniWagnerL2Attack, LinfPGDAttack
from advertorch.utils import predict_from_logits
import matplotlib.pyplot as plt
# custom imports
import models.cifar as models
from utils import Get_balanced_normed_subset, mkdir_p
# custom imports for Only_test function
import time
import torch
from utils import Bar, AverageMeter, accuracy, Logger
# seed
np.random.seed(25)
torch.manual_seed(25) 
torch.cuda.manual_seed_all(25)

#%% functions to create and show adversarial images (not copied)
def tensor2npimg(tensor):
    # print(tensor.shape) # of the shape 3 x 32 x 32
    channel_first_np_img = tensor.cpu().numpy() 
    channel_last_np_img = np.transpose(channel_first_np_img, (1, 2, 0)) # of the shape 32 x 32 x 3
    return channel_last_np_img

def _show_images(model, img, advimg, enhance=127):
    np_img = tensor2npimg(img)
    np_advimg = tensor2npimg(advimg)
    np_perturb = tensor2npimg(advimg - img)

    pred = predict_from_logits(model(img))
    advpred = predict_from_logits(model(advimg))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np_img) # plt.imshow(np_img, cmap='gray')
    plt.axis("off")
    plt.title("original image\n prediction: {}".format(pred))

    plt.subplot(1, 3, 2)
    plt.imshow(np_perturb * enhance + 0.5)
    plt.axis("off")
    plt.title("the perturbation,\n enhanced {} times".format(enhance))

    plt.subplot(1, 3, 3)
    plt.imshow(np_advimg)
    plt.axis("off")
    plt.title("perturbed image\n prediction: {}".format(advpred))
    plt.show()

def analyze_batch(set, loader, info):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print("\n==> "+info+" with batch of {}, {} | no. of images {} | no. of batches {} | pixels with min {} and max {}".format(images.shape, labels.shape, len(set), len(loader), torch.min(images[0]), torch.max(images[0])))

def make_folders(loc):
    mkdir_p(loc)
    for i in range(num_classes):
        mkdir_p(loc+'/{0:04}'.format(i))

def create_PGD_Adv(model, dataset, folder, test_single_image = False):
    make_folders(folder)
    loader_1 = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=3)
    analyze_batch(dataset, loader_1, "dataset to perturb")    
    #adversary = LinfPGDAttack(model, eps=1. / 255, nb_iter=7, eps_iter=0.1 / 255, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False) 
    adversary = LinfPGDAttack(model, eps=1. / 255, nb_iter=20, eps_iter=0.1 / 255, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False) 

    i = 0; tot = len(dataset)
    for input, label in loader_1:
        input = input.cuda().float()
        label = label.cuda()
        adv_untargeted_input = adversary.perturb(input, label)
        if test_single_image:
            _show_images(model, input, adv_untargeted_input)
            break
        for j, image in enumerate(adv_untargeted_input): # save every image in perturbed batch of images
            adv_untargeted_np_img = tensor2npimg(image)
            four_digit_label = "{0:04}".format(label[j].item()) # 4 digit label, e.g., 0003
            plt.imsave(folder+"/"+four_digit_label+"/"+str(i+j)+".png", adv_untargeted_np_img)
        i += len(input)
        print("\r", "Progress {} / {}".format(i, tot), end="")


#%% parsing and args (copied cifar.py, lines 27-82) 
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += ['LeNet5-tanh', 'LeNet5-relu']
# Create parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Adversarial Image Generation')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--downloaded_model', action='store_true', default=False, help='this option is given if you want to evaluate a downloaded model.')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock', help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
# Adversarial image generation options
parser.add_argument("--adv_folder", default="PGD/LeNet5_relu_0.01", help="destination from current directory to a folder where you want save adversarial data and results")
parser.add_argument("--show_single_image", dest='show_single_image', default = False, action='store_true', help="Add this tag (with no arg) to show an adv image.")
# Testing on Adv images options
parser.add_argument("--do-only-test", action='store_true', default=False, help="Add option if you only want to test on already generated adversarial images and save adv_folder/log.txt")
# Store in args
args = parser.parse_args()

#%% dataset
# CIFAR 10 or 100
print('==> Preparing dataset %s' % args.dataset)

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100
testset = dataloader(root='./data', train=False, download=True, transform=transform_test)

#%% use_cuda % currently requires GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = 1

#%% model (copied cifar.py, lines 138-174) 
# CIFAR Models
print("==> creating model '{}'".format(args.arch))
if args.arch.startswith('resnext'):
    model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
elif args.arch.startswith('densenet'):
    model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
elif args.arch.startswith('wrn'):
    model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
elif args.arch.endswith('resnet'):
    model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
            )
elif 'LeNet5' in args.arch:
    if 'relu' in args.arch:
        model = models.lenet5('relu')
    else:
        model = models.lenet5('tanh')
else:
    model = models.__dict__[args.arch](num_classes=num_classes)

model = torch.nn.DataParallel(model).cuda() 
cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

#%% Load checkpoint 
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint...')
    assert os.path.isfile(args.resume), 'Error: no checkpoint found!'
    if args.downloaded_model:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
    else:
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
else:
    print('==? Unspecified model checkpoint..')

# confirm evaluate mode
model.eval()

#%% Generate and save adversarial images
print("Only test? ", args.do_only_test)
if not args.do_only_test:
    outer_folder, mtype_folder = args.adv_folder.rsplit('/') # e.g., PGD/cifar10_densenet_{0.1}
    mkdir_p(outer_folder)
    create_PGD_Adv(model, testset, args.adv_folder, test_single_image = args.show_single_image)

#%% Test function copied from cifar.py 
def Only_test(testloader, model, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad(): # previously, didn't exist. Instead, volatile=True (line 310) was thought to be sufficient. This no longer has any effect in PyTorch.
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets) # previously, torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size(0)) # previously, losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1, inputs.size(0)) # previously, top1.update(prec1[0], inputs.size(0))
            top5.update(prec5, inputs.size(0)) # previously, top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

#%% Test accuracy of model on clean images
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.workers)
#analyze_batch(testset, testloader, "loaded testset")
criterion = nn.CrossEntropyLoss()
test_loss, test_top1, test_top5 = Only_test(testloader, model, criterion, use_cuda)
print('\n')
print('Clean Results: top 1 accuracy = {} | top 5 accuracy = {}'.format(test_top1, test_top5))

# Test accuracy of model on adversarial images 
Advset = datasets.ImageFolder(args.adv_folder, transform=transform_test)
Advloader = DataLoader(Advset, batch_size=256, shuffle=False, num_workers=args.workers)
analyze_batch(Advset, Advloader, "loaded advset")
criterion = nn.CrossEntropyLoss()
adv_loss, top1, top5 = Only_test(Advloader, model, criterion, use_cuda)
print('\n')
print('PGD Results: top 1 accuracy = {} | top 5 accuracy = {}'.format(top1, top5))

# create logger
title = 'cifar-10-' + args.arch
logger = Logger(os.path.join(args.adv_folder, 'log.txt'), title=title) # previously, args.checkpoint # I want to save in adv_folder instead.
logger.set_names(['Adv Loss', 'Adv Acc (Top 1)', 'Adv Acc (Top 5)'])
# save to adv_folder/log.txt
logger.append([adv_loss, top1, top5])
logger.close()
