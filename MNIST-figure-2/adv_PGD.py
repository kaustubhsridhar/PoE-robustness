  # followed advertorch tutorial: https://github.com/BorealisAI/advertorch/blob/master/advertorch_examples/tutorial_attack_imagenet.ipynb 
# imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import argparse
import sys
import os
from advertorch.attacks import L2PGDAttack, CarliniWagnerL2Attack, LinfPGDAttack
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc
import matplotlib.pyplot as plt
import logging
# custom imports
from utils import get_balanced_mnist784, get_balanced_mnist_28x28
from network import LeNet5, Net

def flat784_tensor2npimg(tensor):
    return tensor.cpu().numpy().reshape((28,28))

def _show_images(model, img, advimg, enhance=127):
    np_img = flat784_tensor2npimg(img)
    np_advimg = flat784_tensor2npimg(advimg)
    np_perturb = flat784_tensor2npimg(advimg - img)

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

def create_PGD_MNIST_Adv(model, dataset, folder, test_single_image = False):
    os.system("bash adv.sh -f "+folder)
    loader_1 = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=3)

    # L2PGDAttack(model, loss_fn = nn.MSELoss(), eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    # CarliniWagnerL2Attack(model, num_classes=10, confidence=0, targeted=False, learning_rate=0.1, binary_search_steps=4, max_iterations=500, abort_early=True, initial_const=0.01, clip_min=0.0, clip_max=1.0)
    adversary = LinfPGDAttack(model, eps=0.3, eps_iter=0.01, nb_iter=40,rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    i = 0; tot = len(dataset)
    for input, label in loader_1:
        input = input.to(device).float()
        label = label.to(device) 
        #label_one_hot = F.one_hot(label, num_classes=10)
        #label_one_hot = label_one_hot.float()

        adv_untargeted_input = adversary.perturb(input, label)
        if test_single_image:
            _show_images(model, input, adv_untargeted_input)
            break
        #adv_untargeted_np_img = flat784_tensor2npimg(adv_untargeted_input)
        #plt.imsave(folder+"/000"+str(label.item())+"/"+str(i)+".png", adv_untargeted_np_img, cmap='gray')
        for j, image in enumerate(adv_untargeted_input): # save every image in perturbed batch of images
            adv_untargeted_np_img = flat784_tensor2npimg(image)
            four_digit_label = "{0:04}".format(label[j].item())
            plt.imsave(folder+"/"+four_digit_label+"/"+str(i+j)+".png", adv_untargeted_np_img)
        i += len(input)
        print("\r", "Progress {} / {}".format(i, tot), end="")

# PARSE INPUTS
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parsing input options...") # ref: https://opensource.com/article/19/7/parse-arguments-python, scroll down to function definition in https://docs.python.org/3/library/argparse.html 
    #parser.add_argument("-b", "--batch_size", type=int, default = 64, help="Batch size used in training.")
    parser.add_argument("-n", "--network", default = "LeNet5", help="Network: LeNet5 or OneLayer")
    parser.add_argument("-a", "--activation", default = "relu", help="Activation function used in network (all lowercase).")
    parser.add_argument("-l", "--learning_rate", type=float, default = 0.01, help="Learning rate (float).")
    #parser.add_argument("-r2", "--second_round",dest='round2', default = False, action='store_true', help="Add this tag (no arg) in second round of training with testing on Adv data.")
    parser.add_argument("-f", "--adv_folder", default="PGD/LeNet5_relu_0.01", help="destination from current directory to folder with adversarial data (for second round, adv data generation!)")
    parser.add_argument("-i", "--show_single_image",dest='show_single_image', default = False, action='store_true', help="Add this tag (no arg) to show an adv image.")
    parser.add_argument("-g", "--gpu", type=int, default = 0, help="GPU number (0-3)")
    options = parser.parse_args(args)
    return options
options = getOptions(sys.argv[1:])
print("network -> {} : {}, activation -> {} : {}, learning_rate -> {} : {}, adv_folder-> {} : {}".format(type(options.network), options.network, type(options.activation), options.activation, type(options.learning_rate), options.learning_rate, type(options.adv_folder), options.adv_folder))

# load device
device = torch.device("cuda:"+str(options.gpu) if torch.cuda.is_available() else "cpu")
print(device)

# load model
if "OneLayer" in options.network:
    model = Net(options.activation)
elif "LeNet5" in options.network:
    model = LeNet5(options.activation)
model = model.to(device)

model_location = "./model_statedicts/mnist_"+options.network+"_"+options.activation+"_"+str(options.learning_rate)+".pth"
print("loading model at "+model_location)
model.load_state_dict(torch.load(model_location))
# load dataset
if "OneLayer" in options.network:
    full_val = datasets.MNIST('./', download=True, train=False)
    valset, _ = get_balanced_mnist784(full_val, 1000, data_normed = 1, batch_size = 16, shuffle = False)
elif "LeNet5" in options.network:
    # transforms = transforms.Compose([transforms.ToTensor()]) # in [2], transforms.Resize((32, 32)) was added at start of transforms
    # valset = datasets.MNIST('./', download=True, train=False, transform=transforms) # full MNIST valset
    full_val = datasets.MNIST('./', download=True, train=False)
    print('==> full_val has {} images'.format(len(full_val)))
    valset, _ = get_balanced_mnist_28x28(full_val, 8000, batch_size = 16, shuffle = False)
# create adv dataset
os.system("mkdir ./PGD")
create_PGD_MNIST_Adv(model, valset, options.adv_folder, test_single_image = options.show_single_image)