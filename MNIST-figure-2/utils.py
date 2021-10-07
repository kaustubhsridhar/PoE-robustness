import numpy as np
import torchvision as thv
import torch
import cv2
#np.random.seed(20)
#torch.manual_seed(20)

def check_data_balance(X, Y):
  n_classes = len(np.unique(Y))
  label_ct = np.zeros(n_classes)
  for i in range(len(Y)):
    label = Y[i]
    label_ct[label] += 1
  return label_ct

def resample_data(X, Y, n_samples = 30000):
  new_X = []
  new_Y = []
  n_classes = len(np.unique(Y))
  label_ct = np.zeros(n_classes)
  samples_ct = 0
  for i in range(len(Y)):
    label = Y[i]
    if label_ct[label] <= n_samples/n_classes and samples_ct < n_samples:
      new_X.append(X[i])
      new_Y.append(Y[i])
      label_ct[label] += 1
      samples_ct += 1
  final_X = np.asarray(new_X)
  final_Y = np.asarray(new_Y)

  return final_X, final_Y

def get_balanced_mnist784(fullset, n_samples, data_normed, batch_size = 32, shuffle = True):
    # resample
    X, Y = resample_data(fullset.data.numpy(), fullset.targets.numpy(), n_samples) # ~50k / 60k is upper limit for balanced train dataset, ~8k/10k for val
    #print("resampled data class balance: ", check_data_balance(X, Y))

    # flatten
    X = X.reshape((X.shape[0], -1))/1.0 # currently [0, 255]
    if data_normed == 1:
      X = X/255.0
    elif data_normed == -1:
      X = 2*X/255.0 - 1.0

    # create tensor dataset, dataloader
    set = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = torch.utils.data.DataLoader(set, batch_size= batch_size, shuffle= shuffle, num_workers=3) 

    return set, loader

def get_balanced_mnist_28x28(fullset, n_samples, batch_size = 32, shuffle = True):
    # resample
    X, Y = resample_data(fullset.data.numpy(), fullset.targets.numpy(), n_samples) # ~50k / 60k is upper limit for balanced train dataset, ~8k/10k for val
    # above is n_smaples x 28 x 28 and n_smaples
    # we add dimesnion in between for LeNet5 model
    X = X[:, None, :, :]

    print("l: ", X.shape, Y.shape)
    print("l2: ",torch.tensor(X).shape, torch.tensor(Y).shape)
    # create tensor dataset, dataloader
    set = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = torch.utils.data.DataLoader(set, batch_size= batch_size, shuffle= shuffle, num_workers=3) 

    return set, loader