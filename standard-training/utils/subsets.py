import numpy as np
import torchvision as thv
import torch
import cv2

def check_data_balance(X, Y):
  n_classes = len(np.unique(Y))
  label_ct = np.zeros(n_classes)
  for i in range(len(Y)):
    label = Y[i]
    label_ct[label] += 1
  return label_ct

def resample_data(X, Y, n_samples = 30000):
  """
    Creates balanced subset of X, Y
  """
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

def Get_balanced_normed_subset(fullset, n_samples, batch_size = 32, shuffle = True):
    # resample
    X, Y = resample_data(fullset.data, fullset.targets, n_samples) # ~50k is upper limit for balanced train dataset, ~10k for val
    # Above is n_samples x 32 x 32 x 3. We transpose (in numpy) / permute (in torch) dimensions to get n_samples x 3 x 32 x 32
    # Also, normalize pixels to [0.0, 1.0] floats for adv_PGD.py > create_PGD_Adv() > adversary.perturb()
    X = np.transpose(X, (0, 3, 1, 2)) / 255.0

    # create tensor dataset, dataloader
    set = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    #loader = torch.utils.data.DataLoader(set, batch_size= batch_size, shuffle= shuffle, num_workers=3)

    return set