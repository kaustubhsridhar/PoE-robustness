import os
import sys
import glob
from functools import partial
from multiprocessing import Pool
import scipy
import scipy.io as sio
from scipy.stats import weibull_min
import scipy.optimize
import numpy as np
import argparse
import matplotlib
import random
random.seed(25)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import time


def fit_and_test(rescaled_sample, sample, loc_shift, shape_rescale, optimizer, c_i):
    [c, loc, scale] = weibull_min.fit(-rescaled_sample, c_i, optimizer=optimizer)
    loc = - loc_shift + loc * shape_rescale
    scale *= shape_rescale
    ks, pVal = scipy.stats.kstest(-sample, 'weibull_min', args = (c, loc, scale))
    return c, loc, scale, ks, pVal

c_init = [0.1,1,5,10,20,50,100]


def get_best_weibull_fit(sample, use_reg = False, shape_reg = 0.01):
    
    # initialize dictionary to save the fitting results
    fitted_paras = {"c":[], "loc":[], "scale": [], "ks": [], "pVal": []}
    # reshape the data into a better range 
    # this helps the MLE solver find the solution easier
    loc_shift = np.amax(sample)
    dist_range = np.amax(sample) - np.amin(sample)
    # if dist_range > 2.5:
    shape_rescale = dist_range
    # else:
    #     shape_rescale = 1.0
    print("shape rescale = {}".format(shape_rescale))
    rescaled_sample = np.copy(sample)
    rescaled_sample -= loc_shift
    rescaled_sample /= shape_rescale

    print("loc_shift = {}".format(loc_shift))
    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution
    results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, scipy.optimize.fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print("[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.4f}, ks = {:4.2f}, pVal = {:4.4f}, max = {:7.2f}".format(c_i,c,loc,scale,ks,pVal,loc_shift))
        
        fitted_paras['c'].append(c)
        fitted_paras['loc'].append(loc)
        fitted_paras['scale'].append(scale)
        fitted_paras['ks'].append(ks)
        fitted_paras['pVal'].append(pVal)
    
    
    # get the paras of best pVal among c_init
    max_pVal = np.nanmax(fitted_paras['pVal'])
    if np.isnan(max_pVal) or max_pVal < 0.001:
        print("ill-conditioned samples. Using maximum sample value.")
        # handle the ill conditioned case
        return -1, -1, -max(sample), -1, -1, -1

    max_pVal_idx = fitted_paras['pVal'].index(max_pVal)
    
    c_init_best = c_init[max_pVal_idx]
    c_best = fitted_paras['c'][max_pVal_idx]
    loc_best = fitted_paras['loc'][max_pVal_idx]
    scale_best = fitted_paras['scale'][max_pVal_idx]
    ks_best = fitted_paras['ks'][max_pVal_idx]
    pVal_best = fitted_paras['pVal'][max_pVal_idx]
    
    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best
    

# G_max is the input array of max values
# Return the Weibull position parameter
def get_lipschitz_estimate(G_max, norm = "L2", figname = "", use_reg = False, shape_reg = 0.01):
    # global plot_res 
    c_init, c, loc, scale, ks, pVal = get_best_weibull_fit(G_max, use_reg, shape_reg)
    
    # the norm here is Lipschitz constant norm, not the bound's norm
    if norm == "L1":
        p = "i"; q = "1"
    elif norm == "L2":
        p = "2"; q = "2"
    elif norm == "Li":
        p = "1"; q = "i"
    else:
        print("Lipschitz norm is not in 1, 2, i!")
    
    figname = figname + '_'+ "L"+ p + ".png"
    
    return {'Lips_est':-loc, 'shape':c, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}

# Global variables
pool = Pool(processes = 5)

class Weibull_Fitter(object):
    """
    update()    -   Stores gradients and weights.
        input   :   gradient numpy vector (1D), 
                    params numpy vector (1D)
        no return
    fit()       -   Computes slopes. Computes Lipschitz constant via MLE fit of slopes to reverse weibull distribution.
        input   :   M = number of points to use to fit the reverse weibull distribution,
                    N = *pref. even* number of points to sample (in inner loop) to calculate maximum slope
        returns :   loc parameter of reverse weibull pdf (= Lipschitz constant)
    """
    def __init__(self, M=100, N=100):
        self.reset(M, N)

    def reset(self, M, N):
        self.M = M
        self.N = N
        self.loc = 0
        self.shape = 1
        self.scale = 1
        self.gradients = []
        self.params = []
        self.max_slopes = []
        self.count = 0

    def update(self, gradient_vector, params_vector):
        self.gradients.append(gradient_vector)
        self.params.append(params_vector)
        self.count += 1
    
    def find_slopes(self, M, N):
        print("\n\n==> Sampling {} max_slopes to fit Weibull with {} / {} slopes sampled to find each max_slope".format(M, N, self.count))
        for i in range(M):
            random_gradients = random.sample(self.gradients, N)
            random_params = random.sample(self.params, N)
            slopes = []
            for i in range(1, N, 2): # 1, 3, 5, ..., N-1
                slope = np.linalg.norm(random_gradients[i] - random_gradients[i-1]) / np.linalg.norm(random_params[i] - random_params[i-1])
                slopes.append(slope)
            self.max_slopes.append(max(slopes))

        print("All {} max slopes: {}".format(len(self.max_slopes), self.max_slopes))
        print("||gradient*|| / ||w*|| = {}".format( np.linalg.norm(self.gradients[-1]) / np.linalg.norm(self.params[-1]) ))

    def fit(self): 
        """ 
        """
        Ms = [40,45,50,55,155,160,200]
        largest_M = max(Ms)
        largest_N = self.N # args.epochs in main file
        Ns = [int(largest_N/4), int(largest_N/2), int(largest_N*3/4), largest_N]
        
        t0 = time.time()

        for N in Ns:
            self.max_slopes = []
            self.find_slopes(largest_M, N)
            for M in Ms:
                max_slopes_subset = random.sample(self.max_slopes, M) if M != largest_M else self.max_slopes
                print('[(M, N) = {}]==> Using subset {}/{} slopes, we have .....'.format((M, N), len(max_slopes_subset), len(self.max_slopes)))
                #print("\n==> With regularization...\n", get_lipschitz_estimate(np.array(self.max_slopes), use_reg=True))
                print(get_lipschitz_estimate(np.array(max_slopes_subset)))
        
        print('The time consumed in Lipschitz fitting is {}\n\n'.format(time.time() - t0))
