'''
    [1] https://github.com/IBM/CLEVER-Robustness-Score/blob/master/clever.py 
'''
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
np.random.seed(25)
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# fit using weibull_min.fit and run a K-S test
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
    ##print("rescaled_sample = {}".format(rescaled_sample))

    # fit weibull distn: sample follows reverse weibull dist, so -sample follows weibull distribution
    results = pool.map(partial(fit_and_test, rescaled_sample, sample, loc_shift, shape_rescale, scipy.optimize.fmin), c_init)

    for res, c_i in zip(results, c_init):
        c = res[0]
        loc = res[1]
        scale = res[2]
        ks = res[3]
        pVal = res[4]
        print("[DEBUG][L2] c_init = {:5.5g}, fitted c = {:6.2f}, loc = {:7.2f}, scale = {:7.2f}, ks = {:4.2f}, pVal = {:4.2f}, max = {:7.2f}".format(c_i,c,loc,scale,ks,pVal,loc_shift))
        
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
    
    return c_init_best, c_best, loc_best, scale_best, ks_best, pVal_best, fitted_paras
    

# G_max is the input array of max values
# Return the Weibull position parameter
def get_lipschitz_estimate(G_max, norm = "L2", figname = "", use_reg = False, return_fitted_paras = False, shape_reg = 0.01):
    # global plot_res 
    c_init, c, loc, scale, ks, pVal, fitted_paras = get_best_weibull_fit(G_max, use_reg, shape_reg)
    
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

    if return_fitted_paras:
        return {'Lips_est':-loc, 'shape':c, 'loc': loc, 'scale': scale, 'ks': ks, 'pVal': pVal}, fitted_paras
    else:
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
    
    def find_slopes(self):
        print("==> Sampling {} max_slopes to fit Weibull with {} / {} slopes sampled to find each max_slope".format(self.M, self.N, self.count))
        for i in range(self.M):
            random_gradients = random.sample(self.gradients, self.N)
            random_params = random.sample(self.params, self.N)
            slopes = []
            for i in range(1, self.N, 2): # 1, 3, 5, ..., N-1
                slope = np.linalg.norm(random_gradients[i] - random_gradients[i-1]) / np.linalg.norm(random_params[i] - random_params[i-1])
                slopes.append(slope)
            self.max_slopes.append(max(slopes))

        print("All {} max slopes: {}".format(len(self.max_slopes), self.max_slopes))
        print("\n||gradient*|| / ||w*|| = {}".format( np.linalg.norm(self.gradients[-1]) / np.linalg.norm(self.params[-1]) ))

    def fit(self, checkpoint, Ms=[50, 100, 200], Ns=[100, 164]): 
        """ 
        (M, N) values chosen for different models ==>
        (100,150) for resnet-110, (100, 100) for densenet-bc-40-12, (, ) for WRN-28-10-drop
        """
        if self.N == 300:
            Ns = [100,164,300]

        for M in Ms:
            for N in Ns:
                self.M = M
                self.N = N
                self.max_slopes = []
                self.find_slopes()
                #print("\n==> With regularization...")
                #print(get_lipschitz_estimate(np.array(self.max_slopes), use_reg=True))
                print(get_lipschitz_estimate(np.array(self.max_slopes)))

    def fit_with_heatmap(self, option=1): 
        """ 
        For Figure 4 in the paper.
        """
        Lips_matrix = []; pvalues_matrix = []
        if option==1:
            Ms = [25, 55, 105, 155, 200]
        elif option ==2:
            Ms = [60, 100, 140, 180, 200] # [60, 100, 140, 180, 200] (figure_4 copy.ipynb) # [25, 55, 105, 155, 200] (figure_4.ipynb) # [20, 50, 100, 150, 200] (orig, 1/L beat baseline)
        Ns = [80, 100, 120, 150, 164]
        MN_tuples = [(M, N) for M in Ms for N in Ns]
        L_for_MN_tuples = []; pvalue_for_MN_tuples = []
        for M in Ms:
            Lips_row = []; pvalues_row = []
            for N in Ns:
                self.M = M
                self.N = N
                self.max_slopes = []
                self.find_slopes()
                Lip_data, fitted_paras = get_lipschitz_estimate(np.array(self.max_slopes), return_fitted_paras=True)
                print(Lip_data)
                Lips_row.append(Lip_data['Lips_est'])
                pvalues_row.append(Lip_data['pVal'])

                L_for_MN_tuples.append(fitted_paras["scale"])
                pvalue_for_MN_tuples.append(fitted_paras["pVal"])

            Lips_matrix.append(Lips_row)
            pvalues_matrix.append(pvalues_row)

        return Ms, Ns, np.array(Lips_matrix), np.array(pvalues_matrix), MN_tuples, c_init, np.array(L_for_MN_tuples), np.array(pvalue_for_MN_tuples)

    def track_assumption_directly(self, print_info=False):
        # Note: self.N = no of epochs
        for end in [self.N-1]: # [self.N-11, self.N-10, self.N-9, self.N-8, self.N-7, self.N-6, self.N-5, self.N-4, self.N-3, self.N -2, self.N-1]: # range(int((self.N-1)/2), self.N-1):
            #print('When end = {}'.format(end))
            Ks = []
            for k in range(0, end):
                c = np.dot(self.params[k] - self.params[k+1], self.params[k+1] - self.params[end])
                if c>=0:
                    if print_info:
                        print('For interval ({},{}): c={} >= 0'.format(k, end, c))
                    Ks.append(k)
            consecutive_windows_of_Ks = self.consecutive(np.array(Ks))
            str_store = 'Continous windows are: '
            for window in consecutive_windows_of_Ks:
                str_store += '({} --> {}, {}), '.format(window[0], (window[-1] + 1) if (window[-1] == end-1) else window[-1], end)
            if print_info:
                print(str_store)
        return (len(Ks)+1)/(self.N)

    def consecutive(self, data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)