import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn')
plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title

def get_val_err(IDENTIFIER):
    val_errors_for_epochs = np.load("./np_data/"+IDENTIFIER+"_4.npy", allow_pickle=True)
    val_error_in_last_epoch = val_errors_for_epochs[-1]
    return val_error_in_last_epoch.cpu().item() # .cpu() moves from cuda to cpu | tensor(a).item() gives a

def get_adv_err(IDENTIFIER):
    adv_errors_for_epochs = np.load("./np_data/"+IDENTIFIER+"_6.npy", allow_pickle=True)
    adv_error_in_last_epoch = adv_errors_for_epochs[-1]
    return adv_error_in_last_epoch.cpu().item() # .cpu() moves from cuda to cpu | tensor(a).item() gives a

def lr_plot(list_of_IDENTIFIERS, etas):
    # retrieve data
    final_val_errors = []; final_adv_errors = []
    for IDENTIFIER in list_of_IDENTIFIERS:
            val_errors_for_epochs = np.load("./np_data/"+IDENTIFIER+"_4.npy", allow_pickle=True)
            final_val_errors.append(val_errors_for_epochs[-1].cpu()) # each element of final_... arrays are tensor objects. So we add a .cpu() at the end.

            adv_errors_for_epochs = np.load("./np_data/"+IDENTIFIER+"_6.npy", allow_pickle=True)
            final_adv_errors.append(adv_errors_for_epochs[-1].cpu()) # each element of final_... arrays are tensor objects. So we add a .cpu() at the end.
    # plot
    def acc_from_err(list_in):
        return 100*(1.0-np.array(list_in))
    plt.figure(figsize=(12,7))
    plt.plot(etas, acc_from_err(final_val_errors), linestyle = '--', marker = 'o', linewidth=4.0, markersize = 10.0)
    plt.plot(etas, acc_from_err(final_adv_errors), linestyle = '-', marker = 'o', linewidth=4.0, markersize = 10.0)
    plt.xlabel('learning rate')
    plt.ylabel('Accuracy (%)')
    if 'tanh' in list_of_IDENTIFIERS[0]:
        loc = 'lower center'
        plt.xlim([-0.15, 2.85])
    else:
        loc = 'upper right'
        plt.xlim([-0.075, 1.5])
    plt.legend(["Clean", "PGD"], loc=loc) # previously, loc='lower right'
    plt.show()

def axis_plot(vec, name):
        plt.plot(vec)

def a2a_plot( list_of_IDENTIFIERS = ["tanh", "relu", "sigmoid"]):
    # retrieve data
    train_losses = []; train_errors = []; val_losses = []; val_errors = []; adv_losses = []; adv_errors = []
    mapper = {0: train_losses, 1: train_errors, 2: val_losses, 3: val_errors, 4: adv_losses, 5: adv_errors}
    for IDENTIFIER in list_of_IDENTIFIERS:
        #print("Parsing "+IDENTIFIER)
        for i in range(6):
            load_np_array = np.load("./np_data/"+IDENTIFIER+"_"+str(i+1)+".npy", allow_pickle=True)
            mapper[i].append(load_np_array)
    # plot
    plt.figure(figsize=(14,20))

    N = len(list_of_IDENTIFIERS) # choose less for plotting select few
    plt.subplot(3,2,1)
    for i in range(N):
        axis_plot(train_errors[i], list_of_IDENTIFIERS[i])
    plt.legend(list_of_IDENTIFIERS, fontsize=18, loc='upper left')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('training error', fontsize=18)

    plt.subplot(3,2,2)
    for i in range(N):
        axis_plot(train_losses[i], list_of_IDENTIFIERS[i])
    plt.legend(list_of_IDENTIFIERS, fontsize=18, loc='upper left')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('training loss', fontsize=18)

    plt.subplot(3,2,3)
    #plt.ylim(0, 0.2)
    for i in range(N):
        axis_plot(val_errors[i], list_of_IDENTIFIERS[i])
    plt.legend(list_of_IDENTIFIERS, fontsize=18, loc='upper left')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('validation error', fontsize=18)

    plt.subplot(3,2,4)
    #plt.ylim(0, 0.2)
    for i in range(N):
        axis_plot(val_losses[i], list_of_IDENTIFIERS[i])
    plt.legend(list_of_IDENTIFIERS, fontsize=18, loc='upper left')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('validation loss', fontsize=18)

    plt.subplot(3,2,5)
    #plt.ylim(0, 0.2)
    for i in range(N):
        axis_plot(adv_errors[i], list_of_IDENTIFIERS[i])
    plt.legend(list_of_IDENTIFIERS, fontsize=18, loc='upper left')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Adv error', fontsize=18)

    plt.subplot(3,2,6)
    #plt.ylim(0, 0.2)
    for i in range(N):
        axis_plot(adv_losses[i], list_of_IDENTIFIERS[i])
    plt.legend(list_of_IDENTIFIERS, fontsize=18, loc='upper left')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Adv loss', fontsize=18)
    os.system('mkdir Plots')
    plt.savefig("./Plots/figure.png")
    
    plt.show()