import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title

def get_adv_err_for_ep(EPOCH, LR, loc_PGD):
    with open(loc_PGD+'/resnet-20_ep{}_{}/log.txt'.format(EPOCH, LR)) as f:
        for line in f:
            pass
        last_line = line
    return float(last_line.rsplit('\t')[-3]) # the last character is endline

def get_adv_errs(LR, loc_PGD):
    adv_errs = []
    epoch_numbers = list(np.arange(5,164,5)) + [164] 
    for epoch in epoch_numbers:
        adv_errs.append( get_adv_err_for_ep(epoch, LR, loc_PGD) )
    return epoch_numbers, adv_errs

def get_val_errs(LR, loc_checkpoint):
    val_errs = []
    with open(loc_checkpoint+'/resnet-20_{}/log.txt'.format(LR)) as f:
        for i, line in enumerate(f):
            if i>0:
                val_err_value = float(line.rsplit('\t')[-2])
                val_errs.append(val_err_value)
    return val_errs

def make_figure_3(Lvals, loc_checkpoint, loc_PGD, option=1):
    # Lvals = [10.9005, 7.6474, 3.6954, 7.9951, 6.0226]
    lrs = [(round(1./Ls, 4), round(2./Ls, 4)) for Ls in Lvals]
    marks = ['o', 's', '^', 'd', 'v', 'p', 'h', '>', '<']
    clr = ['b', 'g', 'k', 'r', 'p']
    legend_list = []

    fig = plt.figure(figsize=(24,7))
    plt.subplot(1, 2, 1)
    yc = get_val_errs(0.1, loc_checkpoint)
    plt.plot(yc, linestyle = '-', linewidth=4.0, markersize = 10.0)
    legend_list.append( '$\eta^1$ = 0.1' )
    for i, (lr_conv, lr_pers) in enumerate(lrs):
        y1c = get_val_errs(lr_conv, loc_checkpoint)
        plt.plot(y1c, linestyle = '--', linewidth=4.0, markersize = 10.0)
        legend_list.append( '$\eta^1$ = 2/{}'.format(Lvals[i]) )
        y2c = get_val_errs(lr_pers, loc_checkpoint)
        plt.plot(y2c, linestyle = '-', linewidth=4.0, markersize = 10.0)
        legend_list.append( '$\eta^1$ = 1/{}'.format(Lvals[i]) )
    plt.legend(legend_list)
    plt.xlabel('epochs')
    plt.ylabel('Clean Accuracy (%)')

    plt.subplot(1, 2, 2)
    x, y = get_adv_errs(0.1, loc_PGD)
    plt.plot(x, y, linestyle = '-', marker = marks[0], linewidth=4.0, markersize = 10.0)
    legend_list.append( '$\eta^1$ = 0.1' )
    for i, (lr_conv, lr_pers) in enumerate(lrs):
        x1, y1 = get_adv_errs(lr_conv, loc_PGD)
        x2, y2 = get_adv_errs(lr_pers, loc_PGD)
        plt.plot(x1, y1, linestyle = '--', marker = marks[i+1], linewidth=4.0, markersize = 10.0)
        legend_list.append( '$\eta^1$ = 2/{}'.format(Lvals[i]) )
        plt.plot(x2, y2, linestyle = '-', marker = marks[i+1], linewidth=4.0, markersize = 10.0)
        legend_list.append( '$\eta^1$ = 1/{}'.format(Lvals[i]) )
    plt.legend(legend_list, loc='lower right')
    plt.xlabel('epochs')
    plt.ylabel('PGD Accuracy (%)')
    if option==1:
        plt.ylim(bottom=7)
    elif option==2:
        plt.ylim(bottom=12)

    print('Clean: basline {} | 2/L {} | 1/L {}'.format(yc[-1], y1c[-1], y2c[-1]))
    print('PGD: basline {} | 2/L {} | 1/L {}'.format(y[-1], y1[-1], y2[-1]))


