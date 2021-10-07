import os 
import argparse 


parser = argparse.ArgumentParser(description='Wrapper for easy evaluation of downloaded models.')
parser.add_argument('--folder', default='train_checkpoints/5000_cifar10_resnet20_0.1', type=str)
args = parser.parse_args()

LR = float(args.folder.rsplit('_')[-1])
checkpoints_slash_B = args.folder.rsplit('_')[-4]
B = int(checkpoints_slash_B.split('/')[-1])
print(LR, B)
if 'cifar10_resnet20' in args.folder:
    cmd = 'python cifar_plus.py -a resnet --lr {} --train-batch {} --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint {}'.format(LR, B, args.folder)
elif 'cifar10_resnet50' in args.folder:
    cmd = 'python cifar_plus.py -a resnet --lr {} --train-batch {} --depth 50 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint {}'.format(LR, B, args.folder)
elif 'cifar10_resnet110' in args.folder:
    cmd = 'python cifar_plus.py -a resnet --lr {} --train-batch {} --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint {}'.format(LR, B, args.folder)
elif 'cifar10_densenet' in args.folder:
    cmd = 'python cifar_plus.py -a densenet --lr {} --train-batch {} --depth 40 --growthRate 12 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint {}'.format(LR, B, args.folder)
elif 'cifar100_resnet50' in args.folder:
    cmd = 'python cifar_plus.py -a resnet --dataset cifar100 --lr {} --train-batch {} --depth 50 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint {}'.format(LR, B, args.folder)
elif 'cifar100_resnet110' in args.folder:
    cmd = 'python cifar_plus.py -a resnet --dataset cifar100 --lr {} --train-batch {} --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint {}'.format(LR, B, args.folder)
elif 'cifar100_densenet' in args.folder:
    cmd = 'python cifar_plus.py -a densenet --dataset cifar100 --lr {} --train-batch {} --depth 40 --growthRate 12 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint {}'.format(LR, B, args.folder)

os.system(cmd)