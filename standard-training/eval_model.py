import os 
import argparse 


parser = argparse.ArgumentParser(description='Wrapper for easy evaluation of downloaded models.')
parser.add_argument('--file', default='checkpoints/5000_cifar10_resnet20_0.1.pth.tar', type=str)
parser.add_argument('--gpu-id', default=0, type=int)
parser.add_argument('--trained-from-scratch', action='store_true', default=False, help='set if you want to evaluate a model trained from scratch (instead of one downloaded)')
args = parser.parse_args()

if not args.trained_from_scratch:
    LR_dotpth_dottar = args.file.rsplit('_')[-1]
    LR = float(LR_dotpth_dottar[:-8]) # 0.123.pth.tar becomes 0.123
    checkpoints_slash_B = args.file.rsplit('_')[-4]
    B = int(checkpoints_slash_B.split('/')[-1])
else: # --file is like train_checkpoints/5000_cifar10_resnet20_0.1/checkpoint.pth.tar
    LR_slash_checkpoint_dotpth_dottar = args.file.rsplit('_')[-1]
    LR = float(LR_slash_checkpoint_dotpth_dottar[:-19]) # 0.123/checkpoint.pth.tar becomes 0.123
    checkpoints_slash_B = args.file.rsplit('_')[-4]
    B = int(checkpoints_slash_B.split('/')[-1])

print(LR)
if 'cifar10_resnet20' in args.file:
    cmd = 'python cifar_adv.py -a resnet --depth 20 --resume {} --adv_folder PGD/{}_cifar10_resnet20_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)
elif 'cifar10_resnet50' in args.file:
    cmd = 'python cifar_adv.py -a resnet --depth 50 --resume {} --adv_folder PGD/{}_cifar10_resnet50_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)
elif 'cifar10_resnet110' in args.file:
    cmd = 'python cifar_adv.py -a resnet --depth 110 --resume {} --adv_folder PGD/{}_cifar10_resnet110_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)
elif 'cifar10_densenet' in args.file:
    cmd = 'python cifar_adv.py -a densenet --depth 40 --growthRate 12 --resume {} --adv_folder PGD/{}_cifar10_densenet_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)
elif 'cifar100_resnet50' in args.file:
    cmd = 'python cifar_adv.py -a resnet --dataset cifar100 --depth 50 --resume {} --adv_folder PGD/{}_cifar100_resnet50_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)
elif 'cifar100_resnet110' in args.file:
    cmd = 'python cifar_adv.py -a resnet --dataset cifar100 --depth 110 --resume {} --adv_folder PGD/{}_cifar100_resnet110_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)
elif 'cifar100_densenet' in args.file:
    cmd = 'python cifar_adv.py -a densenet --dataset cifar100 --depth 40 --growthRate 12 --resume {} --adv_folder PGD/{}_cifar100_densenet_{} --gpu-id {}'.format(args.file, B, LR, args.gpu_id)


if not args.trained_from_scratch:
    cmd = cmd + ' --downloaded_model'
os.system(cmd)