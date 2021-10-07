from robustness.datasets import DATASETS
from robustness.model_utils import make_and_restore_model
from robustness.train_plus import train_model
from robustness.defaults import check_and_fill_args
from robustness.tools import constants, helpers
from robustness import defaults

from cox import utils
import cox

import torch as ch
from argparse import ArgumentParser
import os
import pickle 

# Step 2.1: Setting up command-line args
parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser.add_argument('--only-lipschitz', action='store_true', default=False, help='Only perform calculation of Lipschitz constant') # New:  
# Note that we can add whatever extra arguments we want to the parser here
args = parser.parse_args()

# ------------------------------ # New: ONLY LIPSCHITZ -------------------------------
def read_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        return obj

if args.only_lipschitz:
    print('Only calculating the lipschitz constant using saved Lipschitz object...')
    Lipschitz = read_object(os.path.join(args.out_dir,'lipschitz.pkl'))
    Lipschitz.fit()
    exit(0)
# ------------------------------------------------------------------------------

# Step 2.2: Sanity checks and defaults
assert args.dataset is not None, "Must provide a dataset"
ds_class = DATASETS[args.dataset]

args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)
if args.adv_train or args.adv_eval:
  args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)
args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

# Step 3a: Load up the dataset
data_path = os.path.expandvars(args.data)
dataset = DATASETS[args.dataset](data_path)

# Step 3b: Make the data loaders
train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug))

# Step 3c: Prefetches data to improve performance
train_loader = helpers.DataPrefetcher(train_loader)
val_loader = helpers.DataPrefetcher(val_loader)

model, _ = make_and_restore_model(arch=args.arch, dataset=dataset)

# Step 4: Training the model
# Create the cox store, and save the arguments in a table # Updated code below from https://github.com/microsoft/robust-models-transfer/blob/master/src/main.py
store = cox.store.Store(args.out_dir, args.exp_name)
if 'metadata' not in store.keys:
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)
else:
    print('[Found existing metadata in store. Skipping this part.]')

model = train_model(args, model, (train_loader, val_loader), store=store)