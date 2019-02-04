#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homemade PyTorch tutorial, aka (working) skeleton code.
  python run_demo.py --help

@author: Benjamin Cowen, 31 Jan 2019
@contact: ben.cowen@nyu.edu, bencowen.com
"""

#######################################################
# (0) Import modules.
#######################################################
# Scientific Computing
import numpy as np
import torch
import torch.nn as nn
import random as rand # needed for validation shuffle

# DATA-LOADERS
from datasets import load_dataset

# TRAINING
import model_classes as models

# PLOTTING
import matplotlib.pyplot as plt

# ARGUMENTS
import argparse

# MISC
import gc
from utils import ddict

#######################################################
# (0.5) Parse user inputs.
#######################################################
parser = argparse.ArgumentParser(description='tutorial_benCowen')

###### LOGISTICS
# Forget CUDA.
# This means you just put "--use-cpu" instead of "--use-cpu=True"
parser.add_argument('--use-cpu', action='store_true', default=False,
                    help='Disables CUDA training.')
# Control random seed.
parser.add_argument('--seed', default=1, type=int, metavar='S',
                    help='random seed (default: 1)')
# Controls the validation set selection.
parser.add_argument('--data-seed', type=int, default=1, metavar='S',
                    help='random seed FOR DATA (default: 1)')
# Determine save filename.
parser.add_argument('--save-filename', type=str, default='saves/no-name_', metavar='F',
                    help='Path to file where results will be saved.')
# Prints training progress every few batches.
parser.add_argument('--print-frequency', type=int, default=4, metavar='N',
                    help='How many times per epoch to print training progress.')
# Whether to save the whole model
parser.add_argument('--save-model',  action='store_true',
                    help='Saves trained model.')

###### DATA
# Choose dataset.
parser.add_argument('--dataset', type=str, default='mnist',
                    help='"mnist", "fashion-mnist", "cifar10", etc.')
parser.add_argument('--batch-size', default=256,
                    help='Batch size.')
parser.add_argument('--valid-size', type=int, default=-1,
                    help='Size of validation set in samples.')

###### OPTIMIZATION
# Optimization method.
parser.add_argument('--opt-method', type=str, default='Adam',
                    help='Name of optimizer module, e.g. "SGD", "Adam".')
parser.add_argument('--init-learn-rate', type=float, default='0.01', metavar='LR',
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number epochs.')

###### MODEL
# Model Parameters.
parser.add_argument('--modelName', type=str, default='LeNet5',
                    help='Model name (see "model_classes.py").')
parser.add_argument('--hidden-size', type=int, default=20,
                    help='Number hidden units.')

#######################################################
# (1) Get the logistics set up.
#######################################################
# "args" will now have each argument as an attribute.
args = parser.parse_args()


# (1.0) Check cuda.
# Now you can just use the string stored in "device" with
#    exampleTensor.to(device)
# instead of (e.g.) if use_cuda:.... etc.
if (not args.use_cpu) and torch.cuda.is_available():
  device = 'cuda:0'
else:
  devuce = 'cpu'

# (1.1) Set RNG seeds for repeatability.
torch.manual_seed(args.seed)
if device != 'cpu':
    print('\033[93m'+'Using CUDA'+'\033[0m')
    torch.cuda.manual_seed(args.seed)
# Reminder:
# Don't forget Python's built-in RNG if you use it ANYWHERE!
rand.seed(args.data_seed)

# First thing to do is load the data.
# IMPORTANT: don't let randomness in the data
#   mess up your repeatability! We use Python
#   random module to select the validation set.

# If we have a convolutional net, we don't want to vectorize incoming samples.
if args.modelName=='LeNet5':
  vect = False
else:
  vect = True

train_loader, test_loader, len_sample, num_classes = load_dataset(args.dataset,
            batch_size=args.batch_size, vectorize=vect,
            num_workers=1, valid_size=args.valid_size)

# Get some addition information about the data size (mainly for LeNet5)
window_size = train_loader.dataset.train_data[0].shape[0]
# If there are three dimensions, we have (channels x height x width):
if len(train_loader.dataset.train_data[0].shape) == 3:
    num_input_channels = train_loader.dataset.train_data[0].shape[2]
# Else, it's just (heigh x width), no need to worry about color channels.
else:
    num_input_channels = 1

if hasattr(train_loader, 'numSamples'):
  numTrData = train_loader.numSamples
  numTeData = test_loader.numSamples
else:
  numTrData = len(train_loader.dataset)
  numTeData = len(test_loader.dataset)


# Main
if __name__ == "__main__":
    # (1.2) The first thing to do is save the arguments you used!!!
    # You will thank yourself later for this repeatability. 
    print('********************')
    print('Saving shelf to:')
    print(args.save_filename)
    print('********************')
    SH = ddict(args=args.__dict__)
    if args.save_filename:
        SH._save(args.save_filename, date=True)
    # More explanation about this "ddict" thing.
    # It's similar to a "dict" but has built-in saving methods
    #  which is super useful to keep track of arguments/results.
    # Also it uses "attribute" syntax instead of "dictionary" so
    #  it's less cumbersome.

    # Here we give it two attributes which are both also ddicts():
    #   SH.args
    #   SH.perf
    # The idea is that "args" will be populated with the commands/options
    #  used to perform the experiment, while "perf" will contain experimental results.

    # (1.3) So, finally, set up whatever things you want to record in your 
    #       ddict "SH".
    SH.perf = ddict( epoch_train=[], epoch_test=[])
    # Now we can add results like
    #  SH.perf.epoch_train += [ results ]


    ############################################################
    # (2) Load the model and set up loss function and optimizer.
    ############################################################
    # Create model.
    if args.modelName=='LeNet5':
      model = models.LeNet5(num_input_channels=num_input_channels,
                            num_classes = 10, bias=True,
                            window_size=window_size).to(device)
    else:
      # This instantiates a model with the name "args.modelName"
      #   from the module "models". 
      model = getattr(models, args.modelName)(len_sample, args.hidden_size,
                                              num_classes = 10).to(device)

    # For classification we put this in the loss function.
    loss_function = nn.CrossEntropyLoss()

    # Create optimizer.
    # This creates an optimizer with the name
    #    "args.opt_method" from the module "torch.optim".
    optimizer = getattr(torch.optim, args.opt_method)(model.parameters(),
                                                      lr=args.init_learn_rate)

    #######################################################
    # (BONUS): Best practice is to evaluate your
    #           model *before* training too!!
    #######################################################
    SH.perf.epoch_train += [models.test(model, data_loader=train_loader, label="Training")]
    SH.perf.epoch_test += [models.test(model, data_loader=test_loader, label="Test")]

    model.train()
    for epoch in range(1, args.epochs+1):
      print('\nEpoch {} of {}.'.format(epoch, args.epochs,))

      for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # First step: clear gradient from last batch.
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(data)
        loss_value = loss_function(outputs, targets)

        # Backward propagation
        loss_value.backward()
        optimizer.step()


        # Outputs to terminal
        if batch_idx % int(len(train_loader)/args.print_frequency) == 0:
          print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), numTrData,
                  100. * batch_idx / len(train_loader), loss_value.item()))


        # END BATCH LOOP

      # Print performances
      SH.perf.epoch_train += [models.test(model, data_loader=train_loader, label="Training")]
      SH.perf.epoch_test += [models.test(model, data_loader=test_loader, label="Test")]

      # Save data after every epoch
      if args.save_filename:
          SH._save()

    if args.save_model:
      print('Saving model...')
      SH.model = model
      SH._save()















