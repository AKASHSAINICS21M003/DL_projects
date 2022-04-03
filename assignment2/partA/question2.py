#! /usr/bin/env python3

import numpy as np
from utils import get_data
from os.path import join, abspath, realpath
from os import listdir
from model import CNN
import wandb
from wandb.keras import WandbCallback


DATA_PATH = abspath(realpath(__file__) + "/../data")

# This input size does not refer to the size of the actual image 
# but rather the desired input size that we want to feed to the network
INPUT_SIZE = (256, 256, 3)

OUTPUT_SIZE = 10

PROJECT = "CS6910_ASSIGNMENT_2"

ENTITY = "cs21m003_cs21d406"

NUM_EXPERIMENTS = 20


class Runner(object):
  def __init__(self, config, data_path, seed=0):
    self.config = config
    self.seed = seed
    self.model = CNN(config)
    self.train_data, self.val_data, self.test_data = self.get_data(data_path)

  def get_data(self, data_path):
    return get_data(data_path, self.config.input_size, self.config.augmentation, self.seed)

  def run(self, epochs, callbacks=None):
    self.model.train(self.train_data, self.val_data, epochs, callbacks)


def run_wandb():
  wandb.init()
  config = wandb.config
  wandb.run.name=f"e_{config.epochs}_bs_{config.batch_size}_" +\
                 f"kernel_size_{config.kernel_size}_filters_{config.num_filters}_" +\
                 f"ac_{config.activation_func}_rate_{config.lr}_aug_{config.augmentation}_" + \
                 f"BN_{config.batch_norm}_drp_{config.drop_out}_pad_{config.padding}_" +
                 f"dense_{config.dense_size}_metric_{config.filter_org}_type_{config.type}"
  runner = Runner(config, DATA_PATH)
  runner.run(config.epochs, callbacks=[WandbCallback()])


def do_hyperparameter_search_using_wandb():
  sweep_config = {
    "name": "Search Hyperparameters",
    "method": "random", # we also explored searching using bayes and grid_search too
    "metric": {
      "name": "ValidationAccuracy",
      "goal": "maximize"
    },
    "parameters": {
      "type": {"values": ['random']},
      "epochs": {"values": [5, 10]},
      "batch_size": {"values": [64]},
      "kernel_size": {"values": [(3, 3), (4,4), (5,5)]},
      "num_filters": {"values": [32, 64, 128]},
      "activation_func": {"values": ['elu', 'relu', 'selu']},
      "lr": {"values": [1e-3, 1e-4]},
      "augmentation": {"values": [True, False]},
      "batch_norm": {"values": [True, False]},
      "drop_out": {"values": [0.2, 0.3, 0.4, 0.5]},
      "padding": {"values": ['same', 'valid']},
      "dense_size": {"values": [32, 64, 128]},
      "filter_org": {"values": [1, 2, 0.5]},  # filter organisation: same num of filters, double num of filters, half num of filters
      "input_size": {"values": [INPUT_SIZE]},
      "output_size": {"values": [OUTPUT_SIZE]}
    }
  }
  sweep_id = wandb.sweep(sweep_config, project=PROJECT,entity=ENTITY)
  wandb.agent(sweep_id, function=run_wandb, count=NUM_EXPERIMENTS)


if __name__ == '__main__':
  do_hyperparameter_search_using_wandb()


