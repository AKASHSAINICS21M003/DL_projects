#! /usr/bin/env python3

import sys
import numpy as np
import yaml
from model import CNN
from utils import get_data

from config import Config


def main(filepath):
  config = Config.from_yaml(filepath)
  train_data, val_data, test_data = get_data(config.data_path, config.target_size, config.augmentation, 42)
  model = CNN(config)
  model.train(train_data, val_data, config.epochs)
  # save path
  model.save(config.save_path)
  # evaluate test data
  loss, accuracy = model.evaluate(test_data, batch_size=config.batch_size)
  print(f"Test Loss: {loss}, Test Acc: {accuracy}")


if __name__ == '__main__':
  config_file = sys.argv[1]
  main(config_file)


