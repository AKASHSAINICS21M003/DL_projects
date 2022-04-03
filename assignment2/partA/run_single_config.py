#! /usr/bin/env python3

import sys
import numpy as np
import yaml

from model import CNN
from utils import get_data, Config


def main(config_path, save_path):
  config = Config.from_yaml(config_path)
  train_data, val_data, test_data = get_data(config.data_path, config.input_size, config.augmentation, 42)
  model = CNN(config)
  model.train(train_data, val_data, config.epochs)
  # save path
  model.save(save_path)
  # evaluate test data
  loss, accuracy = model.evaluate(test_data, batch_size=config.batch_size)
  print(f"Test Loss: {loss}, Test Acc: {accuracy}")


if __name__ == '__main__':
  config_file, save_path = sys.argv[1], sys.argv[2]
  main(config_file)


