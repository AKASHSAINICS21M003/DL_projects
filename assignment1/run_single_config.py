#!/usr/bin/env python

import sys
import numpy as np
import yaml
from keras.datasets import fashion_mnist, mnist
from feed_forward_nn import FNN
from optimizer import SGD, MomentumGD, NesterovGD, Rmsprop, Adam, Nadam
from activation_func import Sigmoid, Relu, Tanh
from loss_func import CrossEntropy, MeanSquaredError
from measure import accuracy
from sklearn.model_selection import train_test_split


OPTIMIZER_MAP = {
  "sgd": SGD,
  "momentum_gd": MomentumGD,
  "nesterov_gd": NesterovGD,
  "rmsprop": Rmsprop,
  "adam": Adam,
  "nadam": Nadam
}

ACTIVATION_MAP = {
  "sigmoid": Sigmoid,
  "relu": Relu,
  "tanh": Tanh
}

LOSS_MAP = {
  "cross_entropy": CrossEntropy,
  "mse": MeanSquaredError
}


def read_config(filepath):
  with open(filepath, 'r') as fp:
    try:
      params = yaml.safe_load(fp)
    except yaml.YAMLError as err:
      raise
  return params


def get_data(data_set):
  data = fashion_mnist if data_set == "fashion_mnist" else mnist
  (X, y), (X_test, y_test) = data.load_data()
  X = np.array([x.flatten() for x in X]) / 255
  # 10% validation data
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=10)
  X_test = np.array([x.flatten() for x in X_test]) / 255
  return (X_train, y_train, X_val, y_val, X_test, y_test)


def main(filepath, do_test=False):
  params = read_config(filepath)

  X_train, y_train, X_val, y_val, X_test, y_test = get_data(params['data_set'])
  data_size, data_dim = X_train.shape
  class_size = len(set(y_train))

  batch_size, epochs = params['batch_size'], params['epochs']
  act_func, loss_func = ACTIVATION_MAP[params['act_func']](), LOSS_MAP[params['loss_func']]()
  init, reg = params['weight_init'], params['reg']
  hl = [params['hidden_nodes']] * params['hidden_layers']

  model = FNN(data_dim, class_size, hl, act_func=act_func, loss_func=loss_func, reg=reg, init=init)
  opt = OPTIMIZER_MAP[params['optimizer']](model, params['learning_rate'])

  # training 
  for ep in range(1, epochs+1):
    ids = np.arange(data_size)
    np.random.shuffle(ids)
    start, end = 0, batch_size
    while end > start:
      x, y = X_train[ids[start:end]], y_train[ids[start:end]]
      opt.optimize(x, y)
      start, end = end, min(data_size, end+batch_size)
    err = loss_func.error(X_train, y_train, model)
    print(f'epoch: {ep}, error: {err}')

  # Validation
  prob = model.forward(X_val)[-1]
  yesti = np.argmax(prob, axis=1)
  print(f'After training validation accuracy: {accuracy(yesti, y_val)}')  

  if do_test:
    prob_test = model.forward(X_test)[-1]
    yesti_test = np.argmax(prob, axis=1)
    print(f'After training test accuracy: {accuracy(yesti_test, y_test)}') 


if __name__ == '__main__':
  config_file, do_test = sys.argv[1], sys.argv[2]
  do_test = True if do_test.lower() == "true" else False

  main(config_file, do_test)


