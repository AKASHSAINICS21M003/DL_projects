#! /usr/bin/env python

import numpy as np
from keras.datasets import fashion_mnist

import wandb

from feed_forward_nn import FNN
from measure import accuracy
from optimizer import SGD, MomentumGD, NesterovGD, Rmsprop, Adam, Nadam
from activation_func import Sigmoid, Relu, Tanh
from loss_func import CrossEntropy, MeanSquaredError
from sklearn.model_selection import train_test_split


class Runner(object):
  def __init__(self, isFashionMnistData=True):
    if isFashionMnistData:
      self.initialize_fashion_mnist_data()
    else:
      self.initialize_data()

  def initialize_fashion_mnist_data(self):
    (X, self.y), (X_test, self.y_test) = fashion_mnist.load_data()
    self.X = np.array([x.flatten() for x in X]) / 255
    self.X_test = np.array([x.flatten() for x in X_test]) / 255

  def initialize_data(self):
    raise NotImplementedError("Please implement this method if you need other dataset.")

  @staticmethod
  def get_loss_function(key):
    mapper = {
      "cross_entropy": CrossEntropy,
      "mse": MeanSquaredError
    }
    assert key in mapper
    return mapper[key]

  @staticmethod
  def get_activation_function(key):
    mapper = {
      "sigmoid": Sigmoid,
      "relu": Relu,
      "tanh": Tanh
    }
    assert key in mapper
    return mapper[key]

  @staticmethod
  def get_optimizer(key):
    mapper = {
      "sgd": SGD,
      "momentum_gd": MomentumGD,
      "nesterov_gd": NesterovGD,
      "rmsprop": Rmsprop,
      "adam": Adam,
      "nadam": Nadam
    }
    assert key in mapper
    return mapper[key]

  @staticmethod
  def train(X_train, y_train, params, do_val, wandb_log, print_after=500):
    """
    X_train: (batch_size(B), data_size(N))
    y_train: (batch_size(B))
    params: dict(
      batch_size: int,
      epochs: int,
      alpha: float,
      optimizer: One of (SGD, MomentumGD, NesterovGD, Rmsprop, Adam, Nadam),
      hidden_layers_size: list(layer_size),
      act_func: One of (Sigmoid, Relu, Tanh),
      reg: float,
      init: One of (random, xavier),
      loss_func: One of (CrossEntropy, MeanSquaredError)
    )
    """
    if do_val:
      X_train, X_val, y_train, y_val = train_test_split(
          X_train, y_train, test_size=0.1, random_state=10)

    data_size, input_size = X_train.shape
    output_size = 10
    batch_size, epochs = params['batch_size'], params['epochs']
    act_func, loss_func = params['act_func'](), params['loss_func']()
    model = FNN(
      input_size         = input_size,
      output_size        = output_size,
      hidden_layers_size = params['hidden_layers_size'],
      act_func           = act_func,
      reg                = params['reg'],
      init               = params['init'],
      loss_func          = loss_func,
    )
    optimizer = params['optimizer'](model, params['alpha'])
    for ep in range(1, epochs+1):
      ids = np.arange(data_size)
      np.random.shuffle(ids)
      start, end = 0, batch_size
      while end > start:
        x, y = X_train[ids[start:end]], y_train[ids[start:end]]
        optimizer.optimize(x, y)
        start, end = end, min(data_size, end+batch_size)
      # log
      train_loss = loss_func.error(X_train, y_train, model)
      estimate_y_train = Runner.predict(X_train, model)
      train_acc = accuracy(estimate_y_train, y_train)
      val_loss, val_acc = "NotDefined", "NotDefined"
      if do_val:
        val_loss = loss_func.error(X_val, y_val, model)
        estimate_y_val = Runner.predict(X_val, model)
        val_acc = accuracy(estimate_y_val, y_val)
      Runner.logger(train_loss, train_acc, val_loss, val_acc, ep, wandb_log)
    return model

  def logger(train_loss, train_acc, val_loss, val_acc, step, wandb_log):
    # print(f"TrainingLoss: {train_loss}, TrainingAccuracy: {train_acc}")
    # print(f"ValidationLoss: {val_loss}, ValidationAccuracy: {val_acc}")
    if wandb_log:
      wandb.log({
        "step": step,
        "TrainingLoss": train_loss,
        "TrainingAccuracy": train_acc,
        "ValidationLoss": val_loss,
        "ValidationAccuracy": val_acc,  
      })

  @staticmethod
  def predict(X, model):
    """
    X: (batch_size(B), data_size(N))
    return np.array of size: (batch_size(B),)
    """
    prob = model.forward(X)[-1]
    return np.argmax(prob, axis=1)


def run_wandb():
  wandb.init()
  config = wandb.config
  loss_name = "ce" if config.loss_func == "cross_entropy" else "mse"
  wandb.run.name=f"e_{config.epochs}_bs_{config.batch_size}_hl_{config.hidden_layers}_hn_{config.hidden_nodes}_init_{config.weight_init}_ac_{config.act_func}_ls_{loss_name}_opt_{config.optimizer}"
  hidden_layers_size = [config.hidden_nodes] * config.hidden_layers
  runner = Runner()
  params = {
    "batch_size"        : config.batch_size,
    "epochs"            : config.epochs,
    "alpha"             : config.learning_rate,
    "optimizer"         : runner.get_optimizer(config.optimizer),
    "hidden_layers_size": hidden_layers_size,
    "act_func"          : runner.get_activation_function(config.act_func),
    "reg"               : config.reg,
    "init"              : config.weight_init,
    "loss_func"         : runner.get_loss_function(config.loss_func),
  }
  runner.train(runner.X, runner.y, params, do_val=True, wandb_log=True)


def do_hyperparameter_search_using_wandb():
  sweep_config = {
    "name": "Bayesian Sweep",
    "method": "bayes",
    "metric":{
      "name": "ValidationAccuracy",
      "goal": "maximize"
    },
    "parameters":{
      "epochs": {"values": [5, 10]}, 
      "batch_size": {"values": [16, 32, 64]}, 
      "hidden_layers": {"values": [3, 4, 5]}, 
      "hidden_nodes": {"values": [32, 64, 128]},
      "reg": {"values": [0, 0.0005, 0.5]},
      "weight_init": {"values": ['random', 'xavier']} , 
      "act_func": {"values": ["sigmoid", "tanh", "relu"]}, 
      "loss_func": {"values": ["cross_entropy"]}, 
      "learning_rate": {"values": [1e-3, 1e-4]},   
      "optimizer": {"values": ["sgd", "momentum_gd", "nesterov_gd", "rmsprop", "adam", "nadam"]}, 
    }
  }
  sweep_id = wandb.sweep(sweep_config, project = "cs6910")
  wandb.agent(sweep_id, function=run_wandb, count=15)


if __name__ == '__main__':
  do_hyperparameter_search_using_wandb()


