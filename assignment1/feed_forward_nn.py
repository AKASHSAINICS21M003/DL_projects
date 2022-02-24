#!/usr/bin/env python

import numpy as np
from keras.datasets import fashion_mnist
from optimizer import SGD, MomentumGD, NesterovGD, Rmsprop, Adam, Nadam
from activation_func import Sigmoid, Relu, Tanh
from loss_func import CrossEntropy, MeanSquaredError
from measure import accuracy


np.random.seed(2)


class FNN(object):
  def __init__(self, input_size, output_size, hidden_layers_size, act_func, loss_func, reg=0, init='random'):
    self.input_size = input_size
    self.output_size = output_size
    self.weight, self.bias = None, None
    self.act_func = act_func
    self.loss_func = loss_func
    self.reg = reg
    self.initialize(input_size, hidden_layers_size, output_size, init.lower())

  def initialize(self, input_size, hidden_layers_size, output_size, type):
    self.weight, self.bias = [], []
    prev_layer_size = input_size
    hidden_layers_size.append(output_size)
    for curr_layer_size in hidden_layers_size:
      std = np.sqrt(prev_layer_size * curr_layer_size) if type == 'xavier' else 1
      self.weight.append(np.random.randn(prev_layer_size, curr_layer_size)/std)
      self.bias.append(np.zeros(curr_layer_size))
      prev_layer_size = curr_layer_size

  @staticmethod
  def softmax(x):
    """
    x: (batch_size(B), data_size(N))
    """
    max_x = np.max(x, axis=1, keepdims=True)
    exp_prob = np.exp(x - max_x)
    prob = exp_prob / np.sum(exp_prob, axis=1, keepdims=True)
    return prob

  def forward(self, X):
    """
    X: (batch_size(B), data_size(N))
    """
    layer_output = []
    prev_layer = X
    num_hidden_layers = last_layer = len(self.weight) - 1
    for t in range(num_hidden_layers):
      w, b = self.weight[t], self.bias[t]
      next_layer = self.act_func.apply(np.dot(prev_layer, w) + b)
      layer_output.append(next_layer)
      prev_layer = next_layer
    w, b = self.weight[last_layer], self.bias[last_layer]
    prob = self.softmax(np.dot(prev_layer, w) + b)
    layer_output.append(prob)
    return layer_output

  def backward(self, X, y, layer_output):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    batch_size, _ = X.shape
    num_hidden_layers = last_layer = len(layer_output)-1
    dw, db = [None]*(num_hidden_layers+1), [None]*(num_hidden_layers+1)
    for t in range(num_hidden_layers, -1, -1):
      if t == last_layer:
        dh = self.loss_func.grad(layer_output[t], y)
      else:
        dh = np.dot(dh_fwd, self.weight[t+1].T) * self.act_func.grad(layer_output[t])
      prev_layer_output = X if t==0 else layer_output[t-1]
      dw[t] = np.dot(prev_layer_output.T, dh) + self.reg * self.weight[t]
      db[t] = np.sum(dh, axis=0)
      dh_fwd = dh
    return dw, db


if __name__ == '__main__':
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  consider = len(x_train)
  X = np.array([x_train[i].flatten() for i in range(consider)]) / 255
  X_test = np.array([x.flatten() for x in x_test]) / 255

  Y = y_train[:consider]
  batch_size, epochs = 16, 5
  act_func, loss_func = Tanh(), CrossEntropy()
  init, reg = "random", 0.0005
  hl = [128] * 3
  model = FNN(784, 10, hl, act_func=act_func, loss_func=loss_func, reg=reg, init=init)
  opt = Adam(model, 0.0001)

  prob = model.forward(X_test)[-1]
  yesti = np.argmax(prob, axis=1)
  print(f"Before training: {accuracy(yesti, y_test)}")

  for ep in range(1, epochs+1):
    ids = np.arange(consider)
    np.random.shuffle(ids)
    start, end = 0, batch_size
    while end > start:
      x, y = X[ids[start:end]], Y[ids[start:end]]
      opt.optimize(x, y)
      start, end = end, min(consider, end+batch_size)
    err = loss_func.error(X, Y, model)
    print(f'epoch: {ep}, error: {err}')

  prob = model.forward(X_test)[-1]
  yesti = np.argmax(prob, axis=1)
  print(f'After training: {accuracy(yesti, y_test)}')
  
