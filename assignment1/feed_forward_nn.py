#!/usr/bin/env python

import numpy as np
from keras.datasets import fashion_mnist
from optimizer import SGD


class FNN(object):
  def __init__(self, input_size, output_size, hidden_layers_size):
    self.input_size = input_size
    self.output_size = output_size
    self.weight, self.bias = None, None
    self.initialize(input_size, hidden_layers_size, output_size)

  def initialize(self, input_size, hidden_layers_size, output_size):
    self.weight, self.bias = [], []
    prev_layer_size = input_size
    hidden_layers_size.append(output_size)
    for curr_layer_size in hidden_layers_size:
      self.weight.append(np.random.normal(0, 1, size=(prev_layer_size, curr_layer_size)))
      self.bias.append(np.zeros(curr_layer_size))
      prev_layer_size = curr_layer_size

  def reset(self):
    num_layers = len(self.weight)
    for l in range(num_layers):
      m, n = self.weight[l].shape
      self.weight[l] = np.random.normal(0, 1, size=(m, n))
      self.bias[l] = np.zeros(n)

  @staticmethod
  def sigmoid(x):
    return 1./(1+np.exp(-x))

  @staticmethod
  def softmax(x):
    """
    x: (batch_size(B), data_size(N))
    """
    exp_prob = np.exp(x)
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
      next_layer = self.sigmoid(np.dot(prev_layer, w) + b)
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
        dh = layer_output[t] / batch_size
        dh[np.arange(batch_size), y] -= 1/batch_size
      else:
        dh = np.dot(dh_fwd, self.weight[t+1].T) * layer_output[t] * (1-layer_output[t])
      prev_layer_output = X if t==0 else layer_output[t-1]
      dw[t] = np.dot(prev_layer_output.T, dh)
      db[t] = np.sum(dh, axis=0)
      dh_fwd = dh
    return dw, db


if __name__ == '__main__':
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  consider = 1000
  X = np.array([x_train[i].flatten() for i in range(consider)])
  Y = y_train[:consider]
  batch_size, epochs = 16, 20
  model = FNN(784, 10, [50, 20])
  sgd = SGD(model, 0.01)
  for ep in range(1, epochs+1):
    ids = np.arange(consider)
    np.random.shuffle(ids)
    start, end = 0, batch_size
    while end > start:
      x, y = X[ids[start:end]], Y[ids[start:end]]
      sgd.optimize(x, y)
      start, end = end, min(consider, end+batch_size)
    err = sgd.error(X, Y)
    print(f'epoch: {ep}, error: {err}')

