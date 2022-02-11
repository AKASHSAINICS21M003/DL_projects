#!/usr/bin/env python

import numpy as np


class SGD(object):
  def __init__(self, model, alpha):
    self.model = model
    self.alpha = alpha

  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    layer_output = model.forward(X)
    dw, db = model.backward(X, y, layer_output)
    num_layers = len(model.weight)
    for l in range(num_layers):
      model.weight[l] -= self.alpha * dw[l]
      model.bias[l] -= self.alpha * db[l]

  def error(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    batch_size = X.shape[0]
    prob = self.model.forward(X)[-1]
    err = - np.sum(np.log(prob[np.arange(batch_size), y])) / batch_size
    return err


