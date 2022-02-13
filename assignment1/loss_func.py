#!/usr/bin/env python

import numpy as np


class BaseLossFunc(object):
  @staticmethod
  def error(X, y, model):
    raise NotImplementedError("error() method not implemented")

  @staticmethod
  def grad(layer, y):
    raise NotImplementedError("grad() method not implemented")


class CrossEntropy(BaseLossFunc):
  @staticmethod
  def error(X, y, model):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    batch_size = X.shape[0]
    prob = model.forward(X)[-1]
    err = - np.sum(np.log(prob[np.arange(batch_size), y])) / batch_size
    for w in model.weight:
      err += model.reg * np.sum(w**2)
    return err

  @staticmethod
  def grad(layer, y):
    """
    layer: (batch_size(B), output_size(O))
    y: (batch_size(B))
    """
    batch_size = layer.shape[0]
    dl = layer / batch_size
    dl[np.arange(batch_size), y] -= 1/batch_size
    return dl


class MinSquaredError(BaseLossFunc):
  @staticmethod
  def error(X, y, model):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    prob = model.forward(X)[-1]
    batch_size, output_size = prob.shape
    prob[np.arange(batch_size), y] -= 1
    err = np.sum(prob ** 2) / (2 * batch_size * output_size)
    for w in model.weight:
      err += model.reg * np.sum(w**2)
    return err

  @staticmethod
  def grad(layer, y):
    """
    layer: (batch_size(B), output_size(O))
    y: (batch_size(B))
    """
    batch_size, output_size = layer.shape
    normalize = batch_size * output_size
    dl = np.array(layer)
    dl[np.arange(batch_size), y] -= 1
    ret = dl * layer * (1 - layer) / normalize
    return ret 

