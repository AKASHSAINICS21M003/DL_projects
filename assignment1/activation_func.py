#! /usr/bin/env python

import numpy as np


class BaseActivationFunc(object):
  @staticmethod
  def apply(x):
    
    raise NotImplementedError("apply() method not implemented")

  @staticmethod
  def grad(layer):
    
    raise NotImplementedError("grad() method not implemented")


class Sigmoid(BaseActivationFunc):
  @staticmethod
  def apply(x):
    # stackoverflow: https://stackoverflow.com/a/57178527
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

  @staticmethod
  def grad(layer):
    return layer * (1 - layer)


class Relu(BaseActivationFunc):
  @staticmethod
  def apply(x):
    return np.maximum(x, 0.0)

  @staticmethod
  def grad(layer):
    return (layer > 0) * 1.0


class Tanh(BaseActivationFunc):
  @staticmethod
  def apply(x):
    return np.tanh(x)

  @staticmethod
  def grad(layer):
    return 1 - layer**2

