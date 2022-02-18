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


class MomentumGD(object):
  def __init__(self, model, alpha, gamma=0.9):
    self.model = model
    self.alpha = alpha
    self.gamma = gamma
    self.initialize()

  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    num_layers = len(self.model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))

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
      self.v_w[l] = self.gamma*self.v_w[l] + self.alpha*dw[l]
      self.v_b[l] = self.gamma*self.v_b[l] + self.alpha*db[l]
      model.weight[l] -= self.v_w[l]
      model.bias[l] -= self.v_b[l]


class NesterovGD(object):
  def __init__(self, model, alpha, gamma=0.9):
    self.model = model
    self.alpha = alpha
    self.gamma = gamma
    self.initialize()

  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    num_layers = len(self.model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))

  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    num_layers = len(model.weight)
    for l in range(num_layers):
      self.v_w[l] = self.gamma * self.v_w[l]
      self.v_b[l] = self.gamma * self.v_b[l]
      model.weight[l] -= self.v_w[l]
      model.bias[l] -= self.v_b[l]
    layer_output = model.forward(X)
    dw, db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l] += self.alpha*dw[l]
      self.v_b[l] += self.alpha*db[l]
      model.weight[l] -= self.alpha*dw[l]
      model.bias[l] -= self.alpha*db[l]


class Adagrad(object):
  def __init__(self, model, alpha, epsilon=0.000001):
    self.model = model
    self.alpha = alpha
    self.epsilon = epsilon
    self.initialize()

  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    num_layers = len(self.model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))

  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    num_layers = len(model.weight)
    layer_output = model.forward(X)
    dw,db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l] = self.v_w[l] + np.power(dw[l],2)
      self.v_b[l] = self.v_b[l] + np.power(db[l],2)
      model.weight[l] -= (self.alpha/np.sqrt(self.v_w[l]+self.epsilon))*dw[l]
      model.bias[l] -= (self.alpha/np.sqrt(self.v_b[l]+self.epsilon))*db[l]


class Rmsprop(object):
  def __init__(self, model, alpha, beta=0.9, epsilon=0.000001):
    self.model = model
    self.alpha = alpha
    self.beta=beta
    self.epsilon=epsilon
    self.initialize()

  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    num_layers = len(self.model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))

  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    num_layers = len(model.weight)
    layer_output = model.forward(X)
    dw,db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l]= self.beta*self.v_w[l] + (1-self.beta)*np.power(dw[l],2)
      self.v_b[l]=self.beta*self.v_b[l] + (1-self.beta)*np.power(db[l],2)
      model.weight[l]-=(self.alpha/np.sqrt(self.v_w[l]+self.epsilon))*dw[l]
      model.bias[l]-=(self.alpha/np.sqrt(self.v_b[l]+self.epsilon))*db[l]


class Adam(object):
  def __init__(self, model, alpha, beta1=0.9, beta2=0.99, epsilon=0.0000001):
    self.model = model
    self.alpha = alpha
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.found = 0
    self.initialize()
  
  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    self.m_w=[]
    self.m_b=[]
    num_layers = len(self.model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))
      self.m_w.append(np.zeros((m,n)))
      self.m_b.append(np.zeros(n))
    
  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    num_layers = len(model.weight)
    layer_output = model.forward(X)
    dw,db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l] = self.beta2*self.v_w[l]+(1-self.beta2)*np.power(dw[l],2)
      self.v_b[l] = self.beta2*self.v_b[l]+(1-self.beta2)*np.power(db[l],2)
      self.m_w[l] = self.beta1*self.m_w[l]+(1-self.beta1)*dw[l]
      self.m_b[l] = self.beta1*self.m_b[l]+(1-self.beta1)*db[l]
      m_w_hat = (1/(1-(self.beta1**(self.found+1))))*self.m_w[l]
      m_b_hat = (1/(1-(self.beta1**(self.found+1))))*self.m_b[l]
      v_w_hat = (1/(1-(self.beta2**(self.found+1))))*self.v_w[l]
      v_b_hat = (1/(1-(self.beta2**(self.found+1))))*self.v_b[l]
      model.weight[l] -= (self.alpha/np.sqrt(v_w_hat+self.epsilon))*m_w_hat
      model.bias[l] -= (self.alpha/np.sqrt(v_b_hat+self.epsilon))*m_b_hat
    self.found=self.found+1


class Nadam(object):
  def __init__(self, model, alpha, beta1=0.9, beta2=0.99, epsilon=0.0000001):
    self.model = model
    self.alpha = alpha
    self.beta1=beta1
    self.beta2=beta2
    self.epsilon=epsilon
    self.found=0
    self.initialize()
  
  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    self.m_w=[]
    self.m_b=[]
    self.m_w_hat=[]
    self.m_b_hat=[]
    self.v_w_hat=[]
    self.v_b_hat=[]
    self.mw_cap=[]
    self.mb_cap=[]
    num_layers = len(self.model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))
      self.m_w.append(np.zeros((m,n)))
      self.m_b.append(np.zeros(n))
    
  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    num_layers = len(model.weight)
    layer_output = model.forward(X)
    dw,db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l]=self.beta2*self.v_w[l]+(1-self.beta2)*np.power(dw[l],2)
      self.v_b[l]=self.beta2*self.v_b[l]+(1-self.beta2)*np.power(db[l],2)
      self.m_w[l]=self.beta1*self.m_w[l]+(1-self.beta1)*dw[l]
      self.m_b[l]=self.beta1*self.m_b[l]+(1-self.beta1)*db[l]
      m_w_hat=(1/(1-(self.beta1**(self.found+1))))*self.m_w[l]
      m_b_hat=(1/(1-(self.beta1**(self.found+1))))*self.m_b[l]
      v_w_hat=(1/(1-(self.beta2**(self.found+1))))*self.v_w[l]
      v_b_hat=(1/(1-(self.beta2**(self.found+1))))*self.v_b[l]
      mw_cap=self.beta1*m_w_hat+(1-self.beta1)*dw[l]
      mb_cap=self.beta1*m_b_hat+(1-self.beta1)*db[l]
      model.weight[l]-=(self.alpha/np.sqrt(v_w_hat+self.epsilon))*mw_cap
      model.bias[l]-=(self.alpha/np.sqrt(v_b_hat+self.epsilon))*mb_cap
    self.found=self.found+1

