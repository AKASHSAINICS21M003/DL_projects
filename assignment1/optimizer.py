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

class MGD(object):
  def __init__(self, model, alpha,gamma=0.9):
    self.model = model
    self.alpha=alpha
    self.gamma=gamma
    self.initialize()
  def initialize(self):
    self.prev_w=[]
    self.prev_b=[]
    self.v_w=[]
    self.v_b=[]
    num_layers = len(model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.prev_w.append(np.zeros((m,n)))
      self.prev_b.append(np.zeros(n))
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))

  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    gamma=0.9
    model = self.model
    layer_output = model.forward(X)
    dw, db = model.backward(X, y, layer_output)
    num_layers = len(model.weight)
    layers=num_layers
    v_w=[]
    v_b=[]
    for l in range(num_layers):
      self.v_w[l]=self.gamma*self.prev_w[l]+self.alpha*dw[l]
      self.v_b[l]=self.gamma*self.prev_b[l]+self.alpha*db[l]
      model.weight[l]-=self.v_w[l]
      model.bias[l]-=self.v_b[l]
      self.prev_w[l]=self.v_w[l]
      self.prev_b[l]=self.v_b[l]
class NESTGD(object):
  def __init__(self, model, alpha,gamma=0.9):
    self.model = model
    self.alpha = alpha
    self.gamma=gamma
    self.initialize()
  def initialize(self):
    self.prev_w=[]
    self.prev_b=[]
    num_layers = len(model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.prev_w.append(np.zeros((m,n)))
      self.prev_b.append(np.zeros(n))
  def optimize(self, X, y):
    """
    X: (batch_size(B), data_size(N))
    y: (batch_size(B))
    """
    model = self.model
    v_w=[]
    v_b=[]
    num_layers = len(model.weight)
    layers=num_layers
    for j in range(num_layers):
      v_w.append(self.gamma*self.prev_w[j])
      v_b.append(self.gamma*self.prev_b[j])
    w=model.weight
    b=model.bias
    model.bias=[]
    model.weight=[]
    for k in range(num_layers):
      model.weight.append(w[k]-v_w[k])
      model.bias.append(b[k]-v_b[k])
    layer_output = model.forward(X)
    dw, db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      v_w[l]=gamma*self.prev_w[l]+self.alpha*dw[l]
      v_b[l]=gamma*self.prev_b[l]+self.alpha*db[l]
      model.weight[l]-=v_w[l]
      model.bias[l]-=v_b[l]
      self.prev_w[l]=v_w[l]
      self.prev_b[l]=v_b[l]
class ADAGRAD(object):
  def __init__(self, model, alpha,epsilon=0.000001):
    self.model = model
    self.alpha = alpha
    self.epsilon=epsilon
    self.initialize()
  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    num_layers = len(model.weight)
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
    layers=num_layers
    layer_output = model.forward(X)
    dw,db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l]= self.v_w[l]+np.power(dw[l],2)
      self.v_b[l]=self.v_b[l]+np.power(db[l],2)
      model.weight[l]-=(self.alpha/np.sqrt(self.v_w[l]+self.epsilon))*dw[l]
      model.bias[l]-=(self.alpha/np.sqrt(self.v_b[l]+self.epsilon))*db[l]
class RMSPROP(object):
  def __init__(self, model, alpha,beta=0.9,epsilon=0.000001):
    self.model = model
    self.alpha = alpha
    self.beta=beta
    self.epsilon=epsilon
    self.initialize()

  def initialize(self):
    self.v_w=[]
    self.v_b=[]
    num_layers = len(model.weight)
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
    layers=num_layers
    layer_output = model.forward(X)
    dw,db = model.backward(X, y, layer_output)
    for l in range(num_layers):
      self.v_w[l]= self.beta*self.v_w[l]+(1-self.beta)*np.power(dw[l],2)
      self.v_b[l]=self.beta*self.v_b[l]+(1-self.beta)*np.power(db[l],2)
      model.weight[l]-=(self.alpha/np.sqrt(self.v_w[l]+self.epsilon))*dw[l]
      model.bias[l]-=(self.alpha/np.sqrt(self.v_b[l]+self.epsilon))*db[l]
class ADAM(object):
  def __init__(self, model, alpha,beta1=0.9,beta2=0.99,epsilon=0.0000001):
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
    num_layers = len(model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))
      self.m_w.append(np.zeros((m,n)))
      self.m_b.append(np.zeros(n))
      self.m_w_hat.append(np.zeros((m,n)))
      self.m_b_hat.append(np.zeros(n))
      self.v_w_hat.append(np.zeros((m,n)))
      self.v_b_hat.append(np.zeros(n))
    
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
      self.m_w_hat[l]=(1/(1-(self.beta1**(self.found+1))))*self.m_w[l]
      self.m_b_hat[l]=(1/(1-(self.beta1**(self.found+1))))*self.m_b[l]
      self.v_w_hat[l]=(1/(1-(self.beta2**(self.found+1))))*self.v_w[l]
      self.v_b_hat[l]=(1/(1-(self.beta2**(self.found+1))))*self.v_b[l]
      model.weight[l]-=(self.alpha/np.sqrt(self.v_w_hat[l]+self.epsilon))*self.m_w_hat[l]
      model.bias[l]-=(self.alpha/np.sqrt(self.v_b_hat[l]+self.epsilon))*self.m_b_hat[l]
    self.found=self.found+1
class NADAM(object):
  def __init__(self, model, alpha,beta1=0.9,beta2=0.99,epsilon=0.0000001):
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
    num_layers = len(model.weight)
    for i in range(num_layers):
      m, n = self.model.weight[i].shape
      self.v_w.append(np.zeros((m,n)))
      self.v_b.append(np.zeros(n))
      self.m_w.append(np.zeros((m,n)))
      self.m_b.append(np.zeros(n))
      self.m_w_hat.append(np.zeros((m,n)))
      self.m_b_hat.append(np.zeros(n))
      self.v_w_hat.append(np.zeros((m,n)))
      self.v_b_hat.append(np.zeros(n))
      self.mw_cap.append(np.zeros((m,n)))
      self.mb_cap.append(np.zeros(n))
    
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
      self.m_w_hat[l]=(1/(1-(self.beta1**(self.found+1))))*self.m_w[l]
      self.m_b_hat[l]=(1/(1-(self.beta1**(self.found+1))))*self.m_b[l]
      self.v_w_hat[l]=(1/(1-(self.beta2**(self.found+1))))*self.v_w[l]
      self.v_b_hat[l]=(1/(1-(self.beta2**(self.found+1))))*self.v_b[l]
      self.mw_cap[l]=self.beta1*self.m_w_hat[l]+(1-self.beta1)*dw[l]
      self.mb_cap[l]=self.beta1*self.m_b_hat[l]+(1-self.beta1)*db[l]
      model.weight[l]-=(self.alpha/np.sqrt(self.v_w_hat[l]+self.epsilon))*self.mw_cap[l]
      model.bias[l]-=(self.alpha/np.sqrt(self.v_b_hat[l]+self.epsilon))*self.mb_cap[l]
    self.found=self.found+1
   
  
