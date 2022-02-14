#!/usr/bin/env python
import numpy as np

class ACCURACY():
  @staticmethod
  def predict(X,Y_true,model):
    data_size=X.shape[0]
    prob=model.forward(X)[-1]
    Y_pred=[]
    iter=0
    for i in range(data_size):
      Y_pred.append(np.argmax(prob[i]))
    for j in range(data_size):
      if Y_pred[j]==Y_true[j]:
        iter=iter+1
    accuracy=iter/data_size
    accuracy=accuracy*100
    print(f'accuracy is:{accuracy}')
    return  Y_pred