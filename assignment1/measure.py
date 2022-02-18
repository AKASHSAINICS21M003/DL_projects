#!/usr/bin/env python

import numpy as np


def accuracy(y_estimate, y_true):
    cnt = 0
    for esti, true in zip(y_estimate, y_true):
      if esti == true:
        cnt = cnt + 1
        
    accuracy = cnt / len(y_true)
    
    return accuracy
