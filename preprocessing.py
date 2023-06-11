# Modules
import torch as nn
import numpy as np

# Function imported

# Dataset preprocessing

def preprocessing():

 # Define the Dirichlet Problem:    u'(x) = f(x) = 2*x for x ∈ (0,6]
 #                                   u(0) = 1  board condition
 # The solution has the form:
 # u(x) = u(0) + xN(x,p)
 # u'(x) = N(x,p) + x*(dN(x,p)/dx)

 # Define True Solution u(x) (found analitically)
 def u(x):
  return x*x + 1

 # Define of f(x)
 def f(x):
  return 2*x

 # Input training set
 x_train = nn.reshape(nn.from_numpy(np.linspace(0.01, 6, 600)).float(),(600,1))

 # Output training set (target for u'(x))
 y_train = f(x_train)

 # Input test set (thicker than the choice made for the training set)
 x_test = nn.reshape(nn.from_numpy(np.linspace(0.001, 6, 6000)).float(),(6000,1))

 # Output test set (target for u(x))
 y_test = u(x_test)

 return x_train, y_train, x_test, y_test


