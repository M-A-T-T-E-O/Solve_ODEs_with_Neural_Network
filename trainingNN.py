# Modules
import torch
import numpy as np
import torch.nn as nn

# Function imported
from torch import optim

# Training algorithm for the Neural Network

def trainingNN(MyNN, x_train, y_train, epoch):

 # Define epsilon
 epsilon = np.sqrt(np.finfo(np.float32).eps)

 # The input data
 idata = x_train

 # The output data (target)
 odata = y_train

 # Define the Squared Error ( ||x-y||^2 with x, y vectors,||·|| Euclidean norm)
 loss = torch.nn.MSELoss(reduction = 'sum')

 # Define the optimizer (lr := learning rate)
 optimizer = optim.SGD(MyNN.parameters(), lr=1e-7)

 print('\nStart training:')

 for i in range(epoch):

  # Calculate the output of the Neural Network from the given input dataset
  ynn = MyNN(idata)

  # Calculate the derivative of the NN wrt the unique input
  ynn_derivative = (MyNN(idata + epsilon) - MyNN(idata)) / epsilon

  # Calculate u'(x) = N(x,p) + x*(dN(x,p)/dx)
  u_derivative = ynn + x_train*ynn_derivative

  # Calculate the error between the target and the output
  error = loss(u_derivative, odata)

  # Calculate the gradient for each tensor of weight and bias
  error.backward()  
  
  # Update the parameters
  optimizer.step()
  optimizer.zero_grad()

  # Print the error every 500 iterations
  if (i == 0):
   print("\n","The error (every 500 steps):")
  if np.mod(i, 500) == 0:
   print(error)

 return print('\nTraining has finished.\n')


