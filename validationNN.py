import matplotlib.pyplot as plt

# Validation algorithm for the Neural Network

def validationNN(MyNN, x_test, y_test):

 u = MyNN(x_test)*x_test + 1

 # Plot both approximated u(x) and the u'(x) = exp(x) = u(x)
 plt.plot(x_test,y_test, color='red',label='u(x) [ideal]')
 plt.plot(x_test,u.detach().numpy(), color='green',label='u(x) [approximated]')
 plt.legend()
 plt.show()

 # Plot the error ( u(x) - u_NN(x) )
 plt.plot(x_test, y_test - u.detach().numpy(), color='blue', label='error')
 plt.legend()
 plt.show()

 # Show both the target and the output from the trained NN
 print("\nThe output of the NN (predictions) is:\n", u, "\n\n",
       'The target output (from measurements) is:',
       "\n", y_test, "\n")


