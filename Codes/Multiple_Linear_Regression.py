import copy, math
from re import X # For mathematical operations # copy is used to create deep copies of objects
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting graphs

X_train = np.array([[2104, 5, 1, 45],[1416, 3, 2, 40],[852, 2, 1, 35]]) # Features: Size (sq ft), Number of bedrooms, Number of bathrooms, Age of the house
y_train = np.array([460, 232, 178])

print(f"X_Shape: {X_train.shape}, X Type: {type(X_train)}")
print(X_train)
print(f"y_Shape: {y_train.shape}, y Type: {type(y_train)}")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predict_single_loop(x, w, b): # x is a 1D array of n features
    """
    Single predict using linear regression
    Args:
      x (ndarray (n,)): Input data of n features
      w (ndarray (n,)): nodel parameters
      b (scalar): Model parameter

      returns:
      p(scalar): predicted value

    """

    n = x.shape[0] # Number of features
    p = 0.0 # Initialize prediction to zero

    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b 
    return p

#Get a row from out training data 
x_vec = X_train[0,:] # First row of X_train
print(f"x_vec shape: {x_vec.shape}, x_vec type: {type(x_vec)}")

#Make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape:{f_wb.shape}, prediction: {f_wb}")

def predict(x, w, b): # DOT product version
    """
    Single predict using linear regression
    Args:
      x (ndarray (n,)): Input data of n features
      w (ndarray (n,)): nodel parameters
      b (scalar): Model parameter

      returns:
      p(scalar): predicted value

    """
    p = np.dot(x, w) + b
    return p

f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape:{f_wb.shape}, prediction: {f_wb}")


def compute_cost(X, y, w, b):
    """
     compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost

    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2 * m)
    return cost

# Compute cost with initial parameters
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"Cost at optimal w: {cost}")

def compute_gradient(X, y, w, b):
    """
       Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.

    """

    m, n = X.shape #m is number of examples, n is number of features
    dj_dw = np.zeros((n,)) # Initialize the gradient for w #np.zeros creates an array of zeros
    dj_db = 0 # Initialize the gradient for b

    for i in range(m): 
        err = (np.dot(X[i], w) + b) - y[i] # Error for the i-th example
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j] # Accumulate the gradient for w
        dj_db = dj_db + err     # Accumulate the gradient for b
    dj_dw = dj_dw / m # Average the gradient for w
    dj_db = dj_db / m  # Average the gradient for b

    return dj_dw, dj_db

# Compute gradient with initial parameters
tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_dw at initial w,b: \n{tmp_dj_dw}')
print(f'dj_db at initial w,b: {tmp_dj_db}')


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
     """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
      # An array to store cost J and w's at each iteration primarily for graphing later
     J_history = []
     w = copy.deepcopy(w_in)  #avoid modifying global w within function
     b = b_in
    
     for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw,dj_db = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[0]:8.2f}   ")
            0
     return w, b, J_history  
    
 # Initialize parameters for gradient descent
initial_w = np.zeros_like(w_init)
initial_b = 0.0

# Set hyperparameters for gradient descent
iterations = 1000
alpha = 5.0e-7

# Run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train,initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_final},{b_final})")
m,_ = X_train.shape
for i in range (m):
         print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")


#Plot the cost versus iteration

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True, figsize = (12, 4))
#constrained_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])

ax1.set_title("Cost vs. Iteration") ; ax2.set_title("Cost vs Iteration(tail)")
ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step'); ax2.set_xlabel('interation step')
plt.show()