import numpy as np
import matplotlib.pyplot as plt


def load_heart_dataset(div):
    # The larger the div the more distributed the samples
    np.random.seed(1)
    m = 400 # number of samples
    N = int(m/div) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector

    for j in range(div):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*np.pi/(div/2),(j+1)*np.pi/(div/2),N) + np.random.randn(N)*0.2
        X[ix] = np.c_[16*np.sin(t)**3, 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)]
        Y[ix] = j % 2

    return X.T, Y.T

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean)#OrRd_r)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Greys)
