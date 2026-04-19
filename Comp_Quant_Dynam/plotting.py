import numpy as np
import matplotlib.pyplot as plt

#################### Exercise sheet 1 ####################

def plot_func(func, k):
    """
    Plots the function 'func' for a given value of 'k'.
    Useful for creating interactive plots in Jupyter notebooks.
    """

    plt.figure(2)
    plt.clf()
    x = np.linspace(-10,10, num=401)
    plt.plot(x,func(x, k))
    plt.xlabel('x')
    plt.ylabel('func(x, k)')

    return plt.gcf()
