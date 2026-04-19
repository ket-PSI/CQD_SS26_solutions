import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(0.5,1.5,1000)
y = 1/(1-x**2)
plt.plot(x,y)
plt.grid()
plt.show()
