import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = [0,1,2,3,4,5,6,7]
y = [0.443,.2013,0.1342,0.1141,0.1074,0.0872,0.0738,0.0671]
z = [0.506,0.2169,0.1566,0.1687,0.2048,0.1687,0.1928,0.2048]

plt.title('Result Analysis')
plt.plot(x, y, color='green', label='training error')
plt.plot(x, z, color='red', label='testing error')
plt.legend()
plt.xlabel('Max-Depth')
plt.ylabel('Error Rate')
plt.show()