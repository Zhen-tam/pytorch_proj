import numpy as np
import matplotlib.pyplot as plt

x = ["10", "100", "1000", "10000"]

train = [0.83282, 0.83369, 0.86077, 0.93786]

test = [0.83250, 0.83355, 0.85647, 0.92257]

plt.plot(x, train)
plt.plot(x,test)
plt.legend(('Train Accuracy', 'Test Acuracy'), loc='upper left')
plt.title("Number of Training Sequence VS. Predicted Accuracy")
plt.ylabel("Predicted Accuracy")
plt.xlabel("Number of Training Sequence")
plt.show()
