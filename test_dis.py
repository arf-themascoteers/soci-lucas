import matplotlib.pyplot as plt
import numpy as np
x = 0.991

ar = [((i*0.001) + x) for i in range(100)]
ar = np.array(ar)
new_ar = 100000**ar

plt.plot(new_ar)
plt.show()