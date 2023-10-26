import numpy as np
import matplotlib.pyplot as plt

from clearml import Task
task = Task.init(project_name="SF_M2.2_02_1_model", task_name='test2.2_1')

xs = np.linspace(0, 10, 100)
ys = np.sin(xs)
            
plt.scatter(xs, ys)
plt.show()