import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch

# TODO 实验数据
s_curve , _  = make_s_curve(10**4 , noise = 0.1)
s_curve = s_curve[:,[0,2] ]/10.0
print("shape of moons :",np.shape(s_curve))
data = s_curve.T
fig,ax = plt.subplots()
ax.scatter(*data ,color='red',edgecolor='white')
ax.axis('off')
plt.show()

dataset = torch.Tensor(s_curve).float()
DATA_PATH = r"D:\tsinghua_me\diffusion model\test\DATASET3.pt"
torch.save({'dataset': dataset},DATA_PATH)
