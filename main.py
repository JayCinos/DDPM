import numpy as np
import torch as th
import enum

# x =np.array([[1,0,0],[2,2,2],[2,2,2]])
# x = th.tensor(x)
#
# print(x)
# print(len(x.shape))
# print(*([1] * (3)))
# t = 1
# nonzero_mask = (
#             (x != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
#         )
# print(nonzero_mask)


import numpy as np

arr = np.random.random((2,2,2))
print(arr)
print(arr[...,  0])




