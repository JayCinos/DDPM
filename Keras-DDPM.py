import numpy as np

T = 4

alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
x = np.arange(1,5)
bar_alpha = np.cumprod(x)
batch_size = 3

batch_steps = np.random.choice(T, batch_size)

batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]

print(bar_alpha)
print(batch_steps)
print(bar_alpha[batch_steps])
print(batch_bar_alpha)

y = np.array([[[1,2,3],[3,4,5]],[[6,7,8],[9,10,11]]])
print(y)

sun_y = np.sum(y,axis = [0 ,1 ])
