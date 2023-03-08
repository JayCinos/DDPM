import matplotlib.pyplot as plt

from Config import *
from DDPM import *

DATA_PATH = r"D:\tsinghua_me\diffusion model\test\DATASET.pt"
dataset = torch.load(DATA_PATH)['dataset']

model = MLPDiffusion(num_steps) # 输出维度是2 输入是x 和 step
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

num_steps = 100 # 可以由beta alpha 分布 均值 标准差 进行估算
SAVE_PATH = r"D:\tsinghua_me\diffusion model\test\save.pt"
checkpoint = torch.load(SAVE_PATH)

model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print('epoch = ' + str(epoch))
print('loss = ' + str(loss))

x_seq = p_sample_loop(model, dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)
fig, axs = plt.subplots(1, 10, figsize=(28, 3))
for i in range(1, 11):
    cur_x = x_seq[i * 10].detach()
    axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1],color='white',edgecolor='gray', s=5);
    axs[i-1].set_axis_off();
    axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
plt.show()
