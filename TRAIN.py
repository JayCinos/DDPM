import matplotlib.pyplot as plt

from DDPM_MODEL import *
from config import *


# TODO 演示原始数据分布加噪100步后的效果
# num_shows = 20
# fig , axs = plt.subplots(2,10,figsize=(28,3))
# plt.rc('text',color='blue')
# # 共有10000个点，每个点包含两个坐标
# # 生成100步以内每隔5步加噪声后的图像
# for i in range(num_shows):
#     j = i // 10
#     k = i % 10
#     t = i*num_steps//num_shows # t=i*5
#     q_i = q_x(dataset ,torch.tensor( [t] )) # 使用刚才定义的扩散函数，生成t时刻的采样数据  x_0为dataset
#     axs[j,k].scatter(q_i[:,0],q_i[:,1],color='red',edgecolor='white')
#
#     axs[j,k].set_axis_off()
#     axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')
# plt.show()

# TODO 编写拟合逆扩散过程 高斯分布 的模型
# \varepsilon_\theta(x_0,t)

DATA_PATH = r"D:\tsinghua_me\diffusion model\test\DATASET.pt"
dataset = torch.load(DATA_PATH)['dataset']

print('Training model ……')
'''
'''
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1000,shuffle = False)
num_epoch = 6000
epoch = 6000
plt.rc('text',color='blue')

model = MLPDiffusion(num_steps) # 输出维度是2 输入是x 和 step
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

# for checkpoint
# SAVE_PATH = r"D:\tsinghua_me\diffusion model\test\save.pt"
# checkpoint = torch.load(SAVE_PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# epoch = checkpoint['epoch'] + num_epoch
# loss = checkpoint['loss']

for t in range(num_epoch):
    for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.) #
        optimizer.step()

    # print loss
    if (t% 100 == 0):
        print(loss)
        x_seq = p_sample_loop(model,dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)


SAVE_PATH = r"D:\tsinghua_me\diffusion model\test\save2.pt"
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, SAVE_PATH)
