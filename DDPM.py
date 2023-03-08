import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(2,num_units),
            nn.ReLU(),
            nn.Linear(num_units,num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2),]

        )
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps,num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units)
        ])
    def forward(self,x,t):
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx +1](x)

        x = self.linears[-1](x)
        return x

# TODO loss　使用最简单的　loss
def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):# n_steps 用于随机生成t
    '''对任意时刻t进行采样计算loss'''
    batch_size = x_0.shape[0]

    # 随机采样一个时刻t,为了体检训练效率，需确保t不重复
    # weights = torch.ones(n_steps).expand(batch_size,-1)
    # t = torch.multinomial(weights,num_samples=1,replacement=False) # [barch_size, 1]
    t = torch.randint(0,n_steps,size=(batch_size//2,)) # 先生成一半
    t = torch.cat([t,n_steps-1-t],dim=0) # 【batchsize,1】
    t = t.unsqueeze(-1)# batchsieze
    # print(t.shape)

    # x0的系数
    a = alphas_bar_sqrt[t]
    # 生成的随机噪音eps
    e = torch.randn_like(x_0)
    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]
    # 构造模型的输入
    x = x_0* a + e *aml
    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x,t.squeeze(-1))


    # 与真实噪声一起计算误差，求平均值
    return (e-output).square().mean()


# TODO 编写逆扩散采样函数（inference过程）
def p_sample_loop(model ,shape ,n_steps,betas ,one_minus_alphas_bar_sqrt):
    '''从x[T]恢复x[T-1],x[T-2],……，x[0]'''

    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x, i ,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    '''从x[T]采样时刻t的重构值'''
    t = torch.tensor(t)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t]).sqrt() * (x-(coeff * eps_theta)))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)

