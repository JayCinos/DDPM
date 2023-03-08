import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch

# TODO 确定超参数的值
num_steps = 100 # 可以由beta alpha 分布 均值 标准差 进行估算

# 学习的超参数 动态的在（0，1）之间逐渐增大
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)* (0.5e-2 - 1e-5) + 1e-5

# 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod( alphas ,dim=0 ) # 累积连乘  https://pytorch.org/docs/stable/generated/torch.cumprod.html
alphas_prod_p = torch.cat([torch.tensor([1]).float() ,alphas_prod[:-1]],0) # p means previous
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1-alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1-alphas_prod)

assert  alphas_prod.shape == alphas_prod.shape == alphas_prod_p.shape \
        == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
        == one_minus_alphas_bar_sqrt.shape
print("all the same shape:",betas.shape)  #


def q_x(x_0 ,t):
    noise = torch.randn_like(x_0) # noise 是从正太分布中生成的随机噪声
    alphas_t = alphas_bar_sqrt[t] ## 均值 \sqrt{\bar \alpha_t}
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t] ## 标准差  \sqrt{ 1 - \bar \alpha_t}
    # alphas_t = extract(alphas_bar_sqrt , t, x_0) # 得到sqrt(alphas_bar[t]) ,x_0的作用是传入shape
    # alphas_l_m_t = extract(one_minus_alphas_bar_sqrt , t, x_0) # 得到sqrt(1-alphas_bart[t])
    return (alphas_t * x_0 + alphas_l_m_t * noise)


