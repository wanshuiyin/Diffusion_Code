# import numpy as np
# from scipy.stats import multivariate_normal

# def generate


# def generate_gaussian_mixture(dim, num_components, num_samples):
#     # 生成混合系数
#     weights = np.random.uniform(0, 1, num_components)
#     weights /= np.sum(weights)

#     # 生成混合的高斯分布参数
#     means = np.random.uniform(-10, 10, (num_components, dim))
#     covs = np.random.uniform(0.1, 10, (num_components, dim, dim))
#     covs = np.array([np.eye(dim) * v for v in covs])

#     # 生成样本
#     samples = np.zeros((num_samples, dim))
#     for i in range(num_samples):
#         component = np.random.choice(num_components, p=weights)
#         samples[i] = multivariate_normal.rvs(mean=means[component], cov=covs[component])

#     return samples

# # dim = 8
# # num_components = 3
# # num_samples = 1000

# # data = generate_gaussian_mixture(dim, num_components, num_samples)
# # print("Generated data shape:", data.shape)

# def generate_orthogonal_matrix(rows, cols):
#     Q, R = np.linalg.qr(np.random.rand(rows, cols))
#     return Q

# rows = 5
# cols = 3
# orthogonal_matrix = generate_orthogonal_matrix(rows, cols) ## D times d
# print(np.dot(orthogonal_matrix.T,orthogonal_matrix))

import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNetwork, self).__init__()
        # 创建一个线性层，稍后用作编码器和解码器
        self.shared_linear = nn.Linear(input_size, hidden_size)

        # 其他网络层
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 编码器部分
        encoded = self.shared_linear(x)

        # 其他层的操作
        hidden = torch.relu(self.hidden_layer(encoded))

        # 解码器部分，使用相同的线性层
        decoded = self.shared_linear(hidden)

        # 最后的输出层
        output = self.output_layer(decoded)
        return output

# 示例：输入大小10，隐藏层大小20，输出大小5
net = MyNetwork(10, 20, 5)

x= torch.randn(1, 10)
output = net(x)
print(output)