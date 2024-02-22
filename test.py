# import numpy as np

# # 候选人1，候选人2
# data = np.array([1,2,1,2,1,1,2,1,1,1]) ## [1,2,1] [1,2,2]

# for i in range(len(data)):
#     if i == 0:  #### 第一步单独启动
#         winner  = data[i]
#         vote_different =1
#     if i >0:
#         if data[i] == winner: #### 假如这一票和之前的winner一致，只需要加，不会触发抵消
#             vote_different +=1
#         if data[i] != winner:  #### 这一票和之前的winner不一致，需要考虑在进这一步之前是否抵消玩了，抵消玩了就和i==0时一样的操作， 即第10，11行和第17 18行一样的操作。
#             if vote_different ==0:
#                 winner = data[i]
#                 vote_different =1
#             else: #### 这一票和之前的winner不一致，需要考虑在进这一步之前是否抵消玩了，抵消不完就把vote_differnence 减1
#                 vote_different -=1
# if  vote_different!=0:       
#     print("赢家",winner)
#     print("票数差距",vote_different)
# else:
#     print('平局')


# import numpy as np

# # 创建一个三维数组示例，形状为 (2, 3, 4)
# data = np.array([
#     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
#     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
# ])

# # 计算在第三个维度上的平均值
# mean_along_dim2 = np.mean(data, axis=2)

# print("原始数据：")
# print(data)
# print("在第三个维度上的平均值：")
# print(mean_along_dim2)

# import torch
# import torch.nn as nn

# # 创建一个示例的三维输入张量
# input_tensor = torch.randn(32, 3, 5, 5)  # 假设输入是 32 个 3x5x5 的图像

# # 定义 Flatten 模块
# flatten_layer = nn.Flatten()

# # 使用 Flatten 模块将输入张量展平为一维张量
# output_tensor = flatten_layer(input_tensor)

# print("原始输入形状:", input_tensor.shape)
# print("展平后输出形状:", output_tensor.shape)

# A = [0]*3

# for i in range(3):
#     A[i] = [1] *4
#     for j in range(4):
#         A[i][j] =[0] *2
# print(A)

# import torch
# import torch.nn.functional as F

# # 创建一个示例的二维输出张量
# two_dim_output = torch.tensor([[3, 4], [6, 8]])

# # 对二维输出张量进行标准化，在 dim=1 维度上进行
# normalized_output = F.normalize(two_dim_output.float(), dim=1)

# print("原始输出张量：")
# print(two_dim_output)
# print("在第二个维度上标准化后输出张量：")
# print(normalized_output)
x ="亲亲券"
copied_x = x[:]
print(copied_x) 
copied_x_v2 = x.copy("亲亲券")
print(copied_x_v2)