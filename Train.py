import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import sklearn
from scipy import integrate
from Model import ScoreNet
import matplotlib.pyplot as plt

def loss_fn(model, x, marginal_prob_std_fn,marginal_prob_mean_fn, eps=1e-5):  ##### For VESDE, VPSDE, Agg_VESDE, loss function will not change
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std_fn: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
#   print('random_tx', x.shape)
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  ##### 均匀选的，不用match 前向和后向
  z = torch.randn_like(x)
#   print('random_z', z.shape)
  std = marginal_prob_std_fn(random_t).unsqueeze(1)
  decay = marginal_prob_mean_fn(random_t).unsqueeze(1)
#   print("std.shape", std.shape)
#   print('random_z_std', (z * std).shape)
  perturbed_x = decay*x + z * std
#   print("perturbed_x.shape", perturbed_x.shape)
  score = model(perturbed_x, random_t)
  # print(score.shape)
#   print("score * std.shape", (score * std).shape)
  # print('sum shape',((score * std + z)**2).shape)
  loss = torch.mean(torch.sum((score * std + z)**2, dim=(1)))  ########## 这个要根据dim改变？
  return loss

@torch.no_grad()
def _dequeue_and_enqueue(keys,queue, queue_ptr,queue_length):	# 出队和入队操作, 每次输入的是一个batch [batch, 3*32*32]
    batch_size = keys.shape[0]
    ptr = int(queue_ptr)	# ptr 是指针(pointer)的缩写
    # assert: 检查条件，不符合就终止程序
    assert queue_length % batch_size == 0  # for simplicity
    # replace the keys at ptr (dequeue and enqueue)
    queue[ptr:ptr + batch_size,:] = keys
    ptr = (ptr + batch_size) % queue_length  # move pointer
    queue_ptr[0] = ptr
    return queue_ptr,queue

def loss_fn_CL(model, x, marginal_prob_std_fn,marginal_prob_mean_fn,balance_alpha, aug_coe,queue, queue_ptr,queue_length, device, eps=1e-5):  ##### For VESDE, VPSDE, Agg_VESDE, loss function will not change
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std_fn: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
#   print('random_tx', x.shape)
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  ##### 均匀选的，不用match 前向和后向
  z = torch.randn_like(x)
  z_aug = torch.randn_like(x)
#   print('random_z', z.shape)
  std = marginal_prob_std_fn(random_t).unsqueeze(1)
  decay = marginal_prob_mean_fn(random_t).unsqueeze(1)
#   print("std.shape", std.shape)
#   print('random_z_std', (z * std).shape)
  perturbed_x = decay*x + z * std  
  perturbed_x_aug = perturbed_x + z_aug*aug_coe  #### 这里就不需要flatten了，本来就不是图像
#   print("perturbed_x.shape", perturbed_x.shape)
  score = model(perturbed_x, random_t)
  score_aug = model(perturbed_x_aug, random_t)
  loss = torch.mean(torch.sum((score * std + z)**2, dim=(1)))  ########## 这个要根据dim改变？
  
  two_dim_output= F.normalize(score.float(),dim=1)
  two_dim_output_aug= F.normalize(score_aug.float(),dim=1)
  two_dim_output = two_dim_output.to(device)
  neg_matrix = torch.matmul(two_dim_output, queue.clone().detach().t())
  n = neg_matrix.shape[0] ########### n 是batch size
  pos_matrix  = (two_dim_output*two_dim_output_aug).sum(dim=1).view(n,1)  #######
  logits =  torch.cat([pos_matrix,neg_matrix],dim=1) ###### (n,queue_length+1)
  t = 0.07
  logits = logits/t
  labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
  loss_nce = F.cross_entropy(logits, labels)
  copy_score = F.normalize(score.clone().detach().float(),dim =1)
  queue_ptr,queue=_dequeue_and_enqueue(copy_score,queue, queue_ptr,queue_length)
#   print(queue)
  return loss, balance_alpha*loss_nce,queue_ptr, queue


def train(loss_fn, n_epochs, score_model,model_path, marginal_prob_std_fn,marginal_prob_mean_fn, START_TIME,data_loader,device,lr):
    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = tqdm.trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x.to(device)
            # print("data shpae", x.shape)    
            loss = loss_fn(score_model, x, marginal_prob_std_fn, marginal_prob_mean_fn, START_TIME)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), model_path)
        
def train_CL(loss_fn, n_epochs, score_model,model_path, marginal_prob_std_fn,marginal_prob_mean_fn,balance_alpha, aug_coe,queue, queue_ptr,queue_length, START_TIME,data_loader,device,lr):
    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = tqdm.trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        
        avg_loss_CL = 0.
        num_items_CL = 0
        for x in data_loader:
            x = x.to(device)
            # print("data shpae", x.shape)    
            loss, loss_nce, queue_ptr, queue = loss_fn(score_model, x, marginal_prob_std_fn, marginal_prob_mean_fn,balance_alpha, aug_coe,queue, queue_ptr,queue_length,device, START_TIME)
            optimizer.zero_grad()
            totoal_loss  = loss+loss_nce
            totoal_loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            avg_loss_CL += loss_nce.item() * x.shape[0]
            num_items_CL += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f},Average NCE Loss: {:5f}'.format(avg_loss / num_items, avg_loss_CL / num_items_CL))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), model_path)