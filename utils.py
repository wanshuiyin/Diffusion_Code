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

def plt_samples(samples,diffusion_type, sample_method,npts=100):
    plt.hist(samples[:,0].cpu().detach().numpy(), bins=npts, density=True, alpha=0.6, edgecolor='black')
    plt.title("MOG,"+diffusion_type+','+sample_method)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
    
def plt_samples_2d(data_name,samples,diffusion_type, sample_method,LOW_x,HIGH_x,LOW_y,HIGH_y,npts=400):
    plt.hist2d(samples[:,0].cpu().detach().numpy(),samples[:,1].cpu().detach().numpy(), bins=npts, density=True,cmap='inferno',range=[[LOW_x, HIGH_x], [LOW_y, HIGH_y]])
    plt.title(data_name+","+diffusion_type+','+sample_method)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    
def plt_dataset_2d(data_name,samples,diffusion_type, sample_method,LOW_x,HIGH_x,LOW_y,HIGH_y,npts=100):
    plt.hist2d(samples[:,0],samples[:,1], bins=npts, density=True,cmap='inferno',range=[[LOW_x, HIGH_x], [LOW_y, HIGH_y]])
    plt.title(data_name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    
def plt_dataset(samples,npts=100):
    plt.hist(samples, bins=npts, density=True, alpha=0.6, edgecolor='black')
    plt.title("Dataset")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
    
def kl_wrapper(data1, data2):
    def kl_divergence(p, q):
        return np.sum(p * np.log(p / q))
    hist1, _ = np.histogram(data1, bins=100, range=(-10, 10))
    hist2, _ = np.histogram(data2, bins=100, range=(-10, 10))

    # 归一化为概率质量函数（PMF）
    pmf1 = hist1 / np.sum(hist1)
    pmf2 = hist2 / np.sum(hist2)
    # print("hist1",hist1)
    # print("hist2",hist2)
    pmf1  = pmf1+0.00001
    pmf2  = pmf2+0.00001
    kl_div = kl_divergence(pmf1, pmf2)
    return kl_div

def kl_wrapper_2d(data1, data2,LOW_x,HIGH_x,LOW_y,HIGH_y):
    pdf_data1, _ = np.histogramdd(data1, bins=100, range=[[LOW_x, HIGH_x], [LOW_y, HIGH_y]], density=True)
    pdf_data2, _ = np.histogramdd(data2, bins=100, range=[[LOW_x, HIGH_x], [LOW_y, HIGH_y]], density=True)

    # 计算概率密度函数的比值
    pdf_ratio = (pdf_data1+1e-5) / (pdf_data2 + 1e-5)  # 添加一个小常数以避免除以零

    # 计算 KL 散度
    kl_divergence = np.sum((pdf_data1) * np.log(pdf_ratio))
    return kl_divergence