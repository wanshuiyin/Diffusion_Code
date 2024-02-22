#### ICML 2024
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
import sklearn.datasets
from scipy import integrate
from Model import ScoreNet
import matplotlib.pyplot as plt
from Diffusion import marginal_prob_std,marginal_prob_mean, diffusion_coeff, pc_sampler, Euler_Maruyama_sampler, ode_sampler, sample
from Train import loss_fn, train, loss_fn_CL, train_CL
from utils import kl_wrapper, plt_dataset, plt_samples, plt_samples_2d, kl_wrapper_2d, plt_dataset_2d

def inf_train_gen(data='mog', dim=1, batch_size=5000): ### dataset
    if data == 'mog':
        assert batch_size % 10 == 0
        # mus = torch.from_numpy(np.array([-2., -6., 4.])).to(device).reshape((-1, 1)).float() / 7.
        mus = torch.from_numpy(np.array([-16, -8., 6.])).to(device).reshape((-1, 1)).float()/2
        # stds = torch.from_numpy(np.array([0.1, 0.1, 1.])).to(device).reshape((-1, 1)).float() / 7. ** 2 
        stds = torch.from_numpy(np.array([0.1, 0.1, 1.])).to(device).reshape((-1, 1)).float()
        probs = torch.from_numpy(np.array([0.3, 0.3, 0.4])).to(device).reshape((-1, 1)).float()  #### Parameter for GMM
        dim = int(dim)
        x = torch.cat([torch.randn(int(batch_size * probs[i]), dim).to(device) * torch.sqrt(stds[i]) + mus[i] for i in range(len(mus))], axis=0).reshape((-1, dim))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        return x[indices].float()
    elif data == 'point':
        assert dim == 1
        p = [0.2, 0.2, 0.2, 0.2, 0.2]
        points = [-6., -3, 0., 3, 6.]
        length = 0.001
        def uniform(shape, middle, length):
            return torch.rand(shape).to(device) * length + middle - length / 2.
        x = torch.cat([uniform((int(batch_size * p[i]), dim), points[i], length) for i in range(len(points))], axis=0).reshape((-1, dim))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        return x[indices].float()
    elif data == 'checkerboard':
        assert dim == 2
        ## x1 = np.random.rand(batch_size) * 4 - 2
        x1 = np.random.rand(batch_size) * 8 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return torch.from_numpy(np.concatenate([x1[:, None], x2[:, None]], 1) * 2).float()
    elif data == "swissroll":
        assert dim == 2
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 10.
        data = torch.from_numpy(data).float()
        r = 4.5
        data1 = data.clone() + torch.tensor([-r, -r])
        data2 = data.clone() + torch.tensor([-r, r])
        data3 = data.clone() + torch.tensor([r, -r])
        data4 = data.clone() + torch.tensor([r, r])
        data = torch.cat([data, data1, data2, data3, data4], axis=0)
        return data



    
if __name__ == '__main__':
    device = 'cuda'
    diffusion_type = "VESDE"  ### VESDE, VPSDE, Agg_VESDE, Drift_VESDE
    mode = 'sample' ### train sample
    ########### Parameter for diffusion
    # sigma =  10.0#@param {'type':'number'}
    sigma =  25.0#@param {'type':'number'}
    model_path = 'D:\\2023-spring\\Diffusion_model\\score matching\\Synthetic_experiments\\Synthetic_Gaussian\\Model\\Test_VESDE_MOG_sigma80_new_1114.pth'
    
    ckpt_path = 'D:\\2023-spring\\Diffusion_model\\score matching\\Synthetic_experiments\\Synthetic_Gaussian\\Model\\Test_VESDE_swissroll_sigma25_new_1114.pth'
    
    beta_0 = 0.1
    beta_1 = 20.
    
    sigma_agg = 625
    sigma_min_agg =  0.0001
   
    step_number = 500
    
    START_TIME=1e-5
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma,device=device,diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg)
    marginal_prob_mean_fn = functools.partial(marginal_prob_mean, sigma=sigma,device=device,diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma,device=device,diffusion_type = diffusion_type, beta_0=beta_0,beta_1 = beta_1,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg)
    
    ########### Parameter for dataset
    sample_num = 50000
    # data_name = 'mog'
    # x_dim = 1

    data_name = 'swissroll'
    LOW_x = -7
    HIGH_x = 7
    LOW_y = -7
    HIGH_y = 7
    x_dim = 2


    # data_name = 'checkerboard'
    # LOW_x = -4
    # HIGH_x = 12
    # LOW_y = -4
    # HIGH_y = 4
    # x_dim = 2
    
    dataset = inf_train_gen(data_name, dim=x_dim, batch_size=sample_num).to(device)
    print(dataset.shape)
    mean_dataset = torch.mean(dataset)
    var_dataset = torch.var(dataset)
    print('dataset mean', mean_dataset)
    print('dataset variance', var_dataset)
    
    ########### Parameter for training
    n_epochs =   200#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  200 #@param {'type':'integer'}
    ## learning rate
    lr=1e-4 #@param {'type':'number'}
    
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn,encoder_layers=[128, 128], pos_dim=64, decoder_layers=[512, 512], x_dim=x_dim, act_fn=nn.SiLU).to(device))
    score_model = score_model.to(device)
    
    optimizer = Adam(score_model.parameters(), lr=lr)
    


    ########### Parameter for sample
    sample_batch_size = 50000 #@param {'type':'integer'}
    sample_method = 'ode_sampler'  #### ode_sampler, pc_sampler, Euler_Maruyama_sampler
    num_steps =  500#@param {'type':'integer'}  ## The number of sampling steps.
 
    signal_to_noise_ratio = 0.16 #@param {'type':'number'} ### PC sample
    
    error_tolerance = 1e-5 #@param {'type': 'number'} ## The error tolerance for the black-box ODE solver
    sample_epoch  =1
    sample_result = np.zeros(sample_epoch)

    

    # if mode == 'train':
    #     train(loss_fn, n_epochs, score_model,model_path,marginal_prob_std_fn,marginal_prob_mean_fn,  START_TIME= START_TIME,data_loader=data_loader,device=device,lr=lr)
        
    # if mode == 'train_CL':
    #     balance_alpha= 0.2
    #     aug_coe = 0.01
    #     queue_length  = 5000
    #     queue  = torch.randn(queue_length, 2).to(device)
    #     queue_ptr = torch.zeros(1, dtype=torch.long)
    #     train_CL(loss_fn_CL, n_epochs, score_model,model_path,marginal_prob_std_fn,marginal_prob_mean_fn,balance_alpha, aug_coe,queue, queue_ptr,queue_length, START_TIME= START_TIME,data_loader=data_loader,device=device,lr=lr)
    # if mode == 'sample':
    #     for i in range(sample_epoch):
    #         # ckpt_path = 'D:\\2023-spring\\Diffusion_model\\score matching\\Synthetic_experiments\\Synthetic_Gaussian\\Model\\Test_VPSDE.pth'
    #         samples_eval= sample(ckpt_path=ckpt_path, sample_method=sample_method, score_model=score_model, 
    #                             marginal_prob_std_fn=marginal_prob_std_fn,
    #                             diffusion_coeff_fn=diffusion_coeff_fn, 
    #                             sample_batch_size=sample_batch_size, 
    #                             device=device,
    #                             atol=error_tolerance,
    #                             rtol= error_tolerance,
    #                             diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1,
    #                             num_steps=num_steps, 
    #                             snr=signal_to_noise_ratio,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg, x_dim=x_dim)
    #         # plt_samples(samples_eval,diffusion_type,sample_method,npts=100)
    #         true_dataset_eval = inf_train_gen(data_name, dim=x_dim, batch_size=sample_batch_size).cpu().detach().numpy()
    #         samples_eval_kl  =samples_eval.cpu().detach().numpy()
    #         # print(true_dataset_eval.shape)
    #         # print(diffusion_type,sample_method,kl_wrapper(true_dataset_eval,samples_eval_kl))
    #         print(diffusion_type,sample_method,kl_wrapper_2d(true_dataset_eval,samples_eval_kl,LOW_x,HIGH_x,LOW_y,HIGH_y))
    #         plt_samples_2d(data_name,samples_eval,diffusion_type,sample_method,LOW_x,HIGH_x,LOW_y,HIGH_y,npts=100)
    #         # plt_dataset_2d(data_name,true_dataset_eval,diffusion_type,sample_method,LOW_x,HIGH_x,LOW_y,HIGH_y,npts=100)
    #         # sample_result[i]  = kl_wrapper_2d(true_dataset_eval,samples_eval_kl,LOW_x,HIGH_x,LOW_y,HIGH_y)
    #         sample_result[i]  = kl_wrapper(true_dataset_eval,samples_eval_kl)
    # print("mean",np.mean(sample_result))

    

    
    