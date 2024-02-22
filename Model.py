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

def get_timestep_embedding(timesteps, embedding_dim=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float().view(-1, 1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        # print("x before",x.shape)
        for i, layer in enumerate(self.layers[:-1]):
            # print("x after",x.shape)
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x
    
class MLP_one_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        layer_widths= output_dim
        layers.append(torch.nn.Linear(prev_width, layer_widths))
        self.layers = torch.nn.ModuleList(layers)
    def forward(self, x):
        # print("x before",x.shape)
        layer = self.layers[:-1]
        x = layer(x)
        x = self.layers[-1](x)
        return x

class ScoreNet(torch.nn.Module):

    def __init__(self, marginal_prob_std, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=1, act_fn=nn.SiLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        print("self.locals",self.locals)

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=act_fn())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # print("len t shape", t.shape)
        # print("len x shape", x.shape)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 
        # print('out shape', out.shape)
        out = out/self.marginal_prob_std(t).unsqueeze(1)
        # print('out shape', out.shape)
        return out
    
    

class ScoreNet_for_linear_distribution(torch.nn.Module):

    def __init__(self, marginal_prob_std, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=1,x_out_dim =8, act_fn=nn.SiLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        print("self.locals",self.locals)

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=act_fn())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())
        self.shared_linear_one_layer = nn.Linear(x_out_dim, x_dim)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # print("len t shape", t.shape)
        # print("len x shape", x.shape)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x  = self.linear_one_layer(x)
        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 
        # print('out shape', out.shape)
        out = out/self.marginal_prob_std(t).unsqueeze(1)
        # print('out shape', out.shape)
        return out
    
class ScoreNet_CL(torch.nn.Module):

    def __init__(self, marginal_prob_std, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=1, act_fn=nn.SiLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        ## print("self.locals",self.locals)

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=act_fn())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=act_fn())
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # print("len t shape", t.shape)
        # print("len x shape", x.shape)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 
        # print('out shape', out.shape)
        out = out/self.marginal_prob_std(t).unsqueeze(1) #### 这个也应该不需要变
        # print('out shape', out.shape)
        return out