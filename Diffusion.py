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

def beta_fn(t,beta_0,beta_1): ### DDPM
    return beta_0 + t * (beta_1 - beta_0)


def alpha_fn(t,beta_0,beta_1): ### DDPM
    exp_term = beta_0 * t + 0.5 * t * t * (beta_1 - beta_0)
    return torch.where(exp_term <= 1e-3, torch.sqrt(exp_term), torch.sqrt(1. - torch.exp(-exp_term)))  ### Eq 33. DDIM std

# def alpha_fn_agg(t,sigma_agg,sigma_min_agg): ### AGG_VESDE
#     tau  = sigma_agg**2
#     exp_term = 2/tau*(sigma_min_agg*sigma_agg*t+1/3*(sigma_agg*t)**3)
#     return torch.sqrt(tau*(1. - torch.exp(-exp_term))) 

# def alpha_fn_agg(t,sigma_agg,sigma_min_agg): ### AGG_VESDE
#     tau  = sigma_agg**2
#     exp_term = 2/tau*(sigma_min_agg*sigma_agg*t+1/2*(sigma_agg*t)**2)
#     return torch.sqrt(tau*(1. - torch.exp(-exp_term))) 
  
def alpha_fn_agg(t,sigma_agg,sigma_min_agg): ### AGG_VESDE 20231112
    tau  = sigma_agg
    exp_term = 2/tau*(sigma_min_agg*sigma_agg*t+1/2*(sigma_agg*t)**2)
    return torch.where(exp_term <= 1e-3, torch.sqrt(exp_term), torch.sqrt(tau*(1. - torch.exp(-exp_term))) )
    # return torch.sqrt(tau*(1. - torch.exp(-exp_term))) 

def square_alpha_fn(t,beta_0,beta_1): ### DDPM
    exp_term = beta_0 * t + 0.5 * t * t * (beta_1 - beta_0)
    return torch.where(exp_term <= 1e-3, exp_term, 1. - torch.exp(-exp_term))  ### Eq 33. DDIM std**2

# def square_alpha_fn_agg(t,sigma_agg,sigma_min_agg): ### AGG_VESDE
#     tau  = sigma_agg**2
#     exp_term = 2/tau*(sigma_min_agg*sigma_agg*t+1/3*(sigma_agg*t)**3)
#     return tau*(1. - torch.exp(-exp_term)) 

# def square_alpha_fn_agg(t,sigma_agg,sigma_min_agg): ### AGG_VESDE
#     tau  = sigma_agg**2
#     exp_term = 2/tau*(sigma_min_agg*sigma_agg*t+1/2*(sigma_agg*t)**2)
#     return tau*(1. - torch.exp(-exp_term)) 
  
def square_alpha_fn_agg(t,sigma_agg,sigma_min_agg): ### AGG_VESDE 20231112
    tau  = sigma_agg
    exp_term = 2/tau*(sigma_min_agg*sigma_agg*t+1/2*(sigma_agg*t)**2)
    return torch.where(exp_term <= 1e-3, exp_term, tau*(1. - torch.exp(-exp_term)) )
    # return tau*(1. - torch.exp(-exp_term)) 

def sqrt_one_minus_square_alpha_fn(t,beta_0,beta_1):
    return torch.sqrt(1. - square_alpha_fn(t,beta_0,beta_1))  #### VPSDE means 和 variance的关系.

# def sqrt_one_minus_square_alpha_fn_agg(t,sigma_agg,sigma_min_agg):
#     tau  = sigma_agg**2
#     return torch.sqrt((tau - square_alpha_fn_agg(t,sigma_agg,sigma_min_agg))/tau)  #### VPSDE means 和 variance的关系.

def sqrt_one_minus_square_alpha_fn_agg(t,sigma_agg,sigma_min_agg): #### 20231112
    tau  = sigma_agg 
    return torch.sqrt((tau - square_alpha_fn_agg(t,sigma_agg,sigma_min_agg))/tau)  #### AGG_VESDE means 和 variance的关系.

def marginal_prob_std(t, sigma, device, diffusion_type="VESDE",beta_0=0.1,beta_1 = 20, sigma_agg = 10, sigma_min_agg = 0.1):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
#   t = torch.tensor(t, device=device)
  t = t.clone().detach().to(device)
  # print('t!!!!!!!!!!!!!!!!!!!!!!!!', t)
  if diffusion_type=="VESDE":
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

  if diffusion_type=="Drift_VESDE":
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

  if diffusion_type=="VPSDE":
    # print(alpha_fn(t,beta_0,beta_1).shape)
    return alpha_fn(t,beta_0,beta_1)

  if diffusion_type=="Agg_VESDE":
    return alpha_fn_agg(t, sigma_agg, sigma_min_agg) #### T = sigma_agg

def marginal_prob_mean(t, sigma, device, diffusion_type="VESDE",beta_0=0.1,beta_1 = 20, sigma_agg = 10, sigma_min_agg = 0.1):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
#   t = torch.tensor(t, device=device)
  t = t.clone().detach().to(device)
  # print('t!!!!!!!!!!!!!!!!!!!!!!!!', t)
  if diffusion_type=="VESDE":
    # print(torch.ones_like(t).shape)
    return torch.ones_like(t)

  if diffusion_type=="Drift_VESDE":
    # print(torch.ones_like(t).shape)
    return torch.ones_like(t)

  if diffusion_type=="VPSDE":
    # print(sqrt_one_minus_square_alpha_fn(t).reshape((-1, 1)).shape)
    return sqrt_one_minus_square_alpha_fn(t,beta_0,beta_1)

  if diffusion_type=="Agg_VESDE":
    return sqrt_one_minus_square_alpha_fn_agg(t,sigma_agg,sigma_min_agg)

def diffusion_coeff(t, sigma, device, diffusion_type="VESDE",beta_0=0.1,beta_1 = 20, sigma_agg = 10, sigma_min_agg = 0.1):   ## 这是g，不是g**2
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  if diffusion_type=="VESDE":
    return torch.tensor(sigma**t, device=device)

  if diffusion_type=="Drift_VESDE":
    return torch.tensor(sigma**t, device=device)

  if diffusion_type=="VPSDE":
    return torch.tensor(torch.sqrt(beta_fn(t,beta_0,beta_1)), device=device)

#   if diffusion_type=="Agg_VESDE":
#     return torch.tensor(torch.sqrt(2*((t*sigma_agg)**2+sigma_min_agg)), device=device)

  if diffusion_type=="Agg_VESDE":
    return torch.tensor(torch.sqrt(2*((t*sigma_agg)+sigma_min_agg)), device=device)

  # if diffusion_type=="Agg_VESDE":
  #   return torch.tensor(sigma**t*sigma_agg*t, device=device)
  
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3, diffusion_type="VESDE",beta_0=0.1,beta_1 = 20,sigma_agg=10,sigma_min_agg=0.1, x_dim=2):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, x_dim, device=device) * marginal_prob_std(t).unsqueeze(1)
  time_steps = np.linspace(1., eps, num_steps)
#   print(time_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  
#   init_x = torch.randn(batch_size, 1, device=device) * marginal_prob_std(t).unsqueeze(1)
  with torch.no_grad():
    for time_step in time_steps:      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
    #   print("batch_time_step", batch_time_step.shape)
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      if diffusion_type=="VESDE":
        x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        x_mean = x + (g**2)[:] * score_model(x, batch_time_step) * step_size
        x = x_mean + torch.sqrt((g**2)[:] * step_size) * torch.randn_like(x)     
        # The last step does not include any noise
        
      if diffusion_type=="Drift_VESDE":
        x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        x_mean = x + ((g**2)[:] * score_model(x, batch_time_step)+1/625*(g**2)[:]*x) * step_size
        x = x_mean + torch.sqrt((g**2)[:] * step_size) * torch.randn_like(x)     
        # The last step does not include any noise
      if diffusion_type=="VPSDE":
        x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        x_mean = x + (1/2*(g**2)[:]*x+(g**2)[:] * score_model(x, batch_time_step)) * step_size
        x = x_mean + torch.sqrt((g**2)[:] * step_size) * torch.randn_like(x)      
        # The last step does not include any noise
    return x_mean 


def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=500, 
                           device='cuda', 
                           eps=1e-3, diffusion_type="VESDE",beta_0=0.1,beta_1 = 20,sigma_agg=10,sigma_min_agg=0.1, x_dim=2):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, x_dim, device=device) \
    * marginal_prob_std(t).unsqueeze(1)
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in time_steps:      
      if diffusion_type=="VESDE":
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        mean_x = x + (g**2)[:] * score_model(x, batch_time_step) * step_size
        x = mean_x + torch.sqrt(step_size) * g[:] * torch.randn_like(x)      

        
      if diffusion_type=="Drift_VESDE":
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        mean_x = x + ((g**2)[:] * score_model(x, batch_time_step)+1/625*(g**2)[:]*x) * step_size
        x = mean_x + torch.sqrt(step_size) * g[:] * torch.randn_like(x)
        
      if diffusion_type=="Agg_VESDE":
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        # mean_x = x + ((g**2)[:] * score_model(x, batch_time_step)+1/6400*(g**2)[:]*x) * step_size*80*batch_time_step
        mean_x = x + ((g**2)[:] * score_model(x, batch_time_step)+1/sigma_agg*(g**2)[:]*x) * step_size
        x = mean_x + torch.sqrt(step_size) * g[:] * torch.randn_like(x)  
        
      if diffusion_type=="VPSDE":
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step).unsqueeze(1)
        mean_x = x + (1/2*(g**2)[:]*x+(g**2)[:] * score_model(x, batch_time_step)) * step_size
        x = mean_x + torch.sqrt(step_size) * g[:] * torch.randn_like(x)     
  # Do not include any noise in the last sampling step.
  return mean_x

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                eps=1e-3, diffusion_type="VESDE",beta_0=0.1,beta_1 = 20,sigma_agg=10,sigma_min_agg=0.1, x_dim=2):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, x_dim, device=device) \
      * marginal_prob_std(t).unsqueeze(1)
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    print("sample_shape", sample.shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    print('time_steps_shape',time_steps.shape)    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def timesteps_wrapper(sample, time_steps):
    sample = sample.reshape(shape)
    time_steps = time_steps.reshape((sample.shape[0], ))
    # print('time_steps_shape_wrapper',time_steps.shape)    
    return time_steps

#   def x_eval_wrapper(sample):
#     """A wrapper of the score-based model for use by the ODE solver."""
#     sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
#     return sample.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    print('time_steps', time_steps.shape)
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    if diffusion_type=="VESDE":
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
        # return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)-1/(50) * (g**2) *x ### 50 Gau Good
    if diffusion_type=="Drift_VESDE":
        print("shape!!!!!",((g**2) *x).shape)
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)-1/(625) * (g**2) *x ### 50 Gau Good
    # if diffusion_type=="Drift_VESDE":
    #     print("shape!!!!!",((g**2) *x).shape)
    #     return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)-1/(80*80) * (g**2) *x ### 50 Gau Good
    if diffusion_type=="VPSDE":
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)-0.5 * (g**2) *x
    # if diffusion_type=="Agg_VESDE":
    #     return  -(g**2) * score_eval_wrapper(x, time_steps)-1/(sigma_agg**2) * (g**2) *x
    if diffusion_type=="Agg_VESDE":
      return  -(g**2) * score_eval_wrapper(x, time_steps)-1/(sigma_agg) * (g**2) *x
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x

def sample(ckpt_path, sample_method, score_model, 
                            marginal_prob_std_fn,
                            diffusion_coeff_fn, 
                            sample_batch_size, 
                            device,
                            atol,
                            rtol,
                            diffusion_type,beta_0,beta_1,
                            num_steps, 
                            snr,sigma_agg,sigma_min_agg, x_dim=2):
    ckpt = torch.load(ckpt_path, map_location=device)
    score_model.load_state_dict(ckpt)
    
    if sample_method == 'ode_sampler':
        sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples_eval = sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn, 
                        sample_batch_size, 
                        device=device,
                        atol=atol,
                        rtol= rtol,
                        diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg, x_dim=x_dim)
        
    if sample_method == 'pc_sampler':
        sampler = pc_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples_eval = sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn, 
                        sample_batch_size,
                        num_steps=num_steps, 
                        snr=snr,
                        device=device,
                        diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg, x_dim=x_dim)
        
    if sample_method == 'Euler_Maruyama_sampler':
        sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples_eval = sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn, 
                        sample_batch_size, 
                        num_steps=num_steps,
                        device=device,
                        diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1,sigma_agg=sigma_agg,sigma_min_agg=sigma_min_agg, x_dim=x_dim)
    
    return samples_eval


if __name__ == '__main__':
    device = 'cuda'
    diffusion_type = "VPSDE"  ### VESDE, VPSDE, Agg_VESDE
    mode = 'sample' ### train sample
    ########### Parameter for diffusion
    sigma =  25.0#@param {'type':'number'}
    
    beta_0 = 0.1
    beta_1 = 20.
    
    step_number = 500
    
    START_TIME=1e-5
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma,device=device,diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1)
    marginal_prob_mean_fn = functools.partial(marginal_prob_mean, sigma=sigma,device=device,diffusion_type = diffusion_type,beta_0=beta_0,beta_1 = beta_1)
    eps= START_TIME
    random_t = torch.rand(200, device=device) * (1. - eps) + eps  
    std = marginal_prob_mean_fn(random_t)