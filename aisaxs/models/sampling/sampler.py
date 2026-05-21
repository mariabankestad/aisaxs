import numpy as np
import torch
from scipy.stats import lognorm
import matplotlib.pyplot as plt

def real_to_log_params(mean, std):
    """Convert real-space mean and std to log-space parameters for a log-normal distribution."""
    var = std ** 2
    sigma_log = np.sqrt(np.log(1 + var / mean**2))
    mu_log = np.log(mean**2 / np.sqrt(var + mean**2))
    return mu_log, sigma_log

def sample_truncated_lognormal(u_tensor, mean, std, a, b):
    """
    Inverse transform sampling from a multivariate truncated log-normal distribution.

    Inputs:
        u_tensor: torch.Tensor [N, D], uniform samples in [0, 1]
        mean, std: torch.Tensor [D], real-space means and stds
        a, b: torch.Tensor [D], truncation bounds

    Returns:
        x: [N, D] tensor of samples
        Z: scalar torch.Tensor, product of normalization constants
    """
    # Convert to log-space
    mu_log, sigma_log = real_to_log_params(mean.numpy(), std.numpy())
    dist = lognorm(s=sigma_log, scale=np.exp(mu_log))

    # Compute truncation normalization constants per dimension
    Fa = dist.cdf(a.numpy())
    Fb = dist.cdf(b.numpy())
    Z = Fb - Fa

    # Vectorized inverse transform sampling
    x_np = dist.ppf(Fa + u_tensor.numpy() * Z)

    return torch.tensor(x_np, dtype=u_tensor.dtype), torch.tensor(Z.prod(), dtype=u_tensor.dtype)



def get_output_curve(grf_parameters, 
                             r_mean,
                             r_std,
                             shell_thickness,
                             min_parameter_values, 
                             max_parameter_values,
                             model,
                             N=1000,
                             curve_mean=0.0,
                             curve_std=1.0,
                             device="cuda"):
    """
    Computes the expected model output under a truncated log-normal prior.

    Inputs:
        parameter_means, parameter_stds: torch.Tensor [D]
        min_parameter_values, max_parameter_values: torch.Tensor [D]
        model: torch.nn.Module
        N: int, number of samples
        curve_mean, curve_std: denormalization constants
        device: computation device

    Returns:
        out_curve: torch.Tensor [M], expected curve
    """
    seed = torch.randint(0, 2**16, (1,)).item()
    sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
    u = sobol.draw(N)

    x, Z = sample_truncated_lognormal(
        u, r_mean, r_std, min_parameter_values[:,3:4], max_parameter_values[:,3:4]
    )
    grf_parameters_ = (grf_parameters.reshape(1,-1)).repeat(N,1)
    shell_thickness = (shell_thickness.reshape(1,-1)).repeat(N,1)
    x = torch.cat([grf_parameters_, x, shell_thickness], dim = 1)
    # Normalize input
    x_norm = (x - min_parameter_values) / (max_parameter_values - min_parameter_values)

    with torch.no_grad():
        out = model(x_norm.to(device)).cpu()
        out = out * curve_std + curve_mean  # denormalize

    out_real = 10 ** out
    out_curve = out_real.mean(dim=0) 
    return out_curve