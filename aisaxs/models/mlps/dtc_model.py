import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_dct2_ortho_uniform(N, M):
    j = np.arange(N, dtype=np.float64)[:, None]      # 0..N-1
    k = np.arange(M, dtype=np.float64)[None, :]      # 0..M-1
    Phi = np.cos(np.pi / N * (j + 0.5) * k)          # (N, M)
    alpha = np.sqrt(2.0 / N) * np.ones((1, M), dtype=np.float64)
    alpha[0, 0] = 1.0 / np.sqrt(N)                   # DC scaling
    return Phi * alpha    



class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + x)  # residual + nonlinearity


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512, n_blocks=6):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, x):
        h = F.silu(self.input(x))
        h = self.blocks(h)
        y_hat = self.output(h)
        return y_hat
    
class DCTModel(nn.Module):
    def __init__(self, Phi_full, r, hidden=512):
        super().__init__()
        self.register_buffer("Phi_ac_T", Phi_full[:, 1:].t())  # (M_ac, N)
        M_ac = self.Phi_ac_T.shape[0]
        self.net = Net(r, M_ac + 2,hidden, n_blocks=8)

    def forward(self, x):
        z_= self.net(x)
        mu, log_sigma =z_[:, :1], z_[:, 1:2]
            
        sigma = torch.exp(log_sigma)
        d =z_[:,2:]                                         # (B, M_ac)
        core_std = d @ self.Phi_ac_T                                # (B, N)
        return core_std, mu, sigma
    

class CurvePredictor(nn.Module):
    def __init__(self,y_mean, y_std, model):
        super().__init__()

        self.model = model
        self.y_mean = y_mean
        self.y_std = y_std

    def forward(self, parameter):
        core_std, mu, sigma= self.model(parameter)   
        curve = mu + sigma * core_std 
        
        return curve*self.y_std + self.y_mean
    

class DirectResidualModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 512, n_blocks: int = 8):
        super().__init__()
        self.net = Net(input_dim, output_dim, hidden=hidden, n_blocks=n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)