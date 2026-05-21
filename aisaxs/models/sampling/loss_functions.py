import torch


def cdf_loss(sim, exp, eps=1e-8):
    sim_d = sim / (sim.sum() + eps)
    exp_d = exp / (exp.sum() + eps)
    sim_c = torch.cumsum(sim_d, dim=0)
    exp_c = torch.cumsum(exp_d, dim=0)
    return torch.mean((sim_c - exp_c)**2)

def corr_loss(sim, exp, eps=1e-8):
    sim_n = sim - sim.mean()
    exp_n = exp - exp.mean()
    num = torch.dot(sim_n, exp_n)
    den = (sim_n.norm() * exp_n.norm() + eps)
    r = num / den
    return 1.0 - r.clamp(-1, 1)  # in [0, 2]



def corr_loss_batched(sim: torch.Tensor, exp: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    sim: [R, Q]
    exp: [Q] or [R, Q]
    returns: [R]  (1 - Pearson correlation), clamped to [0,2]
    """
    if exp.ndim == 1:
        exp = exp.unsqueeze(0).expand(sim.shape[0], -1)
    elif exp.ndim == 2 and exp.shape[0] == 1:
        exp = exp.expand(sim.shape[0], -1)

    # center per row
    sim_n = sim - sim.mean(dim=1, keepdim=True)
    exp_n = exp - exp.mean(dim=1, keepdim=True)

    # dot product per row
    num = (sim_n * exp_n).sum(dim=1)  # [R]
    den = (sim_n.norm(dim=1) * exp_n.norm(dim=1) + eps)  # [R]

    r = num / den
    return 1.0 - r.clamp(-1.0, 1.0)

def cdf_loss_batched(sim: torch.Tensor, exp: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    sim: [R, Q]
    exp: [Q] or [R, Q]
    returns: [R]  mean squared CDF difference per row
    """
    if exp.ndim == 1:
        exp = exp.unsqueeze(0).expand(sim.shape[0], -1)
    elif exp.ndim == 2 and exp.shape[0] == 1:
        exp = exp.expand(sim.shape[0], -1)

    sim_d = sim / (sim.sum(dim=1, keepdim=True) + eps)
    exp_d = exp / (exp.sum(dim=1, keepdim=True) + eps)

    sim_c = torch.cumsum(sim_d, dim=1)
    exp_c = torch.cumsum(exp_d, dim=1)

    return ((sim_c - exp_c) ** 2).mean(dim=1)  # [R]