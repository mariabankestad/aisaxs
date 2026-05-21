import torch

def frequency_loss_old(pred_curve, target_curve, mode="complex", focus=None, eps=1e-6):
    """
    Computes a loss in the Fourier domain between predicted and target curves.
    
    Args:
        pred_curve (Tensor): [B, L] predicted curves
        target_curve (Tensor): [B, L] ground-truth curves
        mode (str): "magnitude", "log_magnitude", or "complex"
        focus (int or None): if set, only compares the last `focus` frequency bins (high-frequency)
        eps (float): small number to avoid log(0)

    Returns:
        Tensor: scalar loss
    """
    pred_fft = torch.fft.rfft(pred_curve, dim=-1)
    target_fft = torch.fft.rfft(target_curve, dim=-1)

    if mode == "magnitude":
        pred_feat = torch.abs(pred_fft)
        target_feat = torch.abs(target_fft)
    elif mode == "log_magnitude":
        pred_feat = torch.log1p(torch.abs(pred_fft) + eps)
        target_feat = torch.log1p(torch.abs(target_fft) + eps)
    elif mode == "complex":
        pred_feat = pred_fft
        target_feat = target_fft
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optionally focus only on high-frequency bins
    if focus is not None:
        pred_feat = pred_feat[:, -focus:]
        target_feat = target_feat[:, -focus:]

    return torch.mean(torch.abs(pred_feat - target_feat))



def frequency_loss(pred_curve, target_curve, mode="complex", focus=None, eps=1e-6):
    """
    Computes a loss in the Fourier domain between predicted and target curves.
    
    Args:
        pred_curve (Tensor): [B, L] predicted curves
        target_curve (Tensor): [B, L] ground-truth curves
        mode (str): "magnitude", "log_magnitude", or "complex"
        focus (int or None): if set, only compares the last `focus` frequency bins (high-frequency)
        eps (float): small number to avoid log(0)

    Returns:
        Tensor: scalar loss
    """
    pred_fft = torch.fft.rfft(pred_curve, dim=-1)
    target_fft = torch.fft.rfft(target_curve, dim=-1)

    if mode == "magnitude":
        pred_feat = torch.abs(pred_fft)
        target_feat = torch.abs(target_fft)
    elif mode == "log_magnitude":
        pred_feat = torch.log1p(torch.abs(pred_fft) + eps)
        target_feat = torch.log1p(torch.abs(target_fft) + eps)
    elif mode == "complex":
        pred_feat = pred_fft
        target_feat = target_fft
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optionally focus only on high-frequency bins
    if focus is not None:
        pred_feat = pred_feat[:, -focus:]
        target_feat = target_feat[:, -focus:]

    return torch.mean(torch.abs(pred_feat - target_feat))
