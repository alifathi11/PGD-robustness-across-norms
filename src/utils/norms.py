import torch 

def l2_norm(delta: torch.Tensor) -> torch.Tensor: 
    """
    Compute the L2 norm of a perturbation vector
    """
    return torch.linalg.vector_norm(delta.reshape(-1), ord=2)

def linf_norm(delta: torch.Tensor) -> torch.Tensor: 
    """
    Compute the L_inf norm of a perturbation vector
    """
    return torch.linalg.vector_norm(delta.reshape(-1), ord=float("inf"))