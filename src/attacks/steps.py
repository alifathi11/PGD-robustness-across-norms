import torch 

def linf_step(
    gradient: torch.Tensor,
    step_size: float, 
) -> torch.Tensor: 
    """
    Compute a step under an L_inf constraint
    """
    return step_size * gradient.sign()

def l2_normalize(
    tensor: torch.Tensor,
    dim=None,
    eps: float = 1e-12
) -> torch.Tensor: 
    """
    Normalize a tensor with respect to its L2 norm 
    """
    if dim is None: 
        norm = torch.linalg.vector_norm(
            tensor.reshape(-1),
            ord=2
        )

    else: 
        norm = torch.linalg.vector_norm(
            tensor, 
            ord=2,
            dim=dim,
            keepdim=True
        )

    return tensor / norm.clamp_min(eps)

def l2_step(
    gradient: torch.Tensor,
    step_size: float,
    dim=None,
) -> torch.Tensor: 
    """
    Compute a step under L2 constraint 
    """

    direction = l2_normalize(
        gradient, 
        dim=dim
    )

    return step_size * direction