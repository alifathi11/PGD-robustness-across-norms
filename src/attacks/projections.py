import torch 

def project_linf(
    delta: torch.Tensor, 
    epsilon: float,
) -> torch.Tensor: 
    """
    Project perturbation onto an L_inf ball 
    """
    return delta.clamp(
        min=-epsilon,
        max=epsilon
    )

def project_l2(
    delta: torch.Tensor,
    epsilon: float,
    dim=None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Project perturbation onto an L2 ball 
    """
    if dim is None: 
        norm = torch.linalg.vector_norm(
            delta.reshape(-1), 
            ord=2
        )

        if norm <= epsilon: 
            return delta

        return delta * (epsilon / norm.clamp_min(eps))

    else: 
        norm = torch.linalg.vector_norm(
            delta,
            ord=2,
            dim=dim,
            keepdim=True
        )

        scale = torch.clamp(
            epsilon / norm.clamp_min(eps),
            max=1.0
        )

        return delta * scale 
