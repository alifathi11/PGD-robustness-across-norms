import torch 
import torch.nn.functional as F 

from src.attacks.steps import (
    linf_step,
    l2_step,
)
from src.attacks.projections import (
    project_linf,
    project_l2,
)

def _random_l2_delta(
    x: torch.Tensor, 
    epsilon: float,
) -> torch.Tensor:
    """
    Sample random perturbations approximately uniformly
    from the L2 ball for each sample in a batch
    """

    noise = torch.randn_like(x)

    norm = torch.linalg.vector_norm(
        noise.flatten(1),
        ord=2,
        dim=1,
        keepdim=True
    ).view(-1, 1, 1, 1)

    direction = noise / norm.clamp_min(1e-12)

    dimension = x[0].numel()

    radius = torch.rand(
        x.size(0),
        1,
        1,
        1,
        device=x.device,
        dtype=x.dtype
    )

    radius = epsilon * radius.pow(1.0 / dimension)

    return direction * radius 

def pgd_linf(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps:int, 
    random_start: bool = True
) -> torch.Tensor:
    """
    Untargeted PGD attack under an L-infinity constraint.
    """
    x = images.detach()
    y = labels.detach()

    if random_start: 
        delta = torch.empty_like(x).uniform_(
            -epsilon,
            epsilon
        )

        x_adv = torch.clamp(
            x + delta, 
            0.0,
            1.0
        )

    else:
        x_adv = x.clone()

    was_training = model.training 
    model.eval()

    try: 
        for _ in range(steps): 

            x_adv = x_adv.detach()
            x_adv.requires_grad_(True)

            logits = model(x_adv)

            loss = F.cross_entropy(
                logits,
                y
            )

            gradient = torch.autograd.grad(
                loss, 
                x_adv,
                only_inputs=True
            )[0]

            with torch.no_grad(): 

                x_adv = x_adv + linf_step(
                    gradient,
                    step_size
                )

                delta = x_adv - x 

                delta = project_linf(
                    delta,
                    epsilon
                )

                x_adv = torch.clamp(
                    x + delta,
                    0.0,
                    1.0
                )

    finally: 
        model.train(was_training)

    return x_adv.detach()

def pgd_l2(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True
) -> torch.Tensor:
    """
    Untargeted PGD attack under an L2 constraint.
    """
    x = images.detach()
    y = labels.detach()

    if random_start: 
        delta = _random_l2_delta(
            x, 
            epsilon
        )

        x_adv = torch.clamp(
            x + delta,
            0.0,
            1.0
        )

    else: 
        x_adv = x.clone()

    was_training = model.training
    model.eval()

    try: 
        for _ in range(steps): 
            x_adv = x_adv.detach()
            x_adv.requires_grad_(True)

            logits = model(x_adv)

            loss = F.cross_entropy(
                logits,
                y
            )

            gradient = torch.autograd.grad(
                loss,
                x_adv, 
                only_inputs=True
            )[0]

            with torch.no_grad():

                x_adv = x_adv + l2_step(
                    gradient,
                    step_size,
                    dim=(1, 2, 3)
                )

                delta = x_adv - x 

                delta = project_l2(
                    delta,
                    epsilon,
                    dim=(1, 2, 3)
                )

                x_adv = torch.clamp(
                    x + delta,
                    0.0,
                    1.0
                )

    finally:
        model.train(was_training)

    return x_adv.detach()

def pgd_linf_targeted(
    model,
    images: torch.Tensor,
    target_labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Targeted PGD attack under an L_inf constraint 
    """
    x = images.detach()
    y_target = target_labels.detach()

    if random_start: 
        delta = torch.empty_like(x).uniform_(
            -epsilon,
            epsilon
        )

        x_adv = torch.clamp(
            x + delta,
            0.0, 
            1.0
        )

    else: 
        x_adv = x.clone()

    was_training = model.training
    model.eval()

    try: 
        for _ in range(steps): 
            x_adv = x_adv.detach()
            x_adv.requires_grad_(True)

            logits = model(x_adv)

            loss = F.cross_entropy(
                logits,
                y_target
            )

            gradient = torch.autograd.grad(
                loss,
                x_adv,
                only_inputs=True
            )[0]

            with torch.no_grad(): 
                x_adv = x_adv - linf_step(
                    gradient,
                    step_size
                )

                delta = x_adv - x

                delta = project_linf(
                    delta,
                    epsilon
                )

                x_adv = torch.clamp(
                    x + delta,
                    0.0,
                    1.0
                )
    finally:
        model.train(was_training)

    return x_adv.detach()

def pgd_l2_targeted(
    model,
    images: torch.Tensor,
    target_labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Targeted PGD attack under an L2 constraint
    """
    x = images.detach()
    y_target = target_labels.detach()

    if random_start:
        delta = _random_l2_delta(
            x,
            epsilon,
        )

        x_adv = torch.clamp(
            x + delta,
            0.0,
            1.0,
        )

    else:
        x_adv = x.clone()

    was_training = model.training
    model.eval()

    try:
        for _ in range(steps):

            x_adv = x_adv.detach()
            x_adv.requires_grad_(True)

            logits = model(x_adv)

            loss = F.cross_entropy(
                logits,
                y_target,
            )

            gradient = torch.autograd.grad(
                loss,
                x_adv,
                only_inputs=True,
            )[0]

            with torch.no_grad():
                x_adv = x_adv - l2_step(
                    gradient,
                    step_size,
                    dim=(1, 2, 3),
                )

                delta = x_adv - x

                delta = project_l2(
                    delta,
                    epsilon,
                    dim=(1, 2, 3),
                )

                x_adv = torch.clamp(
                    x + delta,
                    0.0,
                    1.0,
                )

    finally:
        model.train(was_training)

    return x_adv.detach()