import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Computes approximate zero-power (orthogonal) matrix via 5th-order Newton-Schulz iteration
    Uses a bfloat16-friendly cubic polynomial with coefficients: a=3.4445, b=-4.7750, c=2.0315
    Handles tall matrices (m > n) by transposing before/after iteration
    Normalizes input before iteration: X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    Returns bfloat16 tensor
    Must assert G.ndim >= 2
    """
    assert G.ndim >= 2, "Input must be at least 2-dimensional"

    # Handle tall matrices (m > n) by transposing
    transposed = False
    if G.size(-2) > G.size(-1):
        G = G.transpose(-2, -1)
        transposed = True

    # Normalize input
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Initialize Y with X transpose
    Y = X.transpose(-2, -1)

    # Coefficients for the 5th-order Newton-Schulz iteration
    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        # Compute X*Y
        XY = X @ Y

        # Compute cubic polynomial: a*XY^2 + b*XY + c*I
        XY2 = XY @ XY
        poly = a * XY2 + b * XY + c * torch.eye(XY.size(-1), device=XY.device, dtype=XY.dtype)

        # Update Y
        Y = Y @ poly

    # Compute final orthogonal matrix
    Z = X @ Y

    # Transpose back if needed
    if transposed:
        Z = Z.transpose(-2, -1)

    # Convert to bfloat16
    return Z.to(torch.bfloat16)


def muon_update(grad, momentum_buffer, beta=0.95, ns_steps=5, nesterov=True) -> torch.Tensor:
    """
    Updates momentum_buffer in-place via momentum_buffer.lerp_(grad, 1 - beta)
    If nesterov=True: update = grad.lerp_(momentum_buffer, beta), else update = momentum_buffer
    If update.ndim == 4, flatten to 2D: update = update.view(len(update), -1)
    Apply Newton-Schulz: update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    Scale by aspect ratio: update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    Return update
    """
    # Update momentum buffer in-place
    momentum_buffer.lerp_(grad, 1 - beta)

    # Compute update
    if nesterov:
        update = grad.lerp_(momentum_buffer, beta)
    else:
        update = momentum_buffer

    # Flatten 4D tensors to 2D
    if update.ndim == 4:
        update = update.view(len(update), -1)

    # Apply Newton-Schulz iteration
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Scale by aspect ratio
    aspect_ratio = max(1, update.size(-2) / update.size(-1))
    update *= aspect_ratio ** 0.5

    return update


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict], ns_steps: int = 5, nesterov: bool = True):
        # Process each parameter group
        for group in param_groups:
            # Assert "use_muon" key exists
            assert "use_muon" in group, "Each param group must have 'use_muon' key"

            if group["use_muon"]:
                # Muon group defaults
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
            else:
                # Adam group defaults
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)

        # Initialize optimizer
        super().__init__(param_groups, dict())

        # Store hyperparameters
        self.ns_steps = ns_steps
        self.nesterov = nesterov

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Muon update logic
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # Initialize momentum buffer if not exists
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    # Compute muon update
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=self.ns_steps,
                        nesterov=self.nesterov
                    )

                    # Apply weight decay if needed
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    # Apply update
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # Adam update logic
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # Initialize state if not exists
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    step = state["step"]

                    grad = p.grad

                    # Apply weight decay if needed
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    # Update first and second moments
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # Compute step size and denominator
                    step_size = group["lr"] / bias_correction1
                    denom = (exp_avg_sq / bias_correction2).sqrt() + group["eps"]

                    # Apply update
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss