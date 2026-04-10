"""Combined GRPO and Position-Level GRPO Loss Function.

This module provides a loss function that supports both standard GRPO training and
position-level GRPO using position_rewards from token_reward/cache.py.

The position-level GRPO loss treats candidates at each position as different samples
and computes reward-weighted log probability: -EGRPO[log p(·) * A(·)]

For multi-candidate training:
- Engine computes multi-candidate logprobs: [seq_len, num_candidates]
- Loss function receives logprobs directly (already has gradients)
- Uses old logprobs from rollout for off-policy importance weighting
"""

from __future__ import annotations

import torch
from typing import Any

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker


def grpo_distill_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    config: PPOActorConfig,
    current_version: int | None = None,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
):
    """Combined GRPO and distillation loss function.

    This function computes:
    1. GRPO loss using standard PPO objective with advantages
    2. Position-level GRPO loss using position_rewards:
       - Uses pre-computed logprobs (with gradients) for all candidates
       - Uses old logprobs from rollout for off-policy importance weighting
       - Computes: -E[importance_weight * reward * logp]

    Parameters
    ----------
    logprobs : torch.Tensor
        Log probabilities for all candidates [seq_len, num_candidates] or [seq_len].
        These are computed by the engine and have gradient information.
    entropy : torch.Tensor
        Entropy values for the current policy [seq_len].
    input_data : dict
        Dictionary containing:
        - logprobs: Old log probabilities from rollout (for importance sampling)
        - advantages: Advantage estimates
        - loss_mask: Mask indicating which positions to compute loss on
        - position_rewards: Position-wise rewards with candidate info
          Each PositionRewardInfo should have:
          - candidate_token_ids: list[int] - token IDs for all candidates
          - rewards: list[float] - rewards for each candidate
          - logprobs: list[float] - OLD logprobs from rollout (for importance weighting)
        - rl_loss_weight: Weight for GRPO loss (default: 1.0)
        - distill_loss_weight: Weight for distillation loss (default: 0.005)
    config : PPOActorConfig
        PPO actor configuration.
    current_version : int | None, optional
        Current weight version for version alignment.
    vocab_min_logits, vocab_max_logits : torch.Tensor | None
        Min/max logits for numerical stability (passed by engine).

    Returns
    -------
    torch.Tensor
        Combined loss (GRPO + position-level GRPO).
    """
    from areal.trainer.ppo.actor import grpo_loss_fn

    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")

    entropy = entropy.detach()

    coeffs = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=config.prox_clip,
        old_logp=old_logp,
        logprobs=logprobs.detach() if logprobs.dim() == 1 else logprobs[:, 0].detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # For standard GRPO loss, use logprobs of chosen tokens only
    # If logprobs is 2D [seq_len, num_candidates], use chosen token (index 0)
    chosen_logprobs = logprobs if logprobs.dim() == 1 else logprobs[:, 0]

    loss, stat = _compute_grpo_loss(
        logprobs=chosen_logprobs,
        old_logp=old_logp,
        advantages=advantages,
        eps_clip=config.eps_clip,
        eps_clip_higher=config.eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=config.c_clip,
        proximal_logprobs=coeffs,
        behave_imp_weight_cap=config.behave_imp_weight_cap,
        importance_sampling_level=config.importance_sampling_level,
        cu_seqlens=input_data.get("cu_seqlens"),
    )

    position_rewards = input_data.get("position_rewards")
    distill_stat = None

    if position_rewards is not None and logprobs.dim() == 2:
        # Multi-candidate logprobs available
        rl_loss_weight = input_data.get("rl_loss_weight", 1.0)
        distill_loss_weight = input_data.get("distill_loss_weight", 0.005)

        # Determine output length from loss_mask
        loss_mask_list = loss_mask.squeeze(0).tolist() if loss_mask.dim() > 1 else loss_mask.tolist()
        try:
            first_one_idx = loss_mask_list.index(1)
            output_len = len(loss_mask_list) - first_one_idx
        except ValueError:
            output_len = chosen_logprobs.shape[-1]

        position_grpo_loss = _compute_position_level_grpo_loss(
            position_rewards=position_rewards,
            logprobs=logprobs,
            loss_mask=loss_mask,
            output_len=output_len,
        )

        loss = rl_loss_weight * loss + distill_loss_weight * position_grpo_loss
        distill_stat = position_grpo_loss

    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    if distill_stat is not None:
        stats_tracker.stat(
            distill_loss=distill_stat,
            denominator="n_valid_tokens",
        )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=chosen_logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )

    return loss


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor | None:
    """Resolve proximal log probabilities based on method."""
    if prox_logp_gt is not None:
        return prox_logp_gt

    if prox_logp_method == "recompute":
        return old_logp

    if versions is not None and current_version is not None:
        logprobs = logprobs[versions == current_version]

    return old_logp


def _compute_grpo_loss(
    logprobs: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_higher: float | None,
    loss_mask: torch.Tensor,
    c_clip: float | None,
    proximal_logprobs: torch.Tensor | None,
    behave_imp_weight_cap: float | None,
    importance_sampling_level: str,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO/PPO loss."""
    from areal.utils.functional import ppo_actor_loss_fn

    proximal_logp = old_logp if proximal_logprobs is None else proximal_logprobs

    return ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logp,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        behave_imp_weight_cap=behave_imp_weight_cap,
        importance_sampling_level=importance_sampling_level,
        cu_seqlens=cu_seqlens,
    )


def _compute_position_level_grpo_loss(
    position_rewards: list,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    output_len: int,
) -> torch.Tensor:
    """Compute position-level GRPO loss using pre-computed multi-candidate logprobs.

    This function uses the logprobs computed by the engine (which have gradients)
    and combines them with rewards and old logprobs from rollout.

    Parameters
    ----------
    position_rewards : list
        List of PositionRewardInfo objects with:
        - candidate_token_ids: list[int] - token IDs for all candidates
        - rewards: list[float] - rewards for each candidate
        - logprobs: list[float] - OLD logprobs from rollout (for importance weighting)
    logprobs : torch.Tensor
        Current policy logprobs for all candidates [seq_len, num_candidates].
        These are computed by the engine and have gradient information.
    loss_mask : torch.Tensor
        Mask indicating which tokens to compute loss on.
    output_len : int
        Number of output tokens.

    Returns
    -------
    torch.Tensor
        GRPO loss tensor (scalar).
    """
    if not position_rewards:
        return torch.tensor(0.0, dtype=torch.float32, device=loss_mask.device)

    # Gather valid positions and data into padded tensors
    positions = []
    reward_rows = []
    old_logprob_rows = []
    has_old_mask_rows = []
    max_candidates = logprobs.shape[1]

    for pr in position_rewards:
        if not pr.rewards:
            continue
        position = pr.position
        if position >= logprobs.shape[0]:
            continue
        num_candidates = min(len(pr.rewards), max_candidates)
        positions.append(position)
        reward_rows.append(pr.rewards[:num_candidates])
        if pr.logprobs and len(pr.logprobs) >= num_candidates:
            old_logprob_rows.append(pr.logprobs[:num_candidates])
            has_old_mask_rows.append([True] * num_candidates)
        else:
            old_logprob_rows.append([0.0] * num_candidates)
            has_old_mask_rows.append([False] * num_candidates)

    if not positions:
        return torch.tensor(0.0, dtype=torch.float32, device=loss_mask.device)

    n_pos = len(positions)
    device = logprobs.device
    positions_t = torch.tensor(positions, device=device)

    # Build padded tensors [n_pos, max_candidates]
    rewards_t = torch.zeros(n_pos, max_candidates, dtype=torch.float32, device=device)
    old_logprobs_t = torch.zeros(n_pos, max_candidates, dtype=torch.float32, device=device)
    candidate_mask = torch.zeros(n_pos, max_candidates, dtype=torch.bool, device=device)
    has_old_mask = torch.zeros(n_pos, max_candidates, dtype=torch.bool, device=device)

    for i in range(n_pos):
        num = len(reward_rows[i])
        rewards_t[i, :num] = torch.tensor(reward_rows[i], dtype=torch.float32, device=device)
        old_logprobs_t[i, :num] = torch.tensor(old_logprob_rows[i], dtype=torch.float32, device=device)
        candidate_mask[i, :num] = True
        has_old_mask[i, :num] = torch.tensor(has_old_mask_rows[i], dtype=torch.bool, device=device)

    # Get new logprobs for all positions at once [n_pos, max_candidates]
    new_logprobs = logprobs[positions_t, :]

    # Compute importance weights with clipping
    importance_weights = torch.ones_like(new_logprobs)
    iw = torch.exp(new_logprobs.detach() - old_logprobs_t).clamp(max=10.0)
    importance_weights = torch.where(has_old_mask, iw, importance_weights)

    # Compute GRPO advantages: normalize rewards within each position group
    num_valid = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1)
    reward_mean = (rewards_t * candidate_mask).sum(dim=1, keepdim=True) / num_valid

    # Unbiased std (match original torch.std behavior)
    var = ((rewards_t - reward_mean) ** 2 * candidate_mask).sum(dim=1, keepdim=True) / (num_valid - 1).clamp(min=1)
    reward_std = torch.sqrt(var)
    reward_std = torch.where(num_valid > 1, reward_std, torch.zeros_like(reward_std))

    advantages = (rewards_t - reward_mean) / (reward_std + 1e-8)
    advantages = advantages * candidate_mask  # mask padding

    # Compute weighted loss with importance sampling
    weighted_advantages = importance_weights * advantages
    loss_per_position = -(weighted_advantages * new_logprobs).sum(dim=1) / num_valid.squeeze(1)

    total_weight = importance_weights.sum(dim=1)
    loss_per_position = torch.where(total_weight > 0, loss_per_position / total_weight, loss_per_position)

    # Pad or truncate to output_len
    n_loss = loss_per_position.shape[0]
    if n_loss < output_len:
        padding = torch.zeros(output_len - n_loss, dtype=torch.float32, device=device)
        loss_per_position = torch.cat([loss_per_position, padding])
    elif n_loss > output_len:
        loss_per_position = loss_per_position[:output_len]

    grpo_loss = loss_per_position.sum() / loss_mask.sum().clamp(min=1)
    return grpo_loss
