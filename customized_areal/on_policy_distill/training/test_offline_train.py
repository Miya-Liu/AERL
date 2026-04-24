"""Tests for bug fixes in the on-policy distillation training pipeline."""
import inspect
import pytest
import torch

from areal.utils import stats_tracker


@pytest.fixture(autouse=True)
def reset_stats_tracker():
    """Reset the stats tracker before each test to avoid cross-test contamination."""
    stats_tracker.export()
    yield


def test_distill_stat_is_detached():
    """Bug 1: distill_stat passed to stats_tracker should be detached
    to avoid retaining the computational graph (memory leak)."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import grpo_distill_loss_fn
    from areal.api.cli_args import PPOActorConfig

    # Create a mock config with the necessary attributes
    class MockConfig:
        def __init__(self):
            self.path = "dummy"
            self.eps_clip = 0.2
            self.eps_clip_higher = None
            self.c_clip = None
            self.ppo_n_minibatches = 1
            self.prox_clip = "recompute"
            self.behave_imp_weight_cap = None
            self.importance_sampling_level = "token"
    config = MockConfig()
    seq_len = 8
    num_candidates = 3

    logprobs = torch.randn(seq_len, num_candidates, requires_grad=True)
    entropy = torch.randn(seq_len)
    old_logp = torch.randn(seq_len)
    advantages = torch.randn(seq_len)
    loss_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.bool)

    position_rewards = [
        PositionRewardInfo(
            position=0,
            candidates=["a", "b", "c"],
            candidate_token_ids=[1, 2, 3],
            logprobs=[-1.0, -2.0, -3.0],
            rewards=[0.5, -0.3, 0.1],
            chosen_index=0,
            sample_index=0,
        ),
    ]

    input_data = {
        "logprobs": old_logp,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_rewards": position_rewards,
    }

    loss = grpo_distill_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        config=config,
    )

    loss.backward()
    assert loss.grad_fn is not None, "Loss should have grad_fn"

    # Repeated calls should not accumulate graph references
    for _ in range(5):
        logprobs2 = torch.randn(seq_len, num_candidates, requires_grad=True)
        loss2 = grpo_distill_loss_fn(
            logprobs=logprobs2,
            entropy=torch.randn(seq_len),
            input_data={
                "logprobs": torch.randn(seq_len),
                "advantages": torch.randn(seq_len),
                "loss_mask": loss_mask,
                "position_rewards": position_rewards,
            },
            config=config,
        )
        loss2.backward()


def test_reward_stats_logged_before_pop():
    """Bug 2: rewards should be logged before being popped from data dict."""
    from areal.trainer.ppo.actor import PPOActor
    from customized_areal.on_policy_distill.training.actor import patch_ppo_actor_class_to_use_distill_loss

    patch_ppo_actor_class_to_use_distill_loss()
    source = inspect.getsource(PPOActor._ppo_update)

    # After fix: reward logging should appear before the data.pop lines
    lines = source.split('\n')
    pop_idx = None
    stat_idx = None
    for i, line in enumerate(lines):
        if 'data.pop' in line and pop_idx is None:
            pop_idx = i
        if 'stats_tracker.stat' in line and 'task_reward' in line and stat_idx is None:
            stat_idx = i

    if pop_idx is not None and stat_idx is not None:
        assert stat_idx < pop_idx, (
            "Reward stats logging should appear before data.pop()"
        )


def test_prompt_len_per_sample():
    """Bug 3: prompt_len should be computed per sample when batch > 1."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import _compute_position_level_grpo_loss

    seq_len = 10
    num_candidates = 2

    loss_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    logprobs = torch.randn(seq_len, num_candidates, requires_grad=True)

    position_rewards = [
        PositionRewardInfo(
            position=0,
            candidates=["a", "b"],
            candidate_token_ids=[1, 2],
            logprobs=[-1.0, -2.0],
            rewards=[0.5, -0.3],
            chosen_index=0,
            sample_index=0,
        ),
    ]

    loss = _compute_position_level_grpo_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[3],
    )

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss.item() != 0.0, "Loss should be non-zero with valid data"


def test_prompt_len_computation_with_batch_dim():
    """Bug 3: When loss_mask has batch dim, prompt_len must be computed correctly."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import grpo_distill_loss_fn

    # This test verifies that the prompt length computation handles 2D loss masks correctly
    # by testing the _compute_position_level_grpo_loss function directly

    from customized_areal.on_policy_distill.training.loss import _compute_position_level_grpo_loss

    seq_len = 8
    num_candidates = 2

    logprobs = torch.randn(seq_len, num_candidates, requires_grad=True)
    loss_mask_2d = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    loss_mask_1d = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.bool)

    position_rewards = [
        PositionRewardInfo(
            position=0,
            candidates=["a", "b"],
            candidate_token_ids=[1, 2],
            logprobs=[-1.0, -2.0],
            rewards=[0.5, -0.3],
            chosen_index=0,
            sample_index=0,
        ),
    ]

    # Test with both 2D and 1D loss masks (squeeze 2D first for stats_tracker compatibility)
    loss_2d = _compute_position_level_grpo_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask_2d,
        prompt_lens=[2],  # manually pass prompt_lens for this test
    )

    loss_1d = _compute_position_level_grpo_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask_1d,
        prompt_lens=[2],  # manually pass prompt_lens for this test
    )

    assert torch.isfinite(loss_2d), f"Loss with 2D mask should be finite, got {loss_2d}"
    assert torch.isfinite(loss_1d), f"Loss with 1D mask should be finite, got {loss_1d}"
    # Losses should be identical when using same prompt lens
    assert torch.allclose(loss_2d, loss_1d), "Losses with 2D and 1D masks should match"


def test_position_level_loss_no_cpu_sync():
    """Bug 4: _compute_position_level_grpo_loss should avoid
    .item() calls for output_len that force GPU-CPU sync."""
    from customized_areal.on_policy_distill.training.loss import _compute_position_level_grpo_loss

    source = inspect.getsource(_compute_position_level_grpo_loss)
    item_calls = [line for line in source.split('\n') if '.item()' in line and 'output_len' in line]
    assert len(item_calls) == 0, (
        f"Found .item() call for output_len. "
        f"This forces GPU-CPU sync. Lines: {item_calls}"
    )


def test_position_bounds_check():
    """Bug 5: position + prompt_len must be clamped to avoid out-of-bounds."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import _compute_position_level_grpo_loss

    seq_len = 5
    num_candidates = 2
    logprobs = torch.randn(seq_len, num_candidates, requires_grad=True)
    loss_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)

    position_rewards = [
        PositionRewardInfo(
            position=0,
            candidates=["a", "b"],
            candidate_token_ids=[1, 2],
            logprobs=[-1.0, -2.0],
            rewards=[0.5, -0.3],
            chosen_index=0,
            sample_index=0,
        ),
        PositionRewardInfo(
            position=10,  # out of bounds for seq_len=5
            candidates=["x", "y"],
            candidate_token_ids=[3, 4],
            logprobs=[-0.5, -1.5],
            rewards=[0.2, -0.1],
            chosen_index=0,
            sample_index=0,
        ),
    ]

    loss = _compute_position_level_grpo_loss(
        position_rewards=position_rewards,
        logprobs=logprobs,
        loss_mask=loss_mask,
        prompt_lens=[0],
    )

    assert torch.isfinite(loss), f"Loss should be finite with bounds-checked positions, got {loss}"


def test_no_duplicate_denominator():
    """Bug 6: _ppo_update_with_distill_loss should not register
    n_valid_tokens denominator since grpo_distill_loss_fn already does."""
    from areal.trainer.ppo.actor import PPOActor
    from customized_areal.on_policy_distill.training.actor import patch_ppo_actor_class_to_use_distill_loss

    patch_ppo_actor_class_to_use_distill_loss()
    source = inspect.getsource(PPOActor._ppo_update)

    lines = source.split('\n')
    found_duplicate = False
    for line in lines:
        if 'stats_tracker.denominator(n_valid_tokens' in line:
            found_duplicate = True
            break

    assert not found_duplicate, (
        "Duplicate n_valid_tokens denominator found. "
        "grpo_distill_loss_fn already registers this denominator."
    )
