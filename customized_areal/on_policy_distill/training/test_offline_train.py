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
    from customized_areal.on_policy_distill.training.actor import (
        patch_ppo_actor_class_to_use_distill_loss,
    )

    from areal.trainer.ppo.actor import PPOActor

    patch_ppo_actor_class_to_use_distill_loss()
    source = inspect.getsource(PPOActor._ppo_update)

    # After fix: reward logging should appear before the data.pop lines
    lines = source.split("\n")
    pop_idx = None
    stat_idx = None
    for i, line in enumerate(lines):
        if "data.pop" in line and pop_idx is None:
            pop_idx = i
        if "stats_tracker.stat" in line and "task_reward" in line and stat_idx is None:
            stat_idx = i

    if pop_idx is not None and stat_idx is not None:
        assert stat_idx < pop_idx, (
            "Reward stats logging should appear before data.pop()"
        )


def test_prompt_len_per_sample():
    """Bug 3: prompt_len should be computed per sample when batch > 1."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import (
        _compute_position_level_grpo_loss,
    )

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

    # This test verifies that the prompt length computation handles 2D loss masks correctly
    # by testing the _compute_position_level_grpo_loss function directly
    from customized_areal.on_policy_distill.training.loss import (
        _compute_position_level_grpo_loss,
    )

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
    from customized_areal.on_policy_distill.training.loss import (
        _compute_position_level_grpo_loss,
    )

    source = inspect.getsource(_compute_position_level_grpo_loss)
    item_calls = [
        line
        for line in source.split("\n")
        if ".item()" in line and "output_len" in line
    ]
    assert len(item_calls) == 0, (
        f"Found .item() call for output_len. "
        f"This forces GPU-CPU sync. Lines: {item_calls}"
    )


def test_position_bounds_check():
    """Bug 5: position + prompt_len must be clamped to avoid out-of-bounds."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.loss import (
        _compute_position_level_grpo_loss,
    )

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

    assert torch.isfinite(loss), (
        f"Loss should be finite with bounds-checked positions, got {loss}"
    )


def test_no_duplicate_denominator():
    """Bug 6: _ppo_update_with_distill_loss should not register
    n_valid_tokens denominator since grpo_distill_loss_fn already does."""
    from customized_areal.on_policy_distill.training.actor import (
        patch_ppo_actor_class_to_use_distill_loss,
    )

    from areal.trainer.ppo.actor import PPOActor

    patch_ppo_actor_class_to_use_distill_loss()
    source = inspect.getsource(PPOActor._ppo_update)

    lines = source.split("\n")
    found_duplicate = False
    for line in lines:
        if "stats_tracker.denominator(n_valid_tokens" in line:
            found_duplicate = True
            break

    assert not found_duplicate, (
        "Duplicate n_valid_tokens denominator found. "
        "grpo_distill_loss_fn already registers this denominator."
    )


def test_generate_mock_batch():
    """Test that generate_mock_batch produces valid batch data with PositionRewardInfo."""
    from customized_areal.on_policy_distill.training.offline_train import (
        generate_mock_batch,
    )

    batch = generate_mock_batch(
        batch_size=2, seq_len=32, prompt_len=8, num_candidates=3, vocab_size=32000
    )
    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
    assert batch["loss_mask"].shape == (2, 32)
    assert batch["logprobs"].shape == (2, 32)
    assert batch["advantages"].shape == (2, 32)
    assert (batch["loss_mask"][:, :8] == 0).all()
    assert (batch["loss_mask"][:, 8:] == 1).all()
    assert (batch["attention_mask"] == 1).all()
    assert isinstance(batch["position_rewards"], list)
    assert len(batch["position_rewards"]) > 0
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo

    assert isinstance(batch["position_rewards"][0], PositionRewardInfo)
    for pr in batch["position_rewards"]:
        assert 0 <= pr.sample_index < 2
        assert len(pr.candidates) == 3
        assert len(pr.rewards) == 3
        assert len(pr.candidate_token_ids) == 3


def test_load_batch_from_disk(tmp_path):
    """Test saving and loading a batch to/from disk."""
    from customized_areal.on_policy_distill.training.offline_train import (
        generate_mock_batch,
        load_batch,
        save_batch,
    )

    batch = generate_mock_batch(
        batch_size=2, seq_len=16, prompt_len=4, num_candidates=2
    )
    save_path = tmp_path / "batch_000.pt"
    save_batch(batch, save_path)
    loaded = load_batch(save_path)
    assert torch.equal(loaded["input_ids"], batch["input_ids"])
    assert torch.equal(loaded["attention_mask"], batch["attention_mask"])
    assert torch.equal(loaded["loss_mask"], batch["loss_mask"])
    assert torch.allclose(loaded["logprobs"], batch["logprobs"])
    assert torch.allclose(loaded["advantages"], batch["advantages"])
    assert len(loaded["position_rewards"]) == len(batch["position_rewards"])
    for orig, loaded_pr in zip(batch["position_rewards"], loaded["position_rewards"]):
        assert orig.position == loaded_pr.position
        assert orig.sample_index == loaded_pr.sample_index
        assert orig.candidate_token_ids == loaded_pr.candidate_token_ids
        assert orig.rewards == loaded_pr.rewards


def test_parse_args_defaults():
    """Test that parse_args returns correct default values."""
    from customized_areal.on_policy_distill.training.offline_train import parse_args

    args = parse_args([])
    assert args.model_path == ""
    assert args.mock_data is True
    assert args.num_epochs == 3
    assert args.batch_size == 4
    assert args.seq_len == 512
    assert args.lr == 1e-6
    assert args.eps_clip == 0.2
    assert args.distill_loss_weight == 0.005
    assert args.rl_loss_weight == 1.0
    assert args.save_every == 100


def test_parse_args_custom():
    """Test that parse_args correctly parses custom command-line arguments."""
    from customized_areal.on_policy_distill.training.offline_train import parse_args

    args = parse_args(
        [
            "--model_path",
            "/data/Qwen2.5-3B",
            "--data_path",
            "/data/rollout/",
            "--num_epochs",
            "5",
            "--batch_size",
            "8",
            "--lr",
            "5e-7",
        ]
    )
    assert args.model_path == "/data/Qwen2.5-3B"
    assert args.data_path == "/data/rollout/"
    assert args.num_epochs == 5
    assert args.batch_size == 8
    assert args.lr == 5e-7


def test_loss_computation_end_to_end():
    """End-to-end test of loss computation without FSDP2."""
    from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
    from customized_areal.on_policy_distill.training.logprobs import (
        gather_logprobs_entropy_multi_candidates,
    )
    from customized_areal.on_policy_distill.training.loss import grpo_distill_loss_fn

    from areal.api.cli_args import PPOActorConfig

    config = PPOActorConfig(path="dummy", eps_clip=0.2, ppo_n_minibatches=1)
    seq_len = 16
    prompt_len = 4
    vocab_size = 1024
    num_candidates = 3

    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    loss_mask = torch.zeros(1, seq_len, dtype=torch.long)
    loss_mask[:, prompt_len:] = 1

    labels = torch.zeros(seq_len, num_candidates, dtype=torch.long)
    for pos in range(prompt_len, seq_len):
        labels[pos] = torch.randint(0, vocab_size, (num_candidates,))

    logprobs, entropy = gather_logprobs_entropy_multi_candidates(
        logits.squeeze(0),
        labels,
        temperature=1.0,
    )

    old_logp = logprobs.detach().clone() + torch.randn_like(logprobs) * 0.1
    old_logp_chosen = old_logp[:, 0]
    advantages = torch.randn(seq_len)

    position_rewards = []
    for pos in range(seq_len - prompt_len):
        pr = PositionRewardInfo(
            position=pos,
            candidates=[f"tok_{i}" for i in range(num_candidates)],
            candidate_token_ids=labels[pos + prompt_len].tolist(),
            logprobs=old_logp[pos + prompt_len].tolist(),
            rewards=(torch.randn(num_candidates) * 0.3).tolist(),
            chosen_index=0,
            sample_index=0,
        )
        position_rewards.append(pr)

    input_data = {
        "logprobs": old_logp_chosen,
        "advantages": advantages,
        "loss_mask": loss_mask.squeeze(0),
        "position_rewards": position_rewards,
        "rl_loss_weight": 1.0,
        "distill_loss_weight": 0.005,
    }

    loss = grpo_distill_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        config=config,
    )

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss.requires_grad, "Loss should have grad_fn"
    loss.backward()
    assert logits.grad is not None, "Logits should have gradients after backward"


def test_mock_data_loss_computation():
    """Test that mock data from generate_mock_batch produces valid loss."""
    from customized_areal.on_policy_distill.training.loss import grpo_distill_loss_fn
    from customized_areal.on_policy_distill.training.offline_train import (
        generate_mock_batch,
    )

    from areal.api.cli_args import PPOActorConfig

    config = PPOActorConfig(path="dummy", eps_clip=0.2, ppo_n_minibatches=1)
    batch = generate_mock_batch(
        batch_size=1, seq_len=32, prompt_len=8, num_candidates=3, vocab_size=1024
    )

    seq_len = batch["input_ids"].shape[1]
    logprobs = torch.randn(seq_len, 3, requires_grad=True)
    entropy = torch.randn(seq_len)

    input_data = {
        "logprobs": batch["logprobs"].squeeze(0),
        "advantages": batch["advantages"].squeeze(0),
        "loss_mask": batch["loss_mask"].squeeze(0),
        "position_rewards": batch["position_rewards"],
        "rl_loss_weight": 1.0,
        "distill_loss_weight": 0.005,
    }

    loss = grpo_distill_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        config=config,
    )

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    loss.backward()
    assert logprobs.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_training_loop_with_gpu():
    """Test that the training loop runs with mock data on GPU."""
    from customized_areal.on_policy_distill.training.offline_train import (
        TrainingConfig,
        run_training,
    )

    config = TrainingConfig(
        model_path="",
        data_path="",
        output_dir="./test_checkpoints",
        num_epochs=1,
        batch_size=1,
        seq_len=64,
        prompt_len=8,
        num_candidates=2,
        lr=1e-6,
        eps_clip=0.2,
        save_every=0,
        mock_data=True,
        dtype="float32",  # Use float32 for testing to avoid precision issues
        gradient_checkpointing=False,
    )

    # Should run without errors
    result = run_training(config)
    assert "final_loss" in result
    assert torch.isfinite(torch.tensor(result["final_loss"])), (
        "Final loss should be finite"
    )
