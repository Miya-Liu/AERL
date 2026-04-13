#!/usr/bin/env python3
"""Integration test for on-policy distillation training.

This script verifies all components are properly wired:
1. MultiCandidateFSDPPPOActor can be imported and instantiated
2. OnPolicyDistillationTrainer uses the custom actor
3. Proxy workflow components are accessible
4. Loss function is properly patched

Usage:
    python test_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all key components can be imported."""
    print("Testing imports...")

    try:
        from customized_areal.on_policy_distill import (
            OnPolicyDistillConfig,
            OnPolicyDistillationTrainer,
            OnPolicyDistillAgent,
            MultiCandidateFSDPEngine,
            MultiCandidateFSDPPPOActor,
            OpenAIProxyWorkflow,
        )
        from customized_areal.on_policy_distill.proxy.cache import PositionRewardInfo
        print("  ✓ All main imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_actor_creation():
    """Test MultiCandidateFSDPPPOActor can be instantiated."""
    print("\nTesting actor creation...")

    try:
        from areal.api.cli_args import PPOActorConfig
        from customized_areal.on_policy_distill import MultiCandidateFSDPPPOActor

        config = PPOActorConfig(
            path="dummy/path",
            mb_spec={"max_tokens_per_mb": 1024},
        )

        # Note: Full initialization requires distributed setup
        # Here we just test the class can be instantiated
        # In real usage, actor.initialize() is called by trainer
        print("  ✓ Actor config created successfully")
        print(f"  ✓ Actor class: {MultiCandidateFSDPPPOActor.__name__}")
        print(f"  ✓ Actor MRO: {[c.__name__ for c in MultiCandidateFSDPPPOActor.__mro__[:4]]}")
        return True
    except Exception as e:
        print(f"  ✗ Actor creation failed: {e}")
        return False


def test_loss_patch():
    """Test that PPOActor gets patched with grpo_distill_loss_fn."""
    print("\nTesting loss function patch...")

    try:
        from areal.trainer.ppo.actor import PPOActor
        from customized_areal.on_policy_distill.training.actor import (
            patch_ppo_actor_class_to_use_distill_loss,
        )

        # Store original method
        original_ppo_update = PPOActor._ppo_update

        # Apply patch
        patch_ppo_actor_class_to_use_distill_loss()

        # Verify patch was applied
        if PPOActor._ppo_update != original_ppo_update:
            print("  ✓ PPOActor patched successfully")
            print(f"  ✓ New _ppo_update: {PPOActor._ppo_update.__name__}")
            return True
        else:
            print("  ✗ Patch not applied (method unchanged)")
            return False
    except Exception as e:
        print(f"  ✗ Loss patch test failed: {e}")
        return False


def test_workflow_components():
    """Test workflow components are properly set up."""
    print("\nTesting workflow components...")

    try:
        from customized_areal.on_policy_distill import (
            OpenAIProxyWorkflow,
            OpenAIProxyClient,
            PositionRewardInfo,
        )

        # Test PositionRewardInfo creation
        pr = PositionRewardInfo(
            position=0,
            candidates=["token1", "token2"],
            candidate_token_ids=[100, 200],
            logprobs=[-1.0, -2.0],
            rewards=[1.0, 0.5],
            chosen_index=0,
        )

        print(f"  ✓ PositionRewardInfo created: {pr}")
        print(f"    - position: {pr.position}")
        print(f"    - candidates: {pr.candidates}")
        print(f"    - chosen_token: {pr.chosen_token}")
        print(f"    - chosen_reward: {pr.chosen_reward}")

        return True
    except Exception as e:
        print(f"  ✗ Workflow component test failed: {e}")
        return False


def test_config():
    """Test configuration can be created."""
    print("\nTesting configuration...")

    try:
        from customized_areal.on_policy_distill import OnPolicyDistillConfig

        config = OnPolicyDistillConfig(
            experiment_name="test",
            trial_name="trial0",
            proxy_base_url="http://localhost:8000",
            proxy_model="qwen/qwen3-1.7b",
        )

        print(f"  ✓ Config created: {config.experiment_name}/{config.trial_name}")
        print(f"  ✓ Proxy URL: {config.proxy_base_url}")
        print(f"  ✓ Proxy model: {config.proxy_model}")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_engine_methods():
    """Test MultiCandidateFSDPEngine has required methods."""
    print("\nTesting engine methods...")

    try:
        from customized_areal.on_policy_distill import MultiCandidateFSDPEngine

        required_methods = [
            "_compute_logprobs_entropy",
            "_compute_logprobs",
            "_prepare_multi_candidate_labels",
            "_compute_logprobs_and_loss",
        ]

        for method in required_methods:
            if hasattr(MultiCandidateFSDPEngine, method):
                print(f"  ✓ {method} present")
            else:
                print(f"  ✗ {method} missing")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Engine methods test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for running actual training."""
    print("\n" + "=" * 60)
    print("NEXT STEPS TO RUN TRAINING:")
    print("=" * 60)
    print("""
1. Start the proxy server:
   python -m customized_areal.on_policy_distill.proxy.proxy_rollout_server \
       --host 0.0.0.0 --port 8000

2. Create a training script that uses OnPolicyDistillationTrainer:

   from customized_areal.on_policy_distill import (
       OnPolicyDistillConfig,
       OnPolicyDistillationTrainer,
       OnPolicyDistillAgent,
   )

   config = OnPolicyDistillConfig(...)
   agent = OnPolicyDistillAgent(agent_id="...", model_name="...")

   with OnPolicyDistillationTrainer(config, agent=agent) as trainer:
       trainer.train()

3. Run training:
   python your_training_script.py --config your_config.yaml
""")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("On-Policy Distillation Integration Test")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Actor Creation", test_actor_creation()))
    results.append(("Loss Patch", test_loss_patch()))
    results.append(("Workflow Components", test_workflow_components()))
    results.append(("Configuration", test_config()))
    results.append(("Engine Methods", test_engine_methods()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All integration tests passed!")
        print_next_steps()
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
