
import sys

# Patch: Fix 'set' object is not subscriptable error in FSDP
# This converts any set to a list before calling the original apply_fsdp2
import areal.engine.fsdp_utils as _fsdp_utils

_original_apply_fsdp2 = _fsdp_utils.apply_fsdp2


def _patched_apply_fsdp2(model, fsdp_kwargs, wrap_policy):
    if wrap_policy is not None:
        transformer_cls = getattr(wrap_policy, "transformer_layer_cls_to_wrap", None)
        if isinstance(transformer_cls, set):
            wrap_policy.transformer_layer_cls_to_wrap = list(transformer_cls)
    return _original_apply_fsdp2(model, fsdp_kwargs, wrap_policy)


_fsdp_utils.apply_fsdp2 = _patched_apply_fsdp2
# End patch

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from customized_areal.tpfc.tpfc_config import TPFCConfig


def main(args):
    config, _ = load_expr_config(args, TPFCConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=config.eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
