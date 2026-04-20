# Qwen3.5 Model Spec for Customized Archon Engine

from customized_areal.models.qwen3_5.infra.parallelize import parallelize_qwen35
from customized_areal.models.qwen3_5.model.args import Qwen35ModelArgs
from customized_areal.models.qwen3_5.model.model import Qwen35Model
from customized_areal.models.qwen3_5.model.state_dict_adapter import (
    Qwen35StateDictAdapter,
)

from areal.experimental.models.archon.model_spec import ModelSpec, register_model_spec
from areal.experimental.models.archon.pipeline_parallel import pipeline_llm

# Model spec definition for Qwen3.5
# Note: This does NOT auto-register to avoid conflicts with the areal/ implementation.
# Use register_custom_qwen35() to register when needed, or import the components directly.

QWEN35_SPEC = ModelSpec(
    name="Qwen35",
    model_class=Qwen35Model,
    model_args_class=Qwen35ModelArgs,
    state_dict_adapter_class=Qwen35StateDictAdapter,
    parallelize_fn=parallelize_qwen35,
    supported_model_types=frozenset(
        {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
    ),
    pipelining_fn=pipeline_llm,
)


def register_custom_qwen35(force: bool = False) -> ModelSpec:
    """Register the customized Qwen3.5 model spec.

    This function allows you to register the customized Qwen3.5 model
    with the Archon engine. If there's already a registration for
    'qwen3_5' model_type, this will raise an error unless force=True.

    Args:
        force: If True, unregister any existing spec before registering.

    Returns:
        The registered ModelSpec.

    Example:
        >>> from customized_areal.models.qwen3_5.spec import register_custom_qwen35
        >>> spec = register_custom_qwen35()
        >>> # Now you can use the customized model
    """
    from areal.experimental.models.archon.model_spec import _MODEL_SPEC_REGISTRY

    if force:
        # Remove existing registrations for qwen3_5 model types
        for model_type in QWEN35_SPEC.supported_model_types:
            if model_type in _MODEL_SPEC_REGISTRY:
                del _MODEL_SPEC_REGISTRY[model_type]

    register_model_spec(QWEN35_SPEC)
    return QWEN35_SPEC


# Alternative: Register with a unique model_type to avoid conflicts
QWEN35_CUSTOM_SPEC = ModelSpec(
    name="Qwen35Custom",
    model_class=Qwen35Model,
    model_args_class=Qwen35ModelArgs,
    state_dict_adapter_class=Qwen35StateDictAdapter,
    parallelize_fn=parallelize_qwen35,
    supported_model_types=frozenset(
        {
            "qwen3_5_custom",
            "qwen3_5_custom_text",
            "qwen3_5_custom_moe",
            "qwen3_5_custom_moe_text",
        }
    ),
    pipelining_fn=pipeline_llm,
)

# Auto-register the custom variant (doesn't conflict with areal/)
register_model_spec(QWEN35_CUSTOM_SPEC)

__all__ = [
    "QWEN35_SPEC",
    "QWEN35_CUSTOM_SPEC",
    "register_custom_qwen35",
]
