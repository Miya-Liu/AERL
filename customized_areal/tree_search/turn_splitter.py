# customized_areal/tree_search/turn_splitter.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Turn:
    """A structured turn with a prompt/response split.

    prompt_tokens: shared context tokens (no branching)
    response_tokens: assistant output tokens (branching point)
    """

    prompt_tokens: list[int]
    response_tokens: list[int]


def make_turn_splitter(
    tokenizer, assistant_marker: str = ""
) -> Callable[[list[int]], list[Turn]]:
    """Create a turn splitter that identifies assistant role markers.

    Finds all occurrences of the assistant marker tokens in the input
    sequence and splits into Turn objects where:
    - prompt_tokens = the marker tokens themselves
    - response_tokens = everything after marker to next marker start (or end)

    Args:
        tokenizer: HuggingFace-style tokenizer with an encode() method.
        assistant_marker: String marker identifying assistant turns.
            If empty, auto-detect from tokenizer chat template.

    Returns:
        A function that takes a list of token IDs and returns a list of Turn objects.
    """
    if not assistant_marker:
        assistant_marker = _detect_assistant_marker(tokenizer)
    marker_tokens = tokenizer.encode(assistant_marker, add_special_tokens=False)

    def split(input_ids: list[int]) -> list[Turn]:
        if not input_ids:
            return []
        if not marker_tokens:
            return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]

        # Find all marker positions
        marker_positions = []
        i = 0
        while i <= len(input_ids) - len(marker_tokens):
            if input_ids[i : i + len(marker_tokens)] == marker_tokens:
                marker_positions.append(i)
                i += len(marker_tokens)
            else:
                i += 1

        if not marker_positions:
            return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]

        turns: list[Turn] = []
        for idx, pos in enumerate(marker_positions):
            marker_end = pos + len(marker_tokens)
            prompt_tokens = input_ids[pos:marker_end]

            # Response: from marker end to next marker start (or end of input)
            if idx + 1 < len(marker_positions):
                response_end = marker_positions[idx + 1]
            else:
                response_end = len(input_ids)
            response_tokens = input_ids[marker_end:response_end]

            if not response_tokens:
                continue  # skip markers with no response after them

            turns.append(
                Turn(prompt_tokens=prompt_tokens, response_tokens=response_tokens)
            )

        return turns

    return split


def _detect_assistant_marker(tokenizer) -> str:
    """Auto-detect the assistant role marker from a tokenizer's chat template.

    Checks common patterns in the chat template for assistant markers.
    Falls back to '<|im_start|>assistant' if no template is found.
    """
    common_markers = [
        "<|im_start|>assistant",  # Qwen, Yi, ChatML models
        "<|start_header_id|>assistant<|end_header_id|>",  # Llama-3
        "<|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>",  # Gemma
    ]

    chat_template = getattr(tokenizer, "chat_template", None) or ""
    for marker in common_markers:
        if marker in chat_template:
            return marker

    return "<|im_start|>assistant"
