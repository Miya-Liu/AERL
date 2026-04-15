# customized_areal/tree_search/turn_splitter.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    """A structured turn with a prompt/response split.

    prompt_tokens: shared context tokens (no branching)
    response_tokens: assistant output tokens (branching point)
    """

    prompt_tokens: list[int]
    response_tokens: list[int]
