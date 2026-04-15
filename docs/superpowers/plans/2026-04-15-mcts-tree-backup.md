# MCTS Tree Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCTS tree backup to the RL training loop — rollouts are inserted turn-by-turn into a shared compressed trie, MCTS backup propagates rewards to compute Q-values, which replace GAE as the advantage signal.

**Architecture:** New `customized_areal/tree_search/` package with: TrieNode (pure path indexing, no MCTS stats), Turn dataclass (prompt_tokens/response_tokens split), MCTSTreeStore (holds trie + MCTS stats + cursor-based API), TreeAdvantageComputer (replaces GAE), TreeCheckpointManager (JSON persistence), role-marker-based turn splitter, and config. Cursor-based turn-by-turn insertion: `start_sequence → add_turn → finish_sequence`. Surgical integration into `PPOTrainer.train()`.

**Tech Stack:** Python 3.12+ | PyTorch | JSON (checkpoint) | dataclasses

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `customized_areal/tree_search/__init__.py` | Public exports |
| Create | `customized_areal/tree_search/config.py` | `TreeBackupMode` enum, `TreeBackupConfig` dataclass |
| Create | `customized_areal/tree_search/trie_node.py` | `TrieNode` — turn-level compressed trie node (path indexing only) |
| Create | `customized_areal/tree_search/turn_splitter.py` | `Turn` dataclass, `make_turn_splitter` (role-marker-based) |
| Create | `customized_areal/tree_search/mcts_tree_store.py` | `MCTSTreeStore` — cursor-based API + MCTS backup stats |
| Create | `customized_areal/tree_search/advantage.py` | `TreeAdvantageComputer` — replaces GAE |
| Create | `customized_areal/tree_search/checkpoint.py` | `TreeCheckpointManager` — JSON save/load |
| Create | `tests/test_tree_search/__init__.py` | Test package marker |
| Create | `tests/test_tree_search/test_config.py` | Tests for config |
| Create | `tests/test_tree_search/test_trie_node.py` | Tests for TrieNode |
| Create | `tests/test_tree_search/test_turn_splitter.py` | Tests for turn splitter |
| Create | `tests/test_tree_search/test_mcts_tree_store.py` | Tests for MCTSTreeStore |
| Create | `tests/test_tree_search/test_advantage.py` | Tests for TreeAdvantageComputer |
| Create | `tests/test_tree_search/test_checkpoint.py` | Tests for TreeCheckpointManager |

---

### Task 1: Config

**Files:**
- Create: `customized_areal/tree_search/config.py`
- Create: `tests/test_tree_search/__init__.py`
- Create: `tests/test_tree_search/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_config.py
import pytest
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode


class TestTreeBackupMode:
    def test_off_is_default(self):
        config = TreeBackupConfig()
        assert config.mode == TreeBackupMode.OFF

    def test_enum_values(self):
        assert TreeBackupMode.OFF == "off"
        assert TreeBackupMode.IN_TRAINING == "in_training"
        assert TreeBackupMode.CROSS_TRAINING == "cross_training"

    def test_default_assistant_marker_empty(self):
        config = TreeBackupConfig()
        assert config.assistant_marker == ""

    def test_default_checkpoint_dir_empty(self):
        config = TreeBackupConfig()
        assert config.checkpoint_dir == ""

    def test_custom_values(self):
        config = TreeBackupConfig(
            mode=TreeBackupMode.CROSS_TRAINING,
            assistant_marker="<|im_start|>assistant",
            checkpoint_dir="/tmp/mcts",
        )
        assert config.mode == TreeBackupMode.CROSS_TRAINING
        assert config.assistant_marker == "<|im_start|>assistant"
        assert config.checkpoint_dir == "/tmp/mcts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# customized_areal/tree_search/config.py
from dataclasses import dataclass
from enum import Enum


class TreeBackupMode(str, Enum):
    OFF = "off"
    IN_TRAINING = "in_training"
    CROSS_TRAINING = "cross_training"


@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    assistant_marker: str = ""
    checkpoint_dir: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/__init__.py customized_areal/tree_search/config.py tests/test_tree_search/__init__.py tests/test_tree_search/test_config.py
git commit -m "feat(tree-search): add TreeBackupConfig with assistant_marker field"
```

Note: `__init__.py` should be empty for now (just a package marker). We'll add public exports in Task 7.

---

### Task 2: Turn and TrieNode

**Files:**
- Create: `customized_areal/tree_search/trie_node.py`
- Create: `tests/test_tree_search/test_trie_node.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_trie_node.py
import pytest
from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.turn_splitter import Turn


class TestTrieNodeCreation:
    def test_root_node(self):
        root = TrieNode(tree_id=0)
        assert root.tree_id == 0
        assert root.tokens == []
        assert root.start_idx == -1
        assert root.end_idx == -1
        assert root.children == {}
        assert root.sequence_ids == []
        assert root.ancestors == []
        assert root.nodes == []

    def test_child_node(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        child = root.add_turn(turn, seq_id=0)
        assert child.tokens == [1, 2, 3, 4]
        assert child.ancestors == [root]
        assert 3 in root.children
        assert root.children[3] is child


class TestTrieNodeAddTurn:
    def test_add_single_turn(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2, 3], response_tokens=[4, 5])
        child = root.add_turn(turn, seq_id=0)
        assert 4 in root.children
        assert root.children[4] is child
        assert child.tokens == [1, 2, 3, 4, 5]
        assert 0 in child.sequence_ids
        assert 0 in root.sequence_ids

    def test_add_two_turns_sequentially(self):
        root = TrieNode(tree_id=0)
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn2 = Turn(prompt_tokens=[5, 6], response_tokens=[7, 8])
        child1 = root.add_turn(turn1, seq_id=0)
        child2 = child1.add_turn(turn2, seq_id=0)
        assert 3 in root.children
        assert 7 in child1.children
        assert child2.tokens == [5, 6, 7, 8]
        assert 0 in child2.sequence_ids

    def test_add_turn_shared_prefix_diverges(self):
        root = TrieNode(tree_id=0)
        turn_a1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn_b1 = Turn(prompt_tokens=[1, 2], response_tokens=[5, 6])
        child_a = root.add_turn(turn_a1, seq_id=0)
        child_b = root.add_turn(turn_b1, seq_id=1)
        # Shared prompt means same first token prefix — different response first token
        assert 3 in root.children
        assert 5 in root.children
        assert child_a is root.children[3]
        assert child_b is root.children[5]
        assert 0 in child_a.sequence_ids
        assert 1 in child_b.sequence_ids

    def test_add_turn_empty_response_raises(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[])
        with pytest.raises(ValueError, match="response_tokens must not be empty"):
            root.add_turn(turn, seq_id=0)

    def test_add_turn_existing_child_reuses(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4, 5])
        child1 = root.add_turn(turn, seq_id=0)
        # Same turn again with different seq_id — should reuse child
        child2 = root.add_turn(turn, seq_id=1)
        assert child1 is child2
        assert 0 in child1.sequence_ids
        assert 1 in child1.sequence_ids


class TestTrieNodeGetPathNodes:
    def test_get_path_single_turn(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        root.add_turn(turn, seq_id=0)
        path = root.get_path_nodes(seq_id=0)
        assert len(path) == 1
        assert path[0].tokens == [1, 2, 3, 4]

    def test_get_path_multiple_turns(self):
        root = TrieNode(tree_id=0)
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn2 = Turn(prompt_tokens=[5, 6], response_tokens=[7, 8])
        child1 = root.add_turn(turn1, seq_id=0)
        child1.add_turn(turn2, seq_id=0)
        path = root.get_path_nodes(seq_id=0)
        assert len(path) == 2
        assert path[0].tokens == [1, 2, 3, 4]
        assert path[1].tokens == [5, 6, 7, 8]

    def test_get_path_missing_seq_id_raises(self):
        root = TrieNode(tree_id=0)
        with pytest.raises(KeyError):
            root.get_path_nodes(seq_id=99)


class TestTrieNodeGetTurnBoundaries:
    def test_get_turn_boundaries(self):
        root = TrieNode(tree_id=0)
        turn1 = Turn(prompt_tokens=[1, 2, 3], response_tokens=[4, 5])
        turn2 = Turn(prompt_tokens=[6, 7], response_tokens=[8])
        child1 = root.add_turn(turn1, seq_id=0)
        child1.add_turn(turn2, seq_id=0)
        boundaries = root.get_turn_boundaries(seq_id=0)
        # turn1 has 5 tokens, turn2 has 3 tokens
        assert boundaries == [0, 5, 8]

    def test_get_turn_boundaries_single_turn(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        root.add_turn(turn, seq_id=0)
        boundaries = root.get_turn_boundaries(seq_id=0)
        assert boundaries == [0, 4]

    def test_get_turn_boundaries_missing_seq_id_raises(self):
        root = TrieNode(tree_id=0)
        with pytest.raises(KeyError):
            root.get_turn_boundaries(seq_id=99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_trie_node.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.trie_node'`

- [ ] **Step 3: Write implementation**

First create `turn_splitter.py` with the `Turn` dataclass (needed by `trie_node.py`):

```python
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
```

Then create `trie_node.py`:

```python
# customized_areal/tree_search/trie_node.py
from __future__ import annotations

from dataclasses import dataclass, field

from customized_areal.tree_search.turn_splitter import Turn


@dataclass
class TrieNode:
    """A node in a compressed trie for turn-level MCTS path indexing.

    Each node stores a full turn's tokens (prompt + response concatenated).
    Children are keyed by the first response token. No MCTS statistics —
    those live in MCTSTreeStore.
    """

    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)

    def add_turn(self, turn: Turn, seq_id: int) -> TrieNode:
        """Add a single turn as a child, keyed by first response token.

        Returns the child node (cursor for next turn).
        Tags the child with seq_id. Also tags self with seq_id.
        """
        if not turn.response_tokens:
            raise ValueError("response_tokens must not be empty")
        self.sequence_ids.append(seq_id)
        key = turn.response_tokens[0]
        if key not in self.children:
            combined_tokens = turn.prompt_tokens + turn.response_tokens
            child = TrieNode(
                tree_id=self.tree_id,
                tokens=combined_tokens,
                ancestors=self.ancestors + [self],
            )
            self.children[key] = child
        child = self.children[key]
        if seq_id not in child.sequence_ids:
            child.sequence_ids.append(seq_id)
        return child

    def get_path_nodes(self, seq_id: int) -> list[TrieNode]:
        """Return the non-root nodes on the path for seq_id, root-to-leaf order."""
        if seq_id not in self.sequence_ids:
            raise KeyError(f"seq_id {seq_id} not in this node's sequences")
        nodes: list[TrieNode] = []
        current = self
        while True:
            found = False
            for child in current.children.values():
                if seq_id in child.sequence_ids:
                    nodes.append(child)
                    current = child
                    found = True
                    break
            if not found:
                break
        return nodes

    def get_turn_boundaries(self, seq_id: int) -> list[int]:
        """Return cumulative token positions where turns start/end."""
        path_nodes = self.get_path_nodes(seq_id)
        boundaries = [0]
        cumlen = 0
        for node in path_nodes:
            cumlen += len(node.tokens)
            boundaries.append(cumlen)
        return boundaries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_trie_node.py -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/turn_splitter.py customized_areal/tree_search/trie_node.py tests/test_tree_search/test_trie_node.py
git commit -m "feat(tree-search): add TrieNode and Turn dataclass with cursor-based add_turn API"
```

---

### Task 3: Turn Splitter

**Files:**
- Modify: `customized_areal/tree_search/turn_splitter.py` (add `make_turn_splitter`)
- Create: `tests/test_tree_search/test_turn_splitter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_turn_splitter.py
import pytest
from customized_areal.tree_search.turn_splitter import Turn, make_turn_splitter


class FakeTokenizer:
    """Minimal tokenizer stub for testing.

    Maps characters to their ord() values. Multi-char strings encode
    character-by-character.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class TestTurn:
    def test_turn_creation(self):
        turn = Turn(prompt_tokens=[1, 2, 3], response_tokens=[4, 5])
        assert turn.prompt_tokens == [1, 2, 3]
        assert turn.response_tokens == [4, 5]


class TestMakeTurnSplitter:
    def test_single_assistant_turn(self):
        # "<a>hello<b>world" -> assistant marker is "<a>"
        # Input: [60, 97, 62, 104, 101, 108, 108, 111, 60, 98, 62, 119, 111, 114, 108, 100]
        # marker "<a>" = [60, 97, 62]
        # After marker: [104, 101, 108, 108, 111, 60, 98, 62, 119, 111, 114, 108, 100]
        # No second marker, so response goes to end
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [60, 97, 62, 104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == [60, 97, 62]
        assert turns[0].response_tokens == [104, 101, 108, 108, 111]

    def test_two_assistant_turns(self):
        # Two "<a>" markers create two turns
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        # "<a>yes<a>no" -> [60,97,62,121,101,115,60,97,62,110,111]
        input_ids = [60, 97, 62, 121, 101, 115, 60, 97, 62, 110, 111]
        turns = splitter(input_ids)
        assert len(turns) == 2
        # First turn: prompt includes marker, response is tokens after marker until next marker
        assert turns[0].prompt_tokens == [60, 97, 62]
        assert turns[0].response_tokens == [121, 101, 115]
        assert turns[1].prompt_tokens == [60, 97, 62]
        assert turns[1].response_tokens == [110, 111]

    def test_no_assistant_marker(self):
        # No marker found — entire input is one turn with empty prompt
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == []
        assert turns[0].response_tokens == [104, 101, 108, 108, 111]

    def test_marker_at_start_only(self):
        # "<a>response" with no more markers
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [60, 97, 62, 114, 101, 115, 112]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == [60, 97, 62]
        assert turns[0].response_tokens == [114, 101, 115, 112]

    def test_marker_at_end_no_response(self):
        # "prompt<a>" — last marker has no response after it, skip it
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [112, 114, 111, 109, 112, 116, 60, 97, 62]
        turns = splitter(input_ids)
        # The marker at the end has no response tokens, so it's skipped
        assert len(turns) == 0

    def test_multi_token_marker(self):
        # Marker "<ab>" = [60, 97, 98, 62]
        splitter = make_turn_splitter(FakeTokenizer(), "<ab>")
        # "<ab>hello" = [60, 97, 98, 62, 104, 101, 108, 108, 111]
        input_ids = [60, 97, 98, 62, 104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == [60, 97, 98, 62]
        assert turns[0].response_tokens == [104, 101, 108, 108, 111]

    def test_empty_input(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        turns = splitter([])
        assert turns == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_turn_splitter.py -v`
Expected: FAIL — `AttributeError: module 'customized_areal.tree_search.turn_splitter' has no attribute 'make_turn_splitter'`

- [ ] **Step 3: Write implementation**

```python
# customized_areal/tree_search/turn_splitter.py (updated — add make_turn_splitter to existing file)
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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
    - prompt_tokens = everything from start (or prev response end) to end of marker
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

            turns.append(Turn(prompt_tokens=prompt_tokens, response_tokens=response_tokens))

        return turns

    return split


def _detect_assistant_marker(tokenizer) -> str:
    """Auto-detect the assistant role marker from a tokenizer's chat template.

    Checks common patterns in the chat template for assistant markers.
    Falls back to '<|im_start|>assistant' if no template is found.
    """
    # Check for common markers in order of specificity
    common_markers = [
        "<|im_start|>assistant",  # Qwen, Yi, ChatML models
        "<|start_header_id|>assistant<|end_header_id|>",  # Llama-3
        "<|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>",  # Gemma
        "<|assistant|>",  # Some models
    ]

    # Try to inspect the chat template
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    for marker in common_markers:
        if marker in chat_template:
            return marker

    # Default fallback
    return "<|im_start|>assistant"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_turn_splitter.py -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/turn_splitter.py tests/test_tree_search/test_turn_splitter.py
git commit -m "feat(tree-search): add role-marker-based make_turn_splitter with Turn dataclass"
```

---

### Task 4: MCTSTreeStore

**Files:**
- Create: `customized_areal/tree_search/mcts_tree_store.py`
- Create: `tests/test_tree_search/test_mcts_tree_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_mcts_tree_store.py
import torch
import pytest
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import Turn


def _two_turn_splitter(input_ids: list[int]) -> list[Turn]:
    """Simple splitter for testing: splits at token 10 (newline), first half is prompt, second is response."""
    # Find the first occurrence of token 10
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestMCTSTreeStoreStartSequence:
    def test_start_sequence_creates_root(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        assert seq_id == 0
        assert "q1" in store.trees
        root = store.trees["q1"]
        assert root.tokens == []
        assert root.tree_id == 0

    def test_start_sequence_increments_seq_id(self):
        store = MCTSTreeStore(_two_turn_splitter)
        id0 = store.start_sequence("q1")
        id1 = store.start_sequence("q1")
        id2 = store.start_sequence("q2")
        assert id0 == 0
        assert id1 == 1
        assert id2 == 2

    def test_start_sequence_sets_cursor(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        assert ("q1", seq_id) in store._cursors
        assert store._cursors[("q1", seq_id)] is store.trees["q1"]


class TestMCTSTreeStoreAddTurn:
    def test_add_single_turn(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn = Turn(prompt_tokens=[1, 2, 10], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn)
        root = store.trees["q1"]
        assert 3 in root.children
        child = root.children[3]
        assert child.tokens == [1, 2, 10, 3, 4]
        assert 0 in child.sequence_ids

    def test_add_two_turns(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        turn2 = Turn(prompt_tokens=[5, 6], response_tokens=[7, 8])
        store.add_turn("q1", seq_id, turn1)
        store.add_turn("q1", seq_id, turn2)
        root = store.trees["q1"]
        child1 = root.children[3]
        child2 = child1.children[7]
        assert child2.tokens == [5, 6, 7, 8]

    def test_add_turn_advances_cursor(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn1 = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn1)
        cursor = store._cursors[("q1", seq_id)]
        assert cursor is root.children[3]


class TestMCTSTreeStoreFinishSequence:
    def test_finish_sequence_runs_backup(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn)
        store.finish_sequence("q1", seq_id, reward=1.0)
        # Verify MCTS stats were updated
        root = store.trees["q1"]
        child = root.children[3]
        # Both root and child should have visit_count=1, q_value=1.0
        assert store._visit_counts[("q1", id(root))] == 1
        assert store._q_values[("q1", id(root))] == 1.0
        assert store._visit_counts[("q1", id(child))] == 1
        assert store._q_values[("q1", id(child))] == 1.0

    def test_finish_sequence_clears_cursor(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.start_sequence("q1")
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[3, 4])
        store.add_turn("q1", seq_id, turn)
        store.finish_sequence("q1", seq_id, reward=1.0)
        assert ("q1", seq_id) not in store._cursors


class TestMCTSTreeStoreInsertTrajectory:
    def test_insert_trajectory_convenience(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        root = store.trees["q1"]
        assert 3 in root.children
        # Backup should have run
        assert store._visit_counts[("q1", id(root))] == 1

    def test_insert_two_trajectories_shared_prefix(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q1", [1, 2, 10, 3, 5], reward=0.5)
        root = store.trees["q1"]
        # Two sequences through root
        assert 0 in root.sequence_ids
        assert 1 in root.sequence_ids


class TestMCTSTreeStoreInsertBatch:
    def test_insert_batch(self):
        store = MCTSTreeStore(_two_turn_splitter)
        trajectories = [
            {
                "input_ids": torch.tensor([1, 2, 10, 3, 4]),
                "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
                "rewards": torch.tensor([1.0]),
            },
            {
                "input_ids": torch.tensor([1, 2, 10, 3, 5]),
                "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
                "rewards": torch.tensor([0.5]),
            },
        ]
        store.insert_batch(trajectories)
        assert "_mcts_seq_id" in trajectories[0]
        assert "_mcts_query_id" in trajectories[0]


class TestMCTSTreeStoreGetAdvantages:
    def test_get_advantages_single_trajectory(self):
        store = MCTSTreeStore(_two_turn_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=2.0)
        advantages = store.get_advantages("q1", seq_id)
        # One turn with q_value=2.0, applied to all tokens
        assert advantages.shape == torch.Size([5])
        assert torch.allclose(advantages, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]))


class TestMCTSTreeStoreClear:
    def test_clear(self):
        store = MCTSTreeStore(_two_turn_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        store.clear()
        assert len(store.trees) == 0
        assert store._next_seq_id == 0
        assert len(store._cursors) == 0
        assert len(store._visit_counts) == 0
        assert len(store._q_values) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.mcts_tree_store'`

- [ ] **Step 3: Write implementation**

```python
# customized_areal/tree_search/mcts_tree_store.py
from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch

from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.turn_splitter import Turn


def _get_query_id(traj: dict[str, Any]) -> str:
    """Derive a query ID from the prompt tokens in a trajectory."""
    loss_mask = traj["loss_mask"]
    input_ids = traj["input_ids"]
    prompt_tokens = input_ids[loss_mask == 0].tolist()
    prompt_str = ",".join(str(t) for t in prompt_tokens)
    return hashlib.md5(prompt_str.encode()).hexdigest()


class MCTSTreeStore:
    def __init__(self, turn_splitter: Callable[[list[int]], list[Turn]]):
        self.trees: dict[str, TrieNode] = {}  # query_id -> root node
        self.turn_splitter = turn_splitter
        self._next_seq_id: int = 0

        # Cursor state — tracks current position per (query_id, seq_id)
        self._cursors: dict[tuple[str, int], TrieNode] = {}

        # MCTS statistics — keyed by (query_id, node_id) -> per-node stats
        # node_id is id(node) for now; will be replaced with stable index for checkpointing
        self._visit_counts: dict[tuple[str, int], int] = {}
        self._total_values: dict[tuple[str, int], float] = {}
        self._q_values: dict[tuple[str, int], float] = {}

    def start_sequence(self, query_id: str) -> int:
        """Create root if needed, assign a seq_id, set cursor at root."""
        tree_idx = len(self.trees)
        root = self.trees.setdefault(query_id, TrieNode(tree_id=tree_idx))
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        root.sequence_ids.append(seq_id)
        self._cursors[(query_id, seq_id)] = root
        return seq_id

    def add_turn(self, query_id: str, seq_id: int, turn: Turn) -> None:
        """Add a single turn at the cursor position, advance cursor."""
        cursor = self._cursors[(query_id, seq_id)]
        child = cursor.add_turn(turn, seq_id)
        self._cursors[(query_id, seq_id)] = child

    def finish_sequence(self, query_id: str, seq_id: int, reward: float) -> None:
        """Run MCTS backup along the completed path, clear cursor."""
        self._backup(query_id, seq_id, reward)
        del self._cursors[(query_id, seq_id)]

    def _backup(self, query_id: str, seq_id: int, reward: float) -> None:
        """Walk from leaf to root, updating MCTS stats at each node."""
        root = self.trees[query_id]
        path_nodes = root.get_path_nodes(seq_id)
        # Walk from leaf to root (ancestors + leaf itself)
        leaf = path_nodes[-1] if path_nodes else None
        all_nodes = list(reversed(leaf.ancestors)) + [leaf] if leaf else [root]
        # Include root
        all_nodes = [root] + (path_nodes if path_nodes else [])
        for node in all_nodes:
            key = (query_id, id(node))
            self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
            self._total_values[key] = self._total_values.get(key, 0.0) + reward
            self._q_values[key] = self._total_values[key] / self._visit_counts[key]

    def insert_trajectory(self, query_id: str, input_ids: list[int], reward: float) -> int:
        """Convenience: split -> start_sequence -> add_turn loop -> finish_sequence."""
        turns = self.turn_splitter(input_ids)
        seq_id = self.start_sequence(query_id)
        for turn in turns:
            self.add_turn(query_id, seq_id, turn)
        self.finish_sequence(query_id, seq_id, reward)
        return seq_id

    def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
        """Batch version — group trajectories by query, insert each group."""
        for traj in trajectories:
            query_id = _get_query_id(traj)
            input_ids = traj["input_ids"].tolist()
            reward = traj["rewards"].item() if traj["rewards"].dim() > 0 else traj["rewards"].item()
            seq_id = self.insert_trajectory(query_id, input_ids, reward)
            traj["_mcts_seq_id"] = seq_id
            traj["_mcts_query_id"] = query_id

    def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Get Q-values per turn, expand to per-token advantages."""
        root = self.trees[query_id]
        path_nodes = root.get_path_nodes(seq_id)
        boundaries = root.get_turn_boundaries(seq_id)
        total_len = boundaries[-1]
        advantages = torch.zeros(total_len)
        for i, node in enumerate(path_nodes):
            key = (query_id, id(node))
            q_val = self._q_values.get(key, 0.0)
            advantages[boundaries[i] : boundaries[i + 1]] = q_val
        return advantages

    def clear(self) -> None:
        """Reset all trees, stats, and cursors."""
        self.trees.clear()
        self._next_seq_id = 0
        self._cursors.clear()
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add MCTSTreeStore with cursor-based API and MCTS backup"
```

---

### Task 5: TreeAdvantageComputer

**Files:**
- Create: `customized_areal/tree_search/advantage.py`
- Create: `tests/test_tree_search/test_advantage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_advantage.py
import torch
import pytest
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.turn_splitter import Turn


def _simple_splitter(input_ids: list[int]) -> list[Turn]:
    """Split at token 10 — everything before is prompt, everything after is response."""
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        traj = {
            "input_ids": torch.tensor([1, 2, 10, 3, 4]),
            "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
            "rewards": torch.tensor([2.0]),
        }
        store.insert_batch([traj])
        computer.compute([traj])
        assert "advantages" in traj
        assert "returns" in traj
        # Advantages should be zeroed for prompt tokens
        assert torch.allclose(traj["advantages"][:2], torch.zeros(2))
        # Response tokens should have q_value
        assert torch.allclose(traj["advantages"][2:], torch.full((3,), 2.0))

    def test_compute_returns_equal_advantages(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        traj = {
            "input_ids": torch.tensor([1, 10, 3]),
            "loss_mask": torch.tensor([0, 0, 1]),
            "rewards": torch.tensor([1.0]),
        }
        store.insert_batch([traj])
        computer.compute([traj])
        assert torch.allclose(traj["returns"], traj["advantages"])

    def test_compute_two_trajectories(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        traj1 = {
            "input_ids": torch.tensor([1, 2, 10, 3, 4]),
            "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
            "rewards": torch.tensor([2.0]),
        }
        traj2 = {
            "input_ids": torch.tensor([5, 6, 10, 7, 8]),
            "loss_mask": torch.tensor([0, 0, 0, 1, 1]),
            "rewards": torch.tensor([0.5]),
        }
        store.insert_batch([traj1, traj2])
        computer.compute([traj1, traj2])
        assert "advantages" in traj1
        assert "advantages" in traj2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_advantage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.advantage'`

- [ ] **Step 3: Write implementation**

```python
# customized_areal/tree_search/advantage.py
from __future__ import annotations

from typing import Any

import torch

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


class TreeAdvantageComputer:
    def __init__(self, tree_store: MCTSTreeStore):
        self.tree_store = tree_store

    def compute(self, trajectories: list[dict[str, Any]]) -> None:
        """Replace GAE advantages with tree Q-values. Mutates trajectories in-place."""
        for traj in trajectories:
            query_id = traj["_mcts_query_id"]
            seq_id = traj["_mcts_seq_id"]
            tree_advantages = self.tree_store.get_advantages(query_id, seq_id)

            input_ids = traj["input_ids"]
            seq_len = input_ids.shape[0]

            # Pad or trim tree_advantages to match trajectory length
            advantages = torch.zeros(seq_len, dtype=torch.float32)
            common_len = min(seq_len, tree_advantages.shape[0])
            advantages[:common_len] = tree_advantages[:common_len]

            # Mask prompt tokens — advantages only for response tokens
            response_mask = traj["loss_mask"].bool()
            advantages = advantages * response_mask.float()

            # Match trajectory shape: [1, seq_len] or [group_size, seq_len]
            if input_ids.dim() > 1:
                advantages = advantages.unsqueeze(0)

            traj["advantages"] = advantages
            traj["returns"] = advantages.clone()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_advantage.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/advantage.py tests/test_tree_search/test_advantage.py
git commit -m "feat(tree-search): add TreeAdvantageComputer replacing GAE with tree Q-values"
```

---

### Task 6: TreeCheckpointManager

**Files:**
- Create: `customized_areal/tree_search/checkpoint.py`
- Create: `tests/test_tree_search/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_checkpoint.py
import json
import os
import pytest
from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.turn_splitter import Turn


def _simple_splitter(input_ids: list[int]) -> list[Turn]:
    try:
        split_pos = input_ids.index(10)
        return [Turn(prompt_tokens=input_ids[:split_pos], response_tokens=input_ids[split_pos:])]
    except ValueError:
        return [Turn(prompt_tokens=[], response_tokens=list(input_ids))]


class TestTreeCheckpointManager:
    def test_save_and_load(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=2.0)
        store.insert_trajectory("q1", [1, 2, 10, 5, 6], reward=0.0)
        store.insert_trajectory("q2", [7, 8], reward=1.0)

        manager.save(store)
        assert manager.exists()

        loaded = manager.load(_simple_splitter)
        assert len(loaded.trees) == 2
        assert "q1" in loaded.trees
        assert "q2" in loaded.trees

    def test_exists_false_when_no_dir(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path / "nonexistent"))
        assert not manager.exists()

    def test_save_creates_directory(self, tmp_path):
        save_dir = str(tmp_path / "new_dir")
        manager = TreeCheckpointManager(save_dir)
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        manager.save(store)
        assert os.path.isdir(os.path.join(save_dir, "mcts_trees"))

    def test_load_preserves_seq_id_counter(self, tmp_path):
        manager = TreeCheckpointManager(str(tmp_path))
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        store.insert_trajectory("q2", [3, 4], reward=0.5)
        manager.save(store)

        loaded = manager.load(_simple_splitter)
        seq_id = loaded.insert_trajectory("q3", [5, 6], reward=1.0)
        assert seq_id == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.checkpoint'`

- [ ] **Step 3: Write implementation**

```python
# customized_areal/tree_search/checkpoint.py
from __future__ import annotations

import json
import os
from typing import Callable

from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.turn_splitter import Turn


class TreeCheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, "mcts_trees")

    def exists(self) -> bool:
        return os.path.isdir(self.save_dir) and os.path.isfile(
            os.path.join(self.save_dir, "metadata.json")
        )

    def save(self, tree_store: MCTSTreeStore) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        for query_id, root in tree_store.trees.items():
            tree_data = {"root": self._serialize_node(root)}
            filepath = os.path.join(self.save_dir, f"query_{query_id}.json")
            with open(filepath, "w") as f:
                json.dump(tree_data, f)
        metadata = {"next_seq_id": tree_store._next_seq_id}
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, turn_splitter: Callable[[list[int]], list[Turn]]) -> MCTSTreeStore:
        store = MCTSTreeStore(turn_splitter)
        with open(os.path.join(self.save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        store._next_seq_id = metadata["next_seq_id"]
        for filename in os.listdir(self.save_dir):
            if not filename.startswith("query_") or not filename.endswith(".json"):
                continue
            query_id = filename[len("query_") : -len(".json")]
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath) as f:
                tree_data = json.load(f)
            root = self._deserialize_node(tree_data["root"], parent=None, tree_id=len(store.trees))
            root.sequence_ids = list(root.sequence_ids)  # ensure list type
            store.trees[query_id] = root
        return store

    def _serialize_node(self, node: TrieNode) -> dict:
        return {
            "tree_id": node.tree_id,
            "start_idx": node.start_idx,
            "end_idx": node.end_idx,
            "tokens": node.tokens,
            "sequence_ids": list(node.sequence_ids),
            "children": {
                str(key): self._serialize_node(child)
                for key, child in node.children.items()
            },
        }

    def _deserialize_node(self, data: dict, parent: TrieNode | None, tree_id: int) -> TrieNode:
        node = TrieNode(
            tree_id=tree_id,
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            tokens=data["tokens"],
            sequence_ids=data["sequence_ids"],
        )
        if parent is not None:
            node.ancestors = parent.ancestors + [parent]
        for key_str, child_data in data["children"].items():
            key = int(key_str)
            child = self._deserialize_node(child_data, parent=node, tree_id=tree_id)
            node.children[key] = child
        return node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_checkpoint.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/checkpoint.py tests/test_tree_search/test_checkpoint.py
git commit -m "feat(tree-search): add TreeCheckpointManager with JSON save/load"
```

---

### Task 7: Package __init__.py

**Files:**
- Modify: `customized_areal/tree_search/__init__.py`

- [ ] **Step 1: Write the __init__.py**

```python
# customized_areal/tree_search/__init__.py
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.trie_node import TrieNode
from customized_areal.tree_search.turn_splitter import Turn, make_turn_splitter

__all__ = [
    "MCTSTreeStore",
    "TreeAdvantageComputer",
    "TreeBackupConfig",
    "TreeBackupMode",
    "TreeCheckpointManager",
    "TrieNode",
    "Turn",
    "make_turn_splitter",
]
```

- [ ] **Step 2: Verify imports work**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "from customized_areal.tree_search import TreeBackupConfig, TreeBackupMode, TrieNode, Turn, MCTSTreeStore, TreeAdvantageComputer, TreeCheckpointManager, make_turn_splitter; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Run all tree_search tests**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/ -v`
Expected: PASS (all tests green)

- [ ] **Step 4: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/__init__.py
git commit -m "feat(tree-search): add package __init__.py with public exports"
```

---

### Task 8: PPOTrainer Integration

**Files:**
- Modify: `areal/trainer/rl_trainer.py`

- [ ] **Step 1: Read the current PPOTrainer to find exact insertion points**

Read `areal/trainer/rl_trainer.py` — find `__init__` signature and the `train()` method where advantage computation happens. Identify the exact lines where:
  1. Tree backup config parameter should be added to `__init__`
  2. Critic values computation happens (to conditionally skip in tree mode)
  3. `compute_advantages` is called (to conditionally replace)
  4. Checkpoint save happens (to add tree checkpoint save)

- [ ] **Step 2: Add imports at top of rl_trainer.py**

Add the following after the existing imports block:

```python
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.turn_splitter import make_turn_splitter
```

- [ ] **Step 3: Add tree_backup_config parameter to PPOTrainer.__init__**

In `PPOTrainer.__init__`, add `tree_backup_config: TreeBackupConfig | None = None` parameter. After the line where `self.tokenizer` is set, add:

```python
# Tree backup setup
self.tree_backup_config = tree_backup_config or TreeBackupConfig()
if self.tree_backup_config.mode != TreeBackupMode.OFF:
    turn_splitter = make_turn_splitter(
        self.tokenizer, self.tree_backup_config.assistant_marker
    )
    self.tree_store = MCTSTreeStore(turn_splitter)
    self.tree_advantage_computer = TreeAdvantageComputer(self.tree_store)
    self.tree_checkpoint_manager = TreeCheckpointManager(
        self.tree_backup_config.checkpoint_dir
    )
    if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
        if self.tree_checkpoint_manager.exists():
            self.tree_store = self.tree_checkpoint_manager.load(turn_splitter)
```

- [ ] **Step 4: Branch the train loop for advantage computation**

In `PPOTrainer.train()`, find the existing `compute_advantages` call and the critic values computation. Replace with conditional branching:

Find the critic values block and wrap it:
```python
if self.critic is not None and self.tree_backup_config.mode == TreeBackupMode.OFF:
    values = self.critic.compute_values(rollout_batch)
    for traj, v in zip(rollout_batch, values):
        traj["values"] = v
```

Find the `compute_advantages` call and replace:
```python
if self.tree_backup_config.mode != TreeBackupMode.OFF:
    self.tree_store.insert_batch(rollout_batch)
    self.tree_advantage_computer.compute(rollout_batch)
    adv_batch = rollout_batch
else:
    adv_batch = self.actor.compute_advantages(rollout_batch)
```

- [ ] **Step 5: Add checkpoint save hook**

In the checkpoint save method, add after existing model checkpoint logic:

```python
if self.tree_backup_config.mode == TreeBackupMode.CROSS_TRAINING:
    self.tree_checkpoint_manager.save(self.tree_store)
```

- [ ] **Step 6: Verify integration doesn't break existing tests**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/ -v -k "not gpu" --timeout=30`
Expected: All existing tests still pass

- [ ] **Step 7: Run all tree_search tests again**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/ -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add areal/trainer/rl_trainer.py
git commit -m "feat(trainer): integrate MCTS tree backup into PPOTrainer train loop"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: Each component in the spec (TrieNode, Turn, MCTSTreeStore, TreeAdvantageComputer, TreeCheckpointManager, config, turn_splitter, RLTrainer integration) has a corresponding task.
- [x] **Placeholder scan**: No TBD, TODO, or vague steps. Every step has complete code.
- [x] **Type consistency**: `add_turn` takes `Turn` and returns `TrieNode`. `MCTSTreeStore` uses cursor-based API consistently. `insert_batch` adds `_mcts_seq_id` and `_mcts_query_id`, which `compute` reads. `get_advantages` returns `torch.Tensor` as expected by `TreeAdvantageComputer`.
- [x] **Config field**: `assistant_marker` replaces `turn_delimiter` throughout.
- [x] **TrieNode has no MCTS stats**: Backup stats are in `MCTSTreeStore._visit_counts` etc., keyed by `(query_id, id(node))`.
- [x] **Cursor-based API**: `start_sequence`, `add_turn`, `finish_sequence` in MCTSTreeStore, plus convenience `insert_trajectory`.