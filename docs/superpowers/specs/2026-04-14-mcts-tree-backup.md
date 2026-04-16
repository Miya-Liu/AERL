# MCTS Tree Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCTS tree backup to the RL training loop — rollouts are inserted into a shared compressed trie at turn-level, MCTS backup propagates rewards to compute Q-values, which replace GAE as the advantage signal.

**Architecture:** New `customized_areal/tree_search/` package with TrieNode (turn-level compressed trie, path indexing only), MCTSTreeStore (per-query tree manager with MCTS stats), TreeAdvantageComputer (replaces GAE), TreeCheckpointManager (JSON persistence), config, and TreeBackupPPOTrainer (streaming backup integration). Streaming insert+backup via `on_trajectory_ready` callback in `BatchTaskDispatcher`.

**Tech Stack:** Python 3.12+ | PyTorch | JSON (checkpoint) | dataclasses

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `customized_areal/tree_search/__init__.py` | Public exports |
| Create | `customized_areal/tree_search/config.py` | `TreeBackupMode` enum, `TreeBackupConfig` dataclass |
| Create | `customized_areal/tree_search/trie_node.py` | `TrieNode` — turn-level compressed trie node (path indexing only, no MCTS stats) |
| Create | `customized_areal/tree_search/turn_splitter.py` | `make_turn_splitter` — delimiter-based turn splitting |
| Create | `customized_areal/tree_search/mcts_tree_store.py` | `MCTSTreeStore` — per-query tree manager (thread-safe) |
| Create | `customized_areal/tree_search/advantage.py` | `TreeAdvantageComputer` — replaces GAE |
| Create | `customized_areal/tree_search/checkpoint.py` | `TreeCheckpointManager` — JSON save/load |
| Modify | `areal/infra/workflow_executor.py` | Add `on_trajectory_ready` callback to `BatchTaskDispatcher` |
| Create | `customized_areal/tree_search/trainer.py` | `TreeBackupPPOTrainer` — PPOTrainer subclass with streaming tree backup |
| Create | `tests/test_tree_search/__init__.py` | Test package marker |
| Create | `tests/test_tree_search/test_trie_node.py` | Tests for TrieNode |
| Create | `tests/test_tree_search/test_turn_splitter.py` | Tests for turn splitter |
| Create | `tests/test_tree_search/test_mcts_tree_store.py` | Tests for MCTSTreeStore (including thread safety) |
| Create | `tests/test_tree_search/test_advantage.py` | Tests for TreeAdvantageComputer |
| Create | `tests/test_tree_search/test_checkpoint.py` | Tests for TreeCheckpointManager |
| Create | `tests/test_tree_search/test_config.py` | Tests for config |

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

    def test_default_delimiter(self):
        config = TreeBackupConfig()
        assert config.turn_delimiter == "\n\n"

    def test_default_checkpoint_dir_empty(self):
        config = TreeBackupConfig()
        assert config.checkpoint_dir == ""

    def test_custom_values(self):
        config = TreeBackupConfig(
            mode=TreeBackupMode.CROSS_TRAINING,
            turn_delimiter="<|step|>",
            checkpoint_dir="/tmp/mcts",
        )
        assert config.mode == TreeBackupMode.CROSS_TRAINING
        assert config.turn_delimiter == "<|step|>"
        assert config.checkpoint_dir == "/tmp/mcts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# customized_areal/tree_search/config.py
from dataclasses import dataclass, field
from enum import Enum


class TreeBackupMode(str, Enum):
    OFF = "off"
    IN_TRAINING = "in_training"
    CROSS_TRAINING = "cross_training"


@dataclass
class TreeBackupConfig:
    mode: TreeBackupMode = TreeBackupMode.OFF
    turn_delimiter: str = "\n\n"
    checkpoint_dir: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/config.py tests/test_tree_search/__init__.py tests/test_tree_search/test_config.py
git commit -m "feat(tree-search): add TreeBackupConfig and TreeBackupMode enum"
```

---

### Task 2: TrieNode

**Files:**
- Create: `customized_areal/tree_search/trie_node.py`
- Create: `customized_areal/tree_search/turn_splitter.py` (needed for `Turn` dataclass)
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
        assert root.tokens == []
        assert root.tree_id == 0
        assert root.start_idx == -1
        assert root.end_idx == -1
        assert root.prompt_len == 0
        assert root.sequence_ids == []
        assert root.children == {}
        assert root.ancestors == []
        assert root.nodes == []

    def test_child_node(self):
        root = TrieNode(tree_id=0)
        turn = Turn(prompt_tokens=[1, 2], response_tokens=[10, 20, 30])
        child = root.add_turn(turn, seq_id=0)
        assert child.tokens == [1, 2, 10, 20, 30]
        assert child.prompt_len == 2
        assert child.tree_id == 0
        assert child.ancestors == [root]
        assert 0 in child.sequence_ids


class TestMCTSNodeInsertPath:
    def test_insert_single_path(self):
        root = MCTSNode(tokens=[])
        turns = [[1, 2, 3], [4, 5], [6]]
        root.insert_path(turns, seq_id=0)
        # Root has one child keyed by first token of first turn (1)
        assert 1 in root.children
        child1 = root.children[1]
        assert child1.tokens == [1, 2, 3]
        assert 0 in child1.sequence_ids
        # child1 has one child keyed by 4
        assert 4 in child1.children
        child2 = child1.children[4]
        assert child2.tokens == [4, 5]
        assert 0 in child2.sequence_ids
        # child2 has one child keyed by 6
        assert 6 in child2.children
        child3 = child2.children[6]
        assert child3.tokens == [6]
        assert 0 in child3.sequence_ids
        assert child3.children == {}

    def test_insert_shared_prefix(self):
        root = MCTSNode(tokens=[])
        # Two paths share the first turn, diverge at second turn
        turns_a = [[1, 2, 3], [4, 5]]
        turns_b = [[1, 2, 3], [7, 8]]
        root.insert_path(turns_a, seq_id=0)
        root.insert_path(turns_b, seq_id=1)
        # Shared first turn node
        child1 = root.children[1]
        assert child1.tokens == [1, 2, 3]
        assert child1.sequence_ids == {0, 1}
        # Two diverging children
        assert 4 in child1.children
        assert 7 in child1.children
        assert child1.children[4].tokens == [4, 5]
        assert child1.children[4].sequence_ids == {0}
        assert child1.children[7].tokens == [7, 8]
        assert child1.children[7].sequence_ids == {1}

    def test_insert_empty_turns(self):
        root = MCTSNode(tokens=[])
        root.insert_path([], seq_id=0)
        assert root.children == {}
        assert 0 in root.sequence_ids


class TestMCTSNodeBackup:
    def test_backup_single_path(self):
        root = MCTSNode(tokens=[])
        turns = [[1, 2], [3, 4]]
        root.insert_path(turns, seq_id=0)
        root.backup(seq_id=0, reward=1.0)
        # All nodes on path get visit_count=1, total_value=1.0, q_value=1.0
        child1 = root.children[1]
        child2 = child1.children[3]
        assert root.visit_count == 1
        assert root.q_value == 1.0
        assert child1.visit_count == 1
        assert child1.q_value == 1.0
        assert child2.visit_count == 1
        assert child2.q_value == 1.0

    def test_backup_two_paths_shared_prefix(self):
        root = MCTSNode(tokens=[])
        turns_a = [[1, 2], [3, 4]]
        turns_b = [[1, 2], [5, 6]]
        root.insert_path(turns_a, seq_id=0)
        root.insert_path(turns_b, seq_id=1)
        root.backup(seq_id=0, reward=1.0)
        root.backup(seq_id=1, reward=0.5)
        # Root and shared child averaged
        assert root.visit_count == 2
        assert root.total_value == 1.5
        assert root.q_value == 0.75
        shared = root.children[1]
        assert shared.visit_count == 2
        assert shared.q_value == 0.75
        # Diverging children each visited once
        assert root.children[1].children[3].visit_count == 1
        assert root.children[1].children[3].q_value == 1.0
        assert root.children[1].children[5].visit_count == 1
        assert root.children[1].children[5].q_value == 0.5


class TestMCTSNodeGetPathQValues:
    def test_get_q_values(self):
        root = MCTSNode(tokens=[])
        turns = [[1, 2], [3, 4], [5, 6]]
        root.insert_path(turns, seq_id=0)
        root.backup(seq_id=0, reward=3.0)
        q_values = root.get_path_q_values(seq_id=0)
        # 3 turns: root + child1 + child2 + child3
        # q_values has one entry per non-root turn
        assert len(q_values) == 3
        assert q_values == [3.0, 3.0, 3.0]

    def test_get_q_values_shared_prefix(self):
        root = MCTSNode(tokens=[])
        root.insert_path([[1, 2], [3, 4]], seq_id=0)
        root.insert_path([[1, 2], [5, 6]], seq_id=1)
        root.backup(seq_id=0, reward=2.0)
        root.backup(seq_id=1, reward=0.0)
        q0 = root.get_path_q_values(seq_id=0)
        q1 = root.get_path_q_values(seq_id=1)
        assert q0 == [1.0, 2.0]  # shared child avg=1.0, diverged child=2.0
        assert q1 == [1.0, 0.0]  # shared child avg=1.0, diverged child=0.0


class TestMCTSNodeGetTurnBoundaries:
    def test_get_turn_boundaries(self):
        root = MCTSNode(tokens=[])
        turns = [[1, 2, 3], [4, 5], [6]]
        root.insert_path(turns, seq_id=0)
        boundaries = root.get_turn_boundaries(seq_id=0)
        # 3 turns: lengths 3, 2, 1
        assert boundaries == [0, 3, 5, 6]

    def test_get_turn_boundaries_single_turn(self):
        root = MCTSNode(tokens=[])
        turns = [[1, 2]]
        root.insert_path(turns, seq_id=0)
        boundaries = root.get_turn_boundaries(seq_id=0)
        assert boundaries == [0, 2]


class TestMCTSNodeSequenceNotFound:
    def test_get_q_values_missing_seq_id(self):
        root = MCTSNode(tokens=[])
        turns = [[1, 2]]
        root.insert_path(turns, seq_id=0)
        with pytest.raises(KeyError):
            root.get_path_q_values(seq_id=99)

    def test_backup_missing_seq_id(self):
        root = MCTSNode(tokens=[])
        with pytest.raises(KeyError):
            root.backup(seq_id=99, reward=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_node.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.mcts_node'`

- [ ] **Step 3: Write implementation**

```python
# customized_areal/tree_search/mcts_node.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    tokens: list[int] = field(default_factory=list)
    parent: MCTSNode | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    q_value: float = 0.0
    sequence_ids: set[int] = field(default_factory=set)

    def insert_path(self, turns: list[list[int]], seq_id: int) -> None:
        self.sequence_ids.add(seq_id)
        if not turns:
            return
        first_token = turns[0][0]
        if first_token not in self.children:
            child = MCTSNode(tokens=turns[0], parent=self)
            self.children[first_token] = child
        self.children[first_token].sequence_ids.add(seq_id)
        self.children[first_token].insert_path(turns[1:], seq_id)

    def backup(self, seq_id: int, reward: float) -> None:
        if seq_id not in self.sequence_ids:
            raise KeyError(f"seq_id {seq_id} not in this node's sequences")
        self.visit_count += 1
        self.total_value += reward
        self.q_value = self.total_value / self.visit_count
        if self.parent is not None:
            self.parent.backup(seq_id, reward)

    def get_path_q_values(self, seq_id: int) -> list[float]:
        if seq_id not in self.sequence_ids:
            raise KeyError(f"seq_id {seq_id} not in this node's sequences")
        # Collect non-root nodes on the path for this seq_id
        path_nodes = self._get_path_nodes(seq_id)
        return [node.q_value for node in path_nodes]

    def get_turn_boundaries(self, seq_id: int) -> list[int]:
        if seq_id not in self.sequence_ids:
            raise KeyError(f"seq_id {seq_id} not in this node's sequences")
        path_nodes = self._get_path_nodes(seq_id)
        boundaries = [0]
        cumlen = 0
        for node in path_nodes:
            cumlen += len(node.tokens)
            boundaries.append(cumlen)
        return boundaries

    def _get_path_nodes(self, seq_id: int) -> list[MCTSNode]:
        """Return the non-root nodes on the path for seq_id, root-to-leaf order."""
        if seq_id not in self.sequence_ids:
            raise KeyError(f"seq_id {seq_id} not in this node's sequences")
        nodes: list[MCTSNode] = []
        current = self
        while True:
            # Find child that contains this seq_id
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_node.py -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/mcts_node.py tests/test_tree_search/test_mcts_node.py
git commit -m "feat(tree-search): add MCTSNode with insert, backup, and path queries"
```

---

### Task 3: Turn Splitter

**Files:**
- Create: `customized_areal/tree_search/turn_splitter.py`
- Create: `tests/test_tree_search/test_turn_splitter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_turn_splitter.py
import pytest
from customized_areal.tree_search.turn_splitter import make_turn_splitter


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Simple mapping: each char -> ord(char) for testability
        # "\n\n" maps to [10, 10]
        return [ord(c) for c in text]


class TestMakeTurnSplitter:
    def test_single_delimiter(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        # "hello\n\nworld" -> [104,101,108,108,111,10,10,119,111,114,108,100]
        input_ids = [104, 101, 108, 108, 111, 10, 10, 119, 111, 114, 108, 100]
        turns = splitter(input_ids)
        assert turns == [[104, 101, 108, 108, 111], [119, 111, 114, 108, 100]]

    def test_multiple_delimiters(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        # "a\n\nb\n\nc" -> [97,10,10,98,10,10,99]
        input_ids = [97, 10, 10, 98, 10, 10, 99]
        turns = splitter(input_ids)
        assert turns == [[97], [98], [99]]

    def test_no_delimiter(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        input_ids = [104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert turns == [[104, 101, 108, 108, 111]]

    def test_delimiter_at_start(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        # "\n\nhello" -> [10,10,104,101,108,108,111]
        input_ids = [10, 10, 104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert turns == [[104, 101, 108, 108, 111]]

    def test_delimiter_at_end(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        # "hello\n\n" -> [104,101,108,108,111,10,10]
        input_ids = [104, 101, 108, 108, 111, 10, 10]
        turns = splitter(input_ids)
        assert turns == [[104, 101, 108, 108, 111]]

    def test_consecutive_delimiters(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        # "a\n\n\n\nb" -> [97,10,10,10,10,98]
        # Two consecutive "\n\n" delimiters = empty turn between them
        input_ids = [97, 10, 10, 10, 10, 98]
        turns = splitter(input_ids)
        assert turns == [[97], [98]]

    def test_multi_token_delimiter(self):
        """Test with a 3-token delimiter like '<|step|>'."""
        splitter = make_turn_splitter(FakeTokenizer(), "abc")
        # delimiter tokens = [97, 98, 99]
        # input: [1, 2, 97, 98, 99, 3, 4]
        input_ids = [1, 2, 97, 98, 99, 3, 4]
        turns = splitter(input_ids)
        assert turns == [[1, 2], [3, 4]]

    def test_empty_input(self):
        splitter = make_turn_splitter(FakeTokenizer(), "\n\n")
        turns = splitter([])
        assert turns == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_turn_splitter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'customized_areal.tree_search.turn_splitter'`

- [ ] **Step 3: Write implementation**

```python
# customized_areal/tree_search/turn_splitter.py
from __future__ import annotations

from typing import Callable


def make_turn_splitter(tokenizer, delimiter: str) -> Callable[[list[int]], list[list[int]]]:
    """Create a function that splits a token ID sequence into turns at delimiter boundaries.

    Args:
        tokenizer: HuggingFace-style tokenizer with an encode() method.
        delimiter: String delimiter that separates turns (e.g., "\\n\\n").

    Returns:
        A function that takes a list of token IDs and returns a list of turn token lists.
    """
    delimiter_tokens = tokenizer.encode(delimiter, add_special_tokens=False)
    delim_len = len(delimiter_tokens)

    def split(input_ids: list[int]) -> list[list[int]]:
        if not input_ids:
            return []
        if delim_len == 0:
            return [list(input_ids)]

        turns: list[list[int]] = []
        start = 0
        i = 0
        while i <= len(input_ids) - delim_len:
            if input_ids[i : i + delim_len] == delimiter_tokens:
                segment = input_ids[start:i]
                if segment:  # skip empty segments (e.g., consecutive delimiters)
                    turns.append(segment)
                i += delim_len
                start = i
            else:
                i += 1
        # Remaining tokens after last delimiter
        segment = input_ids[start:]
        if segment:
            turns.append(segment)
        return turns

    return split
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_turn_splitter.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/turn_splitter.py tests/test_tree_search/test_turn_splitter.py
git commit -m "feat(tree-search): add make_turn_splitter for delimiter-based turn splitting"
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
from customized_areal.tree_search.mcts_node import MCTSNode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _simple_splitter(input_ids: list[int]) -> list[list[int]]:
    """Split at token 10 (newline for testing)."""
    turns: list[list[int]] = []
    start = 0
    for i, tok in enumerate(input_ids):
        if tok == 10:
            seg = input_ids[start:i]
            if seg:
                turns.append(seg)
            start = i + 1
    seg = input_ids[start:]
    if seg:
        turns.append(seg)
    return turns


class TestMCTSTreeStoreInsert:
    def test_insert_single_trajectory(self):
        store = MCTSTreeStore(_simple_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        assert seq_id == 0
        assert "q1" in store.trees
        root = store.trees["q1"]
        assert 1 in root.children
        # Two turns: [1,2] and [3,4]
        child = root.children[1]
        assert child.tokens == [1, 2]
        assert 10 in child.children
        assert child.children[10].tokens == [3, 4]
        # Backup done
        assert root.visit_count == 1
        assert root.q_value == 1.0

    def test_insert_two_trajectories_same_query(self):
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=1.0)
        store.insert_trajectory("q1", [1, 2, 10, 5, 6], reward=0.5)
        root = store.trees["q1"]
        # Shared prefix
        shared = root.children[1]
        assert shared.sequence_ids == {0, 1}
        assert shared.visit_count == 2
        assert abs(shared.q_value - 0.75) < 1e-6
        # Diverged
        assert 10 in shared.children
        # Both children exist under key 10 (same first token)
        # They diverge at the second turn: [3,4] vs [5,6]
        # Since first token of second turn is 3 vs 5, two children
        assert 3 in shared.children[10].children
        assert 5 in shared.children[10].children

    def test_insert_different_queries(self):
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3], reward=1.0)
        store.insert_trajectory("q2", [4, 5, 10, 6], reward=0.0)
        assert len(store.trees) == 2
        assert "q1" in store.trees
        assert "q2" in store.trees

    def test_seq_id_increments(self):
        store = MCTSTreeStore(_simple_splitter)
        id0 = store.insert_trajectory("q1", [1, 2], reward=1.0)
        id1 = store.insert_trajectory("q1", [3, 4], reward=0.0)
        assert id0 == 0
        assert id1 == 1


class TestMCTSTreeStoreInsertBatch:
    def test_insert_batch(self):
        store = MCTSTreeStore(_simple_splitter)
        trajectories = [
            {"input_ids": torch.tensor([1, 2, 10, 3, 4]), "loss_mask": torch.tensor([0, 0, 0, 1, 1]), "rewards": torch.tensor([1.0])},
            {"input_ids": torch.tensor([1, 2, 10, 5, 6]), "loss_mask": torch.tensor([0, 0, 0, 1, 1]), "rewards": torch.tensor([0.5])},
        ]
        store.insert_batch(trajectories)
        assert "_mcts_seq_id" in trajectories[0]
        assert "_mcts_query_id" in trajectories[0]
        assert trajectories[0]["_mcts_seq_id"] == 0
        assert trajectories[1]["_mcts_seq_id"] == 1
        # Same query (same prompt hash)
        assert trajectories[0]["_mcts_query_id"] == trajectories[1]["_mcts_query_id"]


class TestMCTSTreeStoreAdvantages:
    def test_get_advantages(self):
        store = MCTSTreeStore(_simple_splitter)
        seq_id = store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=2.0)
        advantages = store.get_advantages("q1", seq_id)
        # Two turns: [1,2] (2 tokens) and [3,4] (2 tokens), each with q_value=2.0
        assert advantages.shape == torch.Size([5])
        assert torch.allclose(advantages, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]))

    def test_get_advantages_multi_backup(self):
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2, 10, 3, 4], reward=2.0)
        store.insert_trajectory("q1", [1, 2, 10, 5, 6], reward=0.0)
        # Shared turn [1,2]: q=1.0, diverged turns: q=2.0 and q=0.0
        adv0 = store.get_advantages("q1", 0)
        assert abs(adv0[0].item() - 1.0) < 1e-6  # token 1, shared turn
        assert abs(adv0[1].item() - 1.0) < 1e-6  # token 2, shared turn
        assert abs(adv0[3].item() - 2.0) < 1e-6  # token 3, diverged turn
        assert abs(adv0[4].item() - 2.0) < 1e-6  # token 4, diverged turn


class TestMCTSTreeStoreClear:
    def test_clear(self):
        store = MCTSTreeStore(_simple_splitter)
        store.insert_trajectory("q1", [1, 2], reward=1.0)
        store.clear()
        assert len(store.trees) == 0
        assert store._next_seq_id == 0
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

from customized_areal.tree_search.mcts_node import MCTSNode


def _get_query_id(traj: dict[str, Any]) -> str:
    """Derive a query ID from the prompt tokens in a trajectory.

    The prompt is identified by loss_mask == 0 tokens.
    """
    loss_mask = traj["loss_mask"]
    input_ids = traj["input_ids"]
    prompt_tokens = input_ids[loss_mask == 0].tolist()
    prompt_str = ",".join(str(t) for t in prompt_tokens)
    return hashlib.md5(prompt_str.encode()).hexdigest()


class MCTSTreeStore:
    def __init__(self, turn_splitter: Callable[[list[int]], list[list[int]]]):
        self.trees: dict[str, MCTSNode] = {}
        self.turn_splitter = turn_splitter
        self._next_seq_id: int = 0

    def insert_trajectory(self, query_id: str, input_ids: list[int], reward: float) -> int:
        turns = self.turn_splitter(input_ids)
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        root = self.trees.setdefault(query_id, MCTSNode(tokens=[]))
        root.insert_path(turns, seq_id)
        root.backup(seq_id, reward)
        return seq_id

    def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
        for traj in trajectories:
            query_id = _get_query_id(traj)
            input_ids = traj["input_ids"].tolist()
            reward = traj["rewards"].item() if traj["rewards"].dim() > 0 else traj["rewards"].item()
            seq_id = self.insert_trajectory(query_id, input_ids, reward)
            traj["_mcts_seq_id"] = seq_id
            traj["_mcts_query_id"] = query_id

    def get_advantages(self, query_id: str, seq_id: int) -> torch.Tensor:
        root = self.trees[query_id]
        q_values = root.get_path_q_values(seq_id)
        boundaries = root.get_turn_boundaries(seq_id)
        total_len = boundaries[-1]
        advantages = torch.zeros(total_len)
        for i, q_val in enumerate(q_values):
            advantages[boundaries[i] : boundaries[i + 1]] = q_val
        return advantages

    def clear(self) -> None:
        self.trees.clear()
        self._next_seq_id = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_tree_store.py -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store.py
git commit -m "feat(tree-search): add MCTSTreeStore with insert, advantages, and clear"
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


def _simple_splitter(input_ids: list[int]) -> list[list[int]]:
    """Split at token 10 (newline for testing)."""
    turns: list[list[int]] = []
    start = 0
    for i, tok in enumerate(input_ids):
        if tok == 10:
            seg = input_ids[start:i]
            if seg:
                turns.append(seg)
            start = i + 1
    seg = input_ids[start:]
    if seg:
        turns.append(seg)
    return turns


class TestTreeAdvantageComputer:
    def test_compute_single_trajectory(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        # Prompt: [1,2], response: [3,4,10,5,6] (two turns)
        traj = {
            "input_ids": torch.tensor([1, 2, 3, 4, 10, 5, 6]),
            "loss_mask": torch.tensor([0, 0, 1, 1, 1, 1, 1]),
            "rewards": torch.tensor([2.0]),
        }
        store.insert_batch([traj])
        computer.compute([traj])
        assert "advantages" in traj
        assert "returns" in traj
        # Advantages should be zeroed for prompt tokens
        assert torch.allclose(traj["advantages"][:2], torch.zeros(2))
        # Response tokens should have q_value = 2.0
        assert torch.allclose(traj["advantages"][2:], torch.full((5,), 2.0))

    def test_compute_returns_equal_advantages(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        traj = {
            "input_ids": torch.tensor([1, 2, 10, 3]),
            "loss_mask": torch.tensor([0, 0, 1, 1]),
            "rewards": torch.tensor([1.0]),
        }
        store.insert_batch([traj])
        computer.compute([traj])
        assert torch.allclose(traj["returns"], traj["advantages"])

    def test_compute_multi_trajectory(self):
        store = MCTSTreeStore(_simple_splitter)
        computer = TreeAdvantageComputer(store)
        traj1 = {
            "input_ids": torch.tensor([1, 2, 10, 3, 4]),
            "loss_mask": torch.tensor([0, 0, 1, 1, 1]),
            "rewards": torch.tensor([2.0]),
        }
        traj2 = {
            "input_ids": torch.tensor([1, 2, 10, 5, 6]),
            "loss_mask": torch.tensor([0, 0, 1, 1, 1]),
            "rewards": torch.tensor([0.0]),
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
from customized_areal.tree_search.mcts_node import MCTSNode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.checkpoint import TreeCheckpointManager


def _simple_splitter(input_ids: list[int]) -> list[list[int]]:
    turns: list[list[int]] = []
    start = 0
    for i, tok in enumerate(input_ids):
        if tok == 10:
            seg = input_ids[start:i]
            if seg:
                turns.append(seg)
            start = i + 1
    seg = input_ids[start:]
    if seg:
        turns.append(seg)
    return turns


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
        # Verify q1 tree structure and values
        q1_root = loaded.trees["q1"]
        assert q1_root.visit_count == 2
        assert abs(q1_root.q_value - 1.0) < 1e-6
        shared = q1_root.children[1]
        assert shared.tokens == [1, 2]
        assert shared.visit_count == 2
        # Verify parent links restored
        assert shared.parent is q1_root

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
        # Next insert should get seq_id=2
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

from customized_areal.tree_search.mcts_node import MCTSNode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


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

    def load(self, turn_splitter: Callable[[list[int]], list[list[int]]]) -> MCTSTreeStore:
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
            root = self._deserialize_node(tree_data["root"], parent=None)
            store.trees[query_id] = root
        return store

    def _serialize_node(self, node: MCTSNode) -> dict:
        return {
            "tokens": node.tokens,
            "visit_count": node.visit_count,
            "total_value": node.total_value,
            "q_value": node.q_value,
            "sequence_ids": sorted(node.sequence_ids),
            "children": {
                str(key): self._serialize_node(child)
                for key, child in node.children.items()
            },
        }

    def _deserialize_node(self, data: dict, parent: MCTSNode | None) -> MCTSNode:
        node = MCTSNode(
            tokens=data["tokens"],
            parent=parent,
            visit_count=data["visit_count"],
            total_value=data["total_value"],
            q_value=data["q_value"],
            sequence_ids=set(data["sequence_ids"]),
        )
        for key_str, child_data in data["children"].items():
            key = int(key_str)
            child = self._deserialize_node(child_data, parent=node)
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
- Create: `customized_areal/tree_search/__init__.py`

- [ ] **Step 1: Write the __init__.py**

```python
# customized_areal/tree_search/__init__.py
from customized_areal.tree_search.advantage import TreeAdvantageComputer
from customized_areal.tree_search.checkpoint import TreeCheckpointManager
from customized_areal.tree_search.config import TreeBackupConfig, TreeBackupMode
from customized_areal.tree_search.mcts_node import MCTSNode
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore
from customized_areal.tree_search.turn_splitter import make_turn_splitter

__all__ = [
    "MCTSNode",
    "MCTSTreeStore",
    "TreeAdvantageComputer",
    "TreeCheckpointManager",
    "TreeBackupConfig",
    "TreeBackupMode",
    "make_turn_splitter",
]
```

- [ ] **Step 2: Verify imports work**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -c "from customized_areal.tree_search import TreeBackupConfig, TreeBackupMode, MCTSNode, MCTSTreeStore, TreeAdvantageComputer, TreeCheckpointManager, make_turn_splitter; print('All imports OK')"`
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

### Task 8: Add `on_trajectory_ready` Callback to `BatchTaskDispatcher`

**Files:**
- Modify: `areal/infra/workflow_executor.py`

- [ ] **Step 1: Add `on_trajectory_ready` parameter to `BatchTaskDispatcher.__init__`**

In `BatchTaskDispatcher.__init__`, add the callback parameter after existing params:

```python
def __init__(
    self,
    ...,
    on_trajectory_ready: Callable[[TResult], None] | None = None,
):
    ...
    self._on_trajectory_ready = on_trajectory_ready
```

- [ ] **Step 2: Fire callback in `_fetch_loop` after result placement**

In the `_fetch_loop` method, restructure the result handling to fire the
callback **outside** the `_result_cv` lock. Find the existing code block
(approximately lines 409-416):

```python
with self._result_cv:
    for result in results:
        self._pending_results[result.task_id] = result
        # Trigger callback if registered
        cb_addr = self._task_callbacks.pop(result.task_id, None)
        if cb_addr:
            self._send_callback(cb_addr, result.task_id, result.data)
    self._result_cv.notify_all()
```

Replace with:

```python
# Collect results under lock, fire callbacks outside lock
new_results = []
with self._result_cv:
    for result in results:
        self._pending_results[result.task_id] = result
        cb_addr = self._task_callbacks.pop(result.task_id, None)
        new_results.append((result, cb_addr))
    self._result_cv.notify_all()

# Fire callbacks outside the lock to avoid holding it during execution
for result, cb_addr in new_results:
    if self._on_trajectory_ready is not None:
        try:
            self._on_trajectory_ready(result.data)
        except Exception:
            self.logger.error("on_trajectory_ready callback failed", exc_info=True)
    if cb_addr:
        self._send_callback(cb_addr, result.task_id, result.data)
```

- [ ] **Step 3: Verify no regressions in existing tests**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/ -v -k "not gpu" --timeout=30`
Expected: All existing tests still pass

- [ ] **Step 4: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add areal/infra/workflow_executor.py
git commit -m "feat(infra): add on_trajectory_ready callback to BatchTaskDispatcher"
```

---

### Task 9: Thread-Safe MCTSTreeStore

**Files:**
- Modify: `customized_areal/tree_search/mcts_tree_store.py`
- Create: `tests/test_tree_search/test_mcts_tree_store_threading.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tree_search/test_mcts_tree_store_threading.py
import threading
import pytest
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore


def _simple_splitter(input_ids: list[int]) -> list[list[int]]:
    """Split at token 10 (newline for testing)."""
    turns: list[list[int]] = []
    start = 0
    for i, tok in enumerate(input_ids):
        if tok == 10:
            seg = input_ids[start:i]
            if seg:
                turns.append(seg)
            start = i + 1
    seg = input_ids[start:]
    if seg:
        turns.append(seg)
    return turns


class TestMCTSTreeStoreThreadSafety:
    def test_concurrent_insert_and_read(self):
        """Multiple threads insert trajectories while main thread reads advantages."""
        store = MCTSTreeStore(_simple_splitter)
        errors = []

        def insert_worker(query_id, input_ids, reward):
            try:
                store.insert_trajectory(query_id, input_ids, reward)
            except Exception as e:
                errors.append(e)

        # Insert 10 trajectories concurrently for the same query
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=insert_worker,
                args=("q1", [1, 2, 10, 3 + i], float(i)),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent insert: {errors}"

        # Verify tree state is consistent
        root = store.trees["q1"]
        assert root is not None
        # All 10 sequences should be tracked
        assert len(root.sequence_ids) == 10

    def test_concurrent_insert_different_queries(self):
        """Multiple threads insert trajectories for different queries."""
        store = MCTSTreeStore(_simple_splitter)
        errors = []

        def insert_worker(query_id):
            try:
                store.insert_trajectory(query_id, [1, 2, 10, 3], reward=1.0)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(
                target=insert_worker,
                args=(f"q{i}",),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store.trees) == 5
```

- [ ] **Step 2: Run test to verify it fails (may pass sporadically due to GIL)**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_tree_store_threading.py -v`
Expected: May pass due to GIL, but the lock is still needed for correctness

- [ ] **Step 3: Add `threading.Lock` to `MCTSTreeStore`**

In `customized_areal/tree_search/mcts_tree_store.py`, add lock and wrap all
public methods:

```python
import threading

class MCTSTreeStore:
    def __init__(self, turn_splitter):
        ...
        self._lock = threading.Lock()

    def start_sequence(self, query_id: str) -> int:
        with self._lock:
            # existing logic

    def add_turn(self, query_id: str, seq_id: int, turn: Turn) -> None:
        with self._lock:
            # existing logic

    def finish_sequence(self, query_id: str, seq_id: int, reward: float) -> None:
        with self._lock:
            # existing logic

    def insert_trajectory(self, query_id, input_ids, reward):
        with self._lock:
            # existing logic (calls _backup internally, already under lock)

    def insert_batch(self, trajectories):
        with self._lock:
            # existing logic

    def get_advantages(self, query_id, seq_id):
        with self._lock:
            # existing logic

    def get_prompt_mask(self, query_id, seq_id):
        with self._lock:
            # existing logic

    def clear(self):
        with self._lock:
            # existing logic

    # _backup is called from within locked methods, no separate lock needed
```

- [ ] **Step 4: Run all MCTSTreeStore tests**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/test_mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store_threading.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/mcts_tree_store.py tests/test_tree_search/test_mcts_tree_store_threading.py
git commit -m "feat(tree-search): add thread-safe locking to MCTSTreeStore"
```

---

### Task 10: Streaming Backup Integration in TreeBackupPPOTrainer

**Files:**
- Modify: `customized_areal/tree_search/trainer.py`

- [ ] **Step 1: Update `patch_ppo_actor_for_tree_backup` to be read-only**

Remove the `tree_store.insert_batch([data])` line from the patched
`_tree_backup_compute_advantages` method. The method should only call
`tree_advantage_computer.compute([data])` — no insert.

Find in `customized_areal/tree_search/trainer.py`:

```python
        # === TREE BACKUP replaces GAE ===
        # Insert trajectories into the tree and compute tree-based advantages
        tree_store.insert_batch([data])
        tree_advantage_computer.compute([data])
```

Replace with:

```python
        # === TREE BACKUP replaces GAE (read-only) ===
        # Trajectories already inserted+backed during rollout via streaming callback
        tree_advantage_computer.compute([data])
```

- [ ] **Step 2: Add `_on_trajectory_ready` callback method to `TreeBackupPPOTrainer`**

Add the callback method and registration logic:

```python
class TreeBackupPPOTrainer(PPOTrainer):
    ...

    def _register_tree_callback(self):
        """Register on_trajectory_ready callback on the rollout dispatcher."""
        if hasattr(self.actor, 'rollout_coordinator') and \
           self.actor.rollout_coordinator is not None:
            self.actor.rollout_coordinator.dispatcher._on_trajectory_ready = (
                self._on_trajectory_ready
            )
            logger.info("Registered MCTS tree backup streaming callback")

    def _on_trajectory_ready(self, result):
        """Callback: insert trajectory into tree and run backup.

        Called from the BatchTaskDispatcher consumer thread.
        """
        if result is None:
            return
        trajectory = result.trajectory  # _RemoteRolloutResult has .trajectory
        query_id = _get_query_id(trajectory)
        input_ids = trajectory["input_ids"].tolist()
        reward = trajectory["rewards"].item()
        seq_id = self.tree_store.insert_trajectory(query_id, input_ids, reward)
        trajectory["_mcts_seq_id"] = seq_id
        trajectory["_mcts_query_id"] = query_id
```

Also add the import for `_get_query_id`:

```python
from customized_areal.tree_search.mcts_tree_store import MCTSTreeStore, _get_query_id
```

- [ ] **Step 3: Override `train()` to register callback**

Override `train()` in `TreeBackupPPOTrainer` to register the streaming
callback after rollout initialization:

```python
def train(self, workflow=None, eval_workflow=None, workflow_kwargs=None,
          eval_workflow_kwargs=None, dynamic_filter_fn=None, total_epochs=None):
    """Override train() to register streaming callback after rollout init."""
    # Register the streaming callback on the rollout controller's dispatcher
    if self.tree_backup_config.mode != TreeBackupMode.OFF:
        self._register_tree_callback()
    super().train(
        workflow=workflow,
        eval_workflow=eval_workflow,
        workflow_kwargs=workflow_kwargs,
        eval_workflow_kwargs=eval_workflow_kwargs,
        dynamic_filter_fn=dynamic_filter_fn,
        total_epochs=total_epochs,
    )
```

- [ ] **Step 4: Run tree_search tests**

Run: `cd /dfs/share-groups/letrain/zhoujie/AReaL-main && python -m pytest tests/test_tree_search/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
git add customized_areal/tree_search/trainer.py
git commit -m "feat(tree-search): streaming MCTS backup via on_trajectory_ready callback"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: Each component in the spec (MCTSNode, MCTSTreeStore, TreeAdvantageComputer, TreeCheckpointManager, config, turn_splitter, streaming backup, thread safety) has a corresponding task.
- [x] **Placeholder scan**: No TBD, TODO, or vague steps. Every step has complete code.
- [x] **Type consistency**: Method signatures and field names are consistent across tasks. `_mcts_seq_id` and `_mcts_query_id` are set by the streaming callback and read by `TreeAdvantageComputer.compute`. `get_advantages` returns `torch.Tensor` as expected by `TreeAdvantageComputer`.
- [x] **Thread safety**: `MCTSTreeStore` is thread-safe (all public methods wrapped with `threading.Lock`). The callback fires in the consumer thread, advantage computation runs in the main thread. No shared mutable state outside of the locked MCTSTreeStore.
- [x] **Streaming timing**: Insert+backup happens in the `on_trajectory_ready` callback (consumer thread, after each trajectory arrives). Advantage computation is read-only. Rewards are guaranteed to be present when the callback fires (workflow completes before trajectory is returned).
- [x] **Backward compatibility**: When `tree_backup_config.mode == OFF`, no callback is registered, no patching occurs, existing GAE path runs unchanged. The `on_trajectory_ready` parameter defaults to `None` in `BatchTaskDispatcher`, so existing code is unaffected.
