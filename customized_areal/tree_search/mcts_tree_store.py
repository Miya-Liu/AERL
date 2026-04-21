# customized_areal/tree_search/mcts_tree_store.py
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

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


def get_query_id_from_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> str:
    """Derive a query ID from prompt messages by tokenizing them.

    This produces the same query ID as ``_get_query_id`` would produce
    after rollout, because the proxy server tokenizes the same messages
    with the same tokenizer to produce ``input_ids``.
    """
    # Build the prompt text the same way the chat template does
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_str = ",".join(str(t) for t in prompt_tokens)
    return hashlib.md5(prompt_str.encode()).hexdigest()


class MCTSTreeStore:
    """Trie-backed MCTS tree store with cursor-based API.

    Manages multiple search trees (one per query), tracks MCTS statistics
    (visit counts, Q-values) per node, and provides a cursor-based API for
    incrementally building trajectories through start/add/finish sequences.
    """

    def __init__(self, turn_splitter: Callable[[list[int]], list[Turn]]):
        self.trees: dict[str, TrieNode] = {}
        self.turn_splitter = turn_splitter
        self._next_seq_id: int = 0

        self._cursors: dict[tuple[str, int], TrieNode] = {}

        self._visit_counts: dict[tuple[str, int], int] = {}
        self._total_values: dict[tuple[str, int], float] = {}
        self._q_values: dict[tuple[str, int], float] = {}

        self._trained: dict[tuple[str, int], bool] = {}
        self._rewards: dict[tuple[str, int], float] = {}

    def start_sequence(self, query_id: str) -> int:
        """Create root if needed, assign a seq_id, set cursor at root."""
        tree_idx = len(self.trees)
        root = self.trees.setdefault(query_id, TrieNode(tree_id=tree_idx))
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        root.sequence_ids.append(seq_id)
        self._cursors[(query_id, seq_id)] = root
        return seq_id

    def add_turn(
        self,
        query_id: str,
        seq_id: int,
        turn: Turn,
        logprobs: list[float] | None = None,
        versions: list[int] | None = None,
    ) -> None:
        """Add a single turn at the cursor position, advance cursor."""
        cursor = self._cursors[(query_id, seq_id)]
        child = cursor.add_turn(turn, seq_id, logprobs=logprobs, versions=versions)
        self._cursors[(query_id, seq_id)] = child

    def finish_sequence(self, query_id: str, seq_id: int, reward: float) -> None:
        """Run MCTS backup along the completed path, clear cursor."""
        self._backup(query_id, seq_id, reward)
        self._rewards[(query_id, seq_id)] = reward
        self._trained[(query_id, seq_id)] = False
        del self._cursors[(query_id, seq_id)]

    def _backup(self, query_id: str, seq_id: int, reward: float) -> None:
        """Walk from leaf to root, updating MCTS stats at each node."""
        root = self.trees[query_id]
        path_nodes = root.get_path_nodes(seq_id)
        all_nodes = [root] + path_nodes
        for node in all_nodes:
            key = (query_id, id(node))
            self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
            self._total_values[key] = self._total_values.get(key, 0.0) + reward
            self._q_values[key] = self._total_values[key] / self._visit_counts[key]

    @staticmethod
    def _split_metadata_to_turns(turns: list[Turn], metadata: list) -> list[list]:
        """Split a flat metadata list (logprobs or versions) into per-turn chunks.

        Each turn has prompt_tokens + response_tokens tokens total.
        """
        result = []
        offset = 0
        for turn in turns:
            n = len(turn.prompt_tokens) + len(turn.response_tokens)
            result.append(metadata[offset : offset + n])
            offset += n
        return result

    def insert_trajectory(
        self,
        query_id: str,
        input_ids: list[int],
        reward: float,
        logprobs: list[float] | None = None,
        versions: list[int] | None = None,
    ) -> int:
        """Convenience: split -> start_sequence -> add_turn loop -> finish_sequence."""
        turns = self.turn_splitter(input_ids)
        seq_id = self.start_sequence(query_id)

        # Split logprobs/versions across turns to match token boundaries
        turn_logprobs = (
            self._split_metadata_to_turns(turns, logprobs) if logprobs else None
        )
        turn_versions = (
            self._split_metadata_to_turns(turns, versions) if versions else None
        )

        for i, turn in enumerate(turns):
            lp = turn_logprobs[i] if turn_logprobs is not None else None
            vs = turn_versions[i] if turn_versions is not None else None
            self.add_turn(query_id, seq_id, turn, logprobs=lp, versions=vs)

        self.finish_sequence(query_id, seq_id, reward)
        return seq_id

    def insert_batch(self, trajectories: list[dict[str, Any]]) -> None:
        """Batch version -- insert each trajectory, handling grouped dicts.

        When a trajectory dict has batch_size > 1 (grouped via
        GroupedRolloutWorkflow), it is split into individual samples and
        each is inserted separately. The resulting seq_ids are stored as
        ``_mcts_seq_ids`` (list[int]) on the grouped dict.

        Trajectories that already carry ``_mcts_seq_id`` or
        ``_mcts_seq_ids`` are skipped (they were loaded from cache).
        """
        for traj in trajectories:
            # Skip already-inserted cached trajectories
            if "_mcts_seq_id" in traj or "_mcts_seq_ids" in traj:
                continue

            input_ids = traj["input_ids"]
            rewards = traj["rewards"]
            batch_size = input_ids.shape[0]

            if batch_size == 1:
                # Single trajectory — insert directly
                query_id = _get_query_id(traj)
                ids_flat = input_ids[0].tolist()
                reward = rewards.item() if rewards.dim() > 0 else rewards.item()

                logprobs = traj["logprobs"][0].tolist() if "logprobs" in traj else None
                versions = traj["versions"][0].tolist() if "versions" in traj else None

                seq_id = self.insert_trajectory(
                    query_id, ids_flat, reward, logprobs=logprobs, versions=versions
                )
                traj["_mcts_seq_id"] = seq_id
                traj["_mcts_query_id"] = query_id
            else:
                # Grouped trajectory — insert each sample separately
                seq_ids = []
                query_id = None
                for i in range(batch_size):
                    single = {
                        "input_ids": input_ids[i : i + 1],
                        "loss_mask": traj["loss_mask"][i : i + 1],
                        "rewards": rewards[i : i + 1],
                    }
                    qid = _get_query_id(single)
                    if query_id is None:
                        query_id = qid
                    ids_flat = input_ids[i].tolist()
                    reward = rewards[i].item() if rewards.dim() > 0 else rewards.item()

                    logprobs = (
                        traj["logprobs"][i].tolist() if "logprobs" in traj else None
                    )
                    versions = (
                        traj["versions"][i].tolist() if "versions" in traj else None
                    )

                    seq_id = self.insert_trajectory(
                        qid, ids_flat, reward, logprobs=logprobs, versions=versions
                    )
                    seq_ids.append(seq_id)

                traj["_mcts_seq_ids"] = seq_ids
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

    def get_prompt_mask(self, query_id: str, seq_id: int) -> torch.Tensor:
        """Return a boolean tensor: True for response tokens, False for prompt tokens."""
        root = self.trees[query_id]
        path_nodes = root.get_path_nodes(seq_id)
        boundaries = root.get_turn_boundaries(seq_id)
        total_len = boundaries[-1]
        mask = torch.zeros(total_len, dtype=torch.bool)
        for i, node in enumerate(path_nodes):
            start = boundaries[i]
            response_start = start + node.prompt_len
            mask[response_start : boundaries[i + 1]] = True
        return mask

    def set_trained(self, query_id: str, seq_id: int, trained: bool = True) -> None:
        self._trained[(query_id, seq_id)] = trained

    def is_trained(self, query_id: str, seq_id: int) -> bool:
        return self._trained.get((query_id, seq_id), False)

    def get_reward(self, query_id: str, seq_id: int) -> float:
        return self._rewards.get((query_id, seq_id), 0.0)

    def get_untrained_count(self, query_id: str) -> int:
        if query_id not in self.trees:
            return 0
        root = self.trees[query_id]
        return sum(
            1 for sid in set(root.sequence_ids) if not self.is_trained(query_id, sid)
        )

    def get_untrained_seq_ids(self, query_id: str, n_samples: int) -> list[int]:
        if query_id not in self.trees:
            return []
        root = self.trees[query_id]
        result = []
        seen = set()
        for sid in root.sequence_ids:
            if sid in seen:
                continue
            seen.add(sid)
            if not self.is_trained(query_id, sid):
                result.append(sid)
                if len(result) >= n_samples:
                    break
        return result

    def load_trajectories(self, query_id: str, n_samples: int) -> list[dict[str, Any]]:
        """Extract up to n_samples untrained trajectories from tree as training dicts.

        Returns list of dicts with keys: input_ids, logprobs, loss_mask,
        attention_mask, rewards, versions — each with shape [1, seq_len].
        Also includes _mcts_query_id and _mcts_seq_id for tracking.
        """
        if query_id not in self.trees:
            return []

        untrained_ids = self.get_untrained_seq_ids(query_id, n_samples)
        result = []
        for seq_id in untrained_ids:
            root = self.trees[query_id]
            path_nodes = root.get_path_nodes(seq_id)

            # Reconstruct full token sequence and metadata from path
            all_tokens = []
            all_logprobs = []
            all_versions = []
            prompt_len_total = 0

            for node in path_nodes:
                all_tokens.extend(node.tokens)
                if node.logprobs:
                    all_logprobs.extend(node.logprobs)
                else:
                    all_logprobs.extend([0.0] * len(node.tokens))
                if node.versions:
                    all_versions.extend(node.versions)
                else:
                    all_versions.extend([0] * len(node.tokens))
                prompt_len_total += node.prompt_len

            seq_len = len(all_tokens)

            # Build tensors with [1, seq_len] shape (batch dim)
            input_ids = torch.tensor(all_tokens, dtype=torch.int32).unsqueeze(0)
            logprobs_t = torch.tensor(all_logprobs, dtype=torch.float32).unsqueeze(0)
            versions_t = torch.tensor(all_versions, dtype=torch.int32).unsqueeze(0)
            attention_mask = torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)

            # loss_mask: 0 for prompt tokens, 1 for response tokens
            loss_mask = torch.zeros(seq_len, dtype=torch.int32)
            loss_mask[prompt_len_total:] = 1
            loss_mask = loss_mask.unsqueeze(0)

            reward_val = self.get_reward(query_id, seq_id)
            rewards = torch.tensor([reward_val], dtype=torch.float32).unsqueeze(0)

            result.append(
                {
                    "input_ids": input_ids,
                    "logprobs": logprobs_t,
                    "loss_mask": loss_mask,
                    "attention_mask": attention_mask,
                    "rewards": rewards,
                    "versions": versions_t,
                    "_mcts_query_id": query_id,
                    "_mcts_seq_id": seq_id,
                }
            )

        return result

    def reset_trained_flags(self) -> None:
        for key in self._trained:
            self._trained[key] = False

    def clear(self) -> None:
        """Reset all trees, stats, and cursors."""
        self.trees.clear()
        self._next_seq_id = 0
        self._cursors.clear()
        self._visit_counts.clear()
        self._total_values.clear()
        self._q_values.clear()
        self._trained.clear()
        self._rewards.clear()
