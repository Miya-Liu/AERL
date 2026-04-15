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
    prompt_len: int = 0
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
                prompt_len=len(turn.prompt_tokens),
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
