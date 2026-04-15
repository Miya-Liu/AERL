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
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [60, 97, 62, 104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == [60, 97, 62]
        assert turns[0].response_tokens == [104, 101, 108, 108, 111]

    def test_two_assistant_turns(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [60, 97, 62, 121, 101, 115, 60, 97, 62, 110, 111]
        turns = splitter(input_ids)
        assert len(turns) == 2
        assert turns[0].prompt_tokens == [60, 97, 62]
        assert turns[0].response_tokens == [121, 101, 115]
        assert turns[1].prompt_tokens == [60, 97, 62]
        assert turns[1].response_tokens == [110, 111]

    def test_no_assistant_marker(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == []
        assert turns[0].response_tokens == [104, 101, 108, 108, 111]

    def test_marker_at_start_only(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [60, 97, 62, 114, 101, 115, 112]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == [60, 97, 62]
        assert turns[0].response_tokens == [114, 101, 115, 112]

    def test_marker_at_end_no_response(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        input_ids = [112, 114, 111, 109, 112, 116, 60, 97, 62]
        turns = splitter(input_ids)
        assert len(turns) == 0

    def test_multi_token_marker(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<ab>")
        input_ids = [60, 97, 98, 62, 104, 101, 108, 108, 111]
        turns = splitter(input_ids)
        assert len(turns) == 1
        assert turns[0].prompt_tokens == [60, 97, 98, 62]
        assert turns[0].response_tokens == [104, 101, 108, 108, 111]

    def test_empty_input(self):
        splitter = make_turn_splitter(FakeTokenizer(), "<a>")
        turns = splitter([])
        assert turns == []
