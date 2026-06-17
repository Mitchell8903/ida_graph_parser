"""Tests for src.summarize_graph using a mocked OpenAI client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx

from src.summarize_graph import GraphSummarizer


def _make_mock_response(text: str):
    """Build a fake ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _build_summarizer(graph, responses):
    """Create a GraphSummarizer with a mocked LLM that returns *responses* in order."""
    summarizer = GraphSummarizer(graph, api_key="test-key", model="test-model")
    mock_create = AsyncMock(side_effect=[_make_mock_response(r) for r in responses])
    summarizer.client.chat.completions.create = mock_create
    return summarizer


class TestSingleFunction:
    def test_single_block_function(self):
        """A function with one block and few instructions is summarized without LLM."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["MOV X0, #1", "RET"], label="func @ A", func="my_func")

        summarizer = _build_summarizer(G, [])
        result = asyncio.run(summarizer.summarize_all())

        assert "my_func" in result
        # Few instructions, no LLM call
        assert summarizer.client.chat.completions.create.call_count == 0

    def test_multi_block_function(self):
        """A function with multiple blocks and enough instructions uses one LLM call."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["CMP X0, #0", "B.EQ target"] + ["NOP"] * 10,
                    label="func @ A", func="my_func")
        G.add_node("B", instrs=["MOV X0, #1", "RET"] + ["NOP"] * 10,
                    label="func @ B", func="my_func")
        G.add_edge("A", "B", type="intra-function", conditional=True)

        summarizer = _build_summarizer(G, [
            "Checks X0 and conditionally sets it to 1 before returning.",
        ])
        result = asyncio.run(summarizer.summarize_all())

        assert "my_func" in result
        # One LLM call for the whole function
        assert summarizer.client.chat.completions.create.call_count == 1


class TestInterFunctionDeps:
    def test_caller_includes_callee_summary(self):
        """When func_a calls func_b via inter-function edge, func_b summary is appended."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["BL func_b"] + ["NOP"] * 10,
                    label="func_a @ A", func="func_a")
        G.add_node("B", instrs=["MOV X0, #42", "RET"] + ["NOP"] * 10,
                    label="func_b @ B", func="func_b")
        G.add_edge("A", "B", type="inter-function", conditional=False)

        summarizer = _build_summarizer(G, [
            "Returns 42.",           # func_b (no deps, summarized first)
            "Calls func_b which returns 42.",  # func_a (with func_b summary appended)
        ])
        result = asyncio.run(summarizer.summarize_all())

        assert "func_a" in result
        assert "func_b" in result
        assert summarizer.client.chat.completions.create.call_count == 2

    def test_non_call_edge_includes_summary(self):
        """Non-call edges also trigger appended summaries."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["B target"] + ["NOP"] * 10,
                    label="func_a @ A", func="func_a")
        G.add_node("B", instrs=["RET"] + ["NOP"] * 10,
                    label="func_b @ B", func="func_b")
        G.add_edge("A", "B", type="non-call", conditional=False)

        summarizer = _build_summarizer(G, [
            "Returns.",
            "Jumps to func_b which returns.",
        ])
        result = asyncio.run(summarizer.summarize_all())

        assert "func_a" in result
        assert "func_b" in result
        assert summarizer.client.chat.completions.create.call_count == 2


class TestIntraFunctionOnly:
    def test_no_appended_summaries_for_intra_edges(self):
        """Intra-function edges do NOT cause appended summaries from other functions."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["CMP X0, #0"] + ["NOP"] * 10,
                    label="func @ A", func="my_func")
        G.add_node("B", instrs=["MOV X0, #1"] + ["NOP"] * 10,
                    label="func @ B", func="my_func")
        G.add_edge("A", "B", type="intra-function", conditional=True)

        summarizer = _build_summarizer(G, [
            "Checks X0 and conditionally sets it to 1.",
        ])
        result = asyncio.run(summarizer.summarize_all())

        assert "my_func" in result
        # Only one LLM call for the whole function, no dependency summaries
        assert summarizer.client.chat.completions.create.call_count == 1


class TestCycleHandling:
    def test_inter_function_cycle(self):
        """Mutual inter-function calls don't infinite loop."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["BL func_b"] + ["NOP"] * 10,
                    label="func_a @ A", func="func_a")
        G.add_node("B", instrs=["BL func_a"] + ["NOP"] * 10,
                    label="func_b @ B", func="func_b")
        G.add_edge("A", "B", type="inter-function", conditional=False)
        G.add_edge("B", "A", type="inter-function", conditional=False)

        summarizer = _build_summarizer(G, [
            "Calls func_b recursively.",
            "Calls func_a recursively.",
        ])
        result = asyncio.run(summarizer.summarize_all())

        assert "func_a" in result
        assert "func_b" in result


class TestNoFuncAttribute:
    def test_nodes_without_func_attr(self):
        """Nodes without a 'func' attribute each become their own function group."""
        G = nx.DiGraph()
        G.add_node("A", instrs=["MOV X0, #1", "RET"], label="A")
        G.add_node("B", instrs=["MOV X0, #2", "RET"], label="B")

        summarizer = _build_summarizer(G, [])
        result = asyncio.run(summarizer.summarize_all())

        # Each node becomes its own "function" keyed by node id string
        assert "A" in result
        assert "B" in result
