# graph.py — backward-compatibility shim
# The pipeline has been split into two separate graphs:
#   ingest_graph.py  — document fetching + chunking
#   extract_graph.py — LLM extraction + validation + artifact saving
#
# build_graph() is kept here so existing code that imports it continues to work.

from .extract_graph import build_extract_graph as build_graph, ExtractState as GraphState

__all__ = ["build_graph", "GraphState"]
