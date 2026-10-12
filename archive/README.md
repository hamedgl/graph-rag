# Archive

`stage1_langchain/` — LangChain version of Stage 1 extraction (dropped 2026-10-12).
Kept for the lesson, not runnable: LangChain packages and its Neo4j container were removed.

Results: LLMGraphTransformer out-of-box 64% precision / 65% recall (47 nodes vs 32 true);
with our two-pass entity catalog + code canonicalization 96% / 99%.
Hand-built two-pass (stage1/extract.py): 100% / 100%.
Takeaways: quality came from design (entity catalog, name locking), not the framework;
`langchain-experimental` (LLMGraphTransformer) is being sunset.
