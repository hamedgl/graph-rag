# Graph RAG Learning Plan

Core idea: vector RAG retrieves text that is *similar* to the question; graph RAG retrieves *connections*.
Wins on relational questions: "Who reports to the owner of project X?", "Which suppliers do A and B share?"

## Stage 0 — Concepts (no code)
- [x] 0.1 Why vector RAG fails on relational questions (multi-hop, aggregation, facts scattered across chunks)
- [x] 0.2 Graph basics: nodes, edges, properties, labeled property graph vs triples (subject-predicate-object)
- [x] 0.3 End-to-end pipeline: docs → extraction → graph → retrieval → LLM answer
- [x] 0.4 Retrieval styles:
  - Text2Cypher (LLM writes graph query)
  - Entity-anchored traversal (find entities in question, expand neighbors)
  - Hybrid (vector search for entry points, then graph walk)
  - Community summaries (Microsoft GraphRAG, global "what are the themes" questions)

## Stage 1 — PoC (small, local)
| # | Component | Choice | What you learn |
|---|---|---|---|
| [x] 1.1 | Data | 20–50 docs, one domain | Scoping |
| [x] 1.2 | Schema | 3–5 node types, 5–8 relation types, hand-defined | Ontology thinking |
| [x] 1.3 | Extraction | LLM extracts triples from text | Extraction quality, hallucinated edges |
| [x] 1.4 | Store | Neo4j (Docker) | Cypher, graph modeling |
| [x] 1.5 | Retrieval | Text2Cypher first, then entity traversal | Query generation and failure modes |
| [x] 1.6 | Answer | LLM over retrieved subgraph | Grounding |
| [x] 1.7 | Eval | 15–20 hand-written relational questions vs plain vector RAG | Proof the graph helps |

Exit criterion: can explain, with own examples, where graph beats vectors and where it breaks.

## Stage 2 — Real-world scenario
- [x] 2.0 Benchmark LightRAG vs our pipeline: same 20 docs, same 19 questions, same code scoring (stage1/eval.py). Compare accuracy by question type, indexing cost/time, and failure modes (refuse vs confidently wrong)
- [x] 2.1 Entity resolution ("IBM" = "I.B.M." = "International Business Machines") — hardest real problem
- [x] 2.2 Schema evolution: fixed ontology vs open extraction
- [x] 2.3 Incremental ingestion: updates, deletes, provenance (edge → source doc)
- [x] 2.4 Hybrid retrieval: vector + graph + keyword, query router chooses strategy
- [x] 2.5 Structured (DB/CSV) + unstructured (LLM-extracted) data merged in one graph
- [x] 2.6 Evaluation at scale: golden set, retrieval metrics, answer faithfulness
- [x] 2.7 Ops: latency, cost, Cypher-injection guards, access control, observability

## Open questions (asked one at a time)
- [x] Domain / dataset → toy company org
- [x] LLM → OpenAI API
- [x] Learning format → concepts + minimal code

## Stack (decided 2026-10-12)
Neo4j · own two-pass extraction (stage1/extract.py) · own retrieval/answer code (option: neo4j-graphrag
components) · plain-Python routing/retries · LightRAG as external benchmark (2.0).
LangChain track dropped; code archived in archive/stage1_langchain (see archive/README.md).

## Resume notes
- Start DB: `docker compose up -d` (Neo4j; data persists). Stop: `docker compose down`.
- Run anything with `.venv/bin/python` from the project root.
