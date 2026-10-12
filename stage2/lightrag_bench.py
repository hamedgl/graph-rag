"""Stage 2.0: LightRAG vs our pipeline on the same 20 docs and 19 questions.

LightRAG does its own extraction + graph + retrieval (mode 'mix' = its default: graph + vector).
Fairness: LightRAG returns ONLY its retrieved context (only_need_context=True); the answer is
written by our shared answer step (stage1/answer.py) and scored by stage1/eval.py's correct().

Run: .venv/bin/python stage2/lightrag_bench.py          (index once, then evaluate)
     .venv/bin/python stage2/lightrag_bench.py --reindex
"""
import asyncio
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "stage1"))
import env  # noqa: E402,F401  (loads .env)
from answer import answer_from_context  # noqa: E402
from eval import correct  # noqa: E402
from retrieve import db  # noqa: E402
from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

STORE = Path(__file__).parent / "lightrag_store"
OUT = Path(__file__).parent / "out"
MODE = "mix"
stats = {"llm_calls": 0, "prompt_chars": 0}


async def llm(prompt, system_prompt=None, history_messages=None, **kw):
    stats["llm_calls"] += 1
    stats["prompt_chars"] += len(prompt) + len(system_prompt or "")
    # LightRAG parses delimited text, not JSON schema -> plain chat.completions is fine here.
    return await openai_complete_if_cache(os.environ["LLM_MODEL"], prompt, system_prompt=system_prompt,
                                          history_messages=history_messages or [],
                                          base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                                          **kw)


_model = None


async def embed(texts, context="document"):
    """Same model + query/document prompts as our vector baseline (stage1/retrieve.py)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("google/embeddinggemma-2")
    enc = _model.encode_query if context == "query" else _model.encode_document
    return enc(list(texts), normalize_embeddings=True)


async def build(reindex):
    if reindex and STORE.exists():
        shutil.rmtree(STORE)
    fresh = not STORE.exists()
    STORE.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(working_dir=str(STORE), llm_model_func=llm, llm_model_name=os.environ["LLM_MODEL"],
                   embedding_func=EmbeddingFunc(embedding_dim=768, func=embed, max_token_size=2048,
                                                supports_asymmetric=True, model_name="embeddinggemma-2"))
    await rag.initialize_storages()
    if fresh:
        docs = sorted((ROOT / "stage1" / "data" / "docs").glob("*.md"))
        t = time.time()
        await rag.ainsert([p.read_text() for p in docs], ids=[p.name for p in docs],
                          file_paths=[p.name for p in docs])  # file_paths -> citations in context
        index = {"seconds": round(time.time() - t, 1), **stats}
        (OUT / "lightrag_index.json").write_text(json.dumps(index))
        print(f"indexed {len(docs)} docs: {index}")
    return rag


async def main():
    OUT.mkdir(exist_ok=True)
    rag = await build("--reindex" in sys.argv)
    index = json.loads((OUT / "lightrag_index.json").read_text())
    questions = json.loads((ROOT / "stage1" / "data" / "questions.json").read_text())

    results, score = [], defaultdict(int)
    for i, q in enumerate(questions, 1):
        context = await rag.aquery(q["q"], param=QueryParam(mode=MODE, only_need_context=True))
        a = answer_from_context(q["q"], context or "", f"lightrag/{MODE}: {len(context or '')} chars")
        ok = correct(q, a)
        score[q["type"]] += ok
        results.append({**q, "system": f"lightrag-{MODE}", "correct": ok, "context_chars": len(context or ""), **a})
        print(f"{i:2}. [{q['type']:11}] {q['q']:60} {'✓' if ok else '✗'}  ({len(context or '')} chars)")
    (OUT / "lightrag_eval.json").write_text(json.dumps(results, indent=2))
    await rag.finalize_storages()

    # Side by side with the hand-built systems from stage1/out/eval.json (Stage 1.7).
    prev = defaultdict(lambda: defaultdict(int))
    for r in json.loads((ROOT / "stage1" / "out" / "eval.json").read_text()):
        prev[r["type"]][r["system"]] += r["correct"]
    n = defaultdict(int)
    for q in questions:
        n[q["type"]] += 1
    systems = ["vector", "traverse", "cypher"]
    print(f"\n{'type':12}{'n':>3}" + "".join(f"{s:>10}" for s in systems) + f"{'lightrag':>10}")
    for t in n:
        print(f"{t:12}{n[t]:>3}" + "".join(f"{prev[t][s]:>10}" for s in systems) + f"{score[t]:>10}")
    print(f"{'TOTAL':12}{len(questions):>3}" + "".join(f"{sum(prev[t][s] for t in n):>10}" for s in systems)
          + f"{sum(score.values()):>10}")
    print(f"\nLightRAG indexing: {index['seconds']}s, {index['llm_calls']} LLM calls, "
          f"{index['prompt_chars']:,} prompt chars (ours: 21 calls = 1 entity pass + 20 docs)")
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
