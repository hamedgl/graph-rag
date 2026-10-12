"""Shared LangChain setup: LLM, read-only + admin Neo4j handles, schema from stage1/schema.py."""
import os
import sys
from pathlib import Path

STAGE1 = Path(__file__).parent.parent / "stage1"
sys.path.append(str(STAGE1))  # reuse stage1: env loader, schema, scoring, data

import env  # noqa: E402,F401  (loads .env)
from langchain_neo4j import Neo4jGraph  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from schema import NODES, RELATIONS, RULES  # noqa: E402,F401

DATA = STAGE1 / "data"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# Gateway only enforces structured output on the Responses API (tested in 1.3).
llm = ChatOpenAI(model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_BASE_URL"],
                 api_key=os.environ["LLM_API_KEY"], use_responses_api=True)


def graph(read_only=True):
    """Separate Neo4j (port 7688, APOC) so LangChain's graph never mixes with stage1's."""
    g = Neo4jGraph(url="bolt://localhost:7688", username="neo4j", password=os.environ["NEO4J_PASSWORD"],
                   refresh_schema=False)
    if read_only:
        write_query = g.query
        g.query = lambda q, params=None, session_params=None: write_query(
            q, params or {}, session_params={**(session_params or {}), "default_access_mode": "READ"})
    return g
