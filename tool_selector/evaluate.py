"""Evaluate the English ToolBench retriever: python -m tool_selector.evaluate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tool_selector.catalog import load_catalog, read_jsonl
from tool_selector.search import DEFAULT_CACHE, DEFAULT_CATALOG, MODEL_ID, MODEL_REVISION, ROOT, Selector


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str
    query: str = Field(min_length=1)
    relevant_tool_ids: list[str] = Field(min_length=1)


def metrics(predicted: list[str], relevant: list[str], k: int) -> dict:
    actual = set(relevant)
    selected = predicted[:k]
    return {"recall": len(actual.intersection(selected)) / len(actual),
            "complete": float(actual.issubset(selected))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Fixed seeded sample instead of all queries")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    tools = load_catalog(DEFAULT_CATALOG)
    cases = [EvalCase.model_validate(row) for row in read_jsonl(ROOT / "data/queries.jsonl")]
    if not cases or len({c.query_id for c in cases}) != len(cases):
        raise ValueError("Queries must be nonempty with unique query_id")
    if any(not c.query.strip() for c in cases):
        raise ValueError("Queries must not be blank")
    known = {tool.tool_id for tool in tools}
    if {tid for c in cases for tid in c.relevant_tool_ids} - known:
        raise ValueError("Query labels contain unknown tool IDs")
    if args.limit and args.limit < len(cases):
        cases = random.Random(42).sample(cases, args.limit)
    import tiktoken
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DEFAULT_CACHE / "tokenizer"))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    start = time.perf_counter()
    selector = Selector(tools)
    initialization = time.perf_counter() - start
    ks = (5, 10, 15, 20)
    selector.select(cases[0].query, max(ks))  # Unmeasured warmup.
    rows, token_counts = [], {k: [] for k in ks}
    for index, case in enumerate(cases, 1):
        start = time.perf_counter()
        selected = selector.select(case.query, max(ks))
        latency = (time.perf_counter() - start) * 1000
        rows.append({"query_id": case.query_id, "relevant": case.relevant_tool_ids,
                     "predicted": [tool.tool_id for tool in selected], "latency_ms": latency})
        for k in ks:
            text = json.dumps([tool.model_dump() for tool in selected[:k]], ensure_ascii=False, separators=(",", ":"))
            token_counts[k].append(len(tokenizer.encode(text, disallowed_special=())))
        if index % 100 == 0 or index == len(cases):
            print(f"Evaluated {index}/{len(cases)}", flush=True)
    curve = [{"k": k, "recall": float(np.mean([metrics(r['predicted'], r['relevant'], k)['recall'] for r in rows]))}
             for k in range(1, max(ks) + 1)]
    summary = [{"k": k, "recall": curve[k - 1]["recall"],
                "complete": float(np.mean([metrics(r['predicted'], r['relevant'], k)['complete'] for r in rows])),
                "mean_context_tokens": float(np.mean(token_counts[k]))} for k in ks]
    report = {"model": MODEL_ID, "model_revision": MODEL_REVISION, "language_scope": "English",
              "catalog_version": selector.version, "catalog_size": len(tools),
              "queries": len(cases), "query_ids": [c.query_id for c in cases], "seed": 42,
              "queries_sha256": hashlib.sha256(json.dumps([c.model_dump() for c in cases], sort_keys=True).encode()).hexdigest(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ("numpy", "sentence-transformers", "torch", "transformers", "tiktoken")}},
              "index": selector.index_metadata, "initialization_seconds": initialization,
              "latency_p50_ms": float(np.percentile([r['latency_ms'] for r in rows], 50)),
              "latency_p95_ms": float(np.percentile([r['latency_ms'] for r in rows], 95)),
              "latency_scope": "Warm CPU selection of top-20, once per query; excludes HTTP, initialization and token counting. Other k use prefixes of the same ranking.",
              "tokenizer": "cl100k_base, serialized full tool definitions, excludes API framing",
              "minimum_k_for_recall_95_up_to_20": next((x['k'] for x in curve if x['recall'] >= .95), None),
              "summary": summary, "recall_curve": curve, "cases": rows}
    (ROOT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in summary:
        print(f"k={row['k']}: recall={row['recall']:.4f}, complete={row['complete']:.4f}")
    print(f"p95={report['latency_p95_ms']:.1f} ms; saved results.json")


if __name__ == "__main__":
    main()
