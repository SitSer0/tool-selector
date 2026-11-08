"""Prepare the published model, benchmark data, and local index once."""
import argparse
import json
import os
import random
import re
import tempfile
import urllib.request
from pathlib import Path

from tool_selector.catalog import load_catalog
from tool_selector.search import DEFAULT_CACHE, MODEL_ID, MODEL_REVISION, ROOT, Selector

DATASET_REVISION = "6f868e5b4c46a9f04c89f7179ddc7c8f162c1068"
DATASET_URL = "https://huggingface.co/datasets/mteb/ToolBench"


def retrieval_text(document):
    header = ", ".join([document.get("category_name") or "", document.get("tool_name") or "",
                        document.get("api_name") or document.get("name") or "",
                        document.get("api_description") or document.get("description") or ""])
    return (header + ", required_params: " + json.dumps(document.get("required_parameters", ""))
            + ", optional_params: " + json.dumps(document.get("optional_parameters", ""))
            + ", return_schema: " + json.dumps(document.get("template_response", "")))


def input_schema(document):
    types = {"STRING": "string", "NUMBER": "number", "INTEGER": "integer",
             "BOOLEAN": "boolean", "OBJECT": "object", "ARRAY": "array"}
    properties, required = {}, []
    for field in ("required_parameters", "optional_parameters"):
        for parameter in document.get(field) or []:
            name = parameter["name"]
            kind = types.get(str(parameter.get("type", "STRING")).upper(), "string")
            properties[name] = {"type": kind, "description": str(parameter.get("description") or "")}
            if kind == "array":
                properties[name]["items"] = {}
            if field == "required_parameters" and name not in required:
                required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def prepare_data():
    import pyarrow.parquet as pq

    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    if (output / "tools.jsonl").exists() and (output / "queries.jsonl").exists():
        return
    tables = {}
    with tempfile.TemporaryDirectory() as temporary:
        for part in ("corpus", "queries", "qrels"):
            url = f"{DATASET_URL}/resolve/{DATASET_REVISION}/ToolBench-{part}/test-00000-of-00001.parquet"
            path = Path(temporary) / f"{part}.parquet"
            print(f"Downloading {part}...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as stream:
                import shutil
                shutil.copyfileobj(response, stream)
            tables[part] = pq.read_table(path).to_pylist()
    tools = []
    for row in tables["corpus"]:
        doc = json.loads(row["text"])
        name = str(doc.get("name") or row["id"])
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:100]
        tools.append({"tool_id": row["id"], "name": f"{safe_name}__{row['id']}",
                      "description": f"{doc.get('category_name', '')}. {name}. {str(doc.get('description') or '').strip()}",
                      "input_schema": input_schema(doc), "retrieval_text": retrieval_text(doc)})
    labels = {}
    for row in tables["qrels"]:
        if row["score"] > 0:
            labels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    groups = {}
    for row in tables["queries"]:
        if row["id"] in labels:
            key = " ".join(row["text"].casefold().split())
            groups.setdefault(key, []).append({"query_id": row["id"], "query": row["text"],
                                               "relevant_tool_ids": sorted(labels[row["id"]])})
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    # Preserve the same 80% test partition; keep normalized duplicates together.
    queries = [case for key in keys[max(1, len(keys) // 5):] for case in groups[key]]
    for filename, rows in (("tools.jsonl", tools), ("queries.jsonl", queries)):
        (output / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu", help="Index construction device; queries always use CPU")
    args = parser.parse_args()
    prepare_data()
    DEFAULT_CACHE.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE / "huggingface"))
    marker = DEFAULT_CACHE / "model/.revision"
    if not marker.exists() or marker.read_text().strip() != MODEL_REVISION:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=DEFAULT_CACHE / "model",
                          allow_patterns=["config.json", "config_sentence_transformers.json", "modules.json",
                                          "1_Pooling/config.json", "pytorch_model.bin", "sentence_bert_config.json",
                                          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
        marker.write_text(MODEL_REVISION)
    selector = Selector(load_catalog(ROOT / "data/tools.jsonl"), build_index=True, index_device=args.device)
    import tiktoken
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DEFAULT_CACHE / "tokenizer"))
    tiktoken.get_encoding("cl100k_base")
    print(f"Ready: {len(selector.tools)} tools; index {'cached' if selector.cache_hit else 'built'}")


if __name__ == "__main__":
    main()
