import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from tool_selector.catalog import Tool, catalog_version

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ToolBench/ToolBench_IR_bert_based_uncased"
MODEL_REVISION = "cf4a904c33cde549643aa6d9091146e68ed4932f"
DEFAULT_CACHE = ROOT / ".cache"
DEFAULT_CATALOG = ROOT / "data/tools.jsonl"


def index_fingerprint(tools: list[Tool]) -> str:
    payload = [MODEL_ID, MODEL_REVISION, "mean-cosine-v1",
               [(tool.tool_id, tool.retrieval_text) for tool in tools]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class Selector:
    """Published English ToolBench encoder and exact cosine retrieval."""

    def __init__(self, tools: list[Tool], cache_dir: str | Path = DEFAULT_CACHE, *,
                 build_index: bool = False, index_device: str = "cpu"):
        self.tools = tools
        self.version = catalog_version(tools)
        cache = Path(cache_dir)
        model_path = cache / "model"
        marker = model_path / ".revision"
        if not marker.exists() or marker.read_text().strip() != MODEL_REVISION:
            raise ValueError("Model is not prepared. Run: python -m tool_selector.prepare")
        self.fingerprint = index_fingerprint(tools)
        index_path = cache / "index.npz"
        self.cache_hit = False
        if index_path.exists():
            with np.load(index_path, allow_pickle=False) as saved:
                self.index_metadata = json.loads(str(saved["metadata"].item()))
                if self.index_metadata.get("fingerprint") == self.fingerprint:
                    self.vectors = saved["vectors"]
                    self.cache_hit = True
        if not self.cache_hit and not build_index:
            raise ValueError("Index is missing or stale. Run: python -m tool_selector.prepare")

        os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        self.model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        if not self.cache_hit:
            self.vectors = self.model.encode(
                [tool.retrieval_text for tool in tools], device=index_device,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True, batch_size=64,
            ).astype(np.float32)
            self.index_metadata = {"fingerprint": self.fingerprint, "model_id": MODEL_ID,
                                   "model_revision": MODEL_REVISION, "index_device": index_device}
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, vectors=self.vectors, metadata=json.dumps(self.index_metadata))
            os.replace(temporary, index_path)
        self.model.to("cpu")
        if self.vectors.shape != (len(tools), self.model.get_embedding_dimension()):
            raise ValueError("Index shape does not match the catalog/model; rebuild the index")
        if not np.isfinite(self.vectors).all():
            raise ValueError("Index contains non-finite values; rebuild the index")

    def select(self, query: str, top_k: int = 5) -> list[Tool]:
        if not query.strip() or top_k < 1:
            raise ValueError("Query must be nonempty and top_k must be positive")
        vector = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.vectors @ vector
        indices = np.argsort(-scores, kind="stable")[:min(top_k, len(self.tools))]
        return [self.tools[index] for index in indices]
