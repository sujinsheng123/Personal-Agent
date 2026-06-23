"""EmbeddingMemoryProvider — semantic memory backed by ONNX embeddings.

Storage:
  external_memories.json  — list of {id, text, created_at}
  external_embeddings.npy — (N, 512) float32 array

Model: BAAI/bge-small-zh-v1.5 (Chinese-optimized, 512-dim, ~100MB)
Library: fastembed (ONNX Runtime, no PyTorch)

All methods are thread-safe (asyncio.Lock for writes, numpy read-only for reads).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np

from personal_agent.memory.base import MemoryProvider

logger = logging.getLogger(__name__)

_external_instance: EmbeddingMemoryProvider | None = None


def set_external_instance(instance) -> None:
    global _external_instance
    _external_instance = instance


def get_external_instance():
    return _external_instance


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
RELEVANCE_THRESHOLD = 0.3
MAX_PREFETCH = 3


class EmbeddingMemoryProvider(MemoryProvider):
    """Semantic memory with ONNX embeddings. No vector DB needed."""

    def __init__(self, data_dir: Path, *, model_name: str = DEFAULT_MODEL) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = data_dir / "external_memories.json"
        self._embeddings_path = data_dir / "external_embeddings.npy"
        self._model_name = model_name
        self._model = None  # lazy init — first embed() call downloads model
        self._texts: list[dict] = []       # [{id, text, created_at}]
        self._embeddings: np.ndarray | None = None  # (N, 512)
        self._lock = asyncio.Lock()
        self._load()

    # ── MemoryProvider interface ──────────────────────

    async def save(self, content: str) -> None:
        """Embed and persist a new memory entry."""
        async with self._lock:
            emb = await self._embed(content)
            entry = {
                "id": uuid.uuid4().hex[:12],
                "text": content,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._texts.append(entry)
            e = np.array(emb, dtype=np.float32).reshape(1, -1)
            if self._embeddings is not None and len(self._embeddings) > 0:
                self._embeddings = np.vstack([self._embeddings, e])
            else:
                self._embeddings = e
            self._save_to_disk()
            logger.debug("Embedding saved: %s (%d dims)", content[:50], emb.shape[0])

    async def prefetch(self, user_message: str) -> list[dict]:
        """Search memories → api_messages fragments. NOT persisted."""
        results = await self.search(user_message)
        if not results:
            return []
        return [{
            "role": "user",
            "content": [{"type": "text", "text": f"[相关记忆] {r}"}],
        } for r in results]

    async def search(self, query: str) -> list[str]:
        """Cosine similarity search. Returns texts sorted by relevance."""
        if self._embeddings is None or len(self._texts) == 0:
            return []

        query_vec = await self._embed(query)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Cosine similarity: (A·B) / (||A||·||B||)
        emb_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)
        query_normed = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        scores = (emb_norm @ query_normed).flatten()

        top_k = min(MAX_PREFETCH, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] > RELEVANCE_THRESHOLD:
                results.append(self._texts[i]["text"])
        return results

    async def load_all(self) -> list[str]:
        return [t["text"] for t in self._texts]

    async def ingest_file(self, file_path: str, chunk_size: int = 800) -> int:
        """Chunk a file and embed each chunk. Supports txt, md, pdf, docx, and more.
        Returns number of chunks stored."""
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = _read_file(path)
        chunks = _chunk_text(text, chunk_size)
        count = 0
        for chunk in chunks:
            await self.save(f"[{path.name}] {chunk}")
            count += 1
        logger.info("Ingested %s: %d chunks", path.name, count)
        return count

    def get_system_prompt_text(self) -> str:
        return ""  # external memories go via prefetch → api_messages

    # ── internals ────────────────────────────────────

    async def _embed(self, text: str) -> np.ndarray:
        """Encode a single text → numpy array. Lazy-init model on first call."""
        model = self._get_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: list(model.embed([text]))[0])

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            logger.info("Loading embedding model: %s", self._model_name)
            cache_dir = str(self._dir / ".fastembed_cache")
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=cache_dir)
            _fix_windows_symlinks(cache_dir)  # snapshot for next time
            logger.info("Embedding model ready")
        return self._model

    def _load(self) -> None:
        if self._metadata_path.exists():
            try:
                self._texts = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                logger.info("Loaded %d external memories", len(self._texts))
            except Exception:
                logger.exception("Failed to load external memories")

        if self._embeddings_path.exists():
            try:
                self._embeddings = np.load(self._embeddings_path)
                logger.info("Loaded %d embeddings", len(self._embeddings))
            except Exception:
                logger.exception("Failed to load external embeddings")

    def _save_to_disk(self) -> None:
        self._metadata_path.write_text(
            json.dumps(self._texts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._embeddings is not None:
            np.save(self._embeddings_path, self._embeddings)


def _fix_windows_symlinks(cache_dir: str) -> None:
    """On Windows, copy files from flat cache to snapshot dir.
    Checks file sizes — broken symlinks are 0 bytes but still .exists()."""
    from pathlib import Path as _Path
    cache = _Path(cache_dir)
    flat_dirs = list(cache.glob("fast-*"))
    if not flat_dirs:
        return
    for snap in cache.glob("models--*/snapshots/*"):
        if not snap.is_dir():
            continue
        needs_fix = False
        for f in snap.iterdir():
            if f.is_file() and f.stat().st_size == 0:
                f.unlink()  # remove broken symlink
                needs_fix = True
            elif not f.is_file():
                pass
        if (snap / "model_optimized.onnx").exists() and not needs_fix:
            continue  # all good
        # Copy from flat cache
        for flat in flat_dirs:
            if not flat.is_dir():
                continue
            for f in flat.iterdir():
                if f.is_file():
                    dst = snap / f.name
                    if not dst.exists() or dst.stat().st_size == 0:
                        dst.write_bytes(f.read_bytes())


# ── chunking ──────────────────────────────────────────

def _read_file(path: Path) -> str:
    """Extract text from a file based on extension."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log", ".py", ".rst", ".toml", ".ini", ".cfg"):
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split text into chunks by paragraph, merging short ones."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            if current:
                chunks.append(current[:max_chars])
            current = p
    if current:
        chunks.append(current[:max_chars])
    return chunks or [text[:max_chars]]
