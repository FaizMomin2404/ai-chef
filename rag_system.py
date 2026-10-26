"""
RAG (Retrieval-Augmented Generation) knowledge system for AI Chef.

Provides a local, searchable cooking knowledge base using sentence-embeddings
and a vector store (Chroma when available, FAISS as fallback, or an in-memory
index as last resort). All components are optional-degradation safe: if the
embedding model or vector store is unavailable, RAG simply returns no results
and the system falls back to LLM-only answers.

Knowledge is stored as JSONL documents in `knowledge/` and indexed into the
vector store via `index_knowledge.py` (run separately or on startup).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------
@dataclass
class KnowledgeDoc:
    id: str
    title: str
    content: str
    category: str           # recipe | technique | ingredient | safety | faq | substitution
    cuisine: str            # indian | maharashtrian | general | other
    dish: Optional[str] = None
    source: str = "curated" # curated | generated | user
    verified: bool = False  # True only for hand-tested recipes
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "cuisine": self.cuisine,
            "dish": self.dish,
            "source": self.source,
            "verified": self.verified,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeDoc":
        return cls(
            id=d["id"],
            title=d["title"],
            content=d["content"],
            category=d.get("category", "general"),
            cuisine=d.get("cuisine", "general"),
            dish=d.get("dish"),
            source=d.get("source", "curated"),
            verified=d.get("verified", False),
            metadata=d.get("metadata", {}),
        )

    def doc_id_hash(self) -> str:
        """Stable hash for dedup and re-indexing."""
        return hashlib.sha256(
            f"{self.title}|{self.content[:200]}".encode()
        ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Knowledge base loader (JSONL files)
# ---------------------------------------------------------------------------
def _doc_id(title: str, content: str) -> str:
    return hashlib.sha256(f"{title}|{content[:100]}".encode()).hexdigest()[:12]


def load_knowledge_files(directory: str = KNOWLEDGE_DIR) -> List[KnowledgeDoc]:
    """Load all knowledge documents from JSONL files in the directory."""
    docs: List[KnowledgeDoc] = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return docs

    for path in sorted(dir_path.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[RAG] Skipping malformed line {line_no} in {path.name}")
                    continue
                title = data.get("title", "Untitled")
                content = data.get("content", "")
                if not content:
                    continue
                docs.append(KnowledgeDoc(
                    id=data.get("id") or _doc_id(title, content),
                    title=title,
                    content=content,
                    category=data.get("category", "general"),
                    cuisine=data.get("cuisine", "general"),
                    dish=data.get("dish"),
                    source=data.get("source", "curated"),
                    verified=data.get("verified", False),
                    metadata=data.get("metadata", {}),
                ))
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of roughly `size` characters."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            for sep in (". ", "! ", "? ", "\n"):
                idx = text.rfind(sep, start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def doc_chunks(doc: KnowledgeDoc) -> List[Tuple[str, dict]]:
    """Return (chunk_text, metadata) pairs for a document."""
    base_meta = {
        "doc_id": doc.id,
        "title": doc.title,
        "category": doc.category,
        "cuisine": doc.cuisine,
        "dish": doc.dish or "",
        "source": doc.source,
        "verified": doc.verified,
    }
    out: List[Tuple[str, dict]] = []
    for i, chunk in enumerate(chunk_text(doc.content)):
        meta = dict(base_meta)
        meta["chunk_index"] = i
        out.append((chunk, meta))
    return out


# ---------------------------------------------------------------------------
# Embedding provider (optional, graceful degradation)
# ---------------------------------------------------------------------------
class EmbeddingProvider:
    """Lazily loads a sentence-transformers model. Returns None if unavailable."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._tried = False
        self.dimension = 384  # all-MiniLM-L6-v2 default

    def _ensure_loaded(self) -> bool:
        if self._tried:
            return self._model is not None
        self._tried = True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Probe actual dimension
            test = self._model.encode(["test"])
            self.dimension = int(test.shape[1])
            print(f"[RAG] Embedding model loaded: {self.model_name} (dim={self.dimension})")
            return True
        except Exception as e:
            print(f"[RAG] Embedding model unavailable ({e}); RAG disabled.")
            self._model = None
            return False

    def encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self._ensure_loaded():
            return None
        try:
            emb = self._model.encode(texts, convert_to_numpy=True)
            return emb.tolist()
        except Exception as e:
            print(f"[RAG] Encoding failed: {e}")
            return None

    def encode_one(self, text: str) -> Optional[List[float]]:
        r = self.encode([text])
        return r[0] if r else None


# ---------------------------------------------------------------------------
# Vector store backends
# ---------------------------------------------------------------------------
class VectorStore:
    """Abstract vector store with Chroma / FAISS / in-memory backends."""

    def __init__(self, dimension: int, index_dir: str = INDEX_DIR):
        self.dimension = dimension
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store = None
        self._backend = "none"
        self._initialized = False

    def _ensure(self) -> bool:
        if self._initialized:
            return self._backend != "none"
        self._initialized = True
        # Try Chroma
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
            client = chromadb.PersistentClient(
                path=str(self.index_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection(
                name="ai_chef_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            self._store = collection
            self._backend = "chroma"
            print(f"[RAG] Using Chroma vector store at {self.index_dir}")
            return True
        except Exception as e:
            print(f"[RAG] Chroma unavailable: {e}")

        # Try FAISS
        try:
            import faiss  # type: ignore
            import numpy as np
            index = faiss.IndexFlatIP(self.dimension)
            self._store = {"faiss": index, "texts": [], "metas": [], "ids": []}
            self._backend = "faiss"
            print(f"[RAG] Using FAISS vector store (in-memory)")
            return True
        except Exception as e:
            print(f"[RAG] FAISS unavailable: {e}")

        # In-memory last resort
        self._store = {"texts": [], "metas": [], "ids": [], "embs": []}
        self._backend = "memory"
        print("[RAG] Using in-memory vector store (no persistence)")
        return True

    def add(self, ids: List[str], texts: List[str],
            embeddings: List[List[float]], metadatas: List[dict]) -> None:
        if not self._ensure():
            return
        if self._backend == "chroma":
            self._store.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        elif self._backend == "faiss":
            import numpy as np
            store = self._store
            arr = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(arr)
            store["faiss"].add(arr)
            store["ids"].extend(ids)
            store["texts"].extend(texts)
            store["metas"].extend(metadatas)
        else:  # memory
            store = self._store
            store["ids"].extend(ids)
            store["texts"].extend(texts)
            store["metas"].extend(metadatas)
            store["embs"].extend(embeddings)

    def query(self, query_embedding: List[float], top_k: int = 5,
              min_relevance: float = 0.3) -> List[dict]:
        """Return list of {id, text, metadata, score} sorted by relevance."""
        if not self._ensure():
            return []
        if self._backend == "chroma":
            results = self._store.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            out: List[dict] = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for i, doc_id in enumerate(ids):
                # Chroma distance is cosine distance; convert to similarity
                dist = float(dists[i]) if i < len(dists) else 1.0
                score = max(0.0, 1.0 - dist)
                if score >= min_relevance:
                    out.append({
                        "id": doc_id,
                        "text": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "score": round(score, 3),
                    })
            return out

        if self._backend == "faiss":
            import numpy as np
            store = self._store
            if not store["ids"]:
                return []
            q = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(q)
            scores, idxs = store["faiss"].search(q, min(top_k, len(store["ids"])))
            out = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                s = float(score)
                if s >= min_relevance:
                    out.append({
                        "id": store["ids"][idx],
                        "text": store["texts"][idx],
                        "metadata": store["metas"][idx],
                        "score": round(s, 3),
                    })
            return out

        # memory
        store = self._store
        if not store["ids"]:
            return []
        import numpy as np
        q = np.array([query_embedding], dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        embs = np.array(store["embs"], dtype=np.float32)
        embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        sims = (embs_n @ qn.T).flatten()
        order = np.argsort(-sims)[:top_k]
        out = []
        for idx in order:
            s = float(sims[idx])
            if s >= min_relevance:
                out.append({
                    "id": store["ids"][int(idx)],
                    "text": store["texts"][int(idx)],
                    "metadata": store["metas"][int(idx)],
                    "score": round(s, 3),
                })
        return out

    def count(self) -> int:
        if not self._ensure():
            return 0
        if self._backend == "chroma":
            return self._store.count()
        return len(self._store.get("ids", []))

    def reset(self) -> None:
        self._store = None
        self._initialized = False
        if self._backend == "chroma":
            import shutil
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir, ignore_errors=True)
            self.index_dir.mkdir(parents=True, exist_ok=True)


# Global instances (created lazily)
_embedding_provider: Optional[EmbeddingProvider] = None
_vector_store: Optional[VectorStore] = None


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = EmbeddingProvider()
    return _embedding_provider


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        provider = get_embedding_provider()
        _vector_store = VectorStore(provider.dimension)
    return _vector_store


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_knowledge(docs: Optional[List[KnowledgeDoc]] = None,
                    directory: str = KNOWLEDGE_DIR,
                    clear_first: bool = True) -> dict:
    """
    Load knowledge documents, chunk them, embed, and store in the vector store.
    Returns a summary dict.
    """
    if docs is None:
        docs = load_knowledge_files(directory)

    provider = get_embedding_provider()
    store = get_vector_store()
    if clear_first:
        store.reset()

    if not docs:
        return {"indexed": 0, "chunks": 0, "backend": store._backend}

    all_ids: List[str] = []
    all_texts: List[str] = []
    all_embs: List[List[float]] = []
    all_metas: List[dict] = []

    chunk_count = 0
    for doc in docs:
        for chunk_text_val, meta in doc_chunks(doc):
            chunk_id = f"{doc.id}_c{meta['chunk_index']}"
            all_ids.append(chunk_id)
            all_texts.append(chunk_text_val)
            all_metas.append(meta)
            chunk_count += 1

    # Embed in batches
    batch = 32
    for i in range(0, len(all_texts), batch):
        batch_texts = all_texts[i:i + batch]
        embs = provider.encode(batch_texts)
        if embs is None:
            print(f"[RAG] Embedding failed for batch {i}-{i + len(batch_texts)}; skipping.")
            continue
        all_embs.extend(embs)

    if all_embs:
        store.add(all_ids[:len(all_embs)], all_texts[:len(all_embs)],
                  all_embs, all_metas[:len(all_embs)])

    return {
        "indexed": len(docs),
        "chunks": len(all_embs),
        "backend": store._backend,
        "embedding_model": provider.model_name,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 5, min_relevance: float = 0.3) -> List[dict]:
    """Semantic search for cooking knowledge relevant to `query`."""
    provider = get_embedding_provider()
    store = get_vector_store()
    q_emb = provider.encode_one(query)
    if q_emb is None:
        return []
    return store.query(q_emb, top_k=top_k, min_relevance=min_relevance)


_RAG_STOP_WORDS = {
    "with", "and", "for", "the", "a", "an", "of", "some", "easy",
    "simple", "homemade", "recipe", "dish",
}


def _keyword_retrieve_for_dish(dish: str, top_k: int = 5) -> List[dict]:
    """Zero-download fallback used when sentence-transformers is absent.

    It only returns documents that explicitly mention the dish/head word, so
    the hosted deployment still gets grounded context without a 90 MB model.
    """
    dish_l = " ".join((dish or "").lower().split())
    if not dish_l:
        return []
    tokens = [t for t in re.findall(r"[a-z]{3,}", dish_l)
              if t not in _RAG_STOP_WORDS]
    head = tokens[-1] if tokens else dish_l
    scored: List[Tuple[float, dict]] = []
    for doc in load_knowledge_files():
        meta_dish = (doc.dish or "").lower()
        title = doc.title.lower()
        hay = f"{meta_dish} {title} {doc.content.lower()}"
        if dish_l not in hay and head not in hay:
            continue
        score = 1.0 if dish_l in f"{meta_dish} {title}" else 0.7
        score += 0.1 * sum(1 for t in tokens if t in hay)
        metadata = dict(doc.metadata or {})
        metadata.setdefault("dish", doc.dish or doc.title)
        scored.append((score, {
            "id": doc.id,
            "text": f"{doc.title}. {doc.content}",
            "metadata": metadata,
            "score": round(score, 3),
        }))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def retrieve_for_dish(dish: str, top_k: int = 5) -> List[dict]:
    """Retrieve knowledge that genuinely mentions the requested dish.

    Vector similarity alone is not enough: an unrelated document can be the
    closest neighbour. Results are kept only when the dish's head word (or the
    full dish name) appears in the document text or metadata. When nothing
    relevant exists, return an empty list so the caller falls back to LLM
    generation instead of grounding the answer in an unrelated recipe.
    """
    provider = get_embedding_provider()
    store = get_vector_store()
    q_emb = provider.encode_one(dish)
    if q_emb is None:
        return _keyword_retrieve_for_dish(dish, top_k=top_k)

    dish_l = " ".join((dish or "").lower().split())
    if not dish_l:
        return []
    tokens = [t for t in re.findall(r"[a-z]{3,}", dish_l)
              if t not in _RAG_STOP_WORDS]
    head = tokens[-1] if tokens else dish_l

    results = store.query(q_emb, top_k=top_k * 2, min_relevance=0.25)
    filtered: List[dict] = []
    for r in results:
        text = str(r.get("text", "")).lower()
        meta_dish = str(r.get("metadata", {}).get("dish", "")).lower()
        hay = f"{meta_dish} {text}"
        if dish_l in hay or (head and head in hay):
            filtered.append(r)
    return filtered[:top_k]


def knowledge_stats() -> dict:
    provider = get_embedding_provider()
    store = get_vector_store()
    return {
        "embedding_model": provider.model_name,
        "embedding_loaded": provider._model is not None,
        "backend": store._backend,
        "indexed_chunks": store.count(),
    }