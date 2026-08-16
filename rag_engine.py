

import pickle
import re
from pathlib import Path

import numpy as np
import faiss
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

import config


#parsing
def _chunk_text(all_text):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", all_text) if p.strip()]
    records, words_buf = [], []
    for para in paragraphs:
        words = para.split()
        if words_buf and len(words_buf) + len(words) > config.CHUNK_WORDS:
            records.append(" ".join(words_buf))
            words_buf = words_buf[-config.OVERLAP_WORDS:]
        words_buf.extend(words)
    if words_buf:
        records.append(" ".join(words_buf))
    return records


def parse_pdf(path):
    import pdfplumber
    full_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text.strip())
    return _chunk_text("\n\n".join(full_text))


def load_source(source):
    ext = Path(source).suffix.lower()
    if ext != ".pdf":
        raise ValueError(f"Unsupported file type: {ext} (only .pdf is supported)")
    return parse_pdf(source), Path(source).name


#main engine

class RagEngine:
    """Wraps the notebook's index-building, hybrid search, and QA pipeline
    so the Flask app can call it like any other service object."""

    def __init__(self):
        self.embedder = None
        self.reranker = None
        self.index = None
        self.chunks = []          # [{"text": ..., "source": ...}, ...]
        self.bm25 = None
        self.conversation_history = []
        self.status_log = []      # progress lines surfaced to the UI while loading
        self.ready = False
        self.error = None

    def _log(self, msg):
        self.status_log.append(msg)

    def initialize(self):
        #build or load faiss index 
        try:
            Path(config.INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
            self.embedder = SentenceTransformer(config.EMBED_MODEL)

            if config.USE_RERANKER:
                self._log(f"Loading reranker ({config.RERANK_MODEL})...")
                self.reranker = CrossEncoder(config.RERANK_MODEL)

            if (not config.FORCE_REBUILD
                    and Path(config.INDEX_PATH).exists()
                    and Path(config.META_PATH).exists()):
                self._log("Loading saved index...")
                self.index = faiss.read_index(config.INDEX_PATH)
                with open(config.META_PATH, "rb") as f:
                    self.chunks = pickle.load(f)
                self._log(f"Loaded — {self.index.ntotal} chunks")
            else:
                self._log("Building index from scratch...")
                dim = self.embedder.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(dim)
                self.chunks = []

                for source in config.SOURCES:
                    if not Path(source).exists():
                        self._log(f"Skipping (not found): {source}")
                        continue
                    self._log(f"Loading: {source}")
                    records, label = load_source(source)
                    records = [r for r in records if len(r.strip()) > 20]
                    self._log(f"  -> {len(records)} chunks")
                    if not records:
                        continue

                    embeddings = self.embedder.encode(
                        records, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    for txt, emb in zip(records, embeddings):
                        self.chunks.append({"text": txt, "source": label})
                    self.index.add(np.array(embeddings, dtype="float32"))

                faiss.write_index(self.index, config.INDEX_PATH)
                with open(config.META_PATH, "wb") as f:
                    pickle.dump(self.chunks, f)
                self._log(f"Done — {self.index.ntotal} chunks indexed")

            tokenized = [c["text"].lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized) if tokenized else None
            self.ready = True
        except Exception as e:
            self.error = str(e)
            raise

    def sources_summary(self):
        #sources indexed with chunk counts 
        counts = {}
        for c in self.chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]

    def _dense_search(self, query, top_k):
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_emb, top_k)
        return [
            {"text": self.chunks[i]["text"], "source": self.chunks[i]["source"], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]

    def _bm25_search(self, query, top_k):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_ids = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.chunks[i]["text"], "source": self.chunks[i]["source"], "score": float(scores[i])}
            for i in top_ids
        ]

    def _hybrid_rrf(self, query, pool_k):
        #rrf 
        candidate_k = pool_k * 3
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        dense_scores, dense_ids = self.index.search(q_emb, candidate_k)

        rrf = {}
        for rank, idx in enumerate(dense_ids[0]):
            if idx >= 0:
                rrf[int(idx)] = rrf.get(int(idx), 0) + 0.55 / (rank + config.RRF_K)

        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(query.lower().split())
            bm25_ids = np.argsort(bm25_scores)[::-1][:candidate_k]
            for rank, idx in enumerate(bm25_ids):
                rrf[int(idx)] = rrf.get(int(idx), 0) + 0.45 / (rank + config.RRF_K)

        top = sorted(rrf.items(), key=lambda x: -x[1])[:pool_k]
        return [
            {"text": self.chunks[i]["text"], "source": self.chunks[i]["source"], "score": s}
            for i, s in top
        ]

    def _rerank(self, query, candidates, top_k):
        if not candidates or self.reranker is None:
            return candidates[:top_k]
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: -c["rerank_score"])
        return candidates[:top_k]

    def search(self, query, top_k=None, mode=None):
        #default mode is hybrid rerank 
        top_k = top_k or config.TOP_K
        if mode is None:
            mode = "hybrid_rerank" if (config.USE_RERANKER and self.reranker is not None) else "hybrid"

        if mode == "dense":
            return self._dense_search(query, top_k)
        if mode == "bm25":
            return self._bm25_search(query, top_k)
        if mode == "hybrid":
            return self._hybrid_rrf(query, top_k)
        if mode == "hybrid_rerank":
            pool = self._hybrid_rrf(query, max(top_k, config.RERANK_POOL))
            return self._rerank(query, pool, top_k)
        raise ValueError(f"Unknown search mode: {mode}")

    def ask(self, question, top_k=None):
        top_k = top_k or config.FINAL_K
        results = self.search(question, top_k=max(top_k, config.TOP_K))
        if not results:
            return {"answer": "I couldn't find this in the documents provided.", "sources": []}

        context = "\n\n---\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in results[:config.FINAL_K]
        )
        user_msg = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
        self.conversation_history.append({"role": "user", "content": user_msg})

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT}] + self.conversation_history,
        )
        answer = response["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": answer})

        seen = {}
        for r in results[:config.FINAL_K]:
            display_score = r.get("rerank_score", r["score"])
            seen.setdefault(r["source"], round(display_score, 4))
        sources = [{"name": name, "score": score} for name, score in seen.items()]

        return {"answer": answer, "sources": sources}

    def reset(self):
        self.conversation_history = []


engine = RagEngine() 