"""
reuse/index.py — the RECAP plan library and retrieval.

Plans are indexed by the embedding of the question they solved. Retrieval is
per-dataset (a query only ever matches plans from its own dataset) via one FAISS
IndexFlatIP per dataset over L2-normalized MiniLM vectors, so inner product is
exact cosine similarity. IndexFlatIP is exact (no training, deterministic) and
fast on CPU at seed-split scale; swap in IndexHNSWFlat here if a dataset's
library ever grows large enough to need approximate search.

Retrieval returns the top-k (plan, score) candidates above the similarity
threshold τ; the τ pre-filter avoids a judge call when nothing is plausibly
relevant (important on a cold cache). Scores are surfaced for τ tuning/reporting.
"""

import logging
import numpy as np

from recap.embedding import get_embedding, embed_texts

logger = logging.getLogger(__name__)


def _normalize(vecs):
    """L2-normalize rows so IndexFlatIP inner product == cosine similarity.

    MiniLM already returns unit vectors, but a different backend (e.g. OpenAI)
    may not, so we normalize. Zero vectors are left as-is.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class PlanIndex:
    """per-dataset FAISS index over question embeddings -> compiled Plans.

    usage:
        index = PlanIndex(dim=384)
        index.add(plan)                                  # embeds plan.question
        hits = index.search(q_new, dataset="sql_api_bench", top_k=3, tau=0.5)
        # hits: list[(Plan, score)] sorted by score desc, all >= tau
        # (tau is RECAP's retrieval pre-filter; default 0.5, per-dataset via config)
    """

    def __init__(self, dim=None, embedding_provider=None, embedding_model=None, dedup=False):
        self.dim = dim
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.dedup = dedup                 # fold same-dataset + same-tool-sequence plans as aliases
        self._indexes = {}                 # dataset -> faiss.IndexFlatIP
        self._plans = {}                   # dataset -> list[Plan] (position aligns with faiss id)
        self._last_embed = None            # (text, vec) memo; see _embed_one

    @property
    def size(self):
        return sum(len(v) for v in self._plans.values())

    def size_for(self, dataset):
        return len(self._plans.get(dataset, []))

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def _embed_one(self, text):
        """embed one text, memoizing the most recent call.

        On a cache miss the engine embeds the query in search(), then indexes a
        plan whose question IS that same query — the identical text, embedded
        twice per miss. Embedding is deterministic for a given text/model, so a
        one-entry memo removes that duplicate encode without changing any score.
        """
        if self._last_embed is not None and self._last_embed[0] == text:
            return self._last_embed[1]
        vec = get_embedding(text, provider=self.embedding_provider, model=self.embedding_model)
        vec = _normalize(vec)
        if self.dim is None:
            self.dim = vec.shape[1]
        self._last_embed = (text, vec)
        return vec

    def _ensure_index(self, dataset):
        if dataset not in self._indexes:
            if self.dim is None:
                raise ValueError("index dim unknown; add a plan or pass dim= to PlanIndex")
            import faiss
            self._indexes[dataset] = faiss.IndexFlatIP(self.dim)
            self._plans[dataset] = []
        return self._indexes[dataset]

    def add(self, plan):
        """index a compiled Plan under its dataset, keyed by its question embedding.

        with dedup=True, a same-dataset plan whose tool_sequence matches an
        existing one folds this plan's question in as an alias instead of storing
        a duplicate. returns True if a new entry was
        stored, False if it was folded as an alias.
        """
        if plan is None:
            return False

        if self.dedup:
            existing = self._find_same_sequence(plan)
            if existing is not None:
                if plan.question not in existing.aliases and plan.question != existing.question:
                    existing.aliases.append(plan.question)
                return False

        vec = self._embed_one(plan.question)
        index = self._ensure_index(plan.dataset)
        index.add(vec)
        self._plans[plan.dataset].append(plan)
        return True

    def add_many(self, plans):
        """index several plans; uses batch embedding when not deduping."""
        plans = [p for p in plans if p is not None]
        if not plans:
            return 0
        if self.dedup:
            return sum(1 for p in plans if self.add(p))

        # group by dataset and batch-embed each group's questions
        stored = 0
        by_dataset = {}
        for p in plans:
            by_dataset.setdefault(p.dataset, []).append(p)
        for dataset, group in by_dataset.items():
            vecs = _normalize(embed_texts(
                [p.question for p in group],
                provider=self.embedding_provider, model=self.embedding_model,
            ))
            if self.dim is None:
                self.dim = vecs.shape[1]
            index = self._ensure_index(dataset)
            index.add(vecs)
            self._plans[dataset].extend(group)
            stored += len(group)
        return stored

    def _find_same_sequence(self, plan):
        seq = plan.tool_sequence
        for p in self._plans.get(plan.dataset, []):
            if p.tool_sequence == seq:
                return p
        return None

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(self, question, dataset, top_k=3, tau=0.0):
        """return up to top_k (Plan, score) for `dataset`, sorted by score desc.

        only plans from `dataset` are considered (native per-dataset isolation).
        results are filtered to score >= tau; an empty list means cold cache or
        clear mismatch (engine then skips the judge and routes to the big agent).
        """
        plans = self._plans.get(dataset)
        if not plans:
            return []

        qvec = self._embed_one(question)
        index = self._indexes[dataset]
        k = min(top_k, len(plans))
        scores, ids = index.search(qvec, k)

        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:                    # faiss pads with -1 when k > ntotal
                continue
            if float(score) < tau:
                continue
            hits.append((plans[idx], float(score)))
        return hits

    # ------------------------------------------------------------------
    # persistence (plans only; embeddings recomputed on load)
    # ------------------------------------------------------------------

    def to_dicts(self):
        """serialize all plans (grouped by dataset) for JSON storage."""
        return {ds: [p.to_dict() for p in plans] for ds, plans in self._plans.items()}

    def load_dicts(self, data):
        """rebuild the index from serialized plans (re-embeds each question)."""
        from recap.reuse.plan import Plan
        for ds, plan_dicts in data.items():
            plans = [Plan.from_dict(d) for d in plan_dicts]
            self.add_many(plans) if not self.dedup else [self.add(p) for p in plans]
