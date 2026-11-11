"""Dependency-free tf-idf implemented in pure numpy (no scikit-learn)."""

import re
from collections import Counter

import numpy as np

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or per that the "
    "this to was were will with within after before during above below minimum maximum "
    "the and"
    .split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alpha-numeric tokens with stopwords and 1-char tokens removed."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class TfidfIndex:
    """Cosine-similarity tf-idf retriever over a set of documents, in numpy."""

    def __init__(self, documents):
        """
        documents: list of (doc_id, text) tuples.
        """
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.tokenized = [tokenize(text) for _, text in documents]
        self.vocab = sorted({t for toks in self.tokenized for t in toks})
        self.term_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.matrix = self._build_matrix()

    def _build_matrix(self):
        n_docs = len(self.tokenized)
        n_terms = len(self.vocab)
        if n_terms == 0:
            return np.zeros((n_docs, 0), dtype=np.float64)
        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for i, toks in enumerate(self.tokenized):
            for t, c in Counter(toks).items():
                tf[i, self.term_to_idx[t]] = c
        # l2-normalized term frequency
        norms = np.linalg.norm(tf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tf_norm = tf / norms
        # smoothed idf
        df = (tf > 0).sum(axis=0).astype(np.float64)
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return tf_norm * idf

    def search(self, query, top_k=5):
        """Return [(doc_id, score)] ranked by cosine similarity, best first."""
        q_toks = [t for t in tokenize(query) if t in self.term_to_idx]
        if not q_toks or self.matrix.shape[1] == 0:
            return []
        q = np.zeros(self.matrix.shape[1], dtype=np.float64)
        for t, c in Counter(q_toks).items():
            q[self.term_to_idx[t]] = c
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        scores = self.matrix @ q
        order = np.argsort(-scores)
        results = []
        for i in order:
            score = float(scores[i])
            if score <= 0.0:
                continue
            results.append((self.doc_ids[i], score))
            if len(results) >= top_k:
                break
        return results
