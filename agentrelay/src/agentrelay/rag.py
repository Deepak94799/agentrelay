"""Dependency-free retrieval: paragraph chunking + TF-IDF cosine similarity."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_STOP = frozenset(
    "a an and are as at be by for from how i in is it of on or our that the this to was we what when who with you".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Hit:
    source: str
    text: str
    score: float


class Retriever:
    def __init__(self) -> None:
        self._chunks: list[tuple[str, str]] = []
        self._vecs: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    @classmethod
    def from_dir(cls, directory: str | Path) -> "Retriever":
        r = cls()
        for path in sorted(Path(directory).glob("**/*.md")):
            heading = ""
            for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if chunk.startswith("#") and "\n" not in chunk:
                    heading = chunk  # section context: every paragraph below inherits its heading
                    continue
                r._chunks.append((path.name, f"{heading}\n{chunk}".strip() if heading else chunk))
        r._build()
        return r

    def _build(self) -> None:
        docs = [Counter(tokenize(text)) for _, text in self._chunks]
        n = len(docs)
        df: Counter[str] = Counter(term for d in docs for term in d)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = [self._weigh(d) for d in docs]

    def _weigh(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: c * self._idf[t] for t, c in counts.items() if t in self._idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self._weigh(Counter(tokenize(query)))
        scored = [
            Hit(src, text, round(sum(w * vec.get(t, 0.0) for t, w in q.items()), 4))
            for (src, text), vec in zip(self._chunks, self._vecs)
        ]
        return [h for h in sorted(scored, key=lambda h: h.score, reverse=True)[:k] if h.score > 0]

    def __len__(self) -> int:
        return len(self._chunks)
