"""Next-POI metrics for semantic-ID generation.

Acc@1 is the metric reported by GNPR-SID. Acc@K is the same quantity as HR@K
when each sample has one ground-truth POI. MRR and NDCG@K use the rank of that
POI in the beam-search list.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

SID_TOKEN = re.compile(r"<[abcd]_\d+>")
DEFAULT_KS = (1, 5, 10)


def extract_sid(text: str | None) -> tuple[str, ...] | None:
    """Return the first semantic ID in ``text``.

    A semantic ID is the token span that starts at ``<a_*>`` and continues
    through the following ``<b_*>``, ``<c_*>`` and optional ``<d_*>`` tokens.
    Extra words around the ID are ignored. ``<d_*>`` is part of the ID, so a
    collided code does not match the code without that suffix.
    """
    if not text:
        return None
    tokens = SID_TOKEN.findall(text)
    start = next((i for i, token in enumerate(tokens) if token.startswith("<a_")), None)
    if start is None:
        return None
    sid = [tokens[start]]
    for token in tokens[start + 1 :]:
        if token.startswith("<a_"):
            break
        sid.append(token)
    return tuple(sid)


def sid_rank(gold: str, predictions: Sequence[str]) -> int | None:
    """1-based rank of the gold semantic ID, or None when it is absent."""
    gold_sid = extract_sid(gold)
    if gold_sid is None:
        return None
    seen: list[tuple[str, ...]] = []
    for prediction in predictions:
        sid = extract_sid(prediction)
        if sid is None or sid in seen:
            continue
        seen.append(sid)
        if sid == gold_sid:
            return len(seen)
    return None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_metrics(
    golds: Sequence[str],
    prediction_lists: Sequence[Sequence[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float | int]:
    """Compute Acc@K, MRR and NDCG@K.

    Samples whose gold text has no semantic ID are skipped. A sample whose
    predictions never contain the gold ID contributes 0 to every metric.
    """
    if len(golds) != len(prediction_lists):
        raise ValueError("golds and prediction_lists must have the same length")

    ranks: list[int | None] = []
    skipped = 0
    for gold, predictions in zip(golds, prediction_lists):
        if extract_sid(gold) is None:
            skipped += 1
            continue
        ranks.append(sid_rank(gold, predictions))

    metrics: dict[str, float | int] = {
        "num_samples": len(ranks),
        "num_skipped": skipped,
    }
    for k in ks:
        hits = [1.0 if rank is not None and rank <= k else 0.0 for rank in ranks]
        metrics[f"Acc@{k}"] = _mean(hits)
        ndcg = [
            1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
            for rank in ranks
        ]
        metrics[f"NDCG@{k}"] = _mean(ndcg)
    metrics["MRR"] = _mean(
        [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    )
    return metrics
