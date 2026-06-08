from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def content_entropy(contents: np.ndarray, n_clusters: int = 8) -> float:
    """Shannon entropy over k-means cluster assignments of content vectors.

    High entropy → creators spread across many distinct niches.
    Low entropy → homogenisation into few clusters.

    Args:
        contents: (M, d) l2-normalised content vectors.
        n_clusters: number of k-means clusters.

    Returns:
        Entropy in nats, in [0, ln(n_clusters)].
    """
    n = contents.shape[0]
    k = min(n_clusters, n)
    if k <= 1:
        return 0.0

    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    labels = km.fit_predict(contents)
    counts = np.bincount(labels, minlength=k).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def coverage(
    contents: np.ndarray,
    user_prefs: np.ndarray,
    threshold: float = 0.3,
) -> float:
    """Fraction of user preference clusters with at least one nearby creator.

    Uses cosine similarity; a creator "covers" a user cluster centre if
    similarity > threshold.

    Args:
        contents:   (M, d) l2-normalised creator content vectors.
        user_prefs: (N, d) l2-normalised user preference vectors.
        threshold:  cosine similarity threshold for coverage.

    Returns:
        Coverage ∈ [0, 1].
    """
    # Compute cosine similarities: (N, M)
    sims = user_prefs @ contents.T
    # A user preference is "covered" if any creator exceeds the threshold
    covered = (sims.max(axis=1) >= threshold).mean()
    return float(covered)


def intra_list_diversity(slates: np.ndarray, contents: np.ndarray) -> float:
    """Mean pairwise cosine *distance* within each user's slate.

    Args:
        slates:   (N, k) integer indices into creator dimension.
        contents: (M, d) l2-normalised content vectors.

    Returns:
        Mean ILD ∈ [0, 1] (higher = more diverse slates).
    """
    n_users, k = slates.shape
    if k < 2:
        return 0.0

    total = 0.0
    n_pairs = k * (k - 1) / 2
    for slate in slates:
        vecs = contents[slate]  # (k, d)
        sims = vecs @ vecs.T    # (k, k) — cosine similarities (already normalised)
        upper = sims[np.triu_indices(k, k=1)]
        total += float((1.0 - upper).mean())
    return total / n_users
