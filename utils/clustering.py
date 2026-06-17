"""
Adaptive K-Means Clustering
============================
Clustering of patch features within each object for intra-class structure modeling.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Optional, Tuple


def adaptive_cluster(
    feats_selected: np.ndarray,
    max_k: int = 5,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Adaptively cluster patch features using silhouette score to select best K.

    Args:
        feats_selected: Feature array [M, D] (L2-normalized).
        max_k: Maximum number of clusters to try.
        seed: Random seed for reproducibility.

    Returns:
        tuple: (centers [k, D] or None, k_use) where:
            - centers: Cluster center array (None if k=1 or clustering fails)
            - k_use: Number of clusters used (1 if clustering is not applicable)
    """
    num_patches = feats_selected.shape[0]
    k_max_use = min(max_k, num_patches - 1)

    # Not enough samples for clustering
    if num_patches < 4 or k_max_use < 2:
        return None, 1

    best_score = -1.0
    best_k = 1
    best_labels = None
    best_km = None

    try:
        for k_candidate in range(2, k_max_use + 1):
            km = KMeans(
                n_clusters=k_candidate,
                random_state=seed,
                n_init=10,
            )
            labels = km.fit_predict(feats_selected)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            # Avoid tiny clusters (< 2 samples)
            valid_cluster = True
            for cid in unique_labels:
                cluster_size = np.sum(labels == cid)
                if cluster_size < 2:
                    valid_cluster = False
                    break

            if not valid_cluster:
                continue

            score = silhouette_score(feats_selected, labels)

            if score > best_score:
                best_score = score
                best_k = k_candidate
                best_labels = labels
                best_km = km

        # Build cluster centers
        if best_km is not None and best_labels is not None:
            centers_list = []

            for cid in range(best_k):
                mem = np.where(best_labels == cid)[0]
                if len(mem) == 0:
                    center = best_km.cluster_centers_[cid]
                else:
                    center = feats_selected[mem].mean(axis=0)

                center = center / (np.linalg.norm(center) + 1e-12)
                centers_list.append(center)

            centers = np.stack(centers_list, axis=0).astype(np.float32)
            return centers, best_k

    except Exception:
        pass

    return None, 1
