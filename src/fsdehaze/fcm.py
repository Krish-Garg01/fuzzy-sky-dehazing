"""Vectorized NumPy implementation of Fuzzy C-Means (FCM) clustering (Bezdek et al., 1984)."""

from __future__ import annotations

import numpy as np


class FuzzyCMeans:
    """Fuzzy C-Means clustering implemented in pure NumPy.

    Minimises the objective:
        J_m(U, V) = sum_{i=1}^N sum_{k=1}^c (u_{ik})^m * ||x_i - v_k||^2

    Attributes:
        c: Number of clusters (>= 2).
        m: Fuzzifier exponent (> 1.0, default 2.0).
        tol: Stopping tolerance on max |U_new - U_old| (default 1e-4).
        max_iter: Maximum alternating-optimisation iterations (default 100).
        seed: Random seed for deterministic centre initialisation.
        centers_: Learned cluster centres of shape (c, F), dtype float64.
        objective_: List of objective values J_m across iterations.
        n_iter_: Number of iterations executed during `.fit()`.
    """

    def __init__(
        self,
        c: int = 3,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        if c < 2:
            raise ValueError(f"Number of clusters c must be >= 2, got {c}")
        if m <= 1.0:
            raise ValueError(f"Fuzzifier m must be > 1.0, got {m}")
        if tol <= 0.0:
            raise ValueError(f"Tolerance tol must be > 0, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")

        self.c = int(c)
        self.m = float(m)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

        self.centers_: np.ndarray | None = None
        self.objective_: list[float] = []
        self.n_iter_: int = 0

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """Initialise cluster centres deterministically via seeded k-means++."""
        rng = np.random.default_rng(self.seed)
        N, F = X.shape
        centers = np.empty((self.c, F), dtype=np.float64)

        # Pick first centre deterministically near the median of the first feature, then k-means++
        first_idx = int(rng.integers(0, N))
        centers[0] = X[first_idx]

        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for k in range(1, self.c):
            total = float(np.sum(closest_dist_sq))
            if total <= 1e-12:
                idx = int(rng.integers(0, N))
            else:
                probs = closest_dist_sq / total
                idx = int(rng.choice(N, p=probs))
            centers[k] = X[idx]
            dist_sq_new = np.sum((X - centers[k]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new)

        return centers

    @staticmethod
    def _compute_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix D of shape (N, c)."""
        # (N, 1, F) - (1, c, F) -> (N, c, F)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def _distances_to_memberships(self, dist: np.ndarray) -> np.ndarray:
        """Convert Euclidean distances (N, c) to fuzzy memberships (N, c), handling d_ik = 0."""
        N, c = dist.shape
        U = np.zeros((N, c), dtype=np.float64)
        zero_mask = dist <= 1e-12
        has_zero = np.any(zero_mask, axis=1)

        # 1. Exact centre hits (d_ik == 0): distribute membership 1 equally among zero-distance centres
        if np.any(has_zero):
            z_rows = zero_mask[has_zero]
            counts = np.sum(z_rows, axis=1, keepdims=True)
            U[has_zero] = z_rows.astype(np.float64) / counts

        # 2. Standard FCM update for points not coinciding with a centre
        non_zero = ~has_zero
        if np.any(non_zero):
            d_nz = dist[non_zero]  # (M, c), all > 1e-12
            power = 2.0 / (self.m - 1.0)
            # u_ik = 1 / sum_j (d_ik / d_ij)^power  =  d_ik^(-power) / sum_j d_ij^(-power)
            inv_d_pow = d_nz ** (-power)
            denom = np.sum(inv_d_pow, axis=1, keepdims=True)
            U[non_zero] = inv_d_pow / np.maximum(denom, 1e-300)

        return U

    def fit(self, X: np.ndarray) -> FuzzyCMeans:
        """Fit Fuzzy C-Means cluster centres on data matrix X.

        Args:
            X: Data matrix of shape (N, F), dtype float32 or float64, with N >= c.

        Returns:
            `self` with `.centers_`, `.objective_`, and `.n_iter_` populated.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array (N, F) in FuzzyCMeans.fit, got {X_arr.shape}")
        N, _ = X_arr.shape
        if N < self.c:
            raise ValueError(f"Number of samples N={N} must be >= c={self.c}")

        centers = self._init_centers(X_arr)
        dist = self._compute_distances(X_arr, centers)
        U = self._distances_to_memberships(dist)

        self.objective_ = []
        self.n_iter_ = 0

        for it in range(1, self.max_iter + 1):
            U_old = U
            um = U ** self.m  # (N, c)

            # Update centres: v_k = sum_i u_ik^m x_i / sum_i u_ik^m
            denom = np.sum(um, axis=0)[:, np.newaxis]  # (c, 1)
            centers = (um.T @ X_arr) / np.maximum(denom, 1e-12)

            # Update distances and memberships
            dist = self._compute_distances(X_arr, centers)
            U = self._distances_to_memberships(dist)

            # Record objective J_m = sum_i sum_k u_ik^m d_ik^2
            obj = float(np.sum((U ** self.m) * (dist ** 2)))
            self.objective_.append(obj)
            self.n_iter_ = it

            max_diff = float(np.max(np.abs(U - U_old)))
            if max_diff < self.tol:
                break

        self.centers_ = centers
        return self

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        """Predict fuzzy membership matrix U for arbitrary samples X using learned centres.

        Args:
            X: Data matrix of shape (N, F), dtype float32 or float64.

        Returns:
            Membership matrix `U` of shape (N, c), dtype float64, in [0, 1] with rows summing to 1.
        """
        if self.centers_ is None:
            raise RuntimeError("FuzzyCMeans must be fitted via .fit(X) before calling .predict_membership(X).")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.centers_.shape[1]:
            raise ValueError(
                f"Expected X of shape (N, {self.centers_.shape[1]}), got {X_arr.shape}"
            )
        dist = self._compute_distances(X_arr, self.centers_)
        return self._distances_to_memberships(dist)
