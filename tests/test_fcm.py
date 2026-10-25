"""Unit tests for Fuzzy C-Means clustering (`fsdehaze.fcm`)."""

from __future__ import annotations

import numpy as np
import pytest
import skfuzzy as fuzz

from fsdehaze.fcm import FuzzyCMeans


def _make_two_blobs(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate two well-separated 2D Gaussian blobs."""
    rng = np.random.default_rng(seed)
    c1 = np.array([1.0, 1.0], dtype=np.float64)
    c2 = np.array([8.0, 9.0], dtype=np.float64)
    b1 = rng.normal(loc=c1, scale=0.25, size=(150, 2))
    b2 = rng.normal(loc=c2, scale=0.25, size=(150, 2))
    X = np.vstack([b1, b2])
    true_centers = np.vstack([c1, c2])
    return X, true_centers


def test_fcm_recovers_two_gaussian_blobs() -> None:
    """Two well-separated Gaussian blobs recover the true cluster centres."""
    X, true_centers = _make_two_blobs(seed=42)
    fcm = FuzzyCMeans(c=2, m=2.0, tol=1e-5, max_iter=100, seed=42).fit(X)
    assert fcm.centers_ is not None
    assert fcm.centers_.shape == (2, 2)

    # Sort centres by first coordinate to compare up to cluster permutation
    learned_sorted = fcm.centers_[np.argsort(fcm.centers_[:, 0])]
    true_sorted = true_centers[np.argsort(true_centers[:, 0])]
    np.testing.assert_allclose(learned_sorted, true_sorted, atol=0.1)


def test_fcm_membership_rows_sum_to_one() -> None:
    """Every membership row in predict_membership sums to 1 and lies in [0, 1]."""
    rng = np.random.default_rng(123)
    X = rng.uniform(0.0, 1.0, size=(200, 4))
    fcm = FuzzyCMeans(c=3, m=2.0, tol=1e-4, max_iter=100, seed=42).fit(X)
    U = fcm.predict_membership(X)

    assert U.shape == (200, 3)
    assert np.all(U >= 0.0) and np.all(U <= 1.0)
    np.testing.assert_allclose(np.sum(U, axis=1), np.ones(200), atol=1e-12)


def test_fcm_exact_center_zero_distance() -> None:
    """Points coinciding exactly with a cluster centre (d = 0) get membership 1 for that cluster."""
    X, _ = _make_two_blobs(seed=7)
    fcm = FuzzyCMeans(c=2, m=2.0, tol=1e-5, max_iter=100, seed=42).fit(X)
    assert fcm.centers_ is not None

    # Query points that include exact cluster centres plus a midpoint
    midpoint = np.mean(fcm.centers_, axis=0, keepdims=True)
    queries = np.vstack([fcm.centers_, midpoint])
    U = fcm.predict_membership(queries)

    np.testing.assert_allclose(U[0], np.array([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(U[1], np.array([0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(U[2], np.array([0.5, 0.5]), atol=1e-12)


def test_fcm_crosscheck_against_skfuzzy() -> None:
    """Cross-check centres and memberships against `skfuzzy.cluster.cmeans` on a 3-cluster dataset."""
    rng = np.random.default_rng(99)
    true_c = np.array([[0.1, 0.2, 0.8], [0.8, 0.8, 0.1], [0.2, 0.9, 0.2]])
    blobs = [rng.normal(loc=center, scale=0.06, size=(80, 3)) for center in true_c]
    X = np.vstack(blobs)

    fcm = FuzzyCMeans(c=3, m=2.0, tol=1e-6, max_iter=200, seed=42).fit(X)
    U_ours = fcm.predict_membership(X)
    assert fcm.centers_ is not None

    # skfuzzy expects data of shape (features, samples)
    cntr_sk, u_sk, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=X.T,
        c=3,
        m=2.0,
        error=1e-6,
        maxiter=200,
        seed=42,
    )
    U_sk = u_sk.T  # (N, c)

    # Match cluster permutation by nearest centre
    perm = [int(np.argmin(np.linalg.norm(cntr_sk - c_our, axis=1))) for c_our in fcm.centers_]
    assert sorted(perm) == [0, 1, 2]

    np.testing.assert_allclose(fcm.centers_, cntr_sk[perm], atol=1e-3)
    np.testing.assert_allclose(U_ours, U_sk[:, perm], atol=1e-3)


def test_fcm_deterministic_seed_and_monotone_objective() -> None:
    """Fixed seed gives identical centres, memberships, and non-increasing objective J_m."""
    rng = np.random.default_rng(2025)
    X = rng.uniform(0.0, 1.0, size=(300, 4))

    fcm1 = FuzzyCMeans(c=3, m=2.0, tol=1e-5, max_iter=100, seed=42).fit(X)
    fcm2 = FuzzyCMeans(c=3, m=2.0, tol=1e-5, max_iter=100, seed=42).fit(X)

    assert fcm1.centers_ is not None and fcm2.centers_ is not None
    np.testing.assert_array_equal(fcm1.centers_, fcm2.centers_)
    np.testing.assert_array_equal(fcm1.predict_membership(X), fcm2.predict_membership(X))

    # Objective J_m should decrease across iterations (up to numerical precision)
    diffs = np.diff(fcm1.objective_)
    assert np.all(diffs <= 1e-7)


def test_fcm_parameter_validation() -> None:
    """Invalid parameters raise ValueError or RuntimeError."""
    with pytest.raises(ValueError):
        FuzzyCMeans(c=1)
    with pytest.raises(ValueError):
        FuzzyCMeans(c=3, m=1.0)
    with pytest.raises(RuntimeError):
        FuzzyCMeans(c=3).predict_membership(np.zeros((10, 2)))
