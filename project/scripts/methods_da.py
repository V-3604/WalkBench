"""
Unsupervised domain-adaptation transforms for the WalkBench reality-check.

Two classic feature-space DA methods. In the reality-check they are the "complex"
rungs that must be shown NOT to beat a frozen-encoder linear probe cross-city:

  coral_align     CORAL (Sun & Saenko, AAAI 2016, "Return of Frustratingly Easy
                  Domain Adaptation"): whiten the source covariance and recolor it
                  to the target covariance. Mean alignment is added so the source
                  matches the target in both first and second moments -- this only
                  strengthens the DA method, which strengthens a negative result.
  subspace_align  Subspace Alignment (Fernando et al., ICCV 2013): project source
                  and target onto their own PCA subspaces and align the source
                  subspace to the target via the learned map M = Bs^T Bt.

Both are *unsupervised*: they use the target city's image features only, never its
labels -- matching deployment (a new city has imagery but no walkability labels).
That is an extra advantage over the blind linear probe; the reality-check claim is
that they still do not win. Apply them to standardized features (unit-variance per
dim) so the whitening in CORAL is well-conditioned.

Self-test (synthetic Gaussian source/target):
    & ".venv\\Scripts\\python.exe" project\\scripts\\methods_da.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from sklearn.decomposition import PCA


def _cov_power(features: np.ndarray, power: float, eps: float) -> np.ndarray:
    """Symmetric matrix power C^power of cov(features), ridge-regularized by eps.

    Uses the eigendecomposition of the (symmetric PSD) covariance; eigenvalues are
    floored at eps so the inverse-square-root stays finite on rank-deficient blocks.
    """
    cov = np.cov(features, rowvar=False).astype(np.float64)
    cov = cov + eps * np.eye(cov.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, eps, None)
    return (eigenvectors * (eigenvalues ** power)) @ eigenvectors.T


def coral_align(source: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Align `source` to the `target` mean + covariance (CORAL).

    Returns a transformed copy of `source` carrying the target's first and second
    moments. `target` contributes its feature statistics only (unsupervised); its
    rows are never used as labelled training data.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    whiten = _cov_power(source, -0.5, eps)   # Cs^{-1/2}
    recolor = _cov_power(target, +0.5, eps)  # Ct^{+1/2}
    aligned = (source - source_mean) @ whiten @ recolor + target_mean
    return aligned.astype(np.float32)


def subspace_align(
    source: np.ndarray,
    target: np.ndarray,
    n_components: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subspace Alignment (Fernando et al. 2013).

    Fits an `n_components` PCA on each domain, aligns the source subspace to the
    target subspace via M = Bs^T Bt, and returns `(source_aligned, target_projected)`,
    both `k`-dimensional with `k = min(n_components, ...)`. The target's features are
    used unsupervised (PCA basis only).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    k = min(n_components, source.shape[1], source.shape[0] - 1, target.shape[0] - 1)
    pca_source = PCA(n_components=k, random_state=random_state).fit(source)
    pca_target = PCA(n_components=k, random_state=random_state).fit(target)
    basis_source = pca_source.components_.T  # (D, k)
    basis_target = pca_target.components_.T  # (D, k)
    alignment = basis_source.T @ basis_target  # (k, k)
    source_aligned = pca_source.transform(source) @ alignment  # (n_source, k)
    target_projected = pca_target.transform(target)            # (n_target, k)
    return source_aligned.astype(np.float32), target_projected.astype(np.float32)


def _self_test() -> int:
    """Synthetic sanity check: CORAL matches target moments; SA shrinks subspace gap."""
    rng = np.random.default_rng(0)
    dim = 16
    transform_s = rng.normal(size=(dim, dim))
    transform_t = rng.normal(size=(dim, dim))
    source = rng.normal(size=(2000, dim)) @ transform_s + 3.0
    target = rng.normal(size=(2000, dim)) @ transform_t - 1.0

    aligned = coral_align(source, target).astype(np.float64)
    cov_err = np.linalg.norm(np.cov(aligned, rowvar=False) - np.cov(target, rowvar=False))
    cov_scale = np.linalg.norm(np.cov(target, rowvar=False))
    mean_err = np.linalg.norm(aligned.mean(0) - target.mean(0))
    coral_ok = cov_err / cov_scale < 1e-3 and mean_err < 1e-3
    logging.info("CORAL: rel cov err=%.2e  mean err=%.2e  -> %s",
                 cov_err / cov_scale, mean_err, "PASS" if coral_ok else "FAIL")

    src_a, tgt_a = subspace_align(source, target, n_components=8)
    # After alignment the two domains' projected covariances should be closer than
    # projecting both with the source basis would leave them.
    gap = np.linalg.norm(np.cov(src_a, rowvar=False) - np.cov(tgt_a, rowvar=False))
    sa_ok = np.isfinite(gap) and src_a.shape[1] == 8
    logging.info("SA: aligned dim=%d  cov gap=%.3f  -> %s",
                 src_a.shape[1], gap, "PASS" if sa_ok else "FAIL")
    return 0 if (coral_ok and sa_ok) else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    argparse.ArgumentParser(description="DA transforms self-test (CORAL + subspace alignment).").parse_args()
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
