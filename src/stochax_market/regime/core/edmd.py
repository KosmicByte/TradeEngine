"""Extended DMD (Williams, Kevrekidis, Rowley 2015).

Where DMD acts on raw states, EDMD acts on a dictionary of observables ψ(x), giving
a finite-dimensional approximation of the Koopman operator K:

        Ψ_Y ≈ K Ψ_X         where   (Ψ_X)_ij = ψ_j(x_i)

We use the standard least-squares solution K = Ψ_X⁺ Ψ_Y (computed via SVD with
truncation for numerical stability), then return its eigenstructure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochax_market.regime.core.observables import ObservableConfig, build_observables, standardise


@dataclass(frozen=True)
class EDMDResult:
    """Eigenstructure of the Koopman operator on the chosen dictionary."""

    eigenvalues: np.ndarray  # (r,) complex — discrete-time
    koopman_matrix: np.ndarray  # (k, k) — the K estimate
    eigenvectors: np.ndarray  # (k, r) — Koopman eigenfunctions (in observable space)
    observable_names: list[str]
    rank: int
    singular_values: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @property
    def growth_rates(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.real(np.log(self.eigenvalues + 1e-300))

    @property
    def frequencies(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.imag(np.log(self.eigenvalues + 1e-300)) / (2 * np.pi)

    @property
    def spectral_gap(self) -> float:
        """|λ_(1)| - |λ_(next distinct)|. For real data, eigenvalues come in conjugate
        pairs with the same magnitude, so we look past pairs to find a meaningful gap."""
        mags = np.sort(np.abs(self.eigenvalues))[::-1]
        if len(mags) < 2:
            return 0.0
        top = mags[0]
        for m in mags[1:]:
            if top - m > 1e-6:
                return float(top - m)
        return 0.0

    @property
    def mode_energies(self) -> np.ndarray:
        """Energy = |eigenvalue| × ||eigenvector||, normalised."""
        e = np.abs(self.eigenvalues) * np.linalg.norm(self.eigenvectors, axis=0)
        s = e.sum()
        return e / s if s > 0 else e


def fit_edmd(
    close: np.ndarray,
    config: ObservableConfig | None = None,
    rank: int | None = 10,
    energy: float = 0.99,
) -> EDMDResult:
    """Fit EDMD on a close-price series.

    Builds the observable matrix internally, standardises features, then solves for K
    via truncated SVD.

    Args:
        close: 1-D price series, length T.
        config: observable dictionary configuration.
        rank: explicit truncation rank; if None, energy-based.
        energy: retained energy fraction when rank is None.
    """
    psi, names = build_observables(close, config)
    psi_std, mean, std = standardise(psi)

    if psi_std.shape[0] < 3:
        raise ValueError(f"Too few valid snapshots after warm-up: {psi_std.shape[0]}")

    # Snapshot pairs: rows are time, so transpose to (k, T).
    Psi = psi_std.T  # shape (k, T_valid)
    X = Psi[:, :-1]
    Y = Psi[:, 1:]

    # Truncated-SVD pseudoinverse of X for stability.
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    if rank is not None:
        r = min(rank, len(S))
    else:
        cum = np.cumsum(S**2) / (S**2).sum()
        r = int(np.searchsorted(cum, energy) + 1)
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r, :].conj().T

    # K = Y X⁺  (full k×k)
    X_pinv = Vr @ np.diag(1.0 / Sr) @ Ur.conj().T
    K = Y @ X_pinv

    eigvals, eigvecs = np.linalg.eig(K)
    # Sort by magnitude descending — dominant modes first.
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return EDMDResult(
        eigenvalues=eigvals,
        koopman_matrix=K,
        eigenvectors=eigvecs,
        observable_names=names,
        rank=r,
        singular_values=S,
        feature_mean=mean,
        feature_std=std,
    )
