"""
Score verification for the second autocorrelation inequality.

This exactly replicates the Einstein Arena platform verifier.
"""

import numpy as np


def compute_score(f: np.ndarray) -> float:
    """Compute C(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf).

    Uses the exact same method as the Einstein Arena verifier:
    Simpson's rule for L2^2, sum/(nc+1) for L1, pointwise max for Linf.

    Args:
        f: Non-negative numpy array of function values.

    Returns:
        The score C.
    """
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)

    if np.sum(f) == 0:
        raise ValueError("Function must have positive integral.")

    # Autoconvolution
    g = np.convolve(f, f, mode="full")
    nc = len(g)

    # L2^2 via Simpson's rule with zero-padded boundaries
    # x_points = linspace(-0.5, 0.5, nc+2), so h = 1/(nc+1)
    h = 1.0 / (nc + 1)
    y = np.concatenate(([0.0], g, [0.0]))
    y0, y1 = y[:-1], y[1:]
    l2_squared = (h / 3.0) * np.sum(y0**2 + y0 * y1 + y1**2)

    # L1 norm
    l1 = np.sum(np.abs(g)) / (nc + 1)

    # Linf norm
    linf = np.max(np.abs(g))

    return float(l2_squared / (l1 * linf))


def compute_score_fast(f: np.ndarray) -> float:
    """Fast scoring using FFT for the convolution.

    Equivalent to compute_score but O(n log n) instead of O(n^2).
    """
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)
    n = len(f)
    nc = 2 * n - 1

    # FFT-based autoconvolution
    nfft = 1
    while nfft < nc:
        nfft <<= 1
    F = np.fft.rfft(f, n=nfft)
    g = np.fft.irfft(F * F, n=nfft)[:nc]

    # Scoring (same as platform verifier)
    h = 1.0 / (nc + 1)
    y = np.concatenate(([0.0], g, [0.0]))
    y0, y1 = y[:-1], y[1:]
    l2_squared = (h / 3.0) * np.sum(y0**2 + y0 * y1 + y1**2)
    l1 = np.sum(np.abs(g)) / (nc + 1)
    linf = np.max(np.abs(g))

    return float(l2_squared / (l1 * linf))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <solution.npy>")
        sys.exit(1)

    f = np.load(sys.argv[1])
    print(f"n = {len(f)}")
    print(f"Nonzero (>1e-10): {np.sum(f > 1e-10)}")

    score = compute_score_fast(f)
    print(f"C = {score:.13f}")
