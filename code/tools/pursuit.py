import time
import numpy as np
import pywt
from scipy.fftpack import dct
from scipy.optimize import minimize


# =========================================================================
# 1. DICTIONARY / FRAME GENERATION
# =========================================================================

def build_overcomplete_dictionary(frame_len=256):
    """
    Constructs an overcomplete dictionary D in R^(N x M) by concatenating:
    1. Multiscale DCT bases (Short & Long resolution)
    2. Gabor / STFT atoms (Hanning-windowed sine/cosine tones)
    3. Discrete Wavelet Transform atoms ('db4' and 'sym4')
    """
    N = frame_len
    atoms = []

    # A. DCT Basis (N x N)
    dct_basis = np.eye(N)
    for i in range(N):
        atoms.append(dct(dct_basis[i], type=2, norm='ortho'))

    # B. Gabor / Windowed Fourier Atoms (Multiscale Hanning windows)
    t = np.arange(N)
    for win_len in [N // 2, N]:
        win = np.hanning(win_len)
        pad_len = N - win_len
        win_padded = np.pad(win, (0, pad_len), mode='constant')
        
        # Modulated sines and cosines
        freqs = np.linspace(0, 0.5, 16)
        for f in freqs:
            atoms.append(win_padded * np.cos(2 * np.pi * f * t))
            atoms.append(win_padded * np.sin(2 * np.pi * f * t))

    # C. Wavelet Frame Atoms ('db4' & 'sym4' at multiple levels)
    for family in ['db4', 'sym4']:
        try:
            w = pywt.Wavelet(family)
            max_lev = pywt.dwt_max_level(N, w.dec_len)
            for lev in range(1, max_lev + 1):
                coeffs = pywt.wavedec(np.zeros(N), family, level=lev)
                for band_idx in range(len(coeffs)):
                    for coef_i in range(len(coeffs[band_idx])):
                        c_dummy = [np.zeros_like(c) for c in coeffs]
                        c_dummy[band_idx][coef_i] = 1.0
                        atom = pywt.waverec(c_dummy, family)[:N]
                        atoms.append(atom)
        except Exception:
            pass

    # Assemble and normalize all dictionary atoms (L2 normalization)
    D = np.column_stack(atoms)
    norms = np.linalg.norm(D, axis=0)
    norms[norms == 0] = 1.0
    D = D / norms

    return D


# =========================================================================
# 2. PURSUIT ALGORITHMS
# =========================================================================

def matching_pursuit(y, D, max_iter=30, tol=1e-4):
    """
    Matching Pursuit (MP): Standard greedy selection with non-orthogonal residual update.
    """
    t0 = time.perf_counter()
    N, M = D.shape
    residual = y.copy()
    x = np.zeros(M)

    for _ in range(max_iter):
        if np.linalg.norm(residual) < tol:
            break
        # Correlations with residual
        correlations = D.T @ residual
        best_idx = np.argmax(np.abs(correlations))
        coeff = correlations[best_idx]

        # Update coefficient and residual (non-orthogonal update)
        x[best_idx] += coeff
        residual -= coeff * D[:, best_idx]

    runtime = time.perf_counter() - t0
    return x, runtime


def orthogonal_matching_pursuit(y, D, max_iter=30, mode='chol', tol=1e-4):
    """
    Orthogonal Matching Pursuit (OMP): Greedy selection with orthogonal projection.
    Supports 3 update algorithms: 'standard' (Least-Squares), 'qr' (QR decomposition),
    and 'chol' (Cholesky factorization update).
    """
    t0 = time.perf_counter()
    N, M = D.shape
    residual = y.copy()
    selected_indices = []
    x = np.zeros(M)

    for k in range(max_iter):
        if np.linalg.norm(residual) < tol:
            break
        correlations = D.T @ residual
        best_idx = np.argmax(np.abs(correlations))

        if best_idx in selected_indices:
            break
        selected_indices.append(best_idx)

        D_sub = D[:, selected_indices]

        # OMP Solver variants
        if mode == 'qr':
            Q, R = np.linalg.qr(D_sub)
            gamma = np.linalg.solve(R, Q.T @ y)
        elif mode == 'chol':
            # Cholesky on Gram matrix (D_sub.T @ D_sub)
            Gram = D_sub.T @ D_sub
            L = np.linalg.cholesky(Gram + 1e-10 * np.eye(len(selected_indices)))
            z = np.linalg.solve(L, D_sub.T @ y)
            gamma = np.linalg.solve(L.T, z)
        else:  # 'standard' Least-Squares
            gamma, _, _, _ = np.linalg.lstsq(D_sub, y, rcond=None)

        # Residual update
        residual = y - D_sub @ gamma

    x[selected_indices] = gamma
    runtime = time.perf_counter() - t0
    return x, runtime


def basis_pursuit(y, D, alpha=0.01):
    """
    Basis Pursuit (BP) / Lasso: Convex L1-minimization via L-BFGS-B optimization.
    Solves min_x 0.5 * ||y - D x||_2^2 + alpha * ||x||_1
    """
    t0 = time.perf_counter()
    N, M = D.shape

    def objective(x):
        res = y - D @ x
        return 0.5 * np.dot(res, res) + alpha * np.sum(np.abs(x))

    def gradient(x):
        res = y - D @ x
        return -D.T @ res + alpha * np.sign(x)

    x0 = np.zeros(M)
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 100})
    x = res.x

    runtime = time.perf_counter() - t0
    return x, runtime


# =========================================================================
# 3. METRICS EVALUATION
# =========================================================================

def compute_snr(y, y_rec):
    """Calculates Signal-to-Noise Ratio in dB."""
    noise = y - y_rec
    pow_sig = np.mean(y ** 2)
    pow_noise = np.mean(noise ** 2)
    if pow_noise == 0:
        return float('inf')
    return 10 * np.log10(pow_sig / pow_noise)


def compute_compression_ratio(x, threshold=1e-3):
    """Computes Compression Ratio = Total Atoms / Non-zero Atoms."""
    k = np.sum(np.abs(x) > threshold)
    if k == 0:
        return 1.0, 0
    cr = len(x) / k
    return cr, k