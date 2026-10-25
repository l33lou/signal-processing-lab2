import time
import numpy as np
import pywt
from scipy.fftpack import dct
from scipy.optimize import linprog


# =========================================================================
# 1. DICTIONARY / FRAME GENERATION
# =========================================================================

def build_overcomplete_dictionary(frame_len=256):
    """
    Constructs an overcomplete dictionary D in R^(N x M) by concatenating:
    1. DCT basis (N x N orthogonal)
    2. Multiscale Gabor / STFT atoms (Hanning-windowed sine/cosine tones with time shifts)
    3. Discrete Wavelet Transform atoms ('db4' and 'sym4' across all multiscale subbands)
    """
    N = frame_len
    atoms = []

    # A. DCT Basis (N x N)
    dct_basis = np.eye(N)
    for i in range(N):
        atoms.append(dct(dct_basis[i], type=2, norm='ortho'))

    # B. Multiscale Gabor Atoms (time-shifted Hanning windows with modulated tones)
    t = np.arange(N)
    freqs = np.linspace(0.02, 0.48, 24)
    for win_len in [N // 4, N // 2, N]:
        win = np.hanning(win_len)
        step = win_len // 2 if win_len < N else N
        for shift in range(0, N - win_len + 1, step):
            win_padded = np.pad(win, (shift, N - win_len - shift), mode='constant')
            for f in freqs:
                atoms.append(win_padded * np.cos(2 * np.pi * f * t))
                atoms.append(win_padded * np.sin(2 * np.pi * f * t))

    # C. Multiscale Wavelet Frame Atoms ('db4' and 'sym4')
    for family in ['db4', 'sym4']:
        try:
            w = pywt.Wavelet(family)
            max_lev = pywt.dwt_max_level(N, w.dec_len)
            coeffs_template = pywt.wavedec(np.zeros(N), family, level=max_lev)
            for band_idx in range(len(coeffs_template)):
                for coef_i in range(len(coeffs_template[band_idx])):
                    c_dummy = [np.zeros_like(c) for c in coeffs_template]
                    c_dummy[band_idx][coef_i] = 1.0
                    atom = pywt.waverec(c_dummy, family)[:N]
                    atoms.append(atom)
        except Exception:
            pass

    # Assemble and normalize all dictionary atoms (unit L2 normalization)
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
    Supports 3 update algorithms:
    - 'standard': Least-Squares via lstsq
    - 'qr': QR decomposition update
    - 'chol': Cholesky factorization update on the Gram matrix
    """
    t0 = time.perf_counter()
    N, M = D.shape
    residual = y.copy()
    selected_indices = []
    x = np.zeros(M)
    gamma = np.array([])

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
            try:
                L = np.linalg.cholesky(Gram + 1e-10 * np.eye(len(selected_indices)))
                z = np.linalg.solve(L, D_sub.T @ y)
                gamma = np.linalg.solve(L.T, z)
            except np.linalg.LinAlgError:
                gamma, _, _, _ = np.linalg.lstsq(D_sub, y, rcond=None)
        else:  # 'standard' Least-Squares
            gamma, _, _, _ = np.linalg.lstsq(D_sub, y, rcond=None)

        # Residual update
        residual = y - D_sub @ gamma

    if len(selected_indices) > 0 and len(gamma) > 0:
        x[selected_indices] = gamma

    runtime = time.perf_counter() - t0
    return x, runtime


def basis_pursuit(y, D, method='fista', alpha=0.05, max_iter=250):
    """
    Basis Pursuit (BP) / Lasso:
    - method='fista': Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle 2009).
      Solves min_x 0.5 * ||y - D x||_2^2 + alpha * ||x||_1 via exact soft-thresholding.
    - method='lp': Exact equality-constrained Basis Pursuit via Linear Programming (HiGHS).
      Solves min_x ||x||_1  s.t.  D x = y
    """
    t0 = time.perf_counter()
    N, M = D.shape

    if method == 'lp':
        # Exact Basis Pursuit via HiGHS LP solver
        c = np.ones(2 * M)
        A_eq = np.hstack([D, -D])
        res = linprog(c, A_eq=A_eq, b_eq=y, bounds=(0, None), method='highs')
        if res.success:
            x = res.x[:M] - res.x[M:]
        else:
            x = np.zeros(M)
    else:
        # FISTA for Lasso / BPDN with soft-thresholding
        L = np.linalg.norm(D, 2) ** 2
        x = np.zeros(M)
        z = x.copy()
        t_f = 1.0

        for _ in range(max_iter):
            grad = D.T @ (D @ z - y)
            step = z - grad / L
            # Soft-thresholding operator S_(alpha/L)
            x_next = np.sign(step) * np.maximum(np.abs(step) - alpha / L, 0.0)
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_f ** 2)) / 2.0
            z = x_next + ((t_f - 1.0) / t_next) * (x_next - x)
            x = x_next
            t_f = t_next

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


def compute_compression_ratio(x, original_len=None, threshold=1e-3):
    """
    Computes Signal Compression Ratio = Signal Length N / Non-zero Atoms K.
    If original_len is not specified, defaults to len(x).
    """
    k = np.sum(np.abs(x) > threshold)
    if k == 0:
        return 1.0, 0
    ref_len = original_len if original_len is not None else len(x)
    cr = ref_len / k
    return cr, k

    return cr, k