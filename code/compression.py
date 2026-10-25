import numpy as np
from tools.pursuit import (
    basis_pursuit,
    build_overcomplete_dictionary,
    compute_compression_ratio,
    compute_snr,
    matching_pursuit,
    orthogonal_matching_pursuit,
)
from tools.pursuit_visualisation import (
    plot_omp_solvers_benchmark,
    plot_pursuit_benchmarks,
    plot_waveform_comparisons,
)

### Compressing signals using pursuit algorithms (MP, OMP, BP), visualizing and quantifying results.

FS = 16000             # Sampling frequency suitable for speech (16 kHz)
FRAME_LEN = 256        # Block frame size N (16 ms)
MAX_ATOMS = 25         # Target sparsity budget K (for greedy pursuits)

# Generate synthetic audio signal: combination of speech-like tones and a transient
t = np.arange(FRAME_LEN) / FS
speech_tone = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 1200 * t)
transient = 0.8 * np.exp(-((t - 0.008)**2) / (2 * (0.001**2)))
y_audio = speech_tone + transient

# Building overconpete dictionnary
D = build_overcomplete_dictionary(frame_len=FRAME_LEN)
N, M = D.shape
print(f"Dictionary constructed: Frame dimension R^({N} x {M}) [Redundancy factor: {M/N:.1f}x]")

# Pursuit algorithms: MP, OMP, BP
# Matching Pursuit
x_mp, time_mp = matching_pursuit(y_audio, D, max_iter=MAX_ATOMS)
y_mp = D @ x_mp

# Orthogonal Matching Pursuit
x_omp, time_omp = orthogonal_matching_pursuit(y_audio, D, max_iter=MAX_ATOMS, mode='chol')
y_omp = D @ x_omp

# Basis Pursuit
x_bp, time_bp = basis_pursuit(y_audio, D, alpha=0.01)
y_bp = D @ x_bp

# Benchmark metrics
reconstructions = {'MP': y_mp, 'OMP': y_omp, 'BP': y_bp}
metrics = {}

for name, (x_sol, t_sol, y_sol) in zip(
    ['MP', 'OMP', 'BP'],
    [(x_mp, time_mp, y_mp), (x_omp, time_omp, y_omp), (x_bp, time_bp, y_bp)]
):
    snr = compute_snr(y_audio, y_sol)
    cr, k_nonzeros = compute_compression_ratio(x_sol, threshold=1e-2)
    metrics[name] = {'snr': snr, 'time': t_sol, 'cr': cr, 'k': k_nonzeros}

    print(f"[{name}] SNR: {snr:.2f} dB | Time: {t_sol:.4f} s | Non-zero Atoms: {k_nonzeros}/{M} | CR: {cr:.1f}x")

# --- 6. COMPARE OMP SOLVER ALGORITHMS (Standard vs QR vs Cholesky) ---
omp_solvers_time = {}
for solver_mode in ['standard', 'qr', 'chol']:
    _, t_mode = orthogonal_matching_pursuit(y_audio, D, max_iter=MAX_ATOMS, mode=solver_mode)
    omp_solvers_time[solver_mode.upper()] = t_mode

if __name__ == '__main__':
    plot_waveform_comparisons(y_audio, reconstructions, title="Frame Reconstruction: MP vs OMP vs BP")
    plot_pursuit_benchmarks(metrics)
    plot_omp_solvers_benchmark(omp_solvers_time)