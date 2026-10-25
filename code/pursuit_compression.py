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

if __name__ == '__main__':
    # --- Paramètres ---
    FS = 16000             # Fréquence d'échantillonnage audio (16 kHz)
    FRAME_LEN = 256        # Taille de trame N (16 ms)
    MAX_ATOMS = 25         # Nombre d'atomes K retenus (parcimonie)

    # --- Signal synthétique (transitoire + sinusoïdes) ---
    t = np.arange(FRAME_LEN) / FS
    speech_tone = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 1200 * t)
    transient = 0.8 * np.exp(-((t - 0.008)**2) / (2 * (0.001**2)))
    y_audio = speech_tone + transient

    # --- Dictionnaire sur-complet ---
    D = build_overcomplete_dictionary(frame_len=FRAME_LEN)
    N, M = D.shape
    print(f"Dictionnaire construit : R^({N} x {M}) [Facteur de redondance : {M/N:.1f}x]")

    # --- Algorithmes de Pursuit ---
    # Matching Pursuit (MP)
    x_mp, time_mp = matching_pursuit(y_audio, D, max_iter=MAX_ATOMS)
    y_mp = D @ x_mp

    # Orthogonal Matching Pursuit (OMP)
    x_omp, time_omp = orthogonal_matching_pursuit(y_audio, D, max_iter=MAX_ATOMS, mode='chol')
    y_omp = D @ x_omp

    # Basis Pursuit (BP / Lasso via FISTA)
    x_bp, time_bp = basis_pursuit(y_audio, D, method='fista', alpha=0.05)
    y_bp = D @ x_bp

    # --- Métriques d'évaluation ---
    reconstructions = {'MP': y_mp, 'OMP': y_omp, 'BP': y_bp}
    metrics = {}

    for name, (x_sol, t_sol, y_sol) in zip(
        ['MP', 'OMP', 'BP'],
        [(x_mp, time_mp, y_mp), (x_omp, time_omp, y_omp), (x_bp, time_bp, y_bp)]
    ):
        snr = compute_snr(y_audio, y_sol)
        cr, k_nonzeros = compute_compression_ratio(x_sol, original_len=FRAME_LEN, threshold=1e-3)
        metrics[name] = {'snr': snr, 'time': t_sol, 'cr': cr, 'k': k_nonzeros}
        print(f"[{name}] SNR : {snr:.2f} dB | Temps : {t_sol:.4f} s | Atomes non nuls : {k_nonzeros}/{M} | CR (N/K) : {cr:.1f}x")

    # --- Comparaison des solveurs OMP ---
    omp_solvers_time = {}
    for solver_mode in ['standard', 'qr', 'chol']:
        _, t_mode = orthogonal_matching_pursuit(y_audio, D, max_iter=MAX_ATOMS, mode=solver_mode)
        omp_solvers_time[solver_mode.upper()] = t_mode

    # --- Affichage des graphiques ---
    plot_waveform_comparisons(y_audio, reconstructions, title="Reconstruction d'une trame : MP vs OMP vs BP")
    plot_pursuit_benchmarks(metrics)
    plot_omp_solvers_benchmark(omp_solvers_time)
