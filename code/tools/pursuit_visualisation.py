import matplotlib.pyplot as plt


def plot_waveform_comparisons(y, reconstructions_dict, title="Waveform Reconstruction Comparison"):
    """Plots original signal against reconstructed waveforms from MP, OMP, and BP."""
    plt.figure(figsize=(11, 5))
    plt.plot(y, 'k', label='Original Audio Signal', alpha=0.8, linewidth=1.5)

    colors = {'MP': '#D95319', 'OMP': '#0072BD', 'BP': '#7E2FAE'}
    for name, y_rec in reconstructions_dict.items():
        color = colors.get(name, None)
        plt.plot(y_rec, label=f'Reconstructed ({name})', linestyle='--', color=color, alpha=0.85)

    plt.title(title, fontweight='bold', fontsize=11)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_pursuit_benchmarks(metrics_dict):
    """
    Plots a 3-panel bar chart comparing SNR (dB), Runtime (s), and Compression Ratio across algorithms.
    metrics_dict = {'MP': {'snr': ..., 'time': ..., 'cr': ...}, ...}
    """
    names = list(metrics_dict.keys())
    snrs = [metrics_dict[m]['snr'] for m in names]
    times = [metrics_dict[m]['time'] for m in names]
    crs = [metrics_dict[m]['cr'] for m in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.5))

    # 1. SNR
    bars1 = ax1.bar(names, snrs, color=['#D95319', '#0072BD', '#7E2FAE'], width=0.5)
    ax1.set_title('Reconstruction Quality (SNR)', fontweight='bold')
    ax1.set_ylabel('SNR (dB)')
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f} dB', ha='center', va='bottom', fontsize=9)

    # 2. Computation Time
    bars2 = ax2.bar(names, times, color=['#D95319', '#0072BD', '#7E2FAE'], width=0.5)
    ax2.set_title('Computation Time', fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.3f} s', ha='center', va='bottom', fontsize=9)

    # 3. Compression Ratio
    bars3 = ax3.bar(names, crs, color=['#D95319', '#0072BD', '#7E2FAE'], width=0.5)
    ax3.set_title('Sparsity / Compression Ratio', fontweight='bold')
    ax3.set_ylabel('Compression Ratio (N / K)')
    ax3.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.1f}x', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_omp_solvers_benchmark(omp_times_dict):
    """Compares runtime across OMP solver variants (Standard LS vs QR vs Cholesky)."""
    solvers = list(omp_times_dict.keys())
    runtimes = list(omp_times_dict.values())

    plt.figure(figsize=(7, 4))
    bars = plt.bar(solvers, runtimes, color='#0072BD', width=0.4)
    plt.title('OMP Algorithm Solver Speed Comparison', fontweight='bold')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0001, f'{yval*1000:.2f} ms', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()