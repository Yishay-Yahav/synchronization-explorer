"""
Coupled Oscillators Synchronization Explorer: Automated 6-Experiment Suite (main.py).

Runs 6 distinct oscillator setups:
1. Linear Uncoupled (pure linear, no coupling)
2. Linear Coupled (linear oscillators with coupling)
3. Coupled Van der Pol (reproducing publication Fig. 2)
4. Coupled Rayleigh (Rayleigh oscillators with coupling)
5. Coupled Duffing (linear coupling with Duffing cubic stiffness)
6. Duffing Only (uncoupled oscillators with Duffing cubic stiffness)

For each setup:
- Computes a 10x10 initial condition basin up to max_tau = 60.0 on 8 CPU workers.
- Generates 5 output plots saved in dedicated subdirectories:
  1. 2D Attractors Map (discrete)
  2. 2D Sync Index Map (continuous)
  3. 2D Localization Map (continuous)
  4. Final Physical Coordinates (y1, y2 in final converged window)
  5. Final Modal Envelopes (|mu0|, |mu1| in final converged window)
"""

from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from equations import get_equations, SystemEquation
from plot_basin import compute_basin, plot_basin, BasinData


OUTPUT_BASE_DIR = "experiments"

EXPERIMENTS = [
    {
        "id": 1,
        "folder": "1_linear_uncoupled",
        "title": "1. Linear Uncoupled Oscillators",
        "params": {
            "model": "none",
            "coupling": "none",
            "eps": 0.001,
            "delta": 0.0,
            "eta": 0.0,
            "duffing": 0.0,
        },
    },
    {
        "id": 2,
        "folder": "2_linear_coupled",
        "title": "2. Linear Coupled Oscillators",
        "params": {
            "model": "none",
            "coupling": "linear",
            "eps": 0.001,
            "delta": 0.0,
            "eta": 0.0,
            "duffing": 0.0,
        },
    },
    {
        "id": 3,
        "folder": "3_vdp_coupled",
        "title": "3. Coupled Van der Pol (Fig. 2)",
        "params": {
            "model": "vdp",
            "coupling": "linear",
            "eps": 0.001,
            "delta": 0.1,
            "eta": 0.1,
            "duffing": 0.0,
        },
    },
    {
        "id": 4,
        "folder": "4_rayleigh_coupled",
        "title": "4. Coupled Rayleigh Oscillators",
        "params": {
            "model": "rayleigh",
            "coupling": "linear",
            "eps": 0.001,
            "delta": 0.1,
            "eta": 0.15,
            "duffing": 0.0,
        },
    },
    {
        "id": 5,
        "folder": "5_duffing_coupled",
        "title": "5. Coupled Oscillators with Duffing",
        "params": {
            "model": "none",
            "coupling": "linear",
            "eps": 0.001,
            "delta": 0.0,
            "eta": 0.0,
            "duffing": 1.0,
        },
    },
    {
        "id": 6,
        "folder": "6_duffing_only_uncoupled",
        "title": "6. Uncoupled Oscillators with Duffing Only",
        "params": {
            "model": "none",
            "coupling": "none",
            "eps": 0.001,
            "delta": 0.0,
            "eta": 0.0,
            "duffing": 1.0,
        },
    },
]


def plot_final_physical_coords(basin_data: BasinData, save_path: str, title: str):
    """Plots physical coordinates (y1, y2) in the final converged window for representative points."""
    windows = basin_data.final_windows
    classes = basin_data.final_classes

    sample_indices = []
    seen_labels = set()
    for i, cl in enumerate(classes):
        if cl.label not in seen_labels:
            sample_indices.append(i)
            seen_labels.add(cl.label)
        if len(sample_indices) >= 4:
            break

    if not sample_indices:
        sample_indices = [0]

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2.5 * n_samples), sharex=False)
    if n_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        win = windows[idx]
        cl = classes[idx]
        t = win.times
        y1 = win.states[0]
        y2 = win.states[2]

        ax.plot(t, y1, label="$y_1(t)$ (Osc 1)", color="#0077b6", lw=1.2)
        ax.plot(t, y2, label="$y_2(t)$ (Osc 2)", color="#e76f51", lw=1.2)
        ax.set_title(f"Attractor: {cl.label.upper()} | IC #{idx+1} | RMS={cl.rms_amplitude:.2f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Displacement $y$")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Physical Time $t$ (Final Window)")
    fig.suptitle(f"Physical Coordinates in Final Window • {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final_modal_envelopes(basin_data: BasinData, save_path: str, title: str):
    """Plots modal envelopes (|mu0|, |mu1|) in the final converged window for representative points."""
    windows = basin_data.final_windows
    classes = basin_data.final_classes

    sample_indices = []
    seen_labels = set()
    for i, cl in enumerate(classes):
        if cl.label not in seen_labels:
            sample_indices.append(i)
            seen_labels.add(cl.label)
        if len(sample_indices) >= 4:
            break

    if not sample_indices:
        sample_indices = [0]

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2.5 * n_samples), sharex=False)
    if n_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        win = windows[idx]
        cl = classes[idx]
        tau = win.tau
        abs_mu0 = np.abs(win.mu0)
        abs_mu1 = np.abs(win.mu1)

        ax.plot(tau, abs_mu0, label="$|\mu_0(\tau)|$ (In-Phase)", color="#2a9d8f", lw=1.6)
        ax.plot(tau, abs_mu1, label="$|\mu_1(\tau)|$ (Anti-Phase)", color="#3b82f6", lw=1.6)
        ax.set_title(f"Attractor: {cl.label.upper()} | Sync Index={cl.sync_index:.3f} | Beating Purity={cl.beating_purity:.3f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Modal Amplitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Slow Time $\\tau = \\epsilon t$ (Final Window)")
    fig.suptitle(f"Modal Envelopes in Final Window • {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_single_experiment(
    exp_config: dict,
    resolution: int = 10,
    max_tau: float = 60.0,
    chunk_tau: float = 3.0,
    y_range: tuple[float, float] = (-20.0, 20.0),
    workers: int = 8,
):
    """Runs a single experiment and generates 5 output figures."""
    folder_name = exp_config["folder"]
    target_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    p = exp_config["params"]
    print("\n" + "=" * 60)
    print(f"[EXPERIMENT {exp_config['id']}/6] {exp_config['title']}")
    print(f"[FOLDER] Target Directory: {target_dir}")
    print(f"[PARAMS] Model={p['model']}, Coupling={p['coupling']}, eps={p['eps']}, duffing={p['duffing']}")
    print("=" * 60)

    # 1. Instantiate system equations
    eq = get_equations(
        model=p["model"],
        coupling=p["coupling"],
        eps=p["eps"],
        delta=p["delta"],
        eta=p["eta"],
        duffing=p["duffing"],
    )

    # 2. Parallel basin computation
    basin_data = compute_basin(
        system_eq=eq,
        y1_range=y_range,
        y2_range=y_range,
        resolution=resolution,
        workers=workers,
        method="numba_rk4",
        max_tau=max_tau,
        chunk_tau=chunk_tau,
        record_evolution=True,
    )

    # 3. Generate and save 5 plots
    prefix = f"{folder_name}_tau{int(max_tau)}_res{resolution}x{resolution}"

    # Plot 1: Attractors Map (discrete)
    path_attractor = os.path.join(target_dir, f"{prefix}_1_attractors.png")
    plot_basin(basin_data, mode="attractor", show_plot=False, save_path=path_attractor)

    # Plot 2: Sync Index Map (continuous: 0 to 1)
    path_sync = os.path.join(target_dir, f"{prefix}_2_sync_index.png")
    plot_basin(basin_data, mode="sync_index", show_plot=False, save_path=path_sync)

    # Plot 3: Localization Map (continuous: 0 to 1)
    path_loc = os.path.join(target_dir, f"{prefix}_3_localization.png")
    plot_basin(basin_data, mode="localization", show_plot=False, save_path=path_loc)

    # Plot 4: Physical coordinates in final window
    path_phys = os.path.join(target_dir, f"{prefix}_4_final_physical_coords.png")
    plot_final_physical_coords(basin_data, save_path=path_phys, title=exp_config["title"])

    # Plot 5: Modal envelopes in final window
    path_modal = os.path.join(target_dir, f"{prefix}_5_final_modal_envelopes.png")
    plot_final_modal_envelopes(basin_data, save_path=path_modal, title=exp_config["title"])

    print(f"[DONE] Completed Experiment {exp_config['id']}! 5 plots saved in {target_dir}\n")


def run_all_experiments(
    resolution: int = 10,
    max_tau: float = 60.0,
    chunk_tau: float = 3.0,
    workers: int = 8,
):
    """Runs all 6 experiments sequentially and saves 5 plots for each (total 30 figures)."""
    print("\n" + "#" * 60)
    print(f"[START] Running all 6 Experiments ({resolution}x{resolution} grid, max_tau={max_tau}, workers={workers})")
    print(f"[OUTPUT] Base directory: '{OUTPUT_BASE_DIR}/'")
    print("#" * 60)

    t_start = time.perf_counter()
    for exp in EXPERIMENTS:
        run_single_experiment(
            exp,
            resolution=resolution,
            max_tau=max_tau,
            chunk_tau=chunk_tau,
            workers=workers,
        )

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"[ALL COMPLETE] All 6 experiments finished in {t_total:.2f}s!")
    print(f"[SUMMARY] Total 30 figures saved across 6 directories in '{OUTPUT_BASE_DIR}/'.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments(resolution=10, max_tau=60.0, workers=8)
