"""
Coupled Oscillators Synchronization Explorer: Automated 8-Experiment Suite (main.py).

Pre-Configured Execution Profiles:
- Profile 0 (Quick Validation): Resolution 11x11, max_tau = 100.0 (~15 minutes)
- Profile 1 (1-Hour Run):       Resolution 21x21, max_tau = 100.0 (~58 minutes)
- Profile 2 (3-Hour Run):       Resolution 37x37, max_tau = 100.0 (~2.9 hours)
- Profile 8h (Overnight 8-Hour): Resolution 60x60, max_tau = 100.0 (~7.7 hours)

Usage:
  python main.py                     (interactive menu)
  python main.py --profile 0         (run 11x11 max_tau=100)
  python main.py --profile 1         (run 1-hour profile)
  python main.py --profile 2         (run 3-hour profile)
  python main.py --profile 8h        (run 8-hour overnight suite)
  python main.py --res 25 --tau 80   (custom parameters)
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from equations import get_equations, SystemEquation
from plot_basin import compute_basin, plot_basin, BasinData


EXPERIMENTS = [
    {
        "id": 1,
        "name": "linear_uncoupled",
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
        "name": "linear_coupled",
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
        "name": "vdp_coupled",
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
        "name": "rayleigh_coupled",
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
        "name": "duffing_coupled",
        "title": "5. Coupled Linear with Duffing",
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
        "name": "duffing_only_uncoupled",
        "title": "6. Uncoupled Duffing Only",
        "params": {
            "model": "none",
            "coupling": "none",
            "eps": 0.001,
            "delta": 0.0,
            "eta": 0.0,
            "duffing": 1.0,
        },
    },
    {
        "id": 7,
        "name": "vdp_duffing_coupled",
        "title": "7. Coupled Van der Pol with Duffing",
        "params": {
            "model": "vdp",
            "coupling": "linear",
            "eps": 0.001,
            "delta": 0.1,
            "eta": 0.1,
            "duffing": 1.0,
        },
    },
    {
        "id": 8,
        "name": "rayleigh_duffing_coupled",
        "title": "8. Coupled Rayleigh with Duffing",
        "params": {
            "model": "rayleigh",
            "coupling": "linear",
            "eps": 0.001,
            "delta": 0.1,
            "eta": 0.15,
            "duffing": 1.0,
        },
    },
]


PROFILES = {
    "0": {
        "name": "Quick Validation (11x11, max_tau=100)",
        "resolution": 11,
        "max_tau": 100.0,
        "est_time": "~15 minutes",
    },
    "1": {
        "name": "1-Hour Run (21x21, max_tau=100)",
        "resolution": 21,
        "max_tau": 100.0,
        "est_time": "~58 minutes",
    },
    "2": {
        "name": "3-Hour Run (37x37, max_tau=100)",
        "resolution": 37,
        "max_tau": 100.0,
        "est_time": "~2.9 hours",
    },
    "8h": {
        "name": "Overnight 8-Hour Run (60x60, max_tau=100)",
        "resolution": 60,
        "max_tau": 100.0,
        "est_time": "~7.7 hours",
    },
}


def format_param_tag(exp_id: int, name: str, params: dict) -> str:
    """Generates a clean, unambiguous identifier string containing all key parameters."""
    return (
        f"exp{exp_id}_{name}_"
        f"m_{params['model']}_"
        f"c_{params['coupling']}_"
        f"eps{params['eps']}_"
        f"delta{params['delta']}_"
        f"eta{params['eta']}_"
        f"duff{params['duffing']}"
    )


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
    base_run_dir: str,
    resolution: int = 11,
    max_tau: float = 100.0,
    chunk_tau: float = 3.0,
    y_range: tuple[float, float] = (-20.0, 20.0),
    workers: int = 8,
):
    """Runs a single experiment and generates 5 output figures in a dedicated parameter-encoded subfolder."""
    p = exp_config["params"]
    param_tag = format_param_tag(exp_config["id"], exp_config["name"], p)
    target_dir = os.path.join(base_run_dir, param_tag)
    os.makedirs(target_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[EXPERIMENT {exp_config['id']}/8] {exp_config['title']}")
    print(f"[SUBFOLDER] {target_dir}")
    print(f"[PARAMS] Model={p['model']}, Coupling={p['coupling']}, eps={p['eps']}, delta={p['delta']}, eta={p['eta']}, duffing={p['duffing']}")
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
    file_prefix = f"{param_tag}_tau{int(max_tau)}_res{resolution}x{resolution}"

    # Plot 1: Attractors Map (discrete)
    path_attractor = os.path.join(target_dir, f"{file_prefix}_1_attractors.png")
    plot_basin(basin_data, mode="attractor", show_plot=False, save_path=path_attractor)

    # Plot 2: Sync Index Map (continuous: 0 to 1)
    path_sync = os.path.join(target_dir, f"{file_prefix}_2_sync_index.png")
    plot_basin(basin_data, mode="sync_index", show_plot=False, save_path=path_sync)

    # Plot 3: Localization Map (continuous: 0 to 1)
    path_loc = os.path.join(target_dir, f"{file_prefix}_3_localization.png")
    plot_basin(basin_data, mode="localization", show_plot=False, save_path=path_loc)

    # Plot 4: Physical coordinates in final window
    path_phys = os.path.join(target_dir, f"{file_prefix}_4_final_physical_coords.png")
    plot_final_physical_coords(basin_data, save_path=path_phys, title=exp_config["title"])

    # Plot 5: Modal envelopes in final window
    path_modal = os.path.join(target_dir, f"{file_prefix}_5_final_modal_envelopes.png")
    plot_final_modal_envelopes(basin_data, save_path=path_modal, title=exp_config["title"])

    print(f"[DONE] Completed Experiment {exp_config['id']}! 5 plots saved in: {target_dir}\n")


def run_all_experiments(
    resolution: int = 11,
    max_tau: float = 100.0,
    chunk_tau: float = 3.0,
    workers: int = 8,
):
    """
    Runs all 8 experiments sequentially.
    Root folder is dynamically named: experiments_res{resolution}x{resolution}_tau{int(max_tau)}
    """
    base_run_dir = f"experiments_res{resolution}x{resolution}_tau{int(max_tau)}"
    os.makedirs(base_run_dir, exist_ok=True)

    print("\n" + "#" * 60)
    print(f"[START] Running all 8 Experiments Suite")
    print(f"[RUN CONFIG] Resolution: {resolution}x{resolution} ({resolution**2} pts/exp)")
    print(f"[RUN CONFIG] Max Tau: {max_tau}, Workers: {workers}")
    print(f"[RUN ROOT DIR] '{base_run_dir}/'")
    print("#" * 60)

    t_start = time.perf_counter()
    for exp in EXPERIMENTS:
        run_single_experiment(
            exp,
            base_run_dir=base_run_dir,
            resolution=resolution,
            max_tau=max_tau,
            chunk_tau=chunk_tau,
            workers=workers,
        )

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"[ALL COMPLETE] All 8 experiments finished in {t_total:.2f}s ({t_total/60:.1f} mins)!")
    print(f"[SUMMARY] Total 40 figures saved across 8 subdirectories in '{base_run_dir}/'.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Coupled Oscillators 8-Experiment Batch Runner")
    parser.add_argument("--profile", type=str, choices=["0", "1", "2", "8h"], default=None,
                        help="Execution Profile: '0' (11x11, 15m), '1' (21x21, 1h), '2' (37x37, 3h), '8h' (60x60, 8h)")
    parser.add_argument("--res", type=int, default=None, help="Custom grid resolution (e.g. 25)")
    parser.add_argument("--tau", type=float, default=None, help="Custom max_tau (e.g. 100.0)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers (default: 8)")
    args = parser.parse_args()

    if args.profile is not None:
        prof = PROFILES[args.profile]
        resolution = prof["resolution"]
        max_tau = prof["max_tau"]
    elif args.res is not None:
        resolution = args.res
        max_tau = args.tau if args.tau is not None else 100.0
    else:
        # Interactive selection menu if run with plain `python main.py`
        print("\n" + "=" * 60)
        print("COUPLED OSCILLATORS: 8-EXPERIMENT SUITE RUNNER")
        print("=" * 60)
        print("Select Execution Profile:")
        print(" [0] Profile 0: Quick Validation (11x11, max_tau=100.0) -> ~15 mins")
        print(" [1] Profile 1: 1-Hour Schedule  (21x21, max_tau=100.0) -> ~58 mins")
        print(" [2] Profile 2: 3-Hour Schedule  (37x37, max_tau=100.0) -> ~2.9 hours")
        print(" [3] Profile 8h: Overnight 8-Hour (60x60, max_tau=100.0) -> ~7.7 hours")
        print("=" * 60)

        choice = input("Enter choice [0/1/2/3] (default: 0): ").strip()
        choice_map = {"0": "0", "1": "1", "2": "2", "3": "8h", "8h": "8h", "": "0"}
        selected_prof = choice_map.get(choice, "0")
        prof = PROFILES[selected_prof]
        resolution = prof["resolution"]
        max_tau = prof["max_tau"]

    run_all_experiments(resolution=resolution, max_tau=max_tau, workers=args.workers)


if __name__ == "__main__":
    main()
