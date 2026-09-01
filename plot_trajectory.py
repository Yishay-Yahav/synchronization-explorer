"""
Coupled Oscillators: Single Trajectory Visualizer (Matplotlib).
Plots time series y1(t), y2(t), modal envelopes |mu0|, |mu1|, phase portrait, and classification metrics.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from equations import SystemEquation, get_equations
from solvers import solve, SolutionWindow
from classifier import classify, Classification


def plot_trajectory(
    system_eq: SystemEquation,
    ic: list[float] | np.ndarray = (2.0, 0.0, 0.0, 0.0),
    method: str = "numba_rk4",
    adaptive: bool = True,
    max_tau: float = 30.0,
    chunk_tau: float = 3.0,
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure, SolutionWindow, Classification]:
    """
    Solves a single initial condition and displays an interactive Matplotlib figure:
    1. Time series y1(t), y2(t)
    2. Modal envelopes |mu0(tau)|, |mu1(tau)| over slow time
    3. Phase portrait (y1, v1)
    4. Classification summary card
    """
    # 1. Solve and classify
    windows = solve(
        system_eq,
        [ic],
        method=method,
        adaptive=adaptive,
        max_tau=max_tau,
        chunk_tau=chunk_tau,
    )
    win = windows[0]
    cl = classify([win])[0]

    # 2. Setup figure
    fig = plt.figure(figsize=(13, 8))
    fig.canvas.manager.set_window_title(f"Coupled Oscillators Trajectory - {system_eq.model.capitalize()} (IC={ic})")

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_env = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])

    times = win.times
    y1, v1, y2, v2 = win.states
    tau = win.tau
    abs_mu0 = np.abs(win.mu0)
    abs_mu1 = np.abs(win.mu1)

    # 1. Trajectory time series
    ax_traj.plot(times, y1, label="$y_1(t)$ (Oscillator 1)", color="#0077b6", lw=1.2, alpha=0.85)
    ax_traj.plot(times, y2, label="$y_2(t)$ (Oscillator 2)", color="#e76f51", lw=1.2, alpha=0.85)
    ax_traj.set_title("Physical Time Trajectory", fontsize=11, fontweight="bold")
    ax_traj.set_xlabel("Physical Time $t$")
    ax_traj.set_ylabel("Displacement $y$")
    ax_traj.grid(True, linestyle="--", alpha=0.5)
    ax_traj.legend(loc="upper right", fontsize=9)

    # 2. Slow modal envelopes
    ax_env.plot(tau, abs_mu0, label="$|\mu_0(\tau)|$ (In-Phase)", color="#2a9d8f", lw=1.8)
    ax_env.plot(tau, abs_mu1, label="$|\mu_1(\tau)|$ (Anti-Phase)", color="#457b9d", lw=1.8)
    ax_env.set_title("Slow Modal Envelopes $\\tau = \\epsilon t$", fontsize=11, fontweight="bold")
    ax_env.set_xlabel("Slow Time $\\tau$")
    ax_env.set_ylabel("Modal Amplitude")
    ax_env.grid(True, linestyle="--", alpha=0.5)
    ax_env.legend(loc="upper right", fontsize=9)

    # 3. Phase portrait
    ax_phase.plot(y1, v1, label="Osc 1: $(y_1, v_1)$", color="#0077b6", lw=1.0, alpha=0.7)
    ax_phase.plot(y2, v2, label="Osc 2: $(y_2, v_2)$", color="#e76f51", lw=1.0, alpha=0.7)
    ax_phase.plot(ic[0], ic[1], "o", color="#0077b6", markersize=6, label="Start 1")
    ax_phase.plot(ic[2], ic[3], "s", color="#e76f51", markersize=6, label="Start 2")
    ax_phase.set_title("Phase Portrait: Velocity $v$ vs Displacement $y$", fontsize=11, fontweight="bold")
    ax_phase.set_xlabel("Displacement $y$")
    ax_phase.set_ylabel("Velocity $v = \\dot{y}$")
    ax_phase.grid(True, linestyle="--", alpha=0.5)
    ax_phase.legend(loc="upper right", fontsize=8)

    # 4. Classification summary card
    ax_info.axis("off")
    color_map = {
        "in-phase": "#2a9d8f",
        "anti-phase": "#457b9d",
        "stationary beating": "#e76f51",
        "zero": "#6c757d",
        "other": "#d62828",
    }
    label_color = color_map.get(cl.label, "#333333")

    info_text = (
        f"Classification & Dynamical Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Attractor Label:      {cl.label.upper()}\n"
        f"• Sync Index:           {cl.sync_index:.4f}\n"
        f"• Beating Purity:       {cl.beating_purity:.4f}\n"
        f"• Localization:         {cl.localization:.4f}\n"
        f"• Final RMS Amplitude:  {cl.rms_amplitude:.4f}\n"
        f"• Stationary & Stable:  {'YES (Stable)' if cl.is_stable else 'NO (Transient)'}\n"
        f"• Convergence Time:     t = {win.convergence_time:.1f}\n"
        f"• Final Envelope Slope: {win.slope:.2e}\n\n"
        f"Parameters:\n"
        f"Model: {system_eq.model}, eps={system_eq.eps}, delta={system_eq.delta}, eta={system_eq.eta}, Duffing={system_eq.duffing}\n"
        f"Initial Condition: [y1={ic[0]}, v1={ic[1]}, y2={ic[2]}, v2={ic[3]}]"
    )

    bbox_props = dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor=label_color, linewidth=2.5)
    ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes, fontsize=10,
                 verticalalignment="center", fontfamily="monospace", bbox=bbox_props)

    plt.suptitle(
        f"Coupled Oscillators Trajectory Explorer • Final Attractor: {cl.label.upper()}",
        fontsize=13, fontweight="bold", color=label_color, y=0.98
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[SAVED] Trajectory plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig, win, cl


if __name__ == "__main__":
    eq = get_equations("rayleigh", eps=0.001, delta=0.1, eta=0.15)
    plot_trajectory(eq, ic=[2.0, 0.0, 0.0, 0.0], show_plot=False, save_path="trajectory_sample.png")
