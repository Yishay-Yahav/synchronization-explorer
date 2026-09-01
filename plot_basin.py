"""
Coupled Oscillators: 2D Basin of Attraction Visualizer & Explorer (Matplotlib).
Features:
- Parallel batch solving on multiple CPU workers
- Discrete Attractor Colormaps (matching publication Fig. 2 standard)
- Interactive Time-Slider (Time Evolution of the Basin)
- Click-to-Trajectory: Click any coordinate in the basin to open its full trajectory plot
- High-res PNG export & Animated GIF export
"""

from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider

from equations import SystemEquation, get_equations
from solvers import solve, SolutionWindow
from classifier import classify, Classification
from plot_trajectory import plot_trajectory


LABEL_TO_ID = {
    "other": 0,
    "in-phase": 1,
    "anti-phase": 2,
    "stationary beating": 3,
    "zero": 4,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# צבעי מפת האגנים התואמים בדיוק למאמר (Fig. 2):
# 0: other / unsettled -> אפור בהיר
# 1: in-phase -> כחול (#0072BD)
# 2: anti-phase -> צהוב/זהב (#E69F00)
# 3: stationary beating -> אדום/חלודה (#D9531E)
# 4: zero -> שחור (#000000)
COLORS = [
    "#b0bec5",  # 0: other / unsettled (אפור בהיר)
    "#0072bd",  # 1: in-phase (כחול)
    "#e69f00",  # 2: anti-phase (צהוב/זהב)
    "#d9531e",  # 3: stationary beating (אדום/חלודה)
    "#000000",  # 4: zero response (שחור)
]
BASIN_CMAP = ListedColormap(COLORS)
BASIN_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], BASIN_CMAP.N)


def generate_grid(
    y1_range: tuple[float, float] = (-20.0, 20.0),
    y2_range: tuple[float, float] = (-20.0, 20.0),
    resolution: int = 50,
    v1: float = 0.0,
    v2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """מייצרת רשת 2D של תנאי התחלה"""
    y1_vals = np.linspace(y1_range[0], y1_range[1], resolution)
    y2_vals = np.linspace(y2_range[0], y2_range[1], resolution)
    Y1, Y2 = np.meshgrid(y1_vals, y2_vals)

    ics = []
    for r in range(resolution):
        for c in range(resolution):
            ics.append([float(Y1[r, c]), float(v1), float(Y2[r, c]), float(v2)])

    return Y1, Y2, ics


class BasinData:
    """מבנה נתונים המחזיק את תוצאות סריקת מפת האגנים"""
    def __init__(
        self,
        system_eq: SystemEquation,
        Y1: np.ndarray,
        Y2: np.ndarray,
        final_classes: list[Classification],
        final_windows: list[SolutionWindow],
        snapshot_grids: list[tuple[float, np.ndarray]] | None = None,
    ):
        self.system_eq = system_eq
        self.Y1 = Y1
        self.Y2 = Y2
        self.resolution = Y1.shape[0]
        self.final_classes = final_classes
        self.final_windows = final_windows
        self.snapshot_grids = snapshot_grids or []

        # מטריצות 2D סופיות
        self.label_grid = np.zeros((self.resolution, self.resolution), dtype=int)
        self.sync_grid = np.zeros((self.resolution, self.resolution), dtype=float)
        self.loc_grid = np.zeros((self.resolution, self.resolution), dtype=float)
        self.time_grid = np.zeros((self.resolution, self.resolution), dtype=float)

        idx = 0
        for r in range(self.resolution):
            for c in range(self.resolution):
                cl = final_classes[idx]
                win = final_windows[idx]
                self.label_grid[r, c] = LABEL_TO_ID.get(cl.label, 0)
                self.sync_grid[r, c] = cl.sync_index
                self.loc_grid[r, c] = cl.localization
                self.time_grid[r, c] = win.convergence_time
                idx += 1


def compute_basin(
    system_eq: SystemEquation,
    y1_range: tuple[float, float] = (-20.0, 20.0),
    y2_range: tuple[float, float] = (-20.0, 20.0),
    resolution: int = 50,
    v1: float = 0.0,
    v2: float = 0.0,
    workers: int = 8,
    method: str = "numba_rk4",
    adaptive: bool = True,
    max_tau: float = 30.0,
    chunk_tau: float = 3.0,
    record_evolution: bool = True,
) -> BasinData:
    """
    מחשבת מפת אגנים מלאה במקביל על פני מספר מעבדים.
    """
    Y1, Y2, ics = generate_grid(y1_range, y2_range, resolution, v1, v2)
    n_points = len(ics)

    print(f"\n==================================================")
    print(f"[BASIN] Computing 2D Basin of Attraction ({resolution}x{resolution} = {n_points} points)")
    print(f"[CONFIG] Model: {system_eq.model}, Coupling: {system_eq.coupling}, Workers: {workers}")
    print(f"[RANGE] y1 in {y1_range}, y2 in {y2_range}")
    print(f"==================================================")

    t0 = time.perf_counter()
    windows = solve(
        system_eq,
        ics,
        method=method,
        workers=workers,
        adaptive=adaptive,
        max_tau=max_tau,
        chunk_tau=chunk_tau,
    )
    t_solve = time.perf_counter() - t0
    print(f"[DONE] Solved {n_points} initial conditions in {t_solve:.2f}s ({n_points/t_solve:.1f} pts/sec).")

    classes = classify(windows)

    # מעקב אחר התפתחות בזמן עבור הסליידר (אם נדרש)
    snapshot_grids = []
    if record_evolution:
        eps = getattr(system_eq, "eps", 0.001)
        n_steps = max(1, int(np.round(max_tau / chunk_tau)))
        tau_steps = [round((s + 1) * chunk_tau, 2) for s in range(n_steps)]

        print("[EVOLUTION] Tracking time steps for time-slider...")
        for tau_s in tau_steps:
            grid_s = np.zeros((resolution, resolution), dtype=int)
            idx = 0
            for r in range(resolution):
                for c in range(resolution):
                    win = windows[idx]
                    conv_tau = eps * win.convergence_time
                    if win.is_converged and conv_tau <= tau_s:
                        grid_s[r, c] = LABEL_TO_ID.get(classes[idx].label, 0)
                    else:
                        grid_s[r, c] = LABEL_TO_ID["other"]
                    idx += 1
            snapshot_grids.append((tau_s, grid_s))

    return BasinData(system_eq, Y1, Y2, classes, windows, snapshot_grids)


def plot_basin(
    basin_data: BasinData,
    mode: str = "attractor",  # "attractor", "sync_index", "localization", "time"
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    מציגה את מפת האגנים ב-Matplotlib עם התאמה מלאה ל-Fig. 2:
    - כחול = In-Phase
    - צהוב = Anti-Phase
    - אדום = Stationary Beating
    - שחור = Zero
    """
    fig = plt.figure(figsize=(9, 9))
    fig.canvas.manager.set_window_title(f"2D Basin of Attraction - {basin_data.system_eq.model.upper()}")

    ax = fig.add_axes([0.12, 0.16, 0.72, 0.76])

    y1_min, y1_max = basin_data.Y1[0, 0], basin_data.Y1[0, -1]
    y2_min, y2_max = basin_data.Y2[0, 0], basin_data.Y2[-1, 0]
    extent = [y1_min, y1_max, y2_min, y2_max]

    if mode == "sync_index":
        im = ax.imshow(basin_data.sync_grid, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Sync Index (1=IP, 0.5=Beating, 0=AP)")
        title_mode = "Sync Index Map"
    elif mode == "localization":
        im = ax.imshow(basin_data.loc_grid, origin="lower", extent=extent, cmap="coolwarm", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Localization (0.5=Equal, 1=Osc 1, 0=Osc 2)")
        title_mode = "Energy Localization Map"
    elif mode == "time":
        im = ax.imshow(basin_data.time_grid, origin="lower", extent=extent, cmap="plasma")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Convergence Time t")
        title_mode = "Convergence Time Map"
    else:  # "attractor" (תואם בדיוק ל-Fig 2)
        im = ax.imshow(basin_data.label_grid, origin="lower", extent=extent, cmap=BASIN_CMAP, norm=BASIN_NORM)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([
            "Other / Unsettled",
            "In-Phase (Blue)",
            "Anti-Phase (Yellow)",
            "Stationary Beating (Red)",
            "Zero (Black)",
        ], fontsize=9)
        title_mode = f"FIG. 2: Map of Attractors ({basin_data.system_eq.model.upper()})"

    ax.set_title(f"{title_mode}\n[Click any point to view its full trajectory]", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("$y_1(0)$", fontsize=11)
    ax.set_ylabel("$y_2(0)$", fontsize=11)
    ax.set_xlim(y1_min, y1_max)
    ax.set_ylim(y2_min, y2_max)

    # סליידר זמן בתחתית החלון
    slider_ax = fig.add_axes([0.18, 0.05, 0.60, 0.035])
    if basin_data.snapshot_grids:
        max_snap = len(basin_data.snapshot_grids) - 1
        time_slider = Slider(slider_ax, "Slow Time $\\tau$", 0, max_snap, valinit=max_snap, valstep=1, color="#0072bd")

        def update_time(val):
            snap_idx = int(time_slider.val)
            tau_val, grid_data = basin_data.snapshot_grids[snap_idx]
            im.set_data(grid_data)
            ax.set_title(f"{title_mode} ($\\tau = {tau_val:.1f}$)\n[Click any point to view trajectory]", fontsize=11, fontweight="bold")
            fig.canvas.draw_idle()

        time_slider.on_changed(update_time)

    # אינטראקציה: לחיצה על נקודה במפה פותחת את גרף המסלול שלה
    def on_click(event):
        if event.inaxes == ax and event.button == 1:
            clicked_y1 = round(float(event.xdata), 3)
            clicked_y2 = round(float(event.ydata), 3)
            clicked_ic = [clicked_y1, 0.0, clicked_y2, 0.0]
            print(f"[CLICK] Selected IC: y1={clicked_y1}, y2={clicked_y2} -> Opening trajectory plot...")
            plot_trajectory(basin_data.system_eq, ic=clicked_ic, show_plot=True)

    fig.canvas.mpl_connect("button_press_event", on_click)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] Figure saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig
