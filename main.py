"""
Coupled Oscillators Synchronization Explorer: Automated 6-Experiment Suite (main.py).

Runs 6 distinct oscillator setups:
1. Linear Uncoupled (לינארי ללא צימוד)
2. Linear Coupled (לינארי עם צימוד)
3. Coupled Van der Pol (ואן דר פול עם צימוד - כמו במאמר)
4. Coupled Rayleigh (ריילי עם צימוד)
5. Coupled Duffing (צימוד עם דאפינג)
6. Duffing Only (דאפינג בלבד ללא צימוד)

For each setup:
- Computes a 10x10 initial condition basin up to max_tau = 60.0 on 8 CPU workers.
- Generates 5 output plots saved in dedicated subdirectories:
  1. 2D Attractors Map (מפת אטרקטורים דיסקרטית)
  2. 2D Sync Index Map (מפת אינדקס סנכרון רציפה)
  3. 2D Localization Map (מפת לוקליזציה רציפה)
  4. Final Physical Coordinates (קואורדינטות פיזיקליות y1, y2 בחלון הזמן הסופי)
  5. Final Modal Envelopes (מעטפות מודליות |mu0|, |mu1| בחלון הזמן הסופי)
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
        "title": "1. מתנדים לינאריים ללא צימוד (Linear Uncoupled)",
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
        "title": "2. מתנדים לינאריים עם צימוד (Linear Coupled)",
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
        "title": "3. מתנדים ואן דר פול עם צימוד (Coupled Van der Pol)",
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
        "title": "4. מתנדים ריילי עם צימוד (Coupled Rayleigh)",
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
        "title": "5. מתנדים עם צימוד ודאפינג (Coupled Duffing)",
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
        "title": "6. מתנדים עם דאפינג בלבד ללא צימוד (Uncoupled Duffing)",
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
    """מציירת את הקואורדינטות הפיזיקליות (y1, y2) בחלון הזמן הסופי עבור מדגם מייצג של נקודות"""
    windows = basin_data.final_windows
    classes = basin_data.final_classes
    n_points = len(windows)

    # בחירת עד 4 נקודות מייצגות שונות של אטרקטורים
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

        ax.plot(t, y1, label="$y_1(t)$ (מתנד 1)", color="#0077b6", lw=1.2)
        ax.plot(t, y2, label="$y_2(t)$ (מתנד 2)", color="#e76f51", lw=1.2)
        ax.set_title(f"אטרקטור: {cl.label.upper()} | תנאי התחלה #{idx+1} | RMS={cl.rms_amplitude:.2f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("מיקום $y$")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("זמן אמיתי $t$ בחלון הסופי")
    fig.suptitle(f"קואורדינטות פיזיקליות בחלון הסופי • {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final_modal_envelopes(basin_data: BasinData, save_path: str, title: str):
    """מציירת את המעטפות המודליות (|mu0|, |mu1|) בחלון הזמן הסופי עבור מדגם מייצג של נקודות"""
    windows = basin_data.final_windows
    classes = basin_data.final_classes

    # בחירת עד 4 נקודות מייצגות שונות
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
        ax.set_title(f"אטרקטור: {cl.label.upper()} | Sync Index={cl.sync_index:.3f} | טוהר פעימה={cl.beating_purity:.3f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("משרעת מודלית")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("זמן איטי $\\tau = \\epsilon t$ בחלון הסופי")
    fig.suptitle(f"מעטפות מודליות בחלון הסופי • {title}", fontsize=11, fontweight="bold")
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
    """מריצה ניסוי בודד ומפיקה 5 סוגי גרפים (3 מפות אגנים + קואורדינטות פיזיקליות + מעטפות מודליות)"""
    folder_name = exp_config["folder"]
    target_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    p = exp_config["params"]
    print("\n" + "=" * 60)
    print(f"🔬 מריץ ניסוי {exp_config['id']}/6: {exp_config['title']}")
    print(f"📁 תיקיית יעד: {target_dir}")
    print(f"⚙️ פרמטרים: model={p['model']}, coupling={p['coupling']}, eps={p['eps']}, duffing={p['duffing']}")
    print("=" * 60)

    # 1. יצירת המשוואה
    eq = get_equations(
        model=p["model"],
        coupling=p["coupling"],
        eps=p["eps"],
        delta=p["delta"],
        eta=p["eta"],
        duffing=p["duffing"],
    )

    # 2. חישוב רשת האגנים (10x10) במקביל
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

    # 3. הפקת ושמירת 5 סוגי הגרפים
    prefix = f"{folder_name}_tau{int(max_tau)}_res{resolution}x{resolution}"

    # מפה 1: אטרקטורים (דיסקרטי)
    path_attractor = os.path.join(target_dir, f"{prefix}_1_attractors.png")
    plot_basin(basin_data, mode="attractor", show_plot=False, save_path=path_attractor)

    # מפה 2: אינדקס סנכרון (רציף: 0 עד 1)
    path_sync = os.path.join(target_dir, f"{prefix}_2_sync_index.png")
    plot_basin(basin_data, mode="sync_index", show_plot=False, save_path=path_sync)

    # מפה 3: לוקליזציה (רציף: 0 עד 1)
    path_loc = os.path.join(target_dir, f"{prefix}_3_localization.png")
    plot_basin(basin_data, mode="localization", show_plot=False, save_path=path_loc)

    # גרף 4: קואורדינטות פיזיקליות בחלון הסופי
    path_phys = os.path.join(target_dir, f"{prefix}_4_final_physical_coords.png")
    plot_final_physical_coords(basin_data, save_path=path_phys, title=exp_config["title"])

    # גרף 5: מעטפות מודליות בחלון הסופי
    path_modal = os.path.join(target_dir, f"{prefix}_5_final_modal_envelopes.png")
    plot_final_modal_envelopes(basin_data, save_path=path_modal, title=exp_config["title"])

    print(f"✅ הושלם בהצלחה ניסוי {exp_config['id']}! 5 הגרפים נשמרו ב-{target_dir}\n")


def run_all_experiments(
    resolution: int = 10,
    max_tau: float = 60.0,
    chunk_tau: float = 3.0,
    workers: int = 8,
):
    """מריצה את כל 6 הניסויים ברצף ושומרת 5 גרפים לכל ניסוי (סה"כ 30 גרפים)"""
    print("\n" + "#" * 60)
    print("🚀 מתחיל הרצה מלאה של 6 הניסויים הפיזיקליים (10x10, max_tau=60)")
    print(f"📁 כל התוצאות יישמרו תחת התיקייה: '{OUTPUT_BASE_DIR}/'")
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
    print(f"🎉 כל 6 הניסויים הסתיימו בהצלחה תוך {t_total:.2f} שניות!")
    print(f"📂 נשמרו בסך הכל 30 גרפים (5 לכל ניסוי) בתוך '{OUTPUT_BASE_DIR}/'.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments(resolution=10, max_tau=60.0, workers=8)
