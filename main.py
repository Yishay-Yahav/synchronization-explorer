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
- Generates 3 plot maps: Attractors (discrete), Sync Index (continuous), Localization (continuous).
- Saves all outputs organized into 6 dedicated subdirectories inside 'experiments/'.
"""

from __future__ import annotations
import os
import time
from equations import get_equations, SystemEquation
from plot_basin import compute_basin, plot_basin


OUTPUT_BASE_DIR = "experiments"

# הגדרת 6 הניסויים הפיזיקליים
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


def run_single_experiment(
    exp_config: dict,
    resolution: int = 10,
    max_tau: float = 60.0,
    chunk_tau: float = 3.0,
    y_range: tuple[float, float] = (-20.0, 20.0),
    workers: int = 8,
):
    """מריצה ניסוי בודד ומפיקה עבורו 3 סוגי מפות (אטרקטורים, סנכרון ולוקליזציה)"""
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

    # 3. הפקת ושמירת 3 סוגי המפות
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

    print(f"✅ הושלם בהצלחה ניסוי {exp_config['id']}! 3 המפות נשמרו ב-{target_dir}\n")


def run_all_experiments(
    resolution: int = 10,
    max_tau: float = 60.0,
    chunk_tau: float = 3.0,
    workers: int = 8,
):
    """מריצה את כל 6 הניסויים ברצף ושומרת אותם ב-6 תיקיות מסודרות"""
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
    print(f"📂 נשמרו בסך הכל 18 מפות ב-6 תיקיות בתוך '{OUTPUT_BASE_DIR}/'.")
    print("=" * 60)


if __name__ == "__main__":
    # הרצת כל 6 הניסויים
    run_all_experiments(resolution=10, max_tau=60.0, workers=8)
