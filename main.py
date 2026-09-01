"""
Coupled Oscillators Synchronization Explorer: Main Entry Point (main.py).
Run this script to launch either the 2D Basin Map or a Single Trajectory plot.
"""

from equations import get_equations
from plot_basin import compute_basin, plot_basin
from plot_trajectory import plot_trajectory


def run_basin_explorer():
    """הרצת מפת אגני משיכה 2D עם סליידר זמן ולחיצה לצפייה במסלול"""
    # 1. הגדרת פרמטרי המודל
    eq = get_equations(
        model="rayleigh",     # אפשרויות: "rayleigh", "vdp", "none"
        coupling="linear",    # אפשרויות: "linear", "none"
        eps=0.001,            # אפסילון
        delta=0.1,            # דלתא
        eta=0.15,             # אטה
        duffing=0.0,          # דאפינג (לוקליזציה)
    )

    # 2. חישוב רשת תנאי התחלה (y1 מול y2) במקביל על 8 מעבדים
    basin_data = compute_basin(
        system_eq=eq,
        y1_range=(-4.0, 4.0),
        y2_range=(-4.0, 4.0),
        resolution=30,        # רזולוציה (30x30 = 900 נקודות)
        workers=8,            # כמה מעבדים ירוצו במקביל
        method="numba_rk4",   # פותרן Numba מואץ
        max_tau=30.0,
        chunk_tau=3.0,
    )

    # 3. הצגת מפת האגנים האינטראקטיבית ב-Matplotlib
    # אפשרויות ל-mode: "attractor", "sync_index", "localization", "time"
    plot_basin(basin_data, mode="attractor", show_plot=True)


def run_single_trajectory():
    """הרצת מסלול תנועה בודד עבור תנאי התחלה ספציפי"""
    eq = get_equations(
        model="rayleigh",
        coupling="linear",
        eps=0.001,
        delta=0.1,
        eta=0.15,
        duffing=0.0,
    )

    # תנאי התחלה: [y1(0), v1(0), y2(0), v2(0)]
    initial_condition = [2.0, 0.0, 0.0, 0.0]

    plot_trajectory(
        system_eq=eq,
        ic=initial_condition,
        method="numba_rk4",
        show_plot=True,
    )


if __name__ == "__main__":
    print("==================================================")
    print("Coupled Oscillators Explorer - Main Launcher")
    print("==================================================")

    # בחר מה להריץ (הסר את ההערה מהפונקציה הרצויה):
    run_basin_explorer()       # 1. מפת אגני משיכה 2D
    # run_single_trajectory()  # 2. מסלול תנועה בודד
