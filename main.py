"""
Coupled Oscillators Synchronization Explorer: Main Entry Point (main.py).
Configured to reproduce FIG. 2 from the publication (Van der Pol Coupled Oscillators).
"""

from equations import get_equations
from plot_basin import compute_basin, plot_basin
from plot_trajectory import plot_trajectory


def reproduce_fig2():
    """
    משחזרת בדיוק את FIG. 2 מהמאמר:
    - מודל: Van der Pol (VdP)
    - פרמטרים: eps = 0.001, delta = 0.1, eta = 0.1, duffing = 0.0
    - תחום תנאי התחלה: y1 in [-20, 20], y2 in [-20, 20], v1(0)=0, v2(0)=0
    - צבעים:
        שחור   -> Zero Response
        כחול   -> In-Phase Mode
        צהוב   -> Anti-Phase Mode
        אדום   -> Stationary Beatings
    """
    # 1. הגדרת המשוואות בדיוק לפי הנתונים ב-FIG. 2
    eq = get_equations(
        model="vdp",          # Van der Pol
        coupling="linear",    # צימוד לינארי
        eps=0.001,            # eps = 0.001
        delta=0.1,            # delta = 0.1
        eta=0.1,              # eta = 0.1 (בדיוק כמו במאמר)
        duffing=0.0,          # duffing = 0.0
    )

    # 2. חישוב רשת 50x50 בתחום [-20, 20] במקביל על 8 מעבדים
    basin_data = compute_basin(
        system_eq=eq,
        y1_range=(-20.0, 20.0),
        y2_range=(-20.0, 20.0),
        resolution=50,        # 50x50 = 2,500 נקודות
        workers=8,            # 8 מעבדים במקביל
        method="numba_rk4",   # Numba JIT מואץ
        max_tau=30.0,
        chunk_tau=3.0,
    )

    # 3. הצגת המפה ב-Matplotlib ושמירתה כתמונה
    plot_basin(
        basin_data,
        mode="attractor",
        show_plot=True,
        save_path="fig2_vdp_basin.png"
    )


if __name__ == "__main__":
    print("==================================================")
    print("Coupled Oscillators Explorer: Reproducing FIG. 2")
    print("==================================================")

    reproduce_fig2()
