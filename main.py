"""
Coupled Oscillators Synchronization Explorer: Main Entry Point (main.py).
Configured to reproduce FIG. 2 from the publication (Van der Pol Coupled Oscillators).
Automatically formats output filename with all core physical and solver parameters.
"""

import os
from equations import get_equations
from plot_basin import compute_basin, plot_basin


def reproduce_fig2():
    """
    משחזרת בדיוק את FIG. 2 מהמאמר:
    - מודל: Van der Pol (VdP)
    - פרמטרים: eps = 0.001, delta = 0.1, eta = 0.1, duffing = 0.0
    - תחום תנאי התחלה: y1 in [-20, 20], y2 in [-20, 20], v1(0)=0, v2(0)=0
    - שמירת התמונה עם כל פרמטרי הריצה בשם הקובץ
    """
    # 1. הגדרת פרמטרים
    model = "vdp"          # אפשרויות: "vdp", "rayleigh", "none"
    coupling = "linear"    # אפשרויות: "linear", "none"
    eps = 0.001            # אפסילון
    delta = 0.1            # דלתא
    eta = 0.1              # אטה
    duffing = 0.0          # דאפינג (לוקליזציה)

    resolution = 50        # רזולוציה: 50x50 = 2,500 נקודות
    max_tau = 30.0         # זמן מקסימלי (זמן איטי tau)
    chunk_tau = 3.0        # גודל מקטע
    workers = 8            # מספר מעבדים במקביל

    # 2. יצירת המשוואות
    eq = get_equations(
        model=model,
        coupling=coupling,
        eps=eps,
        delta=delta,
        eta=eta,
        duffing=duffing,
    )

    # 3. חישוב רשת האגנים במקביל
    basin_data = compute_basin(
        system_eq=eq,
        y1_range=(-20.0, 20.0),
        y2_range=(-20.0, 20.0),
        resolution=resolution,
        workers=workers,
        method="numba_rk4",
        max_tau=max_tau,
        chunk_tau=chunk_tau,
    )

    # 4. יצירת שם קובץ דינמי המכיל את כל הפרמטרים
    filename = (
        f"basin_{model}_{coupling}_eps{eps}_delta{delta}_eta{eta}_"
        f"duff{duffing}_tau{int(max_tau)}_res{resolution}x{resolution}.png"
    )

    # 5. הצגת המפה ושמירתה
    plot_basin(
        basin_data,
        mode="attractor",
        show_plot=True,
        save_path=filename
    )


if __name__ == "__main__":
    print("==================================================")
    print("Coupled Oscillators Explorer: Reproducing FIG. 2")
    print("==================================================")

    reproduce_fig2()
