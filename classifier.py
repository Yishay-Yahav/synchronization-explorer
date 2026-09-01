"""
Coupled Oscillators: Attractor Classification & Dynamic Indices Engine.
Ultra-clean, concise module for classifying solution states (single or list of windows).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from solvers import SolutionWindow


@dataclass
class Classification:
    """תוצאת סיווג האטרקטור ומדדי המצב הרציפים עבור תנאי התחלה"""
    label: str               # "in-phase", "anti-phase", "stationary beating", "zero", "other"
    sync_index: float        # [0, 1] - 1.0=In-Phase, 0.5=Beating, 0.0=Anti-Phase
    beating_purity: float    # [0, 1] - 1.0=פעימה מושלמת ושווה, 0.0=מוד יחיד
    localization: float      # [0, 1] - 0.5=שווה, 1.0=מתנד 1 בלבד, 0.0=מתנד 2 בלבד
    rms_amplitude: float     # משרעת RMS סופית
    is_stable: bool          # האם המצב סטציונרי ויציב


def _classify_single(
    window: SolutionWindow,
    zero_tol: float = 0.05,
    dominance_tol: float = 0.08,
    cv_tol: float = 0.15,
) -> Classification:
    """סיווג של חלון פתרון יחיד"""
    y1, v1, y2, v2 = window.states

    # 1. עוצמת תנודה כוללת (RMS)
    amp_y = np.mean(y1**2 + y2**2)
    amp_v = np.mean(v1**2 + v2**2)
    rms = float(np.sqrt(0.5 * (amp_y + amp_v)))

    # 2. מעטפות מודליות ממוצעות ומקדם השתנות (CV)
    abs_mu0 = np.abs(window.mu0)
    abs_mu1 = np.abs(window.mu1)
    mean_mu0 = float(np.mean(abs_mu0))
    mean_mu1 = float(np.mean(abs_mu1))
    cv_mu0 = float(np.std(abs_mu0) / max(mean_mu0, 1e-12))
    cv_mu1 = float(np.std(abs_mu1) / max(mean_mu1, 1e-12))

    # 3. אינדקס סנכרון מודלי (sync_index) וטוהר פעימה (beating_purity)
    modal_total = max(mean_mu0 + mean_mu1, 1e-12)
    sync_index = float(mean_mu0 / modal_total)
    beating_purity = float(max(0.0, 1.0 - abs(2.0 * sync_index - 1.0)))

    # 4. אינדקס לוקליזציה של האנרגיה הפיזית (localization)
    e1 = float(np.mean(y1**2 + v1**2))
    e2 = float(np.mean(y2**2 + v2**2))
    localization = float(e1 / max(e1 + e2, 1e-12))

    # 5. בדיקת יציבות סטציונרית
    is_stable = bool(window.is_converged and (rms < zero_tol or (cv_mu0 < cv_tol and cv_mu1 < cv_tol)))

    # 6. קביעת תווית האטרקטור (Label)
    if rms < zero_tol:
        label = "zero"
    elif not window.is_converged:
        label = "other"
    elif sync_index > (1.0 - dominance_tol) and cv_mu0 < cv_tol:
        label = "in-phase"
    elif sync_index < dominance_tol and cv_mu1 < cv_tol:
        label = "anti-phase"
    elif beating_purity > 0.80 and cv_mu0 < cv_tol and cv_mu1 < cv_tol:
        label = "stationary beating"
    else:
        label = "other"

    return Classification(
        label=label,
        sync_index=sync_index,
        beating_purity=beating_purity,
        localization=localization,
        rms_amplitude=rms,
        is_stable=is_stable,
    )


def classify(
    windows: SolutionWindow | list[SolutionWindow],
    zero_tol: float = 0.05,          # סף משרעת לדעיכה לאפס
    dominance_tol: float = 0.08,     # סף דומיננטיות לסנכרון מלא
    cv_tol: float = 0.15,            # סף מקדם השתנות מקסימלי
) -> list[Classification]:
    """
    מקבלת רשימת חלונות פתרון (או חלון בודד) ומחזירה רשימת אובייקטי Classification.
    """
    if isinstance(windows, SolutionWindow):
        windows_list = [windows]
    else:
        windows_list = windows

    return [_classify_single(w, zero_tol, dominance_tol, cv_tol) for w in windows_list]


if __name__ == "__main__":
    from equations import get_equations
    from solvers import solve

    eq = get_equations("rayleigh", eps=0.001, delta=0.1, eta=0.15)
    test_ics = [
        [2.0, 0.0, 0.0, 0.0],   # Beating
        [2.0, 0.0, 2.0, 0.0],   # In-Phase
        [2.0, 0.0, -2.0, 0.0],  # Anti-Phase
        [0.01, 0.0, 0.01, 0.0], # Zero
    ]

    # פותר עבור רשימת הנקודות במקביל ומסווג את כולן ביחד:
    windows = solve(eq, test_ics, workers=4)
    results = classify(windows)

    for i, cl in enumerate(results):
        print(f"נקודה {test_ics[i]}: תווית={cl.label}, sync_index={cl.sync_index:.2f}, loc={cl.localization:.2f}")
