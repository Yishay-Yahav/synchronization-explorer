"""
Coupled Oscillators: Attractor Classification & Dynamic Indices Engine.
Analyzes solution windows to classify attractors and compute continuous dynamical metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from solvers import SolutionWindow


@dataclass
class Classification:
    """Attractor classification result and continuous dynamical indices."""
    label: str               # "in-phase", "anti-phase", "stationary beating", "zero", "other"
    sync_index: float        # [0, 1] - 1.0=In-Phase, 0.5=Beating, 0.0=Anti-Phase
    beating_purity: float    # [0, 1] - 1.0=Pure symmetric beating, 0.0=Single mode
    localization: float      # [0, 1] - 0.5=Equal energy, 1.0=Osc 1 only, 0.0=Osc 2 only
    rms_amplitude: float     # Final RMS oscillation amplitude
    is_stable: bool          # True if stationary and stable


def _classify_single(
    window: SolutionWindow,
    zero_tol: float = 0.05,
    dominance_tol: float = 0.08,
    cv_tol: float = 0.15,
) -> Classification:
    """Classifies a single solution window."""
    y1, v1, y2, v2 = window.states

    # 1. Total RMS oscillation amplitude
    amp_y = np.mean(y1**2 + y2**2)
    amp_v = np.mean(v1**2 + v2**2)
    rms = float(np.sqrt(0.5 * (amp_y + amp_v)))

    # 2. Mean modal envelopes and coefficient of variation (CV)
    abs_mu0 = np.abs(window.mu0)
    abs_mu1 = np.abs(window.mu1)
    mean_mu0 = float(np.mean(abs_mu0))
    mean_mu1 = float(np.mean(abs_mu1))
    cv_mu0 = float(np.std(abs_mu0) / max(mean_mu0, 1e-12))
    cv_mu1 = float(np.std(abs_mu1) / max(mean_mu1, 1e-12))

    # 3. Modal synchronization index and beating purity
    modal_total = max(mean_mu0 + mean_mu1, 1e-12)
    sync_index = float(mean_mu0 / modal_total)
    beating_purity = float(max(0.0, 1.0 - abs(2.0 * sync_index - 1.0)))

    # 4. Energy localization index
    e1 = float(np.mean(y1**2 + v1**2))
    e2 = float(np.mean(y2**2 + v2**2))
    localization = float(e1 / max(e1 + e2, 1e-12))

    # 5. Stationary stability check
    is_stable = bool(window.is_converged and (rms < zero_tol or (cv_mu0 < cv_tol and cv_mu1 < cv_tol)))

    # 6. Attractor label assignment
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
    zero_tol: float = 0.05,
    dominance_tol: float = 0.08,
    cv_tol: float = 0.15,
) -> list[Classification]:
    """Classifies a list of SolutionWindows (or a single window)."""
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
        [2.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 2.0, 0.0],
        [2.0, 0.0, -2.0, 0.0],
        [0.01, 0.0, 0.01, 0.0],
    ]
    wins = solve(eq, test_ics, workers=4)
    results = classify(wins)
    for i, cl in enumerate(results):
        print(f"IC {test_ics[i]} -> Label: {cl.label}, Sync Index: {cl.sync_index:.2f}, Loc: {cl.localization:.2f}")
