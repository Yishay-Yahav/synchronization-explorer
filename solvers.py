"""
Coupled Oscillators: Multi-Method & Parallel Batch Solver Engine.

Supported Models:
- "rayleigh": Velocity-dependent nonlinear damping
- "vdp": Displacement-dependent nonlinear damping
- "none": Linear oscillator coupling (no nonlinear damping)

Supported Solver Methods (method):
- "numba_rk4": Accelerated JIT compiled RK4 (50x-100x faster, default)
- "dop853": SciPy 8th-order Runge-Kutta (High precision adaptive)
- "rk45": SciPy 5th-order Runge-Kutta
"""

from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
from scipy.integrate import solve_ivp
import numba

from equations import SystemEquation


@dataclass
class SolutionWindow:
    """תוצאת החלון הסופי לאחר התכנסות עבור תנאי התחלה מסוים"""
    times: np.ndarray             # זמנים בחלון [t]
    states: np.ndarray            # מצבים בחלון [y1, v1, y2, v2]
    tau: np.ndarray               # זמן איטי: tau = eps * t
    mu0: np.ndarray               # מעטפת מוד סימטרי: In-Phase
    mu1: np.ndarray               # מעטפת מוד אנטי-סימטרי: Anti-Phase
    convergence_time: float       # זמן ההתכנסות (t)
    is_converged: bool            # האם התכנס
    slope: float                  # שיפוע המעטפה הסופי


def compute_modal(times: np.ndarray, states: np.ndarray, eps: float):
    """חישוב מעטפות מודליות איטיות וזמן איטי"""
    y1, v1, y2, v2 = states
    phi1 = (v1 + 1j * y1) * np.exp(-1j * times)
    phi2 = (v2 + 1j * y2) * np.exp(-1j * times)
    mu0 = phi1 + phi2    # In-phase
    mu1 = phi1 - phi2    # Anti-phase
    tau = eps * times    # tau
    return tau, mu0, mu1


# --- מנוע JIT מואץ ב-Numba (מהירות קוד C) ---
@numba.njit(fastmath=True)
def _rk4_step_jit(state, eps, delta, eta, duffing, model_id, dt):
    """צעד בודד של RK4 בקוד מכונה (model_id: 0=none, 1=rayleigh, 2=vdp)"""
    def eval_rhs(s):
        y1, v1, y2, v2 = s[0], s[1], s[2], s[3]
        c12 = eps * (y1 - y2)
        c21 = eps * (y2 - y1)
        duff1 = eps * duffing * (y1**3)
        duff2 = eps * duffing * (y2**3)

        if model_id == 2:  # vdp
            nl1 = eps * delta * v1 * (1.0 - y1**2 + eta * y1**4)
            nl2 = eps * delta * v2 * (1.0 - y2**2 + eta * y2**4)
        elif model_id == 1:  # rayleigh
            nl1 = eps * delta * v1 * (1.0 - v1**2 + eta * v1**4)
            nl2 = eps * delta * v2 * (1.0 - v2**2 + eta * v2**4)
        else:  # 0 = none / linear
            nl1 = 0.0
            nl2 = 0.0

        return np.array([v1, -y1 - c12 - duff1 - nl1, v2, -y2 - c21 - duff2 - nl2])

    k1 = eval_rhs(state)
    k2 = eval_rhs(state + 0.5 * dt * k1)
    k3 = eval_rhs(state + 0.5 * dt * k2)
    k4 = eval_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@numba.njit(fastmath=True)
def _numba_integrate_chunk(state, eps, delta, eta, duffing, model_id, t_start, t_end, dt, n_points):
    """אינטגרציה של מקטע שלם בלולאת C מהירה"""
    total_steps = int(np.ceil((t_end - t_start) / dt))
    actual_dt = (t_end - t_start) / total_steps
    save_interval = max(1, total_steps // (n_points - 1))

    times = np.empty(n_points, dtype=np.float64)
    states = np.empty((4, n_points), dtype=np.float64)

    curr_s = state.copy()
    curr_t = t_start
    saved_idx = 0

    times[0] = curr_t
    states[:, 0] = curr_s
    saved_idx += 1

    for step in range(1, total_steps + 1):
        curr_s = _rk4_step_jit(curr_s, eps, delta, eta, duffing, model_id, actual_dt)
        curr_t = t_start + step * actual_dt
        if step % save_interval == 0 and saved_idx < n_points:
            times[saved_idx] = curr_t
            states[:, saved_idx] = curr_s
            saved_idx += 1

    if saved_idx < n_points:
        times[-1] = curr_t
        states[:, -1] = curr_s

    return times, states


def _solve_single_ic(
    system_eq: SystemEquation,
    ic: np.ndarray,
    method: str = "numba_rk4",
    adaptive: bool = True,
    chunk_tau: float = 3.0,
    max_tau: float = 30.0,
    slope_tol: float = 2e-5,
    zero_tol: float = 0.05,
    consecutive: int = 2,
    points_per_chunk: int = 1500,
    rk4_dt: float = 0.05,
) -> SolutionWindow:
    """פונקציה פנימית: פותרת תנאי התחלה יחיד"""
    eps = getattr(system_eq, "eps", 0.001)
    rhs_fn = getattr(system_eq, "rhs", system_eq)
    model = getattr(system_eq, "model", "rayleigh")

    if model in ("vdp", "van_der_pol"):
        model_id = 2
    elif model in ("rayleigh", "coupled_rayleigh"):
        model_id = 1
    else:
        model_id = 0

    delta = getattr(system_eq, "delta", 0.1)
    eta = getattr(system_eq, "eta", 0.15)
    duffing = getattr(system_eq, "duffing", 0.0)

    chunk_time = chunk_tau / eps
    max_time = max_tau / eps
    min_time = consecutive * chunk_time

    state = np.array(ic, dtype=np.float64)
    t_curr = 0.0
    stable_count = 0
    final_slope = 1.0

    last_t, last_y, last_tau, last_mu0, last_mu1 = [None] * 5

    while t_curr < max_time:
        t_next = min(t_curr + chunk_time, max_time)

        if method == "numba_rk4":
            t_chunk, y_chunk = _numba_integrate_chunk(
                state, eps, delta, eta, duffing, model_id, t_curr, t_next, rk4_dt, points_per_chunk
            )
        else:
            t_eval = np.linspace(t_curr, t_next, points_per_chunk)
            sol = solve_ivp(rhs_fn, (t_curr, t_next), state, method=method, t_eval=t_eval, rtol=1e-8, atol=1e-10)
            if not sol.success:
                break
            t_chunk, y_chunk = sol.t, sol.y

        last_t, last_y = t_chunk, y_chunk
        last_tau, last_mu0, last_mu1 = compute_modal(last_t, last_y, eps)
        state = last_y[:, -1]
        t_curr = t_next

        if adaptive:
            rms = np.sqrt(0.5 * np.mean(last_y[0]**2 + last_y[2]**2 + last_y[1]**2 + last_y[3]**2))
            if rms < zero_tol:
                stable_count += 1
                if stable_count >= consecutive:
                    return SolutionWindow(last_t, last_y, last_tau, last_mu0, last_mu1, t_curr, True, 0.0)
                continue

            if t_curr >= min_time:
                dt = last_t - last_t[0]
                slope0 = np.abs(np.polyfit(dt, np.abs(last_mu0), 1)[0])
                slope1 = np.abs(np.polyfit(dt, np.abs(last_mu1), 1)[0])
                final_slope = float(max(slope0, slope1))

                if final_slope < slope_tol:
                    stable_count += 1
                    if stable_count >= consecutive:
                        return SolutionWindow(last_t, last_y, last_tau, last_mu0, last_mu1, t_curr, True, final_slope)
                else:
                    stable_count = 0

    dt = last_t - last_t[0]
    final_slope = float(max(
        np.abs(np.polyfit(dt, np.abs(last_mu0), 1)[0]),
        np.abs(np.polyfit(dt, np.abs(last_mu1), 1)[0])
    ))
    return SolutionWindow(last_t, last_y, last_tau, last_mu0, last_mu1, t_curr, (not adaptive or final_slope < slope_tol), final_slope)


# פונקציית מעטפת להרצה בתוך ProcessPoolExecutor
def _worker_wrapper(args):
    system_eq, ic, kwargs = args
    return _solve_single_ic(system_eq, ic, **kwargs)


def solve(
    system_eq: SystemEquation,
    ics: list[list[float]] | list[float] | np.ndarray,
    method: str = "numba_rk4",     # שיטת הפתרון ("numba_rk4", "dop853", "rk45")
    workers: int = 8,              # מספר המעבדים שיפעלו במקביל (1 עד 8, 16 וכו')
    adaptive: bool = True,         # האם להשתמש בעצירה אדפטיבית לפי שיפוע
    chunk_tau: float = 3.0,        # גודל מקטע בזמן איטי
    max_tau: float = 30.0,         # זמן ריצה מקסימלי בזמן איטי
    slope_tol: float = 2e-5,       # סף שיפוע להתכנסות
    zero_tol: float = 0.05,        # סף דעיכה לאפס
    consecutive: int = 2,          # מספר חלונות רצופים לאימות יציבות
    points_per_chunk: int = 1500,  # נקודות דיגום לכל חלון
    rk4_dt: float = 0.05,          # גודל צעד זמן עבור numba_rk4
) -> list[SolutionWindow]:
    """
    פונקציית הפותרן הראשית והמאוחדת:
    מקבלת תמיד רשימת תנאי התחלה (נקודה אחת או רשימה של עשרות נקודות),
    פותרת אותן במקביל לפי מספר המעבדים שנבחר (workers), ומחזירה רשימת תוצאות.
    """
    raw_ics = np.asarray(ics, dtype=float)
    if raw_ics.ndim == 1:
        ics_list = [raw_ics]
    else:
        ics_list = [row for row in raw_ics]

    solver_kwargs = {
        "method": method,
        "adaptive": adaptive,
        "chunk_tau": chunk_tau,
        "max_tau": max_tau,
        "slope_tol": slope_tol,
        "zero_tol": zero_tol,
        "consecutive": consecutive,
        "points_per_chunk": points_per_chunk,
        "rk4_dt": rk4_dt,
    }

    if len(ics_list) == 1 or workers == 1:
        return [_solve_single_ic(system_eq, ic, **solver_kwargs) for ic in ics_list]

    max_workers = min(workers, len(ics_list))
    tasks = [(system_eq, ic, solver_kwargs) for ic in ics_list]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker_wrapper, tasks))

    return results


if __name__ == "__main__":
    from equations import get_equations
    import time

    eq_none = get_equations("none", eps=0.001, delta=0.1, eta=0.15)
    res_none = solve(eq_none, [[2.0, 0.0, 0.0, 0.0]], method="numba_rk4")
    print(f"מודל ללא (none): התכנס ב-t={res_none[0].convergence_time:.1f}, שיפוע={res_none[0].slope:.2e}")
