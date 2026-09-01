"""
Coupled Oscillators: System Equations Engine.
Supports Rayleigh, Van der Pol (VdP), and Linear models with configurable coupling & Duffing cubic stiffness.
"""

from __future__ import annotations
import numpy as np


class SystemEquation:
    """
    Encapsulates differential equations and parameters for coupled oscillators.
    Fully picklable across parallel multiprocessing workers.
    """
    def __init__(
        self,
        model: str = "rayleigh",
        eps: float = 0.001,
        delta: float = 0.1,
        eta: float = 0.15,
        duffing: float = 0.0,
        coupling: str = "linear",
    ):
        self.model = model.lower() if model else "none"
        self.coupling = coupling.lower() if coupling else "none"
        self.has_coupling = (self.coupling != "none")
        self.eps = eps
        self.delta = delta
        self.eta = eta
        self.duffing = duffing

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.rhs(t, state)

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        y1, v1, y2, v2 = state
        eps = self.eps

        # 1. Coupling term between oscillators (linear: eps*(y1 - y2) or none)
        if self.has_coupling:
            c12 = eps * (y1 - y2)
            c21 = eps * (y2 - y1)
        else:
            c12 = 0.0
            c21 = 0.0

        # 2. Duffing cubic stiffness term (localization: eps * duffing * y^3)
        duff1 = eps * self.duffing * (y1**3)
        duff2 = eps * self.duffing * (y2**3)

        # 3. Nonlinear damping term (VdP, Rayleigh, or None)
        if self.model in ("vdp", "van_der_pol"):
            nl1 = eps * self.delta * v1 * (1.0 - y1**2 + self.eta * y1**4)
            nl2 = eps * self.delta * v2 * (1.0 - y2**2 + self.eta * y2**4)
        elif self.model in ("rayleigh", "coupled_rayleigh"):
            nl1 = eps * self.delta * v1 * (1.0 - v1**2 + self.eta * v1**4)
            nl2 = eps * self.delta * v2 * (1.0 - v2**2 + self.eta * v2**4)
        else:  # none / linear
            nl1 = 0.0
            nl2 = 0.0

        # 4. State derivatives: [dy1/dt, dv1/dt, dy2/dt, dv2/dt]
        dy1 = v1
        dv1 = -y1 - c12 - duff1 - nl1
        dy2 = v2
        dv2 = -y2 - c21 - duff2 - nl2

        return np.array([dy1, dv1, dy2, dv2], dtype=float)


def get_equations(
    model: str = "rayleigh",
    eps: float = 0.001,
    delta: float = 0.1,
    eta: float = 0.15,
    duffing: float = 0.0,
    coupling: str = "linear",
) -> SystemEquation:
    """Factory function returning a configured SystemEquation instance."""
    return SystemEquation(
        model=model,
        eps=eps,
        delta=delta,
        eta=eta,
        duffing=duffing,
        coupling=coupling,
    )


if __name__ == "__main__":
    for m in ["rayleigh", "vdp", "none"]:
        eq = get_equations(m, eps=0.001, delta=0.1, eta=0.15, duffing=0.0)
        print(f"Model '{eq.model}' RHS output:", eq(0.0, [1.0, 0.0, -1.0, 0.5]))
