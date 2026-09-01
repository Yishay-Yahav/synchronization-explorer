"""
Coupled Oscillators: Rayleigh & Van der Pol (VdP) Numerical Equations.
"""

from dataclasses import dataclass
import numpy as np


class SystemEquation:
    """אובייקט משוואה המכיל את פונקציית הנגזרות והפרמטרים (ניתן ל-Pickle מלא בהרצה מקבילית)"""
    def __init__(self, model="rayleigh", eps=0.001, delta=0.1, eta=0.15, duffing=0.0):
        self.model = model
        self.eps = eps
        self.delta = delta
        self.eta = eta
        self.duffing = duffing
        self.is_vdp = (model == "vdp")

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.rhs(t, state)

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        y1, v1, y2, v2 = state
        eps = self.eps

        c12 = eps * (y1 - y2)
        c21 = eps * (y2 - y1)
        duff1 = eps * self.duffing * (y1**3)
        duff2 = eps * self.duffing * (y2**3)

        if self.is_vdp:
            nl1 = eps * self.delta * v1 * (1.0 - y1**2 + self.eta * y1**4)
            nl2 = eps * self.delta * v2 * (1.0 - y2**2 + self.eta * y2**4)
        else:  # rayleigh
            nl1 = eps * self.delta * v1 * (1.0 - v1**2 + self.eta * v1**4)
            nl2 = eps * self.delta * v2 * (1.0 - v2**2 + self.eta * v2**4)

        dy1 = v1
        dv1 = -y1 - c12 - duff1 - nl1
        dy2 = v2
        dv2 = -y2 - c21 - duff2 - nl2

        return np.array([dy1, dv1, dy2, dv2], dtype=float)


def get_equations(model="rayleigh", eps=0.001, delta=0.1, eta=0.15, duffing=0.0) -> SystemEquation:
    """יוצרת ומחזירה אובייקט משוואה SystemEquation"""
    return SystemEquation(model=model, eps=eps, delta=delta, eta=eta, duffing=duffing)


if __name__ == "__main__":
    eq = get_equations("rayleigh", eps=0.001, delta=0.1, eta=0.15, duffing=0.5)
    print(f"נוצר אובייקט משוואה: מודל={eq.model}, אפסילון={eq.eps}")
    print("נגזרות:", eq(0.0, [1.0, 0.0, -1.0, 0.5]))
