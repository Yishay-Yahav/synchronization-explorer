"""
Coupled Oscillators: Unified General Nonlinear Equations.
Supports any simultaneous combination of:
- Linear Coupling: eps_coupling * (y1 - y2) (set to 0 for uncoupled oscillators)
- Duffing Cubic Stiffness: duffing * eps * y^3
- Van der Pol Nonlinear Damping: delta_vdp * eps * v * (1 - y^2 + eta_vdp * y^4)
- Rayleigh Nonlinear Damping: delta_rayleigh * eps * v * (1 - v^2 + eta_rayleigh * v^4)
- Linear Damping: gamma * v
"""

import numpy as np


class SystemEquation:
    """אובייקט משוואה מאוחד וגמיש התומך בכל שילוב של איברים פיזיקליים"""
    def __init__(
        self,
        eps_coupling: float = 0.001,
        duffing: float = 0.0,
        delta_vdp: float = 0.0,
        eta_vdp: float = 0.15,
        delta_rayleigh: float = 0.1,
        eta_rayleigh: float = 0.15,
        gamma: float = 0.0,
        eps_scale: float = 0.001,
    ):
        self.eps_coupling = float(eps_coupling)
        self.duffing = float(duffing)
        self.delta_vdp = float(delta_vdp)
        self.eta_vdp = float(eta_vdp)
        self.delta_rayleigh = float(delta_rayleigh)
        self.eta_rayleigh = float(eta_rayleigh)
        self.gamma = float(gamma)

        # סקאלת אפסילון עבור זמן איטי tau = eps * t
        self.eps = float(eps_scale if eps_scale > 0 else max(self.eps_coupling, 0.001))

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.rhs(t, state)

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        y1, v1, y2, v2 = state
        eps_c = self.eps_coupling
        eps_s = self.eps

        # 1. צימוד לינארי (0 עבור מערכת לא מצומדת)
        c12 = eps_c * (y1 - y2)
        c21 = eps_c * (y2 - y1)

        # 2. איבר דאפינג ממעלה שלישית
        duff1 = eps_s * self.duffing * (y1**3)
        duff2 = eps_s * self.duffing * (y2**3)

        # 3. איבר ואן דר פול (תלוי מיקום)
        if self.delta_vdp > 0:
            nl_vdp1 = eps_s * self.delta_vdp * v1 * (1.0 - y1**2 + self.eta_vdp * y1**4)
            nl_vdp2 = eps_s * self.delta_vdp * v2 * (1.0 - y2**2 + self.eta_vdp * y2**4)
        else:
            nl_vdp1 = nl_vdp2 = 0.0

        # 4. איבר ריילי (תלוי מהירות)
        if self.delta_rayleigh > 0:
            nl_ray1 = eps_s * self.delta_rayleigh * v1 * (1.0 - v1**2 + self.eta_rayleigh * v1**4)
            nl_ray2 = eps_s * self.delta_rayleigh * v2 * (1.0 - v2**2 + self.eta_rayleigh * v2**4)
        else:
            nl_ray1 = nl_ray2 = 0.0

        # 5. שיכוך לינארי
        damp1 = self.gamma * v1
        damp2 = self.gamma * v2

        # 6. נגזרות
        dy1 = v1
        dv1 = -y1 - c12 - duff1 - nl_vdp1 - nl_ray1 - damp1
        dy2 = v2
        dv2 = -y2 - c21 - duff2 - nl_vdp2 - nl_ray2 - damp2

        return np.array([dy1, dv1, dy2, dv2], dtype=float)


def get_equations(**kwargs) -> SystemEquation:
    """יוצרת ומחזירה אובייקט SystemEquation גמיש"""
    return SystemEquation(**kwargs)


if __name__ == "__main__":
    # בדיקה: מערכת לא מצומדת (eps_coupling=0) עם ריילי בלבד
    eq_uncoupled = get_equations(eps_coupling=0.0, delta_rayleigh=0.1)
    print("משוואה לא מצומדת נגזרות:", eq_uncoupled(0.0, [1.0, 0.0, -1.0, 0.0]))
