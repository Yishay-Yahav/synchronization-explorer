# Reference Paper Alignment Policy (Wolfovich & Gendelman, Chaos 2025)

## 1. Reference Paper Foundation
- All dynamic models, experiments, and basin visualizers in this project are benchmarked against:
  **"Modal and wave synchronization in coupled self-excited oscillators", Y. Wolfovich & O. V. Gendelman, Chaos 35, 023139 (2025)**.

## 2. Core Scientific Invariants
- **Bistable Van der Pol (BVdP) Equations (Eq. 3)**:
  $$\ddot{y}_1 + y_1 + \varepsilon(y_1 - y_2) + \varepsilon\delta\dot{y}_1(1 - y_1^2 + \eta y_1^4) = 0$$
  $$\ddot{y}_2 + y_2 + \varepsilon(y_2 - y_1) + \varepsilon\delta\dot{y}_2(1 - y_2^2 + \eta y_2^4) = 0$$
- **Canonical Parameters**:
  - $\varepsilon = 0.001$, $\delta = 0.1$, $\eta = 0.1$, Duffing = $0.0$.
  - Time hierarchy: $0 < \varepsilon \ll \delta \ll 1$. Slow time $\tau = \varepsilon t$, super-slow time $\xi = \delta\tau$.
- **4 Attractor Classification & Colormap Standard (Fig. 2)**:
  - **Red / Rust (`#D9531E`)**: Stationary Beatings (Modal Synchronization)
  - **Blue (`#0072BD`)**: In-phase mode
  - **Yellow / Gold (`#E69F00`)**: Anti-phase mode
  - **Black (`#000000`)**: Zero response (trivial equilibrium)
- **Analytical Amplitude Reference**:
  - Modal beating envelope amplitude $A = 2\sqrt{\frac{3 + \sqrt{9-80\eta}}{10\eta}} = 4.0$ (for $\eta = 0.1$).

## 3. Communication Invariant
- Always notify and explain prior to making any code edits or running long simulations, keeping explanations short, punchy, bulleted, and in Hebrew.
