"""
Coupled Oscillators Web Explorer: Lightweight Local HTTP & API Server.
Zero-dependency server providing simulation API and serving web dashboard.
"""

from __future__ import annotations
import http.server
import socketserver
import json
import os
import webbrowser
import numpy as np

from equations import get_equations
from solvers import solve, compute_modal
from classifier import classify


PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def run_full_timeseries_analysis(params: dict) -> dict:
    """
    Runs simulation for given parameters and returns full timeseries + fine-grained
    data so the client can scrub and compute live metrics at any point in time.
    """
    # 1. חילוץ פרמטרים של המשוואה
    eps_coupling = float(params.get("eps_coupling", 0.001))
    duffing = float(params.get("duffing", 0.0))
    delta_vdp = float(params.get("delta_vdp", 0.0))
    eta_vdp = float(params.get("eta_vdp", 0.15))
    delta_rayleigh = float(params.get("delta_rayleigh", 0.1))
    eta_rayleigh = float(params.get("eta_rayleigh", 0.15))
    gamma = float(params.get("gamma", 0.0))
    eps_scale = float(params.get("eps_scale", 0.001))

    # 2. תנאי התחלה
    y1 = float(params.get("y1", 2.0))
    v1 = float(params.get("v1", 0.0))
    y2 = float(params.get("y2", 0.0))
    v2 = float(params.get("v2", 0.0))
    ic = [y1, v1, y2, v2]

    # 3. הגדרות פותרן
    method = params.get("method", "numba_rk4")
    adaptive = bool(params.get("adaptive", True))
    chunk_tau = float(params.get("chunk_tau", 3.0))
    max_tau = float(params.get("max_tau", 30.0))
    slope_tol = float(params.get("slope_tol", 2e-5))
    zero_tol = float(params.get("zero_tol", 0.05))
    rk4_dt = float(params.get("rk4_dt", 0.05))

    # 4. יצירת המשוואה והרצת הפותרן
    eq = get_equations(
        eps_coupling=eps_coupling,
        duffing=duffing,
        delta_vdp=delta_vdp,
        eta_vdp=eta_vdp,
        delta_rayleigh=delta_rayleigh,
        eta_rayleigh=eta_rayleigh,
        gamma=gamma,
        eps_scale=eps_scale,
    )

    windows = solve(
        eq,
        [ic],
        method=method,
        adaptive=adaptive,
        chunk_tau=chunk_tau,
        max_tau=max_tau,
        slope_tol=slope_tol,
        zero_tol=zero_tol,
        rk4_dt=rk4_dt,
        points_per_chunk=1500,
    )
    final_win = windows[0]
    final_class = classify([final_win])[0]

    # 5. דגימה רציפה של כל המסלול לגרף ולסליידר (עד 1,500 נקודות)
    total_points = len(final_win.times)
    stride = max(1, total_points // 1500)
    idx = np.arange(0, total_points, stride)

    return {
        "success": True,
        "eps": eq.eps,
        "zero_tol": zero_tol,
        "slope_tol": slope_tol,
        "convergence_time": float(final_win.convergence_time),
        "is_converged": bool(final_win.is_converged),
        "times": final_win.times[idx].tolist(),
        "tau": final_win.tau[idx].tolist(),
        "y1": final_win.states[0, idx].tolist(),
        "v1": final_win.states[1, idx].tolist(),
        "y2": final_win.states[2, idx].tolist(),
        "v2": final_win.states[3, idx].tolist(),
        "mu0_abs": np.abs(final_win.mu0[idx]).tolist(),
        "mu1_abs": np.abs(final_win.mu1[idx]).tolist(),
        "final_classification": {
            "label": final_class.label,
            "sync_index": round(final_class.sync_index, 4),
            "beating_purity": round(final_class.beating_purity, 4),
            "localization": round(final_class.localization, 4),
            "rms": round(final_class.rms_amplitude, 4),
            "is_stable": bool(final_class.is_stable),
            "convergence_time": round(final_win.convergence_time, 1),
            "slope": float(f"{final_win.slope:.2e}"),
        }
    }


class ExplorerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/api/simulate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                params = json.loads(body.decode("utf-8"))
                result = run_full_timeseries_analysis(params)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def start_server(open_browser: bool = True):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ExplorerHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"==================================================")
        print(f"Synchronization Explorer Web Dashboard Active!")
        print(f"URL: {url}")
        print(f"==================================================")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    start_server(open_browser=True)
