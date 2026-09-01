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
from classifier import classify, _classify_single, SolutionWindow


PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def run_full_timeseries_analysis(params: dict) -> dict:
    """
    Runs simulation for given parameters and returns full timeseries + step-by-step
    classification snapshots for time slider scrub.
    """
    model = params.get("model", "rayleigh")
    eps = float(params.get("eps", 0.001))
    delta = float(params.get("delta", 0.1))
    eta = float(params.get("eta", 0.15))
    duffing = float(params.get("duffing", 0.0))

    y1 = float(params.get("y1", 2.0))
    v1 = float(params.get("v1", 0.0))
    y2 = float(params.get("y2", 0.0))
    v2 = float(params.get("v2", 0.0))
    ic = [y1, v1, y2, v2]

    method = params.get("method", "numba_rk4")
    adaptive = bool(params.get("adaptive", True))
    chunk_tau = float(params.get("chunk_tau", 3.0))
    max_tau = float(params.get("max_tau", 30.0))
    slope_tol = float(params.get("slope_tol", 2e-5))
    zero_tol = float(params.get("zero_tol", 0.05))
    rk4_dt = float(params.get("rk4_dt", 0.05))

    # 1. יצירת המשוואה והרצת הפותרן
    eq = get_equations(model=model, eps=eps, delta=delta, eta=eta, duffing=duffing)
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
        points_per_chunk=1000,
    )
    final_win = windows[0]
    final_class = classify([final_win])[0]

    # 2. הרצת סנאפשוטים לאורך ציר הזמן (כדי שהסליידר יוכל לעבור על כל שלב)
    chunk_time = chunk_tau / eps
    total_time = float(final_win.times[-1])
    n_chunks = max(1, int(np.round(total_time / chunk_time)))

    snapshots = []
    # נדגום בכל נקודת מקטע זמן את המצב והסיווג
    for step in range(1, n_chunks + 1):
        t_step_end = min(step * chunk_time, total_time)
        tau_step = eps * t_step_end

        # חילוץ תת-חלון של המקטע הנוכחי
        sub_win_results = solve(
            eq,
            [ic],
            method=method,
            adaptive=False,
            max_tau=tau_step,
            chunk_tau=chunk_tau,
            rk4_dt=rk4_dt,
            points_per_chunk=600,
        )
        sub_win = sub_win_results[0]
        sub_class = _classify_single(sub_win, zero_tol=zero_tol)

        snapshots.append({
            "step": step,
            "time": float(t_step_end),
            "tau": float(tau_step),
            "label": sub_class.label,
            "sync_index": round(sub_class.sync_index, 4),
            "beating_purity": round(sub_class.beating_purity, 4),
            "localization": round(sub_class.localization, 4),
            "rms": round(sub_class.rms_amplitude, 4),
            "is_stable": bool(sub_class.is_stable),
            "slope": float(f"{sub_win.slope:.2e}"),
        })

    # 3. דגימת נתוני מסלול מלאים עבור הגרפים (דגימה חסכונית ומהירה ל-JSON)
    # נדגום עד 1,200 נקודות לגרף חלק וסופר מהיר
    total_points = len(final_win.times)
    stride = max(1, total_points // 1200)
    idx = np.arange(0, total_points, stride)

    return {
        "success": True,
        "times": final_win.times[idx].tolist(),
        "tau": final_win.tau[idx].tolist(),
        "y1": final_win.states[0, idx].tolist(),
        "v1": final_win.states[1, idx].tolist(),
        "y2": final_win.states[2, idx].tolist(),
        "v2": final_win.states[3, idx].tolist(),
        "mu0_abs": np.abs(final_win.mu0[idx]).tolist(),
        "mu1_abs": np.abs(final_win.mu1[idx]).tolist(),
        "snapshots": snapshots,
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
        # מניעת Caching כדי ששינויים בדפדפן ייטענו מיד
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
