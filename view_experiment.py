"""
Coupled Oscillators Interactive Basin Viewer (view_experiment.py).

Allows loading and exploring any completed experiment from disk:
- Live interactive Time-Slider (time evolution of the basin of attraction)
- Click-to-Trajectory (click any point on the 2D grid to open its full trajectory & envelopes)
- Switch between Attractor Map, Sync Index Map, Localization Map, and Convergence Time Map

Usage:
  python view_experiment.py               (interactive selector of all saved experiments)
  python view_experiment.py --path <path_to_exp_subfolder>
  python view_experiment.py --mode sync_index
"""

from __future__ import annotations
import os
import sys
import glob
import pickle
import argparse
import matplotlib.pyplot as plt

from plot_basin import plot_basin, BasinData
from plot_trajectory import plot_trajectory


def find_all_saved_experiments() -> list[str]:
    """Finds all subdirectories containing a valid basin_data.pkl file."""
    search_pattern = os.path.join("experiments*", "exp*", "basin_data.pkl")
    pkl_files = glob.glob(search_pattern)
    return sorted([os.path.dirname(f) for f in pkl_files])


def load_experiment_data(exp_dir: str) -> BasinData:
    """Loads serialized BasinData object from an experiment directory."""
    pkl_path = os.path.join(exp_dir, "basin_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find 'basin_data.pkl' in: {exp_dir}")

    with open(pkl_path, "rb") as f:
        basin_data = pickle.load(f)
    return basin_data


def choose_experiment_via_dialog(initial_dir: str = ".") -> str | None:
    """Opens native Windows File Explorer dialog directly in the sync_explorer workspace directory."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Always start directly in current sync_explorer directory
        init_path = os.path.abspath(initial_dir)

        print(f"[EXPLORER] Opening file browser dialog in: {init_path}")
        selected_file = filedialog.askopenfilename(
            initialdir=init_path,
            title="Select 'basin_data.pkl' (or any file inside the experiment folder)",
            filetypes=[
                ("Experiment Data / Images", "*.pkl *.png"),
                ("Pickle Files", "*.pkl"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()

        if not selected_file:
            return None

        # If a file was selected, get its parent directory
        if os.path.isfile(selected_file):
            target_dir = os.path.dirname(selected_file)
        else:
            target_dir = selected_file

        # Verify basin_data.pkl exists in the directory
        if os.path.exists(os.path.join(target_dir, "basin_data.pkl")):
            return target_dir

        return None
    except Exception as e:
        print(f"[WARN] GUI dialog error: {e}")
        return None


def interactive_selector() -> str:
    """Displays an interactive CLI menu listing all available saved experiments."""
    exp_dirs = find_all_saved_experiments()
    if not exp_dirs:
        print("[ERROR] No saved experiments with 'basin_data.pkl' found!")
        print("Run an experiment first using: python main.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SAVED EXPERIMENTS VIEWER - SELECT EXPERIMENT TO EXPLORE")
    print("=" * 60)
    for i, p in enumerate(exp_dirs):
        print(f" [{i + 1}] {p}")
    print("=" * 60)

    choice = input(f"Select experiment [1-{len(exp_dirs)}] (default: 1): ").strip()
    try:
        idx = int(choice) - 1 if choice else 0
        if not (0 <= idx < len(exp_dirs)):
            idx = 0
    except ValueError:
        idx = 0

    return exp_dirs[idx]


def main():
    parser = argparse.ArgumentParser(description="Interactive Basin of Attraction Viewer")
    parser.add_argument("--path", type=str, default=None, help="Path to experiment subfolder containing basin_data.pkl")
    parser.add_argument("--cli", action="store_true", help="Force terminal CLI text menu instead of Windows dialog")
    parser.add_argument("--mode", type=str, choices=["attractor", "sync_index", "localization", "time"],
                        default="attractor", help="Display mode for the 2D basin map")
    args = parser.parse_args()

    target_dir = None

    if args.path:
        target_dir = args.path
    elif not args.cli:
        target_dir = choose_experiment_via_dialog()

    if not target_dir:
        # Fallback to CLI selector if dialog was cancelled or --cli was requested
        target_dir = interactive_selector()

    print(f"\n[LOADING] Loading experiment data from: {target_dir} ...")
    basin_data = load_experiment_data(target_dir)

    print(f"[READY] Launching Interactive Basin Explorer for {basin_data.system_eq.model.upper()}...")
    print(" -> Use the Slider at the bottom to explore time evolution (tau).")
    print(" -> Click any coordinate on the map to pop up its complete trajectory analysis.")
    plot_basin(basin_data, mode=args.mode, show_plot=True)


if __name__ == "__main__":
    main()
