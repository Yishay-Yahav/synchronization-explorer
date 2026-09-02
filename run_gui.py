import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

EXPERIMENTS = [
    "1. Linear Uncoupled Oscillators",
    "2. Linear Coupled Oscillators",
    "3. Coupled Van der Pol (Fig. 2)",
    "4. Coupled Rayleigh Oscillators",
    "5. Coupled Linear with Duffing",
    "6. Uncoupled Duffing Only",
    "7. Coupled Van der Pol with Duffing",
    "8. Coupled Rayleigh with Duffing",
]

def on_run():
    selected_indices = listbox.curselection()
    if not selected_indices:
        messagebox.showwarning("No Selection", "Please select at least one experiment.")
        return
        
    exp_ids = [str(i + 1) for i in selected_indices]
    
    res = res_var.get().strip()
    tau = tau_var.get().strip()
    workers = workers_var.get().strip()
    is_full = full_var.get()
    
    cmd = ["python", "main.py", "--exp"] + exp_ids
    if res:
        cmd.extend(["--res", res])
    if tau:
        cmd.extend(["--tau", tau])
    if workers:
        cmd.extend(["--workers", workers])
    if is_full:
        cmd.append("--full")
        
    cmd_str = " ".join(cmd)
    # Open a new interactive terminal window and execute the command
    full_cmd = f'start cmd /k "{cmd_str}"'
    
    subprocess.Popen(full_cmd, shell=True)
    root.destroy()

root = tk.Tk()
root.title("Sync Explorer Launcher")
root.geometry("420x520")
root.resizable(False, False)
root.configure(padx=20, pady=20)

style = ttk.Style()
style.theme_use('clam')

ttk.Label(root, text="Select Experiments to Run:", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))

frame_list = tk.Frame(root)
frame_list.pack(fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(frame_list)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

listbox = tk.Listbox(frame_list, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, font=("Segoe UI", 10), height=9)
for exp in EXPERIMENTS:
    listbox.insert(tk.END, exp)
listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=listbox.yview)
# Pre-select VdP by default
listbox.selection_set(2)

ttk.Label(root, text="Run Parameters:", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(15, 5))

params_frame = ttk.Frame(root)
params_frame.pack(fill=tk.X)

ttk.Label(params_frame, text="Resolution (e.g. 37):", font=("Segoe UI", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
res_var = tk.StringVar(value="37")
ttk.Entry(params_frame, textvariable=res_var, width=10, font=("Segoe UI", 10)).grid(row=0, column=1, sticky=tk.W, padx=10)

ttk.Label(params_frame, text="Max Tau (e.g. 100):", font=("Segoe UI", 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
tau_var = tk.StringVar(value="100.0")
ttk.Entry(params_frame, textvariable=tau_var, width=10, font=("Segoe UI", 10)).grid(row=1, column=1, sticky=tk.W, padx=10)

ttk.Label(params_frame, text="Workers (e.g. 8):", font=("Segoe UI", 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
workers_var = tk.StringVar(value="8")
ttk.Entry(params_frame, textvariable=workers_var, width=10, font=("Segoe UI", 10)).grid(row=2, column=1, sticky=tk.W, padx=10)

full_var = tk.BooleanVar(value=False)
ttk.Checkbutton(params_frame, text="Disable Adaptive Stopping (--full)", variable=full_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=10)

run_btn = tk.Button(root, text="🚀 Run Experiments", command=on_run, font=("Segoe UI", 12, "bold"), bg="#0072bd", fg="white", activebackground="#005b9f", activeforeground="white", relief=tk.FLAT)
run_btn.pack(fill=tk.X, pady=(20, 0), ipady=8)

root.mainloop()
