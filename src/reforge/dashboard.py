import json
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    ReforgeDashboard().run()


try:
    from scipy.ndimage import uniform_filter1d as _uniform_filter1d
    def smooth_series(y, k):
        return _uniform_filter1d(np.asarray(y, dtype=float),
                                 size=max(1, int(k)), mode="nearest")
except Exception:
    def smooth_series(y, k):
        k = max(1, int(k))
        y = np.asarray(y, dtype=float)
        if k == 1 or len(y) < 2:
            return y.copy()
        kernel = np.ones(k) / k
        pad = k // 2
        ypad = np.pad(y, (pad, k - 1 - pad), mode="edge")
        return np.convolve(ypad, kernel, mode="valid")


def try_parse_loss_line(line: str) -> dict | None:
    line = line.strip()
    if not line or "{" not in line or "'loss'" not in line and '"loss"' not in line:
        return None
    for attempt in (line, re.sub(r"(?<!\\)'", '"', line)):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict) and "loss" in obj:
                return obj
        except Exception:
            pass
    m = re.search(r"\{[^{}]+\}", line)
    if m:
        try:
            obj = json.loads(m.group(0).replace("'", '"'))
            if isinstance(obj, dict) and "loss" in obj:
                return obj
        except Exception:
            pass
    return None


def parse_loss_input(raw_input: str) -> list[dict]:
    parsed = []
    for line in raw_input.splitlines():
        entry = try_parse_loss_line(line)
        if entry:
            parsed.append(entry)
    if parsed:
        return parsed
    try:
        for obj in re.findall(r"\{[^{}]+\}", raw_input):
            entry = json.loads(obj.replace("'", '"'))
            if isinstance(entry, dict) and "loss" in entry:
                parsed.append(entry)
    except Exception:
        pass
    return parsed


def parse_trainer_state(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return [e for e in state.get("log_history", []) if "loss" in e]
    except Exception:
        return []


def calculate_grade(slope_norm, snr, exposure_pct) -> str:
    score = 0
    if   slope_norm <= -3.0: score += 4
    elif slope_norm <= -1.5: score += 3
    elif slope_norm <= -0.5: score += 2
    elif slope_norm <= -0.1: score += 1
    if   snr >= 3.0: score += 3
    elif snr >= 2.0: score += 2
    elif snr >= 1.2: score += 1
    if   exposure_pct >= 36: score += 2
    elif exposure_pct >= 18: score += 1
    return "A" if score >= 8 else "B" if score >= 6 else "C" if score >= 4 else "D"


def trend_label(norm: float) -> str:
    if   norm <= -3.0:     return "Strong Drop"
    elif norm <= -1.0:     return "Moderate Drop"
    elif norm <  -0.1:     return "Weak Drop"
    elif abs(norm) <= 0.1: return "Flat"
    elif norm <   1.0:     return "Weak Rise"
    elif norm <   3.0:     return "Moderate Rise"
    else:                  return "Strong Rise"


class ReforgeDashboard:
    _DEFAULTS = dict(
        base_model = "",
        output_dir = os.environ.get("REFORGE_OUTPUT_DIR",
                                     os.path.join(os.path.expanduser("~"), "Reforge_Output")),
        epoch      = "1",
        steps      = "",
    )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reforge Dashboard")
        self.root.geometry("1180x780")
        self.root.minsize(920, 600)

        self._proc: subprocess.Popen | None = None
        self._running = False
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

        self._line_queue: queue.Queue = queue.Queue()
        self._live_logs: list[dict] = []
        self._step_count = 0
        self._loss_header_printed = False

        self._style_setup()
        self._build_ui()
        self._update_cmd_preview()

        self.root.after(200, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _style_setup(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"))
        style.configure("Run.TButton",   font=("Segoe UI", 10, "bold"))
        style.configure("Stop.TButton",  font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        paned = ttk.PanedWindow(main, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)

        btn = ttk.Frame(left, padding=(6, 6))
        btn.pack(fill="x", side="bottom")
        ttk.Separator(left, orient="horizontal").pack(
            fill="x", side="bottom", pady=(4, 0))

        self._btn_start = ttk.Button(
            btn, text="  Start Training  ", style="Run.TButton",
            command=self._start_training)
        self._btn_start.pack(side="left", padx=4)

        self._btn_stop = ttk.Button(
            btn, text="  Stop (graceful)  ", style="Stop.TButton",
            command=self._stop_training, state="disabled")
        self._btn_stop.pack(side="left", padx=4)

        self._lbl_status = ttk.Label(btn, text="Idle", style="Status.TLabel")
        self._lbl_status.pack(side="left", padx=12)

        scroll_frame = ttk.Frame(left)
        scroll_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_frame, highlightthickness=0, width=370)
        sb = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        cfg = ttk.Frame(canvas, padding=6)

        cfg.bind("<Configure>",
                 lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._cfg_window = canvas.create_window((0, 0), window=cfg, anchor="nw")

        def _on_canvas_resize(event):
            canvas.itemconfig(self._cfg_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_mw(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mw))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        row = 0

        ttk.Label(cfg, text="Paths", style="Header.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)); row += 1

        ttk.Label(cfg, text="Base Model:").grid(row=row, column=0, sticky="w")
        self._var_base_model = tk.StringVar(value=self._DEFAULTS["base_model"])
        ttk.Entry(cfg, textvariable=self._var_base_model, width=38).grid(
            row=row, column=1, sticky="ew", padx=4)
        ttk.Button(cfg, text="\u2026", width=3,
                   command=lambda: self._browse_dir(self._var_base_model)).grid(
            row=row, column=2)
        row += 1

        ttk.Label(cfg, text="Output Dir:").grid(row=row, column=0, sticky="w")
        self._var_output_dir = tk.StringVar(value=self._DEFAULTS["output_dir"])
        e_out = ttk.Entry(cfg, textvariable=self._var_output_dir, width=38)
        e_out.grid(row=row, column=1, sticky="ew", padx=4)
        e_out.configure(state="readonly")
        ttk.Label(cfg, text="(trainer)", foreground="gray").grid(
            row=row, column=2, sticky="w")
        row += 1

        ttk.Label(cfg, text="Dataset (.parquet):").grid(row=row, column=0, sticky="w")
        self._var_dataset = tk.StringVar()
        ttk.Entry(cfg, textvariable=self._var_dataset, width=38).grid(
            row=row, column=1, sticky="ew", padx=4)
        ttk.Button(cfg, text="\u2026", width=3,
                   command=self._browse_dataset).grid(row=row, column=2)
        row += 1

        ttk.Label(cfg, text="HF Dataset ID:").grid(row=row, column=0, sticky="w")
        self._var_hf_dataset = tk.StringVar()
        ttk.Entry(cfg, textvariable=self._var_hf_dataset, width=38).grid(
            row=row, column=1, sticky="ew", padx=4)
        ttk.Label(cfg, text="(optional)", foreground="gray").grid(
            row=row, column=2, sticky="w")
        row += 1

        ttk.Separator(cfg, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8); row += 1

        ttk.Label(cfg, text="Training", style="Header.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)); row += 1

        ttk.Label(cfg, text="Epochs:").grid(row=row, column=0, sticky="w")
        self._var_epoch = tk.StringVar(value=self._DEFAULTS["epoch"])
        ttk.Spinbox(cfg, textvariable=self._var_epoch,
                    from_=1, to=20, width=8).grid(
            row=row, column=1, sticky="w", padx=4)
        row += 1

        ttk.Label(cfg, text="Fixed Steps:").grid(row=row, column=0, sticky="w")
        self._var_steps = tk.StringVar(value=self._DEFAULTS["steps"])
        ttk.Entry(cfg, textvariable=self._var_steps, width=10).grid(
            row=row, column=1, sticky="w", padx=4)
        ttk.Label(cfg, text="(blank = epochs)", foreground="gray").grid(
            row=row, column=2, sticky="w")
        row += 1

        ttk.Separator(cfg, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8); row += 1

        ttk.Label(cfg, text="Flags", style="Header.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)); row += 1

        self._var_force = tk.BooleanVar()
        ttk.Checkbutton(cfg, text="--force  (disable early stopping)",
                        variable=self._var_force).grid(
            row=row, column=0, columnspan=3, sticky="w"); row += 1
        self._var_latex = tk.BooleanVar()
        ttk.Checkbutton(cfg, text="--latex  (preserve LaTeX / equations)",
                        variable=self._var_latex).grid(
            row=row, column=0, columnspan=3, sticky="w"); row += 1
        self._var_code = tk.BooleanVar()
        ttk.Checkbutton(cfg, text="--code   (preserve code formatting)",
                        variable=self._var_code).grid(
            row=row, column=0, columnspan=3, sticky="w"); row += 1
        self._var_image = tk.BooleanVar()
        ttk.Checkbutton(cfg, text="--image  (multimodal)",
                        variable=self._var_image).grid(
            row=row, column=0, columnspan=3, sticky="w"); row += 1

        ttk.Separator(cfg, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8); row += 1

        ttk.Label(cfg, text="Command Preview", style="Header.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)); row += 1
        self._cmd_preview = tk.Text(
            cfg, height=4, width=44, font=("Consolas", 9),
            state="disabled", bg="#f4f4f4", relief="groove", bd=1)
        self._cmd_preview.grid(
            row=row, column=0, columnspan=3, sticky="ew", padx=2); row += 1

        for v in (self._var_base_model, self._var_dataset,
                  self._var_hf_dataset, self._var_epoch, self._var_steps):
            v.trace_add("write", lambda *_: self._update_cmd_preview())
        for v in (self._var_force, self._var_latex, self._var_code, self._var_image):
            v.trace_add("write", lambda *_: self._update_cmd_preview())

        cfg.columnconfigure(1, weight=1)

        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=2)

        ttk.Label(right, text="Live Loss", style="Header.TLabel").pack(anchor="w")

        self._live_fig = Figure(figsize=(6, 4), dpi=96)
        self._live_ax  = self._live_fig.add_subplot(111)
        self._live_ax.set_xlabel("Step")
        self._live_ax.set_ylabel("Loss")
        self._live_ax.set_title("Waiting for training data\u2026")
        self._live_fig.tight_layout()

        self._live_canvas = FigureCanvasTkAgg(self._live_fig, master=right)
        self._live_canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = ttk.Frame(right)
        tb_frame.pack(fill="x")
        NavigationToolbar2Tk(self._live_canvas, tb_frame)

        chart_btn = ttk.Frame(right)
        chart_btn.pack(fill="x", pady=(4, 0))
        ttk.Button(chart_btn, text="Full Analysis",
                   command=self._full_analysis).pack(side="left", padx=4)
        ttk.Button(chart_btn, text="Load JSONL / State",
                   command=self._load_external_logs).pack(side="left", padx=4)
        ttk.Button(chart_btn, text="Clear",
                   command=self._clear_live).pack(side="left", padx=4)

    def _browse_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _browse_dataset(self):
        p = filedialog.askopenfilename(
            filetypes=[("Parquet files", "*.parquet"), ("All", "*.*")])
        if p:
            self._var_dataset.set(p)

    def _build_cmd(self) -> list[str]:
        cmd = [sys.executable, "-m", "reforge"]
        base = self._var_base_model.get().strip()
        if base:
            cmd += ["--base_model", base]
        hf = self._var_hf_dataset.get().strip()
        ds = self._var_dataset.get().strip()
        dataset_path = hf or ds
        if dataset_path:
            cmd += ["--hf_dataset", dataset_path]
        epoch = self._var_epoch.get().strip()
        if epoch:
            cmd += ["--epoch", epoch]
        steps = self._var_steps.get().strip()
        if steps:
            cmd += ["--steps", steps]
        if self._var_force.get():  cmd.append("--force")
        if self._var_latex.get():  cmd.append("--latex")
        if self._var_code.get():   cmd.append("--code")
        if self._var_image.get():  cmd.append("--image")
        return cmd

    def _update_cmd_preview(self):
        cmd = self._build_cmd()
        display = cmd[:]  # Show all including python -m reforge
        grouped = []
        i = 0
        while i < len(display):
            tok = display[i]
            if (tok.startswith("--") and i + 1 < len(display)
                    and not display[i + 1].startswith("--")):
                grouped.append(f"{tok} {display[i+1]}")
                i += 2
            else:
                grouped.append(tok)
                i += 1
        text = " \\\n    ".join(grouped)
        self._cmd_preview.configure(state="normal")
        self._cmd_preview.delete("1.0", "end")
        self._cmd_preview.insert("1.0", text)
        self._cmd_preview.configure(state="disabled")

    def _start_training(self):
        if self._running:
            messagebox.showwarning("Busy", "A training run is already active.")
            return

        ds = self._var_dataset.get().strip()
        hf = self._var_hf_dataset.get().strip()
        if not ds and not hf:
            messagebox.showwarning(
                "No dataset",
                "Select a .parquet file or enter a HuggingFace dataset ID.")
            return

        dataset_path = hf or ds
        is_local = os.path.exists(dataset_path)
        if is_local and not dataset_path.lower().endswith(".parquet"):
            messagebox.showwarning(
                "Invalid file",
                "Local dataset must be a .parquet file.")
            return
        if not is_local and not hf:
            messagebox.showwarning(
                "File not found",
                f"Dataset path does not exist:\n{dataset_path}")
            return

        cmd = self._build_cmd()

        self._live_logs.clear()
        while not self._line_queue.empty():
            try:
                self._line_queue.get_nowait()
            except queue.Empty:
                break
        self._live_ax.clear()
        self._live_ax.set_title("Starting\u2026")
        self._live_canvas.draw_idle()

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"

            kw = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            if platform.system() == "Windows":
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._proc = subprocess.Popen(cmd, **kw)
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))
            return

        self._running = True
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._lbl_status.configure(text="Training\u2026")

        self._step_count = 0
        self._loss_header_printed = False
        self._stdout_thread = threading.Thread(
            target=self._reader_stdout, args=(self._proc.stdout,), daemon=True)
        self._stderr_thread = threading.Thread(
            target=self._reader_stderr, args=(self._proc.stderr,), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    _SUPPRESS_PATTERNS = (
        "FutureWarning: `tokenizer` is deprecated",
        "UserWarning: The PEFT config's `base_model_name_or_path` was renamed",
        "Use `processing_class` instead.",
    )

    def _write_line(self, dest, line: str):
        try:
            dest.write(line)
            dest.flush()
        except UnicodeEncodeError:
            dest.write(line.encode("ascii", errors="replace").decode("ascii"))
            dest.flush()
        except Exception:
            pass

    def _reader_stdout(self, pipe):
        try:
            for raw in iter(pipe.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                if any(p in line for p in self._SUPPRESS_PATTERNS):
                    continue
                entry = try_parse_loss_line(line)
                if entry:
                    self._step_count += 1
                    self._line_queue.put(entry)
                    if not self._loss_header_printed:
                        self._loss_header_printed = True
                        hdr = (
                            f"\n  {'step':<8}  "
                            f"{'loss':<10}  "
                            f"{'lr':<12}  "
                            f"{'grad_norm':<10}  "
                            f"{'epoch'}\n"
                            f"  {'-'*58}\n"
                        )
                        self._write_line(sys.stdout, hdr)
                    loss = entry.get("loss", 0)
                    lr   = entry.get("learning_rate", 0)
                    ep   = entry.get("epoch", 0)
                    gnorm = entry.get("grad_norm", 0)
                    step  = entry.get("step", self._step_count)
                    if lr > 0:
                        lr_str = f"{lr:.2e}"
                    else:
                        lr_str = "0"
                    formatted = (
                        f"  step {step:<6}  "
                        f"loss {loss:<8.4f}  "
                        f"lr {lr_str:<10}  "
                        f"gnorm {gnorm:<8.3f}  "
                        f"ep {ep:.2f}\n"
                    )
                    self._write_line(sys.stdout, formatted)
                else:
                    self._write_line(sys.stdout, line)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _reader_stderr(self, pipe):
        try:
            while True:
                chunk = pipe.read(256)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                if any(p in text for p in self._SUPPRESS_PATTERNS):
                    continue
                self._write_line(sys.stderr, text)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _stop_training(self):
        if not self._proc:
            return
        try:
            if platform.system() == "Windows":
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._proc.send_signal(signal.SIGINT)
        except Exception as e:
            messagebox.showwarning("Stop", f"Could not send interrupt: {e}")
        self._lbl_status.configure(text="Stopping (saving)\u2026")

    def _tick(self):
        dirty = False
        while True:
            try:
                entry = self._line_queue.get_nowait()
                self._live_logs.append(entry)
                dirty = True
            except queue.Empty:
                break

        if dirty:
            self._refresh_live_chart()

        if self._proc is not None:
            retcode = self._proc.poll()
            if retcode is not None:
                self._running = False
                self._btn_start.configure(state="normal")
                self._btn_stop.configure(state="disabled")
                self._lbl_status.configure(
                    text="Done" if retcode == 0 else f"Exited ({retcode})")
                self._proc = None
                self._refresh_live_chart()

        self.root.after(250, self._tick)

    def _refresh_live_chart(self):
        if not self._live_logs:
            return

        logs = self._live_logs
        steps = [e.get("step", i + 1) for i, e in enumerate(logs)]
        y = np.array([e["loss"] for e in logs])
        x = np.array(steps)

        ax = self._live_ax
        ax.clear()

        ax.plot(x, y, alpha=0.25, color="#1f77b4", linewidth=0.8)

        if len(y) >= 5:
            k = min(max(5, len(y) // 20), 51)
            ax.plot(x, smooth_series(y, k), color="#1f77b4", linewidth=1.5)

        bi = int(np.argmin(y))
        ax.scatter([x[bi]], [y[bi]], color="green", s=30, zorder=5)

        cur  = f"{y[-1]:.4f}"
        best = f"{np.min(y):.4f}"
        step = steps[-1]
        ax.set_title(
            f"Step {step}  |  current {cur}  |  best {best}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        self._live_fig.tight_layout()
        self._live_canvas.draw_idle()

    def _clear_live(self):
        self._live_logs.clear()
        self._loss_header_printed = False
        self._step_count = 0
        self._live_ax.clear()
        self._live_ax.set_title("Cleared")
        self._live_ax.set_xlabel("Step")
        self._live_ax.set_ylabel("Loss")
        self._live_canvas.draw_idle()

    def _load_external_logs(self):
        p = filedialog.askopenfilename(
            filetypes=[("JSONL", "*.jsonl"),
                       ("JSON", "*.json"),
                       ("All", "*.*")])
        if not p:
            return
        logs = parse_trainer_state(p)
        if not logs:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    logs = parse_loss_input(f.read())
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
        if not logs:
            messagebox.showwarning("Empty", "No loss entries found.")
            return
        self._live_logs = logs
        self._refresh_live_chart()

    def _full_analysis(self):
        logs = self._live_logs
        if len(logs) < 3:
            messagebox.showwarning("Not enough data",
                                    "Need at least 3 loss entries for analysis.")
            return

        steps  = [e.get("step", i + 1) for i, e in enumerate(logs)]
        losses = [e["loss"] for e in logs]
        epochs = [e.get("epoch", 0.0) for e in logs]

        last_ep = float(epochs[-1]) if epochs else 0.0
        exposure = last_ep - int(last_ep) if last_ep >= 1.0 else last_ep
        exposure_pct = round(exposure * 100.0, 2)

        x = np.array(steps, dtype=float)
        y = np.array(losses, dtype=float)
        sk = 7 if len(y) >= 7 else max(3, (len(y) // 3) * 2 + 1)
        smoothed = smooth_series(y, sk)

        y_mean = float(np.mean(y));  y_med = float(np.median(y))
        y_std  = float(np.std(y))
        y_max  = float(np.max(y))

        drop_abs = float(y_max - float(np.min(y)))
        drop_pct = 100.0 * drop_abs / max(y_max, 1e-8)

        gc       = np.polyfit(x, smoothed, 1)
        gslope_n = float(gc[0]) * 100.0

        win = min(max(10, len(y) // 10), 200)
        xw  = x[-win:]; yw = smoothed[-win:]
        w   = np.linspace(0.5, 1.0, num=len(xw))
        rc  = np.polyfit(xw, yw, 1, w=w)
        rslope_n = float(rc[0]) * 100.0
        r_med    = float(np.median(yw))
        r_mad    = float(1.4826 * np.median(np.abs(yw - r_med)))
        r_noise  = r_mad if r_mad > 1e-8 else float(np.std(yw))
        r_snr    = float(abs(rc[0]) / (r_noise + 1e-8))

        best_i  = int(np.argmin(y))
        best_v  = float(y[best_i])
        since_b = len(y) - (best_i + 1)

        if rc[0] < -1e-6:
            to_t = max(0, int((y[-1] - (best_v + 0.05)) / -rc[0]))
            pred = f"step {steps[-1] + to_t} (~{to_t} more)"
        else:
            pred = "\u2014"

        grade = calculate_grade(rslope_n, r_snr, exposure_pct)

        fig, ax = plt.subplots(figsize=(12, 4))

        ll, = ax.plot(x, y, alpha=0.25)
        sl, = ax.plot(x, smoothed)
        ax.scatter([x[best_i]], [best_v], color="green", s=36, zorder=3)
        ax.axhspan(best_v, best_v + 0.05, color="green", alpha=0.08)

        outlier = False
        if len(yw) > 1:
            sd = float(np.std(yw))
            if sd > 0:
                for j in range(win):
                    idx = len(y) - win + j
                    if 0 < idx < len(smoothed):
                        if abs(y[idx] - smoothed[idx]) > sd * 1.5:
                            sx = float(x[idx])
                            ax.axvspan(sx - 0.5, sx + 0.5,
                                       color="red", alpha=0.05)
                            outlier = True

        epoch_lines = False
        for j in range(1, len(epochs)):
            if int(epochs[j-1] * 100) != int(epochs[j] * 100):
                ax.axvline(x=float(x[j]), color="gray", ls="--", alpha=0.25)
                epoch_lines = True

        ax.set_title("Loss Analysis \u2014 Best, trend, ideal zone")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        summary = (
            f"avg {y_mean:.4f} | med {y_med:.4f} | std {y_std:.4f}\n"
            f"best {best_v:.4f} @ step {steps[best_i]} | since best {since_b} | "
            f"grade {grade}\n"
            f"global slope {gslope_n:+.2f}/100 | recent {rslope_n:+.2f}/100 | "
            f"SNR {r_snr:.2f}\n"
            f"trend {trend_label(gslope_n)}/{trend_label(rslope_n)} | "
            f"var {drop_pct:.1f}% | exposure {exposure_pct:.1f}%\n"
            f"est. ideal zone: {pred}"
        )
        ax.text(0.99, 0.99, summary, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white",
                          alpha=0.7, edgecolor="none"))

        handles = [
            Line2D([0], [0], color=ll.get_color(), alpha=0.25, label="Loss (raw)"),
            Line2D([0], [0], color=sl.get_color(), label=f"Smoothed (k={sk})"),
            Line2D([0], [0], marker="o", color="green", ls="None", ms=6,
                   label="Best"),
            Patch(fc="green", alpha=0.08, label="Ideal zone"),
        ]
        if outlier:
            handles.append(Patch(fc="red", alpha=0.05, label="Outlier"))
        if epoch_lines:
            handles.append(Line2D([0], [0], color="gray", ls="--", alpha=0.5,
                                  label="Epoch boundary"))

        nc = 2 if len(handles) > 5 else 1
        ft = 0.18 if nc == 2 else 0.14
        fig.tight_layout(rect=[0, ft, 1, 1])
        fig.legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(0.01, 0.01),
                   framealpha=0.95, fancybox=True, borderpad=0.4,
                   handlelength=2.2, handletextpad=0.6, ncol=nc)
        plt.show()

    def _on_close(self):
        if self._running and self._proc:
            if not messagebox.askyesno(
                    "Training active",
                    "Training is still running.\nStop and exit?"):
                return
            self._stop_training()
            for pipe in (self._proc.stdout, self._proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ReforgeDashboard().run()
