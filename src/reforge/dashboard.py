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


# ── Theme (Hermes-inspired dark palette) ─────────────────────────────
BG        = "#101014"   # canvas
SURFACE   = "#1a1a2e"   # cards
SURFACE2  = "#202032"   # input fields
SURFACE3  = "#33333f"   # borders
ROW       = "#333355"   # active rows
ACCENT    = "#FFBF00"   # gold accent
PRIMARY   = "#FFD700"   # bright gold
BORDER_BR = "#CD7F32"   # bronze rule
TEXT      = "#FFF8DC"   # warm ink
MUTED     = "#9096ad"   # dim labels
FAINT     = "#6a6a80"   # hints
OK        = "#4caf50"
GOOD      = "#8FBC8F"
WARN      = "#ffa726"
ERR       = "#ef5350"

FONT      = "Segoe UI"
FONT_MONO = "Consolas"


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


def style_analysis_axes(fig, ax):
    """Apply the dashboard dark theme to a matplotlib axes."""
    fig.patch.set_facecolor("#15151f")
    ax.set_facecolor(BG)
    ax.grid(color="#262638", linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(SURFACE3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


class ReforgeDashboard:
    _DEFAULTS = dict(
        base_model = "",
        output_dir = os.environ.get("REFORGE_OUTPUT_DIR",
                                     os.path.join(os.path.expanduser("~"), "Reforge_Output")),
        epoch      = "1",
        steps      = "",
        batch_size = "",
        grad_accum = "",
    )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reforge Dashboard")
        self.root.geometry("1180x780")
        self.root.minsize(920, 600)
        self.root.configure(bg=BG)

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

    # ── Styling ───────────────────────────────────────────────────────

    def _style_setup(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=BG, foreground=TEXT, bordercolor=SURFACE3)

        style.configure("Header.TLabel", font=(FONT, 11, "bold"),
                        background=BG, foreground=PRIMARY)
        style.configure("Status.TLabel", font=(FONT, 10, "italic"),
                        background=BG, foreground=TEXT)
        style.configure("HeroTitle.TLabel", font=(FONT, 20, "bold"),
                        background=BG, foreground=PRIMARY)
        style.configure("HeroSub.TLabel", font=(FONT, 9),
                        background=BG, foreground=MUTED)

        # Card surfaces (ttk widgets living inside cards must match the card bg)
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("CardTitle.TLabel", background=SURFACE, foreground=ACCENT,
                        font=(FONT, 9, "bold"))
        style.configure("Field.TLabel", background=SURFACE, foreground=TEXT,
                        font=(FONT, 9))
        style.configure("Hint.TLabel", background=SURFACE, foreground=FAINT,
                        font=(FONT, 8))

        style.configure("Field.TEntry", fieldbackground=SURFACE2, background=SURFACE2,
                        foreground=TEXT, bordercolor=SURFACE3,
                        lightcolor=SURFACE3, darkcolor=SURFACE3,
                        insertcolor=PRIMARY, selectbackground=ROW,
                        selectforeground=PRIMARY)
        style.map("Field.TEntry",
                  fieldbackground=[("disabled", SURFACE2)],
                  foreground=[("disabled", FAINT)],
                  bordercolor=[("focus", ACCENT)])

        style.configure("Field.TSpinbox", fieldbackground=SURFACE2, background=SURFACE2,
                        foreground=TEXT, bordercolor=SURFACE3, arrowcolor=PRIMARY,
                        lightcolor=SURFACE3, darkcolor=SURFACE3,
                        insertcolor=PRIMARY, selectbackground=ROW,
                        selectforeground=PRIMARY)
        style.map("Field.TSpinbox",
                  fieldbackground=[("disabled", SURFACE2)],
                  bordercolor=[("focus", ACCENT)])

        style.configure("Flag.TCheckbutton", background=SURFACE, foreground=TEXT,
                        font=(FONT, 9), focuscolor=ACCENT)
        style.map("Flag.TCheckbutton",
                  background=[("active", SURFACE)],
                  foreground=[("disabled", FAINT), ("active", PRIMARY)])

        style.configure("Card.TButton", background=SURFACE, foreground=TEXT,
                        bordercolor=SURFACE3, lightcolor=SURFACE3, darkcolor=SURFACE3,
                        font=(FONT, 9))
        style.map("Card.TButton",
                  background=[("pressed", ROW), ("active", ROW), ("disabled", SURFACE)],
                  foreground=[("disabled", FAINT), ("active", PRIMARY)],
                  bordercolor=[("active", ACCENT)])

        style.configure("Accent.TButton", background=ACCENT, foreground="#101014",
                        font=(FONT, 10, "bold"), borderwidth=0,
                        lightcolor=ACCENT, darkcolor=ACCENT)
        style.map("Accent.TButton",
                  background=[("pressed", "#e0a800"), ("active", PRIMARY),
                              ("disabled", "#3a3a55")],
                  foreground=[("disabled", "#6a6a80")])

        style.configure("Danger.TButton", background=SURFACE, foreground=ERR,
                        bordercolor=SURFACE3, lightcolor=SURFACE3, darkcolor=SURFACE3,
                        font=(FONT, 10, "bold"))
        style.map("Danger.TButton",
                  background=[("pressed", ROW), ("active", ROW), ("disabled", SURFACE)],
                  foreground=[("disabled", FAINT), ("active", ERR)],
                  bordercolor=[("active", ERR)])

        style.configure("TSeparator", background=SURFACE3)

        style.configure("Vertical.TScrollbar", background=SURFACE, troughcolor=BG,
                        bordercolor=BG, arrowcolor=MUTED, lightcolor=SURFACE3,
                        darkcolor=SURFACE3)
        style.map("Vertical.TScrollbar", background=[("active", ROW)])

        style.configure("Horizontal.TScrollbar", background=SURFACE, troughcolor=BG,
                        bordercolor=BG, arrowcolor=MUTED)

        style.configure("TPanedwindow", background=BG)

    # ── Layout helpers ────────────────────────────────────────────────

    def _card(self, parent, title, row, col):
        """A bordered surface panel with a gold title wie in Hermes' chrome."""
        card = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                        highlightbackground=SURFACE3, highlightcolor=ACCENT)
        card.grid(row=row, column=col, columnspan=4, sticky="ew", pady=(0, 8))
        inner = tk.Frame(card, bg=SURFACE, padx=10, pady=8)
        inner.pack(fill="both", expand=True)
        tk.Label(inner, text=title.upper(), bg=SURFACE, fg=ACCENT,
                 font=(FONT, 9, "bold")).grid(row=0, column=0, columnspan=4,
                                              sticky="w", pady=(0, 6))
        return inner

    def _field_row(self, cfg, row, label, var, browse=None, hint=None,
                   readonly=False, width=38):
        tk.Label(cfg, text=label, bg=SURFACE, fg=TEXT,
                 font=(FONT, 9)).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(cfg, textvariable=var, width=width, style="Field.TEntry")
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 4))
        if readonly:
            entry.configure(state="readonly")
        if browse:
            ttk.Button(cfg, text="…", width=3, style="Card.TButton",
                       command=browse).grid(row=row, column=2)
        else:
            tk.Label(cfg, text="", bg=SURFACE).grid(row=row, column=2)
        if hint:
            tk.Label(cfg, text=hint, bg=SURFACE, fg=FAINT,
                     font=(FONT, 8)).grid(row=row, column=3, sticky="w", padx=(0, 4))

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG, padx=16, pady=12)
        header.pack(fill="x")
        ttk.Label(header, text="REFORGE", style="HeroTitle.TLabel").pack(side="left")
        ttk.Label(header, text="   LoRA fine-tuning dashboard",
                  style="HeroSub.TLabel").pack(side="left", pady=(8, 0))

        self._lbl_status = tk.Label(
            header, text="\u25cf  Idle", bg=SURFACE, fg=MUTED,
            font=(FONT, 9, "bold"), padx=10, pady=4,
            highlightthickness=1, highlightbackground=SURFACE3)
        self._lbl_status.pack(side="right")

        tk.Frame(self.root, bg=BORDER_BR, height=2).pack(fill="x")

        main = tk.Frame(self.root, bg=BG, padx=10, pady=8)
        main.pack(fill="both", expand=True)

        paned = ttk.PanedWindow(main, orient="horizontal", style="TPanedwindow")
        paned.pack(fill="both", expand=True)

        left = tk.Frame(paned, bg=BG)
        paned.add(left, weight=1)

        # Bottom action bar (packed first to reserve space)
        btn = tk.Frame(left, bg=BG, pady=8)
        btn.pack(fill="x", side="bottom")
        tk.Frame(left, bg=SURFACE3, height=1).pack(fill="x", side="bottom")

        self._btn_start = ttk.Button(btn, text="  Start Training  ",
                                     style="Accent.TButton",
                                     command=self._start_training)
        self._btn_start.pack(side="left", padx=(2, 6))

        self._btn_stop = ttk.Button(btn, text="  Stop (graceful)  ",
                                    style="Danger.TButton",
                                    command=self._stop_training, state="disabled")
        self._btn_stop.pack(side="left", padx=6)

        self._lbl_substatus = tk.Label(btn, text="", bg=BG, fg=MUTED,
                                       font=(FONT, 9, "italic"))
        self._lbl_substatus.pack(side="left", padx=10)

        # Scrollable config column
        scroll_frame = tk.Frame(left, bg=BG)
        scroll_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_frame, highlightthickness=0, width=380,
                           bg=BG, bd=0)
        sb = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        cfg = tk.Frame(canvas, bg=BG)

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

        # – Paths –
        c = self._card(cfg, "Paths", row, 0); row += 1
        self._var_base_model = tk.StringVar(value=self._DEFAULTS["base_model"])
        self._field_row(c, 1, "Base Model", self._var_base_model,
                        browse=lambda: self._browse_dir(self._var_base_model))
        self._var_output_dir = tk.StringVar(value=self._DEFAULTS["output_dir"])
        self._field_row(c, 2, "Output Dir", self._var_output_dir,
                        hint="(trainer)", readonly=True)
        self._var_dataset = tk.StringVar()
        self._field_row(c, 3, "Dataset (.parquet)", self._var_dataset,
                        browse=self._browse_dataset)
        self._var_hf_dataset = tk.StringVar()
        self._field_row(c, 4, "HF Dataset ID", self._var_hf_dataset,
                        hint="(optional)", width=30)

        # – Training –
        c = self._card(cfg, "Training", row, 0); row += 1
        tk.Label(c, text="Epochs", bg=SURFACE, fg=TEXT,
                 font=(FONT, 9)).grid(row=1, column=0, sticky="w")
        self._var_epoch = tk.StringVar(value=self._DEFAULTS["epoch"])
        ttk.Spinbox(c, textvariable=self._var_epoch, from_=1, to=20, width=8,
                    style="Field.TSpinbox").grid(
            row=1, column=1, sticky="w", padx=(8, 4))
        tk.Label(c, text="", bg=SURFACE).grid(row=1, column=2)
        tk.Label(c, text="", bg=SURFACE).grid(row=1, column=3)

        self._var_steps = tk.StringVar(value=self._DEFAULTS["steps"])
        self._field_row(c, 2, "Fixed Steps", self._var_steps,
                        hint="(blank = epochs)", width=10)

        # – Performance (batch size + gradient accumulation) –
        c = self._card(cfg, "Performance", row, 0); row += 1
        tk.Label(c, text="Batch Size", bg=SURFACE, fg=TEXT,
                 font=(FONT, 9)).grid(row=1, column=0, sticky="w")
        self._var_batch = tk.StringVar(value=self._DEFAULTS["batch_size"])
        ttk.Spinbox(c, textvariable=self._var_batch, from_=1, to=64, width=8,
                    style="Field.TSpinbox").grid(
            row=1, column=1, sticky="w", padx=(8, 4))
        tk.Label(c, text="", bg=SURFACE).grid(row=1, column=2)
        tk.Label(c, text="(blank = 1)",
                 bg=SURFACE, fg=FAINT, font=(FONT, 8)).grid(
            row=1, column=3, sticky="w", padx=(0, 4))

        tk.Label(c, text="Grad. Accumulation", bg=SURFACE, fg=TEXT,
                 font=(FONT, 9)).grid(row=2, column=0, sticky="w")
        self._var_grad = tk.StringVar(value=self._DEFAULTS["grad_accum"])
        ttk.Spinbox(c, textvariable=self._var_grad, from_=1, to=64, width=8,
                    style="Field.TSpinbox").grid(
            row=2, column=1, sticky="w", padx=(8, 4))
        tk.Label(c, text="", bg=SURFACE).grid(row=2, column=2)
        tk.Label(c, text="(blank = profile)",
                 bg=SURFACE, fg=FAINT, font=(FONT, 8)).grid(
            row=2, column=3, sticky="w", padx=(0, 4))

        # – Flags –
        c = self._card(cfg, "Flags", row, 0); row += 1
        self._var_force = tk.BooleanVar()
        ttk.Checkbutton(c, text="--force   (disable early stopping)",
                        variable=self._var_force,
                        style="Flag.TCheckbutton").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=2)
        self._var_latex = tk.BooleanVar()
        ttk.Checkbutton(c, text="--latex   (preserve LaTeX / equations)",
                        variable=self._var_latex,
                        style="Flag.TCheckbutton").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=2)
        self._var_code = tk.BooleanVar()
        ttk.Checkbutton(c, text="--code    (preserve code formatting)",
                        variable=self._var_code,
                        style="Flag.TCheckbutton").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=2)
        self._var_image = tk.BooleanVar()
        ttk.Checkbutton(c, text="--image   (multimodal)",
                        variable=self._var_image,
                        style="Flag.TCheckbutton").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=2)

        # – Command preview –
        c = self._card(cfg, "Command Preview", row, 0); row += 1
        self._cmd_preview = tk.Text(
            c, height=5, width=44, font=(FONT_MONO, 9),
            state="disabled", bg="#14141c", fg="#cfd3ea",
            relief="flat", bd=1, highlightthickness=1,
            highlightbackground=SURFACE3, highlightcolor=ACCENT,
            insertbackground=PRIMARY, wrap="word")
        self._cmd_preview.grid(
            row=1, column=0, columnspan=4, sticky="ew")

        cfg.columnconfigure(0, weight=0)
        cfg.columnconfigure(1, weight=1)

        for v in (self._var_base_model, self._var_dataset, self._var_hf_dataset,
                  self._var_epoch, self._var_steps, self._var_batch, self._var_grad):
            v.trace_add("write", lambda *_: self._update_cmd_preview())
        for v in (self._var_force, self._var_latex, self._var_code, self._var_image):
            v.trace_add("write", lambda *_: self._update_cmd_preview())

        # – Right: live chart –
        right = tk.Frame(paned, bg=BG)
        paned.add(right, weight=2)

        top = tk.Frame(right, bg=BG)
        top.pack(fill="x", pady=(0, 4))
        tk.Label(top, text="LIVE LOSS", bg=BG, fg=PRIMARY,
                 font=(FONT, 11, "bold")).pack(side="left")

        self._stat_steps = self._stat(top, "steps", 0)
        self._stat_current = self._stat(top, "loss", "\u2014")
        self._stat_best = self._stat(top, "best", "\u2014")

        self._live_fig = Figure(figsize=(6, 4), dpi=96)
        self._live_fig.patch.set_facecolor("#15151f")
        self._live_ax  = self._live_fig.add_subplot(111)
        self._reset_live_axes(title="Waiting for training data…")

        self._live_canvas = FigureCanvasTkAgg(self._live_fig, master=right)
        self._live_canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(right, bg=SURFACE, height=28)
        tb_frame.pack(fill="x")
        tb = NavigationToolbar2Tk(self._live_canvas, tb_frame)
        for w in tb_frame.winfo_children():
            try:
                w.configure(bg=SURFACE)
                for child in w.winfo_children():
                    try:
                        child.configure(bg=SURFACE, foreground=MUTED)
                    except Exception:
                        pass
            except Exception:
                pass

        chart_btn = tk.Frame(right, bg=BG)
        chart_btn.pack(fill="x", pady=(6, 0))
        ttk.Button(chart_btn, text="Full Analysis", style="Card.TButton",
                   command=self._full_analysis).pack(side="left", padx=(2, 6))
        ttk.Button(chart_btn, text="Load JSONL / State", style="Card.TButton",
                   command=self._load_external_logs).pack(side="left", padx=6)
        ttk.Button(chart_btn, text="Clear", style="Card.TButton",
                   command=self._clear_live).pack(side="left", padx=6)

    def _stat(self, parent, label, value):
        box = tk.Label(
            parent, text=f"  {label.upper()}  {value}  ", bg=SURFACE, fg=PRIMARY,
            font=(FONT, 9, "bold"), padx=6, pady=3,
            highlightthickness=1, highlightbackground=SURFACE3)
        box.pack(side="left", padx=(0, 6))
        return box

    # ── Browsers ──────────────────────────────────────────────────────

    def _browse_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _browse_dataset(self):
        p = filedialog.askopenfilename(
            filetypes=[("Parquet files", "*.parquet"), ("All", "*.*")])
        if p:
            self._var_dataset.set(p)

    # ── Command construction ──────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        cmd = [sys.executable, "-m", "reforge"]
        base = self._var_base_model.get().strip()
        if base:
            cmd += ["--base_model", base]
        hf = self._var_hf_dataset.get().strip()
        ds = self._var_dataset.get().strip()
        if ds:
            cmd += [ds]                      # positional dataset (local file)
        if hf:
            cmd += ["--hf_dataset", hf]      # hub ID stays on its own flag
        epoch = self._var_epoch.get().strip()
        if epoch:
            cmd += ["--epoch", epoch]
        steps = self._var_steps.get().strip()
        if steps:
            cmd += ["--steps", steps]
        batch = self._var_batch.get().strip()
        if batch:
            cmd += ["--batch_size", batch]
        grad = self._var_grad.get().strip()
        if grad:
            cmd += ["--grad_accum", grad]
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

    def _parse_int_field(self, var: tk.StringVar, name: str) -> int | None:
        """Blank → None (backend default); else validated int ≥ 1."""
        raw = var.get().strip()
        if not raw:
            return None
        try:
            val = int(raw)
        except ValueError:
            messagebox.showwarning("Invalid value", f"{name} must be an integer (got {raw!r}).")
            raise ValueError(name)
        if val < 1:
            messagebox.showwarning("Invalid value", f"{name} must be \u2265 1 (got {val}).")
            raise ValueError(name)
        return val

    def _set_status(self, text, color=MUTED):
        self._lbl_status.configure(text=f"\u25cf  {text}", fg=color)

    # ── Training control ──────────────────────────────────────────────

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

        dataset_path = ds or hf
        # Local file goes as the positional arg; hub ID via --hf_dataset.
        if ds:
            if not os.path.exists(ds):
                messagebox.showwarning("File not found",
                                       f"Dataset path does not exist:\n{ds}")
                return
            if not ds.lower().endswith(".parquet"):
                messagebox.showwarning("Invalid file",
                                       "Local dataset must be a .parquet file.")
                return

        try:
            epoch = self._parse_int_field(self._var_epoch, "Epochs") or 1
            steps = self._parse_int_field(self._var_steps, "Fixed Steps")
            batch = self._parse_int_field(self._var_batch, "Batch size")
            grad  = self._parse_int_field(self._var_grad, "Grad. accumulation")
        except ValueError:
            return

        cmd = self._build_cmd()

        self._live_logs.clear()
        while not self._line_queue.empty():
            try:
                self._line_queue.get_nowait()
            except queue.Empty:
                break
        self._stat_steps.configure(text="  STEPS  0  ")
        self._live_ax.clear()
        self._reset_live_axes(title="Starting…")
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
        self._set_status("Training…", ACCENT)

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
        self._set_status("Stopping (saving)…", WARN)

    # ── Poll loop ─────────────────────────────────────────────────────

    def _tick(self):
        try:
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
                    self._set_status(
                        "Done" if retcode == 0 else f"Exited ({retcode})",
                        OK if retcode == 0 else ERR)
                    self._proc = None
                    self._refresh_live_chart()
        except Exception:
            pass
        finally:
            # Always reschedule, even if a tick threw — a broken after-chain
            # used to freeze the live chart permanently.
            self.root.after(250, self._tick)

    # ── Live chart ────────────────────────────────────────────────────

    def _reset_live_axes(self, title):
        ax = self._live_ax
        ax.clear()
        style_analysis_axes(self._live_fig, ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontsize=10)

    def _refresh_live_chart(self):
        if not self._live_logs:
            return

        logs = self._live_logs
        steps = [e.get("step", i + 1) for i, e in enumerate(logs)]
        y = np.array([e["loss"] for e in logs])
        x = np.array(steps)

        # Pixel-decimate very long series to keep redraws cheap.
        if len(y) > 4000:
            idx = np.linspace(0, len(y) - 1, 4000).astype(int)
            y = y[idx]
            x = x[idx]

        ax = self._live_ax
        ax.clear()
        style_analysis_axes(self._live_fig, ax)

        ax.plot(x, y, alpha=0.25, color=PRIMARY, linewidth=0.8)

        if len(y) >= 5:
            k = min(max(5, len(y) // 20), 51)
            ax.plot(x, smooth_series(y, k), color=ACCENT, linewidth=1.8)

        bi = int(np.argmin(y))
        ax.scatter([x[bi]], [y[bi]], color=OK, s=30, zorder=5)

        cur  = f"{y[-1]:.4f}"
        best = f"{np.min(y):.4f}"
        step = steps[-1]
        ax.set_title(
            f"Step {step}  |  current {cur}  |  best {best}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        self._live_fig.tight_layout()
        self._live_canvas.draw_idle()

        self._stat_steps.configure(text=f"  STEPS  {step}  ")
        self._stat_current.configure(text=f"  LOSS  {cur}  ")
        self._stat_best.configure(text=f"  BEST  {best}  ")

    def _clear_live(self):
        self._live_logs.clear()
        self._loss_header_printed = False
        self._step_count = 0
        self._reset_live_axes(title="Cleared")
        self._live_canvas.draw_idle()
        self._stat_steps.configure(text="  STEPS  0  ")
        self._stat_current.configure(text="  LOSS  \u2014  ")
        self._stat_best.configure(text="  BEST  \u2014  ")

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

    # ── Full analysis ─────────────────────────────────────────────────

    def _full_analysis(self):
        logs = self._live_logs
        if len(logs) < 3:
            messagebox.showwarning(
                "Not enough data",
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
        style_analysis_axes(fig, ax)

        ll, = ax.plot(x, y, alpha=0.25, color=PRIMARY)
        sl, = ax.plot(x, smoothed, color=ACCENT)
        ax.scatter([x[best_i]], [best_v], color=OK, s=36, zorder=3)
        ax.axhspan(best_v, best_v + 0.05, color=OK, alpha=0.08)

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
                                       color=ERR, alpha=0.05)
                            outlier = True

        epoch_lines = False
        for j in range(1, len(epochs)):
            if int(epochs[j-1] * 100) != int(epochs[j] * 100):
                ax.axvline(x=float(x[j]), color="gray", ls="--", alpha=0.25)
                epoch_lines = True

        ax.set_title("Loss Analysis — Best, trend, ideal zone")
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
                va="top", ha="right", fontsize=8, color=TEXT,
                bbox=dict(boxstyle="round", facecolor=SURFACE,
                          alpha=0.85, edgecolor=SURFACE3))

        handles = [
            Line2D([0], [0], color=ll.get_color(), alpha=0.25, label="Loss (raw)"),
            Line2D([0], [0], color=sl.get_color(), label=f"Smoothed (k={sk})"),
            Line2D([0], [0], marker="o", color=OK, ls="None", ms=6,
                   label="Best"),
            Patch(fc=OK, alpha=0.08, label="Ideal zone"),
        ]
        if outlier:
            handles.append(Patch(fc=ERR, alpha=0.05, label="Outlier"))
        if epoch_lines:
            handles.append(Line2D([0], [0], color="gray", ls="--", alpha=0.5,
                                  label="Epoch boundary"))

        # Keep the window face dark and free the figure once closed.
        fig.canvas.mpl_connect("close_event", lambda e: plt.close(fig))
        if fig.canvas.manager is not None:
            try:
                win = fig.canvas.manager.window
                win.configure(bg=BG)
                win.title("Reforge \u2014 Loss Analysis")
            except Exception:
                pass

        nc = 2 if len(handles) > 5 else 1
        ft = 0.18 if nc == 2 else 0.14
        fig.tight_layout(rect=[0, ft, 1, 1])
        leg = fig.legend(handles=handles, loc="lower left",
                         bbox_to_anchor=(0.01, 0.01),
                         facecolor=SURFACE, edgecolor=SURFACE3,
                         labelcolor=TEXT,
                         framealpha=0.95, fancybox=True, borderpad=0.4,
                         handlelength=2.2, handletextpad=0.6, ncol=nc)
        leg.set_zorder(100)
        plt.show()

    # ── Shutdown ──────────────────────────────────────────────────────

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
