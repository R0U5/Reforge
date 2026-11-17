import json
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
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


def _fmt_hms(seconds) -> str:
    """tqdm-style clock: 03:54, 1h12m05s. Unknown value -> --:--."""
    try:
        sec = int(float(seconds))
    except (TypeError, ValueError):
        return "--:--"
    if sec < 0:
        sec = 0
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}:{s:02d}"


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
        dpo_beta  = "",
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

        # – DPO –
        c = self._card(cfg, "DPO", row, 0); row += 1
        self._var_dpo = tk.BooleanVar()
        ttk.Checkbutton(c, text="--dpo     (force DPO training)",
                        variable=self._var_dpo,
                        style="Flag.TCheckbutton").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=2)
        self._var_dpo_beta = tk.StringVar(value=self._DEFAULTS["dpo_beta"])
        self._field_row(c, 1, "DPO Beta", self._var_dpo_beta,
                        hint="(blank = 0.1)", width=10)
        tk.Label(c, text="SFT first, then DPO: after SFT finishes, merged/ "
                         "becomes the DPO base model automatically.",
                 bg=SURFACE, fg=FAINT, font=(FONT, 8), wraplength=340,
                 justify="left").grid(row=2, column=0, columnspan=4,
                                      sticky="w", pady=(4, 0))

        # – Batch –
        c = self._card(cfg, "Batch", row, 0); row += 1
        self._var_batch_mode = tk.BooleanVar()
        ttk.Checkbutton(c, text="--batch    (sequential stages, chained merges)",
                        variable=self._var_batch_mode, style="Flag.TCheckbutton",
                        command=self._update_cmd_preview).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=2)
        self._batch_stages: list[dict] = []
        self._batch_manifest_path: str | None = None
        lst = tk.Frame(c, bg=SURFACE)
        lst.grid(row=1, column=0, columnspan=4, sticky="ew")
        self._batch_list = tk.Listbox(
            lst, height=4, font=(FONT_MONO, 8), bg="#14141c", fg="#cfd3ea",
            relief="flat", bd=1, highlightthickness=1,
            highlightbackground=SURFACE3, highlightcolor=ACCENT,
            selectbackground=ROW, selectforeground=PRIMARY, activestyle="none")
        self._batch_list.pack(side="left", fill="x", expand=True)
        _bsb = ttk.Scrollbar(lst, orient="vertical",
                             command=self._batch_list.yview)
        _bsb.pack(side="right", fill="y")
        self._batch_list.configure(yscrollcommand=_bsb.set)

        _bb = tk.Frame(c, bg=SURFACE)
        _bb.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        for text, cb in (("Add", self._batch_add), ("Edit", self._batch_edit),
                         ("Remove", self._batch_remove), ("\u2191", lambda: self._batch_move(-1)),
                         ("\u2193", lambda: self._batch_move(1))):
            ttk.Button(_bb, text=text, width=7 if len(text) > 2 else 4,
                       style="Card.TButton", command=cb).pack(side="left", padx=2)

        _bm = tk.Frame(c, bg=SURFACE)
        _bm.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Button(_bm, text="Save Manifest", width=13, style="Card.TButton",
                   command=self._batch_save).pack(side="left", padx=2)
        ttk.Button(_bm, text="Load Manifest", width=13, style="Card.TButton",
                   command=self._batch_load).pack(side="left", padx=2)
        tk.Label(c, text="Stages run in order; each trains on the previous "
                         "stage's merge (SFT \u2192 ... \u2192 DPO). Batch requires "
                         "a Base Model; the final model is the last stage's "
                         "merged/.",
                 bg=SURFACE, fg=FAINT, font=(FONT, 8), wraplength=340,
                 justify="left").grid(row=4, column=0, columnspan=4,
                                      sticky="w", pady=(4, 0))

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
        for v in (self._var_force, self._var_latex, self._var_code, self._var_image,
                  self._var_dpo, self._var_dpo_beta):
            v.trace_add("write", lambda *_: self._update_cmd_preview())
        self._var_batch_mode.trace_add("write", lambda *_: self._update_cmd_preview())

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
        self._stat_eta = self._stat(top, "eta", "\u2014")

        # Progress card (tqdm/Docker-inspired): phase tracker + bar + ETA line
        self._build_progress_card(right)

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

    # ── Progress panel (tqdm/Docker-inspired) ─────────────────────────

    # Pipeline phases shown as Docker-pull-style segments. Driven by the
    # child's --agent NDJSON event stream (see agent.py for the contract).
    _PHASES = ("dataset", "model", "train", "merge")
    _PHASE_TITLES = {
        "dataset": "DATASET",
        "model":   "MODEL",
        "train":   "TRAIN",
        "merge":   "MERGE",
    }
    _PHASE_WAIT_TEXT = {
        "dataset": "Loading dataset\u2026",
        "model":   "Loading model\u2026",
        "train":   "Training\u2026",
        "merge":   "Merging LoRA\u2026",
    }
    _SPINNER_FRAMES = "|/-\\"

    def _build_progress_card(self, parent):
        card = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                        highlightbackground=SURFACE3, highlightcolor=ACCENT)
        card.pack(fill="x", pady=(0, 6))
        inner = tk.Frame(card, bg=SURFACE, padx=10, pady=8)
        inner.pack(fill="both", expand=True)

        # Phase tracker row: DATASET → MODEL → TRAIN → MERGE (+ stage chip)
        row = tk.Frame(inner, bg=SURFACE)
        row.pack(fill="x")
        self._lbl_stage_chip = tk.Label(
            row, text="", bg=SURFACE, fg=ACCENT, font=(FONT, 8, "bold"))
        self._lbl_stage_chip.pack(side="right")
        self._phase_labels = {}
        for i, phase in enumerate(self._PHASES):
            if i:
                tk.Label(row, text="\u2192", bg=SURFACE, fg=FAINT,
                         font=(FONT, 8)).pack(side="left", padx=3)
            lbl = tk.Label(row, text=self._PHASE_TITLES[phase], bg=SURFACE2,
                           fg=MUTED, font=(FONT, 8, "bold"), padx=8, pady=2)
            lbl.pack(side="left")
            self._phase_labels[phase] = lbl

        # Progress bar (custom canvas; tqdm-style text is drawn inside)
        self._prog_canvas = tk.Canvas(inner, height=26, bg=SURFACE,
                                      highlightthickness=0)
        self._prog_canvas.pack(fill="x", pady=(8, 5))
        self._prog_canvas.bind("<Configure>",
                               lambda e: self._render_progress())

        # tqdm-style info line: left = step counts, right = [elapsed<ETA, it/s]
        info = tk.Frame(inner, bg=SURFACE)
        info.pack(fill="x")
        self._lbl_prog_left = tk.Label(info, text="waiting for run\u2026",
                                       bg=SURFACE, fg=MUTED,
                                       font=(FONT_MONO, 9))
        self._lbl_prog_left.pack(side="left")
        self._lbl_prog_right = tk.Label(info, text="", bg=SURFACE,
                                        fg=PRIMARY, font=(FONT_MONO, 9))
        self._lbl_prog_right.pack(side="right")

        # Live progress state, updated from agent NDJSON events.
        self._prog = dict(
            phase=None,         # active phase name, or None when idle/done
            failed=False,       # last terminal event was a failure
            done=False,         # run reached a terminal event
            step=0, total=0, pct=None, eta=None,
            t_train=None,       # monotonic-ish wall clock of first step
            step0=0, rate=None, # it/s estimate from first step to now
            sweep=0.0, spinner=0,
        )
        self._reset_progress()

    def _reset_progress(self, keep_chip=False):
        p = self._prog
        p.update(phase=None, failed=False, done=False,
                 step=0, total=0, pct=None, eta=None,
                 t_train=None, step0=0, rate=None,
                 sweep=0.0, spinner=0)
        for lbl in self._phase_labels.values():
            lbl.configure(text=lbl.cget("text").split("  ", 1)[-1])
            self._style_phase(lbl, "pending")
        if not keep_chip:
            self._lbl_stage_chip.configure(text="")
        self._lbl_prog_left.configure(text="waiting for run\u2026")
        self._lbl_prog_right.configure(text="")
        self._stat_eta.configure(text="  ETA  \u2014  ")
        self._render_progress()

    def _style_phase(self, lbl, state):
        styles = {
            "pending": (SURFACE2, MUTED),
            "active":  (ROW, ACCENT),
            "done":    (SURFACE2, OK),
            "failed":  (ROW, ERR),
        }
        bg, fg = styles.get(state, styles["pending"])
        lbl.configure(bg=bg, fg=fg)

    def _set_progress_phase(self, name):
        """Activate `name`; earlier phases become done, later ones pending."""
        p = self._prog
        p["phase"] = name
        p["failed"] = False
        p["done"] = False
        idx = self._PHASES.index(name)
        for i, phase in enumerate(self._PHASES):
            lbl = self._phase_labels[phase]
            lbl.configure(text=self._PHASE_TITLES[phase])
            self._style_phase(
                lbl, "done" if i < idx else "active" if i == idx else "pending")
        self._update_prog_labels()
        self._render_progress()

    def _mark_phase(self, name, state):
        lbl = self._phase_labels.get(name)
        if lbl:
            lbl.configure(text=self._PHASE_TITLES[name])
            self._style_phase(lbl, state)

    def _update_prog_labels(self):
        p = self._prog
        phase = p["phase"]
        if phase == "train" and p["total"]:
            now = time.time()
            elapsed = now - p["t_train"] if p["t_train"] else 0.0
            if p["rate"] is None and elapsed > 0 and p["step"] > p["step0"]:
                p["rate"] = (p["step"] - p["step0"]) / elapsed
            eta = p["eta"]
            if eta is None and p["rate"] and p["rate"] > 0:
                eta = (p["total"] - p["step"]) / p["rate"]
            shown_pct = p["pct"] if p["pct"] is not None else 0.0
            left = f"step {p['step']}/{p['total']}"
            right = (f"{shown_pct:.1f}%  \u00b7  [{_fmt_hms(elapsed)}"
                     f"<{_fmt_hms(eta)}"
                     f" \u00b7 {p['rate']:.2f} it/s]" if p["rate"] is not None
                     else f"{shown_pct:.1f}%  \u00b7  [{_fmt_hms(elapsed)}<{_fmt_hms(eta)}]")
            self._lbl_prog_left.configure(text=left)
            self._lbl_prog_right.configure(text=right)
            self._stat_eta.configure(
                text=f"  ETA  {_fmt_hms(eta)}  " if eta is not None else "  ETA  \u2014  ")
        elif p["done"]:
            pass  # terminal text is set by the event handler
        else:
            self._lbl_prog_left.configure(
                text=self._PHASE_WAIT_TEXT.get(phase, "waiting for run\u2026"))
            self._lbl_prog_right.configure(text="")

    def _render_progress(self):
        canvas = self._prog_canvas
        try:
            w = int(canvas.winfo_width())
        except Exception:
            w = 0
        h = 26
        canvas.delete("all")
        if w < 20:
            return
        pad = 2
        track_w = w - 2 * pad
        p = self._prog

        canvas.create_rectangle(pad, pad, w - pad, h - pad,
                                fill=SURFACE2, outline=SURFACE3)

        # Determinate train bar; indeterminate sweep everywhere else
        # (dataset/model load and the merge take minutes with no % to show,
        # so they get Docker-pull-style motion instead of a frozen bar).
        indeterminate = (p["phase"] is not None and not p["done"]
                         and not (p["phase"] == "train" and p["total"]))
        if indeterminate:
            chunk = max(48, track_w // 6)
            travel = track_w + chunk
            x_off = int(p["sweep"] * travel) - chunk
            canvas.create_rectangle(pad + x_off, pad, pad + x_off + chunk,
                                    h - pad, fill=PRIMARY, width=0)
        elif p["phase"] is not None or p["done"]:
            pct = p["pct"]
            if pct is None and p["total"]:
                pct = 100.0 * p["step"] / max(p["total"], 1)
            if pct is None:
                pct = 100.0 if p["done"] else 0.0
            pct = max(0.0, min(100.0, pct))
            fill_color = ERR if p["failed"] else ACCENT
            canvas.create_rectangle(pad, pad, pad + track_w * pct / 100.0,
                                    h - pad, fill=fill_color, width=0)
            # 10% tick marks over the filled region (Docker-style segments)
            for i in range(1, 10):
                x = pad + track_w * i / 10.0
                canvas.create_line(x, pad + 1, x, h - pad - 1,
                                   fill=SURFACE, width=1)
            if p["done"] or p["pct"] is not None:
                label = "100%" if pct >= 100 else f"{pct:.1f}%"
                # shadow pass keeps the caption readable on both gold and dark
                canvas.create_text(w / 2 + 1, h / 2 + 1, text=label,
                                   fill="#101014", font=(FONT_MONO, 9, "bold"))
                canvas.create_text(w / 2, h / 2, text=label,
                                   fill=TEXT, font=(FONT_MONO, 9, "bold"))

    # Event handlers -----------------------------------------------------

    def _apply_agent_event(self, ev: dict):
        """Route one NDJSON event from the child into the progress panel."""
        event = ev.get("event")
        p = self._prog

        status_map = {
            "run_started":       ("Starting\u2026", WARN),
            "dataset_loaded":    ("Preparing\u2026", WARN),
            "training_started":  ("Training\u2026", ACCENT),
            "training_complete": ("Exporting\u2026", WARN),
            "merge_started":     ("Exporting\u2026", WARN),
            "run_complete":      ("Complete", OK),
            "run_interrupted":   ("Interrupted (saving)\u2026", WARN),
        }
        if event in status_map:
            self._set_status(*status_map[event])

        if event == "run_started":
            self._set_progress_phase("dataset")
        elif event == "dataset_loaded":
            self._mark_phase("dataset", "done")
            self._set_progress_phase("model")
        elif event == "training_started":
            self._set_progress_phase("train")
            p["total"] = int(ev.get("total_steps") or 0)
            p["step"] = 0
            p["pct"] = None
            p["eta"] = None
            p["t_train"] = None
            p["rate"] = None
            self._update_prog_labels()
            self._render_progress()
        elif event == "step":
            if p["t_train"] is None:
                p["t_train"] = time.time()
                p["step0"] = int(ev.get("step") or 0)
            p["step"] = int(ev.get("step") or p["step"])
            if ev.get("total_steps"):
                p["total"] = int(ev["total_steps"])
            p["pct"] = float(ev["pct"]) if ev.get("pct") is not None else (
                100.0 * p["step"] / max(p["total"], 1) if p["total"] else None)
            p["eta"] = ev.get("eta_sec")
            self._update_prog_labels()
            self._render_progress()
        elif event in ("training_complete", "merge_started"):
            self._mark_phase("train", "done")
            self._set_progress_phase("merge")
        elif event == "merge_complete":
            self._mark_phase("merge", "done")
        elif event == "run_complete":
            p["done"] = True
            p["phase"] = None
            p["pct"] = 100.0
            for name in self._PHASES:
                self._mark_phase(name, "done")
            self._lbl_prog_left.configure(text="run complete")
            self._lbl_prog_right.configure(text="")
            self._stat_eta.configure(text="  ETA  \u2014  ")
            self._render_progress()
        elif event == "run_interrupted":
            self._lbl_prog_left.configure(text="interrupted \u2014 saving state\u2026")
        elif event == "run_failed":
            p["failed"] = True
            if p["phase"]:
                self._mark_phase(p["phase"], "failed")
            err = str(ev.get("error") or "run failed")
            self._lbl_prog_left.configure(
                text=("failed: " + err)[:80])
            self._lbl_prog_right.configure(text="")
            self._render_progress()
        elif event == "batch_started":
            total = ev.get("total_stages")
            if total:
                self._prog_stage_total = int(total)
                self._prog_stage_done = 0
                self._lbl_stage_chip.configure(text=f"0/{total} STAGES")
        elif event == "stage_started":
            idx = int(ev.get("index", 0)) + 1
            total = self._stage_count() or idx
            self._lbl_stage_chip.configure(text=f"STAGE {idx}/{total}")
            self._reset_progress(keep_chip=True)
            self._set_progress_phase("dataset")
        elif event == "stage_completed":
            self._bump_stage_count()
        elif event == "batch_complete":
            self._lbl_stage_chip.configure(text="BATCH DONE")
        elif event in ("stage_failed", "batch_failed"):
            p["failed"] = True
            self._lbl_prog_left.configure(
                text=("batch failed: " + str(ev.get("error") or ""))[:80])
            self._render_progress()

    def _stage_count(self):
        return getattr(self, "_prog_stage_total", None)

    def _bump_stage_count(self):
        if self._stage_count():
            self._prog_stage_done = getattr(self, "_prog_stage_done", 0) + 1
            self._lbl_stage_chip.configure(
                text=f"{self._prog_stage_done}/{self._prog_stage_total} STAGES")

    def _apply_agent_event_safe(self, ev: dict):
        try:
            self._apply_agent_event(ev)
        except Exception:
            pass


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

    # ── Batch stage editor ────────────────────────────────────────────

    def _batch_refresh(self):
        self._batch_list.delete(0, tk.END)
        for i, s in enumerate(self._batch_stages):
            bits = [s.get("dataset") and os.path.basename(s["dataset"])
                    or s.get("hf_dataset") or "?"]
            flags = []
            if s.get("epochs") is not None:
                flags.append(f"ep={s['epochs']}")
            if s.get("steps") is not None:
                flags.append(f"steps={s['steps']}")
            if s.get("force"):
                flags.append("force")
            if s.get("dpo"):
                flags.append("dpo")
            if s.get("dpo_beta") is not None:
                flags.append(f"beta={s['dpo_beta']}")
            for k in ("latex", "code"):
                if s.get(k):
                    flags.append(k)
            label = f"{i + 1:02d}. {bits[0]}"
            if s.get("name"):
                label += f"  [{s['name']}]"
            if flags:
                label += "  (" + ", ".join(flags) + ")"
            self._batch_list.insert(tk.END, label)
        self._update_cmd_preview()

    def _batch_add(self):
        self._stage_dialog(None)

    def _batch_edit(self):
        sel = self._batch_list.curselection()
        if not sel:
            messagebox.showinfo("Batch", "Select a stage to edit (or Add a new one).")
            return
        self._stage_dialog(sel[0])

    def _batch_remove(self):
        sel = self._batch_list.curselection()
        if not sel:
            return
        del self._batch_stages[sel[0]]
        self._batch_refresh()

    def _batch_move(self, delta):
        sel = self._batch_list.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if not (0 <= j < len(self._batch_stages)):
            return
        self._batch_stages[i], self._batch_stages[j] = \
            self._batch_stages[j], self._batch_stages[i]
        self._batch_refresh()
        self._batch_list.selection_set(j)

    def _stage_dialog(self, index: int | None):
        stage = dict(self._batch_stages[index]) if index is not None else {}
        dlg = tk.Toplevel(self.root)
        dlg.title("Stage" if index is not None else "Add stage")
        dlg.configure(bg=SURFACE, padx=12, pady=10)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        def _row(r, label, widget):
            tk.Label(dlg, text=label, bg=SURFACE, fg=TEXT,
                     font=(FONT, 9)).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="ew", padx=(8, 0))

        v_ds = tk.StringVar(value=stage.get("dataset", ""))
        v_hf = tk.StringVar(value=stage.get("hf_dataset", ""))
        v_name = tk.StringVar(value=stage.get("name", ""))
        v_ep = tk.StringVar(value="" if stage.get("epochs") is None else str(stage["epochs"]))
        v_steps = tk.StringVar(value="" if stage.get("steps") is None else str(stage["steps"]))
        v_beta = tk.StringVar(value="" if stage.get("dpo_beta") is None else str(stage["dpo_beta"]))
        v_force = tk.BooleanVar(value=bool(stage.get("force")))
        v_dpo = tk.BooleanVar(value=bool(stage.get("dpo")))
        v_latex = tk.BooleanVar(value=bool(stage.get("latex")))
        v_code = tk.BooleanVar(value=bool(stage.get("code")))

        _row(0, "Dataset (.parquet)", ttk.Entry(dlg, textvariable=v_ds, width=34,
                                                style="Field.TEntry"))
        ttk.Button(dlg, text="…", width=3, style="Card.TButton",
                   command=lambda: self._browse_to(v_ds, "*.parquet")).grid(
            row=0, column=2, padx=(4, 0))
        _row(1, "HF Dataset ID", ttk.Entry(dlg, textvariable=v_hf, width=34,
                                           style="Field.TEntry"))
        _row(2, "Name", ttk.Entry(dlg, textvariable=v_name, width=34,
                                  style="Field.TEntry"))
        _row(3, "Epochs (blank = 1)", ttk.Entry(dlg, textvariable=v_ep, width=10,
                                                style="Field.TEntry"))
        _row(4, "Steps (blank = epochs)", ttk.Entry(dlg, textvariable=v_steps, width=10,
                                                    style="Field.TEntry"))
        _row(5, "DPO Beta (blank = 0.1)", ttk.Entry(dlg, textvariable=v_beta, width=10,
                                                    style="Field.TEntry"))
        checks = ttk.Frame(dlg, bg=SURFACE)
        checks.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        for col, (text, var) in enumerate((("force", v_force), ("dpo", v_dpo),
                                           ("latex", v_latex), ("code", v_code))):
            ttk.Checkbutton(checks, text=text, variable=var,
                            style="Flag.TCheckbutton").grid(row=0, column=col, padx=4)

        def _ok():
            ds = v_ds.get().strip()
            hf = v_hf.get().strip()
            if not ds and not hf:
                messagebox.showwarning("Stage", "Dataset path or HF dataset ID is required.",
                                       parent=dlg)
                return
            if ds and not os.path.isfile(ds):
                messagebox.showwarning("Stage", f"Dataset not found:\n{ds}", parent=dlg)
                return
            stage_new = dict(stage)  # preserve unknown keys
            if ds:
                stage_new["dataset"] = ds
            else:
                stage_new.pop("dataset", None)
            if hf:
                stage_new["hf_dataset"] = hf
            else:
                stage_new.pop("hf_dataset", None)
            name = v_name.get().strip()
            if name:
                stage_new["name"] = name
            elif "name" in stage_new and index is None:
                stage_new.pop("name", None)
            for var, key, cast in ((v_ep, "epochs", int), (v_steps, "steps", int),
                                   (v_beta, "dpo_beta", float)):
                raw = var.get().strip()
                if raw:
                    try:
                        val = cast(raw)
                    except ValueError:
                        messagebox.showwarning("Stage", f"{key} must be a number.",
                                               parent=dlg)
                        return
                    stage_new[key] = val
                else:
                    stage_new.pop(key, None)
            for var, key in ((v_force, "force"), (v_dpo, "dpo"),
                             (v_latex, "latex"), (v_code, "code")):
                if var.get():
                    stage_new[key] = True
                else:
                    stage_new.pop(key, None)
            if index is None:
                self._batch_stages.append(stage_new)
            else:
                self._batch_stages[index] = stage_new
            self._batch_refresh()
            dlg.destroy()

        ttk.Button(dlg, text="OK", style="Accent.TButton", command=_ok).grid(
            row=7, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(dlg, text="Cancel", style="Card.TButton",
                   command=dlg.destroy).grid(row=7, column=2, sticky="w", pady=(10, 0))
        dlg.columnconfigure(1, weight=1)

    def _browse_to(self, var: tk.StringVar, pattern: str):
        p = filedialog.askopenfilename(filetypes=[("Parquet files", pattern), ("All", "*.*")])
        if p:
            var.set(p)

    def _batch_save(self):
        if not self._batch_stages:
            messagebox.showwarning("Batch", "No stages to save.")
            return
        default = self._batch_manifest_path or os.path.join(
            self._var_output_dir.get().strip() or self._DEFAULTS["output_dir"],
            "batch_manifest.json")
        p = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile=os.path.basename(default),
            initialdir=os.path.dirname(default) or ".",
            filetypes=[("JSON manifest", "*.json")])
        if not p:
            return
        manifest = {"base_model": self._var_base_model.get().strip() or None,
                    "stages": self._batch_stages}
        if not manifest["base_model"]:
            manifest.pop("base_model")
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            messagebox.showerror("Batch", f"Save failed: {e}")
            return
        self._batch_manifest_path = p
        self._update_cmd_preview()

    def _batch_load(self):
        p = filedialog.askopenfilename(
            filetypes=[("JSON manifest", "*.json"), ("All", "*.*")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                manifest = json.load(f)
        except Exception as e:
            messagebox.showerror("Batch", f"Load failed: {e}")
            return
        stages = manifest.get("stages")
        if not isinstance(stages, list) or not stages:
            messagebox.showwarning("Batch", "Manifest has no 'stages' list.")
            return
        self._batch_stages = [s for s in stages if isinstance(s, dict)]
        self._batch_manifest_path = p
        base = manifest.get("base_model")
        if base and not self._var_base_model.get().strip():
            self._var_base_model.set(str(base))
        self._var_batch_mode.set(True)
        self._batch_refresh()

    def _collect_params(self) -> dict:
        """Form state -> the shared ``build_argv`` params dict.

        D9: this is the single source of truth for command construction in
        both UIs; the Tk dashboard no longer duplicates the argv builder or
        the start validations (``webui_config.build_argv``).
        """
        p = {
            "base_model": self._var_base_model.get().strip(),
            "output_dir": self._var_output_dir.get().strip(),
            "dataset": self._var_dataset.get().strip(),
            "hf_dataset": self._var_hf_dataset.get().strip(),
            "epoch": self._var_epoch.get().strip(),
            "steps": self._var_steps.get().strip(),
            "batch_size": self._var_batch.get().strip(),
            "grad_accum": self._var_grad.get().strip(),
            "force": self._var_force.get(),
            "latex": self._var_latex.get(),
            "code": self._var_code.get(),
            "image": self._var_image.get(),
            "dpo": self._var_dpo.get(),
            "dpo_beta": self._var_dpo_beta.get().strip(),
        }
        if self._var_batch_mode.get():
            p["stages"] = [dict(s) for s in self._batch_stages]
        return p

    # ── Command construction ──────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        """Preview argv (strict=False: no filesystem checks while typing)."""
        from .webui_config import build_argv
        try:
            return build_argv(self._collect_params(), strict=False,
                              manifest_path=self._batch_manifest_path or None)
        except ValueError as e:
            # Form incomplete/invalid: show the reason where the command
            # would be (the web UI preview behaves the same).
            return [f"(incomplete form) {e}"]

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

    def _set_status(self, text, color=MUTED):
        self._lbl_status.configure(text=f"\u25cf  {text}", fg=color)

    # ── Training control ──────────────────────────────────────────────

    def _start_training(self):
        if self._running:
            messagebox.showwarning("Busy", "A training run is already active.")
            return

        # D9: validation + argv construction are shared with the web UI.
        # Keep Tk's Save-Manifest target: build_argv writes the manifest
        # there (or to <out>/batch_manifest.json) on this strict build.
        if self._var_batch_mode.get() and not self._batch_manifest_path:
            od = self._var_output_dir.get().strip() or self._DEFAULTS["output_dir"]
            try:
                os.makedirs(od, exist_ok=True)
            except Exception:
                pass
            self._batch_manifest_path = os.path.join(od, "batch_manifest.json")

        from .webui_config import build_argv
        try:
            cmd = build_argv(self._collect_params(), strict=True,
                             manifest_path=self._batch_manifest_path or None)
        except ValueError as e:
            messagebox.showwarning("Invalid input", str(e))
            return

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
        self._prog_stage_total = None
        self._prog_stage_done = 0
        self._reset_progress()

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
                stripped = line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        obj = json.loads(stripped)
                    except Exception:
                        obj = None
                    if isinstance(obj, dict) and "event" in obj:
                        # Agent-mode NDJSON event: drives the progress panel.
                        # (Step events contain "loss", so this check must run
                        # before the loss-dict filter below.)
                        self._line_queue.put(("agent_event", obj))
                        self._write_line(sys.stdout, line)
                        continue
                # Loss dicts are consumed from stderr (LossLineLogger) with
                # line integrity; echoing the stdout copies here would
                # double-count chart points, so filter them on this side.
                if "'loss'" in line or '"loss"' in line:
                    continue
                self._write_line(sys.stdout, line)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _reader_stderr(self, pipe):
        """Consume stderr line-by-line; dict lines feed the live chart.

        Uses readline() rather than a fixed-size read(). BufferedReader.read(n)
        blocks until *n* bytes have accumulated, so the old read(256) sat on the
        first ~4 loss rows (each row is ~84 bytes) before surfacing anything —
        the chart showed "Starting..." for the first several steps of every run.
        readline() returns the moment a newline arrives, so step 1 plots
        immediately. A trailing partial line is flushed at EOF by the iterator.

        Loss rows are child-proof: trainers LossLineLogger writes them via
        _safe_print, so even if this pipe dies the training run keeps going.
        """
        try:
            for raw in iter(pipe.readline, b""):
                self._handle_stderr_line(raw)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _handle_stderr_line(self, raw: bytes):
        text = raw.decode("utf-8", errors="replace").replace("\r", "")
        if not text.strip():
            return
        if any(p in text for p in self._SUPPRESS_PATTERNS):
            return
        entry = try_parse_loss_line(text)
        if entry:
            self._emit_loss_row(entry)
        else:
            self._write_line(sys.stderr, text)

    def _emit_loss_row(self, entry: dict):
        self._step_count += 1
        self._line_queue.put(entry)
        if not self._loss_header_printed:
            self._loss_header_printed = True
            hdr = (
                f"\n  {'step':<8}  "
                f"{'loss':<10}  "
                f"{'lr':<12}  "
                f"{'grad_norm':<10}  "
                f"{'epoch':<8}  "
                f"{'acc'}\n"
                f"  {'-'*66}\n"
            )
            self._write_line(sys.stdout, hdr)
        loss = entry.get("loss", 0)
        lr   = entry.get("learning_rate", 0)
        ep   = entry.get("epoch", 0)
        gnorm = entry.get("grad_norm", 0)
        step  = entry.get("step", self._step_count)
        acc   = entry.get("acc")
        if lr > 0:
            lr_str = f"{lr:.2e}"
        else:
            lr_str = "0"
        formatted = (
            f"  step {step:<6}  "
            f"loss {loss:<8.4f}  "
            f"lr {lr_str:<10}  "
            f"gnorm {gnorm:<8.3f}  "
            f"ep {ep:.2f}"
        )
        # DPO runs log rewards/accuracies (chosen preferred over rejected);
        # SFT rows have no acc and print a dash.
        formatted += f"  acc {acc:.3f}\n" if acc is not None else "  acc \u2014\n"
        self._write_line(sys.stdout, formatted)

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
                    if isinstance(entry, tuple) and entry[0] == "agent_event":
                        self._apply_agent_event_safe(entry[1])
                        continue
                    self._live_logs.append(entry)
                    dirty = True
                except queue.Empty:
                    break

            if dirty:
                self._refresh_live_chart()

            self._progress_tick()

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

    def _progress_tick(self):
        """250 ms animation/refresh for the progress panel.

        Sweeps the indeterminate bar during load/merge phases, spins the
        active phase marker, and keeps [elapsed<ETA] fresh between step
        events (a 25 s wall-clock gap between events would otherwise make
        the ETA read stale).
        """
        p = self._prog
        if not p["phase"] or p["done"] or p["failed"]:
            return
        try:
            p["spinner"] = (p["spinner"] + 1) % len(self._SPINNER_FRAMES)
            active = self._phase_labels.get(p["phase"])
            if active:
                frame = self._SPINNER_FRAMES[p["spinner"]]
                active.configure(
                    text=f"{frame}  {self._PHASE_TITLES[p['phase']]}")
            indeterminate = not (p["phase"] == "train" and p["total"])
            if indeterminate:
                p["sweep"] = (p["sweep"] + 0.012) % 1.0
                self._render_progress()
            elif p["phase"] == "train" and p["total"]:
                self._update_prog_labels()
        except Exception:
            pass

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
    raise SystemExit(
        "Launch the dashboard with:  reforge dashboard"
    )
