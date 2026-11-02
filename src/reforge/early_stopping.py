import math

import numpy as np
from transformers import TrainerCallback


def calc_floor(steps_total: int) -> float:
    if steps_total < 4000:
        return 0.22
    return 0.18


def compute_min_steps(train_dataset, batch_size=1, grad_accum=6):
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
    dyn_frac = calc_floor(steps_per_epoch)
    return max(int(steps_per_epoch * dyn_frac), 50)


def dynamic_early_stop_cap(steps_total):
    min_cap = 0.65
    max_cap = 0.95
    log_steps = math.log10(max(steps_total, 10))
    scale = min(1.0, max(0.0, (log_steps - 1) / 4))
    cap = max_cap - (max_cap - min_cap) * scale
    return round(cap, 4)


def select_scheduler(dataset_rows: int, epochs: int, min_stop_steps: int) -> str:
    total = max(1, dataset_rows) * max(1, epochs)
    pressure = min_stop_steps / total
    EXPOSURE_FLOOR = 0.18
    if total < 1500 or epochs <= 1:
        return "linear"
    if pressure >= 2 * EXPOSURE_FLOOR:
        return "linear"
    if 1500 <= total <= 40000 and epochs >= 2 and pressure <= EXPOSURE_FLOOR:
        if (total / 3) < min_stop_steps:
            return "cosine_with_restarts"
    return "cosine"


class EarlyStopByLoss(TrainerCallback):
    def __init__(
        self,
        steps_total: int,
        mode: str = "sft",
        active: bool = True,
        hard_cap_steps: int | None = None,
        exposure_floor: float = 0.10,
        quality_lr_frac: float = 0.90,
        window: int = 128,
        ema_beta: float = 0.90,
        std_floor: float = 0.10,
        min_abs_improve: float = 0.03,
        min_sigma_improve: float = 0.50,
        slope_window: int | None = None,
        slope_thresh: float = 0.002,
        patience: int = 80,
        cooldown_after_best: int = 15,
        verbose_every: int = 0
    ):
        super().__init__()
        self.steps_total = max(1, int(steps_total))
        self.mode = mode
        self.active = active
        self.hard_cap_steps = hard_cap_steps
        self.exposure_floor = float(exposure_floor)
        self.quality_lr_frac = float(quality_lr_frac)

        base_w = max(96, int(0.015 * self.steps_total))
        self.window = max(min(window, 512), base_w)
        self.ema_beta = float(ema_beta)
        self.std_floor = float(std_floor)
        self.min_abs_improve = float(min_abs_improve)
        self.min_sigma_improve = float(min_sigma_improve)
        self.slope_window = max(48, int(0.75 * self.window)) if slope_window is None else int(slope_window)
        self.slope_thresh = float(slope_thresh)
        self.patience = int(patience)
        self.cooldown_after_best = int(cooldown_after_best)
        self.verbose_every = int(verbose_every)

        self.losses: list[float] = []
        self.ema_series: list[float] = []
        self.ema: float | None = None
        self.prev_ema: float | None = None
        self.best_ema: float = float("inf")
        self.since_best: int = 0
        self.cooldown: int = 0
        self.triggered: bool = False
        self.max_lr_seen: float = 0.0
        self.last_step_seen: int = -1

    @staticmethod
    def _mad_sigma(arr: list[float]) -> float:
        if len(arr) < 3:
            return 0.0
        a = np.asarray(arr, dtype=np.float32)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return 1.4826 * mad

    def _epoch_fraction(self, state) -> float:
        step = max(0, int(getattr(state, "global_step", 0)))
        return step / float(self.steps_total)

    def _update_ema(self, loss: float) -> None:
        if self.ema is None:
            self.ema = loss
        else:
            b = self.ema_beta
            self.ema = b * self.ema + (1.0 - b) * loss

    def _lr_gate_ok(self, logs: dict) -> bool:
        if self.quality_lr_frac >= 1.0:
            return True
        lr = logs.get("learning_rate", None)
        if lr is None:
            return True
        try:
            lr = float(lr)
        except Exception:
            return True
        if lr > self.max_lr_seen:
            self.max_lr_seen = lr
        return lr <= (self.max_lr_seen * self.quality_lr_frac + 1e-12)

    def _recent_slice(self, arr: list[float], k: int) -> list[float]:
        k = max(1, int(k))
        if len(arr) <= k:
            return arr[:]
        return arr[-k:]

    def _slope(self, series: list[float]) -> float:
        if len(series) < 8:
            return 0.0
        x = np.arange(len(series), dtype=np.float32)
        y = np.asarray(series, dtype=np.float32)
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.active or logs is None:
            return

        if self.hard_cap_steps is not None:
            if int(getattr(state, "global_step", 0)) >= int(self.hard_cap_steps):
                control.should_training_stop = True
                self.triggered = True
                return

        if "loss" not in logs:
            return

        step = int(getattr(state, "global_step", 0))
        if step == self.last_step_seen:
            return
        self.last_step_seen = step

        loss = float(logs["loss"])
        warmup = int(getattr(args, "warmup_steps", 0) or 0)

        self._update_ema(loss)
        if self.prev_ema is None:
            self.prev_ema = self.ema
        self.ema_series.append(self.ema)

        if self.ema < self.best_ema - 1e-12:
            self.best_ema = self.ema
            self.since_best = 0
            self.cooldown = self.cooldown_after_best
        else:
            self.since_best += 1
            if self.cooldown > 0:
                self.cooldown -= 1

        if step < warmup:
            self.prev_ema = self.ema
            return
        if self._epoch_fraction(state) < self.exposure_floor:
            self.prev_ema = self.ema
            return
        if not self._lr_gate_ok(logs):
            return

        min_history = max(24, self.window // 3)
        if len(self.ema_series) < min_history:
            self.prev_ema = self.ema
            return

        recent_ema = self._recent_slice(self.ema_series, self.window)
        sigma = max(self._mad_sigma(recent_ema), self.std_floor)

        abs_drop = (self.prev_ema - self.ema)
        sigma_drop = abs_drop / (sigma + 1e-12)
        improved = (abs_drop >= self.min_abs_improve) or (sigma_drop >= self.min_sigma_improve)

        slope_win = self._recent_slice(self.ema_series, self.slope_window)
        slope = self._slope(slope_win)

        worsening = slope >= self.slope_thresh
        plateau = (self.since_best >= self.patience) and (abs(slope) <= self.slope_thresh) and (sigma <= max(0.5 * self.std_floor, 0.05))

        stop_now = False
        if improved:
            self.prev_ema = self.ema
        else:
            stop_now = plateau or worsening

        if self.verbose_every and (step % self.verbose_every == 0):
            print(f"[ES] step={step} ema={self.ema:.4f} best={self.best_ema:.4f} "
                  f"abs_drop={abs_drop:.4f} sigma_drop={sigma_drop:.2f} "
                  f"sigma={sigma:.4f} slope={slope:.5f} since_best={self.since_best} "
                  f"plateau={plateau} worsening={worsening}")

        if stop_now:
            control.should_training_stop = True
            self.triggered = True
            return

    def on_train_end(self, args, state, control, **kwargs):
        if self.hard_cap_steps is not None and int(getattr(state, "global_step", 0)) >= int(self.hard_cap_steps):
            self.triggered = True
