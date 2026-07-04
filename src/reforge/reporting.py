import json
import os

from .display import _ok, _warn, _banner, _table, _c, _ANSI_BCYAN, _ANSI_DIM, _fmt_duration


def _loss_sparkline(values: list, width: int = 48) -> str:
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = list(values)
    lo, hi = min(sampled), max(sampled)
    span = hi - lo if hi > lo else 1.0
    blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    out = []
    for v in sampled:
        idx = int((v - lo) / span * (len(blocks) - 1))
        out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    return "".join(out)


def summarize_training(
    trainer,
    *,
    dataset_size: int,
    num_train_epochs: int,
    batch_size: int,
    grad_accum: int,
    report_path: str | None = None,
) -> dict:
    state = trainer.state
    log = list(state.log_history or [])
    steps = int(state.global_step or 0)

    loss_steps: list[int] = []
    loss_values: list[float] = []
    eval_entries: list[dict] = []
    train_runtime = 0.0
    samples_per_sec = 0.0
    steps_per_sec = 0.0
    for entry in log:
        if "loss" in entry and "eval_loss" not in entry:
            loss_steps.append(int(entry.get("step", len(loss_steps) + 1)))
            loss_values.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_entries.append(entry)
        if "train_runtime" in entry:
            train_runtime = float(entry["train_runtime"])
        if "train_samples_per_second" in entry:
            samples_per_sec = float(entry["train_samples_per_second"])
        if "train_steps_per_second" in entry:
            steps_per_sec = float(entry["train_steps_per_second"])

    first_loss = loss_values[0] if loss_values else None
    final_loss = loss_values[-1] if loss_values else None
    best_loss = min(loss_values) if loss_values else None
    best_step = loss_steps[loss_values.index(best_loss)] if loss_values else None
    worst_loss = max(loss_values) if loss_values else None

    if loss_values:
        tail_n = max(1, len(loss_values) // 10)
        avg_recent = sum(loss_values[-tail_n:]) / tail_n
        head_n = max(1, len(loss_values) // 5)
        avg_head = sum(loss_values[:head_n]) / head_n
        delta_recent = avg_recent - avg_head
        if delta_recent < -0.02:
            trend = f"improving  ({delta_recent:+.4f} vs first 20%)"
        elif delta_recent > 0.02:
            trend = f"worsening  ({delta_recent:+.4f} vs first 20%)"
        else:
            trend = f"plateau    ({delta_recent:+.4f} vs first 20%)"
    else:
        avg_recent = None
        trend = "n/a"

    effective_batch = batch_size * grad_accum
    samples_seen = steps * effective_batch
    epochs_completed = float(state.epoch or 0.0)

    report = {
        "total_steps":         steps,
        "epochs_planned":      num_train_epochs,
        "epochs_completed":    round(epochs_completed, 4),
        "dataset_size":        dataset_size,
        "batch_size":          batch_size,
        "grad_accum":          grad_accum,
        "effective_batch":     effective_batch,
        "samples_seen":        samples_seen,
        "training_time_sec":   round(train_runtime, 2),
        "samples_per_sec":     round(samples_per_sec, 4),
        "steps_per_sec":       round(steps_per_sec, 4),
        "loss": {
            "first":              round(first_loss, 6) if first_loss is not None else None,
            "final":              round(final_loss, 6) if final_loss is not None else None,
            "best":               round(best_loss, 6) if best_loss is not None else None,
            "best_step":          best_step,
            "worst":              round(worst_loss, 6) if worst_loss is not None else None,
            "avg_last_10pct":     round(avg_recent, 6) if avg_recent is not None else None,
            "delta_first_to_final": round(final_loss - first_loss, 6) if (final_loss is not None and first_loss is not None) else None,
            "trend":              trend,
        },
        "eval": [
            {k: v for k, v in e.items() if k.startswith("eval_") or k == "step"}
            for e in eval_entries
        ],
    }

    _banner("Training Results")
    if steps:
        _table("Run", [
            ("Steps",        f"{steps:,}"),
            ("Epochs",       f"{epochs_completed:.2f} / {num_train_epochs}"),
            ("Samples seen", f"{samples_seen:,}"),
        ])
    if train_runtime:
        _table("Throughput", [
            ("Time",       _fmt_duration(train_runtime)),
            ("Samples/s",  f"{samples_per_sec:.2f}"),
            ("Steps/s",    f"{steps_per_sec:.2f}"),
        ])
    if loss_values:
        delta_val = (final_loss - first_loss) if (final_loss is not None and first_loss is not None) else None
        _table("Loss", [
            ("First",           f"{first_loss:.4f}"),
            ("Final",           f"{final_loss:.4f}"),
            ("Best",            f"{best_loss:.4f} @ step {best_step}"),
            ("\u0394 (first\u2192final)",  f"{delta_val:+.4f}" if delta_val is not None else "n/a"),
            ("Avg (last 10%)",  f"{avg_recent:.4f}"),
            ("Trend",           trend),
        ], value_color=_ANSI_BCYAN)
        dot = chr(0xB7)
        print(f"  {_c(_ANSI_BCYAN, dot)}  Curve (n={len(loss_values)}): {_c(_ANSI_BCYAN, _loss_sparkline(loss_values))}")
        print(f"  {_c(_ANSI_DIM, f'    range: min={best_loss:.4f}  max={worst_loss:.4f}  span={worst_loss - best_loss:.4f}')}")
    else:
        _warn("No loss entries found in log history \u2014 was logging_steps > 0?")
    if eval_entries:
        final_eval = eval_entries[-1]
        ev_loss = final_eval.get("eval_loss")
        if ev_loss is not None:
            _table("Evaluation", [
                ("Final eval loss", f"{float(ev_loss):.4f}"),
                ("Eval points",     f"{len(eval_entries)}"),
            ])
    print()

    if report_path:
        try:
            os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            _ok(f"Report written: {report_path}")
        except Exception as e:
            _warn(f"Could not write training_report.json: {e}")

    return report
