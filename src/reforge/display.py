import io
import os
import re
import sys
import threading

_ANSI_RESET   = "\033[0m"
_ANSI_BOLD    = "\033[1m"
_ANSI_DIM     = "\033[2m"
_ANSI_RED     = "\033[31m"
_ANSI_GREEN   = "\033[32m"
_ANSI_YELLOW  = "\033[33m"
_ANSI_BLUE    = "\033[34m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_CYAN    = "\033[36m"
_ANSI_WHITE   = "\033[37m"
_ANSI_GRAY    = "\033[90m"
_ANSI_BRED    = "\033[91m"
_ANSI_BGREEN  = "\033[92m"
_ANSI_BYELLOW = "\033[93m"
_ANSI_BBLUE   = "\033[94m"
_ANSI_BMAG    = "\033[95m"
_ANSI_BCYAN   = "\033[96m"
_ANSI_BWHITE  = "\033[97m"

_ANSI_ESC_RE = re.compile(r"\033\[[0-9;]*m")


def _enable_windows_vt() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:
        return False


def _enable_utf8_stdout() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_enable_utf8_stdout()


# When agent mode is active, pretty/human console output is diverted here
# (stderr) so stdout carries pure NDJSON events for machine parsing.
_DIVERT_STREAM = None


def _set_stream(stream) -> None:
    """Redirect all display output to `stream` (used by agent mode)."""
    global _DIVERT_STREAM
    _DIVERT_STREAM = stream


def _safe_print(*args, **kwargs) -> None:
    """print() that never raises, even if the console handle dies mid-run.

    On Windows, a failing terminal makes *every* subsequent write raise
    `OSError: [Errno 22] Invalid argument`, which cascades through logging
    and can abort training (and even the emergency-save path). Swallow
    OSError and keep going.
    """
    if _DIVERT_STREAM is not None:
        kwargs.setdefault("file", _DIVERT_STREAM)
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def _redirect_output(path: str) -> None:
    """Send all console output to a file.

    Keeps long training runs immune to failing Windows console handles
    (OSError 22) and produces a reviewable log of the run.
    """
    global _USE_COLOR
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    sys.stdout = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    sys.stderr = sys.stdout
    _USE_COLOR = False


class _SafeStream(io.TextIOBase):
    """A text stream wrapper that survives OSError mid-run.

    `tqdm` and the Trainer capture `sys.stdout` / `sys.stderr` when they are
    created. On Windows the underlying console/pipe handle can start failing
    with `OSError: [Errno 22] Invalid argument` mid-run and *every* further
    write fails, which aborts training (and even the exception handlers).
    This wrapper: on the first OSError it silently switches to a fallback
    log file and keeps writing from there, so a dead terminal can never
    kill a training run.
    """

    def __init__(self, delegate, fallback_factory):
        self._delegate = delegate
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._lock = threading.Lock()

    @property
    def encoding(self):
        return getattr(self._delegate, "encoding", "utf-8")

    def isatty(self):
        return False

    def write(self, s):
        try:
            return self._delegate.write(s)
        except OSError:
            pass
        except Exception:
            pass
        try:
            with self._lock:
                if self._fallback is None:
                    self._fallback = self._fallback_factory()
                self._delegate = self._fallback
            return self._delegate.write(s)
        except Exception:
            return len(s)

    def flush(self):
        try:
            self._delegate.flush()
        except Exception:
            pass


def _install_safe_streams(fallback_log: str) -> None:
    """Wrap sys.stdout / sys.stderr so OSError can never escape a write."""
    directory = os.path.dirname(os.path.abspath(fallback_log))
    if directory:
        os.makedirs(directory, exist_ok=True)

    def factory():
        return open(fallback_log, "a", encoding="utf-8", errors="replace", buffering=1)

    if not isinstance(sys.stdout, _SafeStream):
        sys.stdout = _SafeStream(sys.stdout, factory)
    if not isinstance(sys.stderr, _SafeStream):
        sys.stderr = _SafeStream(sys.stderr, factory)


def _detect_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return _enable_windows_vt()


_USE_COLOR = _detect_color()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR or not text:
        return text
    return f"{code}{text}{_ANSI_RESET}"


def _vlen(s: str) -> int:
    return len(_ANSI_ESC_RE.sub("", s))


def _bold(text: str) -> str:
    return _c(_ANSI_BOLD + _ANSI_BWHITE, text)


def _banner(title: str, color: str = _ANSI_BCYAN, width: int = 64) -> None:
    _safe_print()
    _safe_print(_c(color, "=" * width))
    _safe_print(_c(_ANSI_BOLD + _ANSI_WHITE, f"  {title}"))
    _safe_print(_c(_ANSI_GRAY, "-" * width))


def _ok(msg: str)   -> None: _safe_print(f"  {_c(_ANSI_BGREEN, chr(0x2714))}  {msg}")
def _info(msg: str) -> None: _safe_print(f"  {_c(_ANSI_BCYAN, chr(0xB7))}  {msg}")
def _warn(msg: str) -> None: _safe_print(f"  {_c(_ANSI_BYELLOW, chr(0x26A0))}  {msg}")
def _err(msg: str)  -> None: _safe_print(f"  {_c(_ANSI_BRED, chr(0x2716))}  {msg}")


def _title(app: str, info: list = None, subtitle: str = None,
           color: str = _ANSI_BCYAN, width: int = 64) -> None:
    _safe_print()
    _safe_print(_c(color, "=" * width))
    _safe_print()
    app_padded = f"  {app.center(width - 2)}"
    _safe_print(_c(_ANSI_BOLD + _ANSI_BWHITE, app_padded))
    if subtitle:
        sub_padded = f"  {subtitle.center(width - 2)}"
        _safe_print(_c(_ANSI_DIM, sub_padded))
    _safe_print()
    if info:
        key_w = max(_vlen(k) for k, _ in info)
        for key, value in info:
            key_styled = _c(_ANSI_GRAY, key.ljust(key_w + 2))
            value_styled = _c(_ANSI_WHITE, str(value)) if _USE_COLOR else str(value)
            _safe_print(f"  {key_styled}{value_styled}")
    _safe_print()
    _safe_print(_c(_ANSI_GRAY, "-" * width))


def _table(title: str, rows: list, value_color: str = None,
           key_color: str = _ANSI_BCYAN, width: int = 64) -> None:
    _safe_print()
    _safe_print(_c(_ANSI_BOLD + _ANSI_WHITE, f"  {title}"))
    _safe_print(_c(_ANSI_GRAY, "-" * width))
    if not rows:
        _safe_print(_c(_ANSI_DIM, "  (no data)"))
    else:
        key_w = max(_vlen(k) for k, _ in rows)
        for key, value in rows:
            value_str = value() if callable(value) else str(value)
            key_styled = _c(key_color, key.ljust(key_w))
            if value_color:
                value_str = _c(value_color, value_str)
            _safe_print(f"  {key_styled}    {value_str}")
    _safe_print(_c(_ANSI_GRAY, "-" * width))


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"
