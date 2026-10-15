import os
import re
import sys

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
    print()
    print(_c(color, "=" * width))
    print(_c(_ANSI_BOLD + _ANSI_WHITE, f"  {title}"))
    print(_c(_ANSI_GRAY, "-" * width))


def _ok(msg: str)   -> None: print(f"  {_c(_ANSI_BGREEN, chr(0x2714))}  {msg}")
def _info(msg: str) -> None: print(f"  {_c(_ANSI_BCYAN, chr(0xB7))}  {msg}")
def _warn(msg: str) -> None: print(f"  {_c(_ANSI_BYELLOW, chr(0x26A0))}  {msg}")
def _err(msg: str)  -> None: print(f"  {_c(_ANSI_BRED, chr(0x2716))}  {msg}")


def _title(app: str, info: list = None, subtitle: str = None,
           color: str = _ANSI_BCYAN, width: int = 64) -> None:
    print()
    print(_c(color, "=" * width))
    print()
    app_padded = f"  {app.center(width - 2)}"
    print(_c(_ANSI_BOLD + _ANSI_BWHITE, app_padded))
    if subtitle:
        sub_padded = f"  {subtitle.center(width - 2)}"
        print(_c(_ANSI_DIM, sub_padded))
    print()
    if info:
        key_w = max(_vlen(k) for k, _ in info)
        for key, value in info:
            key_styled = _c(_ANSI_GRAY, key.ljust(key_w + 2))
            value_styled = _c(_ANSI_WHITE, str(value)) if _USE_COLOR else str(value)
            print(f"  {key_styled}{value_styled}")
    print()
    print(_c(_ANSI_GRAY, "-" * width))


def _table(title: str, rows: list, value_color: str = None,
           key_color: str = _ANSI_BCYAN, width: int = 64) -> None:
    print()
    print(_c(_ANSI_BOLD + _ANSI_WHITE, f"  {title}"))
    print(_c(_ANSI_GRAY, "-" * width))
    if not rows:
        print(_c(_ANSI_DIM, "  (no data)"))
    else:
        key_w = max(_vlen(k) for k, _ in rows)
        for key, value in rows:
            value_str = value() if callable(value) else str(value)
            key_styled = _c(key_color, key.ljust(key_w))
            if value_color:
                value_str = _c(value_color, value_str)
            print(f"  {key_styled}    {value_str}")
    print(_c(_ANSI_GRAY, "-" * width))


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"
