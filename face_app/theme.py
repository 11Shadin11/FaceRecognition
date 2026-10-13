"""Shared colors and ttk styles, configured once for the application's window."""
from dataclasses import dataclass
from tkinter import font as tkfont, ttk


@dataclass(frozen=True)
class Palette:
    background: str = "#0b1020"
    panel: str = "#141d32"
    raised: str = "#202d48"
    border: str = "#2b3a59"
    preview: str = "#080e1a"
    text: str = "#edf3ff"
    muted: str = "#9bacca"
    disabled: str = "#647593"
    violet: str = "#a995ff"
    primary: str = "#795cf4"
    cyan: str = "#42c8e5"
    green: str = "#55dcb0"
    amber: str = "#f5c36b"
    red: str = "#ff879c"


def apply_theme(window):
    palette = Palette()
    family = tkfont.nametofont("TkDefaultFont", root=window).actual("family")
    style = ttk.Style(window)
    # Aqua ignores many color settings; clam supports the same palette on all OSes.
    style.theme_use("clam")
    window.configure(background=palette.background)
    style.configure("TFrame", background=palette.background)
    style.configure("Card.TFrame", background=palette.panel, bordercolor=palette.border,
                    relief="solid", borderwidth=1)
    style.configure("Surface.TFrame", background=palette.panel)
    style.configure("TLabel", background=palette.background, foreground=palette.text, font=(family, 11))
    style.configure("Card.TLabel", background=palette.panel, foreground=palette.text)
    style.configure("Muted.TLabel", foreground=palette.muted, font=(family, 10))
    style.configure("CardMuted.TLabel", background=palette.panel, foreground=palette.muted, font=(family, 10))
    style.configure("Title.TLabel", font=(family, 24, "bold"))
    style.configure("CardTitle.TLabel", background=palette.panel, font=(family, 14, "bold"))
    style.configure("Result.TLabel", background=palette.panel, font=(family, 20, "bold"))
    style.configure("Eyebrow.TLabel", background=palette.panel, foreground=palette.violet, font=(family, 9, "bold"))
    style.configure("TButton", background=palette.raised, foreground=palette.text,
                    bordercolor=palette.border, lightcolor=palette.raised, darkcolor=palette.raised,
                    borderwidth=0, relief="flat", padding=(14, 11), font=(family, 11, "bold"),
                    focuscolor=palette.violet)
    style.map("TButton", background=[("disabled", palette.panel), ("pressed", "#2d3b5a"), ("active", "#2a3b5c")],
              foreground=[("disabled", palette.disabled)])
    style.configure("Compact.TButton", padding=(10, 7), font=(family, 10, "bold"))
    for name, background, active, foreground in (
        ("Primary", palette.primary, "#8e73ff", "#ffffff"),
        ("Cyan", palette.cyan, "#6ddaf0", "#08212d"),
        ("Stop", "#39283b", "#503347", palette.red),
    ):
        style.configure(f"{name}.TButton", background=background, foreground=foreground,
                        bordercolor=background, lightcolor=background, darkcolor=background)
        style.map(f"{name}.TButton", background=[("disabled", palette.panel), ("active", active)],
                  foreground=[("disabled", palette.disabled)])
    style.configure("Compact.Cyan.TButton", padding=(10, 7), font=(family, 10, "bold"))
    for widget in ("TCombobox", "TSpinbox"):
        style.configure(widget, fieldbackground=palette.preview, background=palette.raised,
                        foreground=palette.text, arrowcolor=palette.muted, bordercolor=palette.border,
                        lightcolor=palette.border, darkcolor=palette.border, padding=8,
                        insertcolor=palette.text, selectbackground=palette.primary, selectforeground="white")
        style.map(widget, fieldbackground=[("disabled", palette.panel), ("readonly", palette.preview)],
                  foreground=[("disabled", palette.disabled)], bordercolor=[("focus", palette.violet)])
    style.configure("App.Horizontal.TProgressbar", background=palette.primary,
                    troughcolor=palette.raised, borderwidth=0, thickness=4,
                    lightcolor=palette.primary, darkcolor=palette.primary)
    for option, value in (("background", palette.raised), ("foreground", palette.text),
                          ("selectBackground", palette.primary), ("selectForeground", "white"),
                          ("font", (family, 11))):
        window.option_add(f"*TCombobox*Listbox.{option}", value)
    return palette, family
