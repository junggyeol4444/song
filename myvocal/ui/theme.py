"""
화면 색과 글꼴.

음악 프로그램은 오래 들여다본다. 밝은 배경은 몇 시간 쓰면 눈이 아프다.
그래서 어두운 바탕을 기본으로 한다. 실제 DAW 들이 전부 그렇게 하는 이유다.

색은 한 군데 모아 둔다. 화면마다 색을 직접 적으면 나중에 바꿀 때 전부
뒤져야 하고, 한두 군데는 반드시 빠진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Palette:
    """색 묶음."""

    # 바탕
    window: str = "#1b1d21"
    surface: str = "#232629"
    surface_raised: str = "#2b2f33"
    surface_sunken: str = "#151719"

    # 글자
    text: str = "#e8eaed"
    text_dim: str = "#9aa0a6"
    text_faint: str = "#6b7176"

    # 선
    border: str = "#3a3f44"
    border_strong: str = "#4d5359"
    grid_bar: str = "#3a3f44"
    grid_beat: str = "#2b2f33"

    # 강조
    accent: str = "#4a9eff"
    accent_dim: str = "#2d6bb0"
    accent_text: str = "#ffffff"

    # 상태
    success: str = "#4caf7d"
    warning: str = "#e0a340"
    danger: str = "#e05555"
    playhead: str = "#ff5a5a"

    # 트랙 종류별 색 (core/tracks.py 의 DEFAULT_COLORS 와 맞춘다)
    vocal: str = "#e8556d"
    instrument: str = "#4a90d9"
    drum: str = "#e8a33d"
    audio: str = "#7aba5a"
    bus: str = "#9b6bd6"

    # 음량 미터
    meter_low: str = "#4caf7d"
    meter_mid: str = "#e0a340"
    meter_high: str = "#e05555"


DARK = Palette()

LIGHT = Palette(
    window="#f2f3f5", surface="#ffffff", surface_raised="#ffffff",
    surface_sunken="#e7e9ec",
    text="#1b1d21", text_dim="#5f6368", text_faint="#9aa0a6",
    border="#d0d4d9", border_strong="#b0b6bd",
    grid_bar="#c8ccd2", grid_beat="#e4e7ea",
    accent="#1a73e8", accent_dim="#a8c7f0", accent_text="#ffffff",
)


# 한글이 들어가므로 한글을 지원하는 글꼴을 먼저 찾는다.
# 윈도우는 맑은 고딕, 맥은 애플 SD 산돌고딕, 리눅스는 배포판마다 다르다.
FONT_CANDIDATES: tuple[str, ...] = (
    "Malgun Gothic",        # 윈도우 기본 한글 글꼴
    "맑은 고딕",
    "Apple SD Gothic Neo",  # 맥
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "NanumGothic",
    "Segoe UI",
    "sans-serif",
)

MONO_CANDIDATES: tuple[str, ...] = (
    "Cascadia Mono", "Consolas", "D2Coding", "Menlo",
    "DejaVu Sans Mono", "monospace",
)


def pick_font(candidates: tuple[str, ...] = FONT_CANDIDATES) -> str:
    """이 컴퓨터에 실제로 있는 글꼴 중 첫 번째를 고른다."""
    try:
        from PySide6 import QtGui

        available = set(QtGui.QFontDatabase.families())
        for name in candidates:
            if name in available:
                return name
    except Exception:
        pass
    return candidates[-1]


def build_stylesheet(palette: Palette = DARK, font_family: str | None = None,
                     base_size: int = 10) -> str:
    """Qt 스타일시트를 만든다."""
    family = font_family or pick_font()
    mono = pick_font(MONO_CANDIDATES)
    return f"""
    * {{
        font-family: "{family}";
        font-size: {base_size}pt;
        outline: none;
    }}
    QWidget {{
        background: {palette.window};
        color: {palette.text};
    }}
    QFrame#Surface, QScrollArea, QStackedWidget {{
        background: {palette.surface};
    }}
    QFrame#Card {{
        background: {palette.surface_raised};
        border: 1px solid {palette.border};
        border-radius: 8px;
    }}
    QLabel#Title {{
        font-size: {base_size + 12}pt;
        font-weight: 600;
    }}
    QLabel#Heading {{
        font-size: {base_size + 3}pt;
        font-weight: 600;
    }}
    QLabel#Dim {{
        color: {palette.text_dim};
    }}
    QLabel#Faint {{
        color: {palette.text_faint};
        font-size: {base_size - 1}pt;
    }}
    QLabel#Mono {{
        font-family: "{mono}";
    }}
    QPushButton {{
        background: {palette.surface_raised};
        border: 1px solid {palette.border};
        border-radius: 6px;
        padding: 6px 14px;
        color: {palette.text};
    }}
    QPushButton:hover {{
        background: {palette.border};
        border-color: {palette.border_strong};
    }}
    QPushButton:pressed {{
        background: {palette.surface_sunken};
    }}
    QPushButton:disabled {{
        color: {palette.text_faint};
        background: {palette.surface};
        border-color: {palette.surface_raised};
    }}
    QPushButton#Primary {{
        background: {palette.accent};
        border-color: {palette.accent};
        color: {palette.accent_text};
        font-weight: 600;
    }}
    QPushButton#Primary:hover {{
        background: {palette.accent_dim};
    }}
    QPushButton#Big {{
        font-size: {base_size + 2}pt;
        padding: 18px 20px;
        text-align: left;
    }}
    /* 작은 정사각 버튼. 기본 좌우 여백(14px)을 그대로 두면 22px 버튼 안에서
       글자가 밀려나 아무것도 안 보인다. */
    QPushButton#Toggle, QPushButton#Mute, QPushButton#Solo {{
        padding: 0;
        font-size: {base_size - 1}pt;
        font-weight: 600;
        color: {palette.text_dim};
    }}
    QPushButton#Toggle:checked {{
        background: {palette.accent};
        border-color: {palette.accent};
        color: {palette.accent_text};
    }}
    QPushButton#Mute:checked {{
        background: {palette.warning};
        border-color: {palette.warning};
        color: #1b1d21;
    }}
    QPushButton#Solo:checked {{
        background: {palette.success};
        border-color: {palette.success};
        color: #1b1d21;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {palette.surface_sunken};
        border: 1px solid {palette.border};
        border-radius: 5px;
        padding: 5px 8px;
        selection-background-color: {palette.accent};
        selection-color: {palette.accent_text};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {palette.accent};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {palette.surface_raised};
        border: 1px solid {palette.border};
        selection-background-color: {palette.accent};
    }}
    QSlider::groove:horizontal {{
        background: {palette.surface_sunken};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {palette.text};
        width: 12px;
        margin: -5px 0;
        border-radius: 6px;
    }}
    QSlider::sub-page:horizontal {{
        background: {palette.accent};
        border-radius: 2px;
    }}
    QSlider::groove:vertical {{
        background: {palette.surface_sunken};
        width: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:vertical {{
        background: {palette.text};
        height: 12px;
        margin: 0 -5px;
        border-radius: 6px;
    }}
    QScrollBar:vertical {{
        background: transparent; width: 11px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {palette.border}; border-radius: 5px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {palette.border_strong}; }}
    QScrollBar:horizontal {{
        background: transparent; height: 11px; margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {palette.border}; border-radius: 5px; min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {palette.border_strong}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    QSplitter::handle {{ background: {palette.border}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    QMenuBar {{ background: {palette.surface}; }}
    QMenuBar::item:selected {{ background: {palette.border}; }}
    QMenu {{
        background: {palette.surface_raised};
        border: 1px solid {palette.border};
        padding: 4px;
    }}
    QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {palette.accent}; color: {palette.accent_text}; }}
    QMenu::separator {{ height: 1px; background: {palette.border}; margin: 4px 8px; }}
    QToolTip {{
        background: {palette.surface_raised};
        color: {palette.text};
        border: 1px solid {palette.border};
        padding: 4px 8px;
    }}
    QProgressBar {{
        background: {palette.surface_sunken};
        border: none; border-radius: 3px; height: 6px; text-align: center;
    }}
    QProgressBar::chunk {{ background: {palette.accent}; border-radius: 3px; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px; height: 15px;
        border: 1px solid {palette.border_strong};
        border-radius: 3px;
        background: {palette.surface_sunken};
    }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {palette.accent};
        border-color: {palette.accent};
    }}
    QGroupBox {{
        border: 1px solid {palette.border};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: {palette.text_dim};
    }}
    QTabWidget::pane {{
        border: 1px solid {palette.border};
        border-radius: 6px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        padding: 7px 16px;
        border-bottom: 2px solid transparent;
        color: {palette.text_dim};
    }}
    QTabBar::tab:selected {{
        color: {palette.text};
        border-bottom-color: {palette.accent};
    }}
    QHeaderView::section {{
        background: {palette.surface};
        border: none;
        border-bottom: 1px solid {palette.border};
        padding: 6px;
        color: {palette.text_dim};
    }}
    QTableWidget, QTreeWidget, QListWidget {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
        alternate-background-color: {palette.surface_raised};
    }}
    QStatusBar {{ background: {palette.surface}; color: {palette.text_dim}; }}
    """


def track_color(kind: str, palette: Palette = DARK) -> str:
    return {
        "vocal": palette.vocal, "instrument": palette.instrument,
        "drum": palette.drum, "audio": palette.audio, "bus": palette.bus,
    }.get(kind, palette.instrument)


def meter_color(db: float, palette: Palette = DARK) -> str:
    """음량에 따른 미터 색. 0dB 가까우면 빨강."""
    if db > -1.0:
        return palette.meter_high
    if db > -6.0:
        return palette.meter_mid
    return palette.meter_low
