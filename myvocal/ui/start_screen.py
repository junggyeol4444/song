"""
시작 화면 — 2번.

프로그램을 켜면 처음 보는 화면이다. 여기서 무엇을 할지 고른다.

버튼만 나열하지 않는다. 각 버튼 아래에 그게 무엇을 하는지 한 줄로 적는다.
아이콘과 이름만 있으면 처음 쓰는 사람은 '내 AI 가수 만들기' 와 'AI로 노래
만들기' 가 어떻게 다른지 알 수 없다.

아직 안 만들어진 기능은 숨기지 않고, 눌리지 않게 하고 이유를 적는다.
숨기면 사용자는 그 기능이 없다고 생각하고, 눌리는데 아무 일도 안 일어나면
고장난 줄 안다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import DARK, Palette


@dataclass(frozen=True, slots=True)
class MenuEntry:
    """시작 화면의 항목 하나."""

    key: str
    icon: str
    title: str
    description: str
    available: bool = True
    unavailable_reason: str = ""


MENU: tuple[MenuEntry, ...] = (
    MenuEntry(
        "auto_song", "✨", "AI로 노래 만들기",
        "설명 한 줄만 쓰면 작사·작곡·편곡까지 한 번에 만듭니다.",
    ),
    MenuEntry(
        "voice_train", "🎤", "내 AI 가수 만들기",
        "내 목소리를 녹음해서 학습시킵니다. 그 목소리로 노래하게 됩니다.",
        available=False,
        unavailable_reason="목소리 학습은 아직 만들고 있습니다.",
    ),
    MenuEntry(
        "compose", "🎹", "직접 작곡하기",
        "빈 프로젝트에서 시작합니다. AI 도움은 필요할 때만 받습니다.",
    ),
    MenuEntry(
        "music_video", "🎬", "뮤직비디오 만들기",
        "완성된 곡에 맞춰 장면을 짜고 영상을 만듭니다.",
        available=False,
        unavailable_reason="영상 제작은 아직 만들고 있습니다.",
    ),
    MenuEntry(
        "register", "👤", "사람 / 캐릭터 등록",
        "뮤직비디오에 나올 사람이나 캐릭터를 등록합니다.",
        available=False,
        unavailable_reason="캐릭터 등록은 아직 만들고 있습니다.",
    ),
    MenuEntry(
        "character_pv", "🕺", "캐릭터 PV 만들기",
        "등록한 캐릭터가 노래하고 움직이는 영상을 만듭니다.",
        available=False,
        unavailable_reason="캐릭터 영상은 아직 만들고 있습니다.",
    ),
    MenuEntry(
        "learning", "📚", "AI 학습실",
        "내 음악, 내 목소리, 내 캐릭터를 학습시켜 스타일을 만듭니다.",
        available=False,
        unavailable_reason="학습실은 아직 만들고 있습니다.",
    ),
    MenuEntry(
        "open", "📂", "프로젝트 열기",
        "저장해 둔 작업을 이어서 합니다.",
    ),
)


class MenuButton(QtWidgets.QPushButton):
    """아이콘 + 제목 + 설명이 함께 있는 큰 버튼."""

    def __init__(self, entry: MenuEntry, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self._palette = palette
        self.setMinimumHeight(74)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor
                       if entry.available else QtCore.Qt.CursorShape.ForbiddenCursor)
        self.setEnabled(entry.available)
        if not entry.available:
            self.setToolTip(entry.unavailable_reason)
        else:
            self.setToolTip(entry.description)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        enabled = self.isEnabled()

        # 바탕
        if not enabled:
            background = QtGui.QColor(self._palette.surface)
            border = QtGui.QColor(self._palette.surface_raised)
        elif self.isDown():
            background = QtGui.QColor(self._palette.surface_sunken)
            border = QtGui.QColor(self._palette.accent)
        elif self.underMouse():
            background = QtGui.QColor(self._palette.border)
            border = QtGui.QColor(self._palette.accent)
        else:
            background = QtGui.QColor(self._palette.surface_raised)
            border = QtGui.QColor(self._palette.border)
        painter.setBrush(background)
        painter.setPen(QtGui.QPen(border, 1))
        painter.drawRoundedRect(QtCore.QRectF(rect), 9, 9)

        # 아이콘
        icon_font = QtGui.QFont(self.font())
        icon_font.setPointSize(max(16, self.font().pointSize() + 12))
        painter.setFont(icon_font)
        painter.setPen(QtGui.QColor(
            self._palette.text if enabled else self._palette.text_faint
        ))
        icon_rect = QtCore.QRect(rect.left() + 14, rect.top(), 48, rect.height())
        painter.drawText(icon_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.entry.icon)

        # 제목
        title_font = QtGui.QFont(self.font())
        title_font.setPointSize(self.font().pointSize() + 2)
        title_font.setWeight(QtGui.QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(
            self._palette.text if enabled else self._palette.text_faint
        ))
        text_left = rect.left() + 70
        painter.drawText(
            QtCore.QRect(text_left, rect.top() + 14, rect.width() - text_left - 14, 24),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            self.entry.title,
        )

        # 설명
        small_font = QtGui.QFont(self.font())
        small_font.setPointSize(max(7, self.font().pointSize() - 1))
        painter.setFont(small_font)
        painter.setPen(QtGui.QColor(
            self._palette.text_dim if enabled else self._palette.text_faint
        ))
        description = self.entry.description if enabled else self.entry.unavailable_reason
        painter.drawText(
            QtCore.QRect(text_left, rect.top() + 38, rect.width() - text_left - 14, 22),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            description,
        )
        painter.end()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


class RecentList(QtWidgets.QWidget):
    """최근 연 프로젝트 목록."""

    opened = QtCore.Signal(Path)

    def __init__(self, entries: list[tuple[Path, str, str]],
                 palette: Palette = DARK, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        heading = QtWidgets.QLabel("최근 작업")
        heading.setObjectName("Heading")
        layout.addWidget(heading)

        if not entries:
            empty = QtWidgets.QLabel("아직 만든 프로젝트가 없습니다.")
            empty.setObjectName("Dim")
            layout.addWidget(empty)
            layout.addStretch(1)
            return

        for path, title, subtitle in entries:
            button = QtWidgets.QPushButton()
            button.setMinimumHeight(48)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            inner = QtWidgets.QVBoxLayout(button)
            inner.setContentsMargins(12, 6, 12, 6)
            inner.setSpacing(1)
            name = QtWidgets.QLabel(title)
            name.setStyleSheet("background: transparent;")
            detail = QtWidgets.QLabel(subtitle)
            detail.setObjectName("Faint")
            detail.setStyleSheet(f"background: transparent; color: {palette.text_faint};")
            inner.addWidget(name)
            inner.addWidget(detail)
            button.clicked.connect(lambda _=False, p=path: self.opened.emit(p))
            button.setToolTip(str(path))
            layout.addWidget(button)
        layout.addStretch(1)


class StartScreen(QtWidgets.QWidget):
    """시작 화면 전체."""

    chosen = QtCore.Signal(str)          # MenuEntry.key
    open_project = QtCore.Signal(Path)

    def __init__(self, recent: list[tuple[Path, str, str]] | None = None,
                 palette: Palette = DARK, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(40, 32, 40, 28)
        outer.setSpacing(0)

        # --- 머리말 ---
        title = QtWidgets.QLabel("MYVOCAL Studio")
        title.setObjectName("Title")
        outer.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "내 목소리와 내 캐릭터로 노래와 뮤직비디오를 만드는 곳"
        )
        subtitle.setObjectName("Dim")
        outer.addWidget(subtitle)
        outer.addSpacing(24)

        # --- 본문: 왼쪽 메뉴 / 오른쪽 최근 작업 ---
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(28)

        # 항목이 8개라 세로로 800픽셀 가까이 된다. 노트북 화면(높이 768)에서는
        # 아래가 잘리므로 스크롤할 수 있게 감싼다. 창을 크게 쓰면 스크롤바는
        # 저절로 사라진다.
        menu_holder = QtWidgets.QWidget()
        menu_area = QtWidgets.QVBoxLayout(menu_holder)
        menu_area.setContentsMargins(0, 0, 6, 0)
        menu_area.setSpacing(9)
        self._buttons: dict[str, MenuButton] = {}
        for entry in MENU:
            button = MenuButton(entry, palette)
            button.clicked.connect(lambda _=False, k=entry.key: self.chosen.emit(k))
            menu_area.addWidget(button)
            self._buttons[entry.key] = button
        menu_area.addStretch(1)

        menu_scroll = QtWidgets.QScrollArea()
        menu_scroll.setWidgetResizable(True)
        menu_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        menu_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        menu_scroll.setWidget(menu_holder)
        menu_scroll.setStyleSheet("background: transparent;")
        columns.addWidget(menu_scroll, 3)

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(12)
        self.recent_list = RecentList(recent or [], palette)
        self.recent_list.opened.connect(self.open_project)
        side.addWidget(self.recent_list, 1)
        side.addWidget(self._status_card(palette))
        columns.addLayout(side, 2)

        outer.addLayout(columns, 1)

        # --- 꼬리말 ---
        outer.addSpacing(16)
        footer = QtWidgets.QLabel(
            "만들어진 곡은 트랙과 음으로 남습니다. 완성된 음원 한 덩어리가 아니라, "
            "언제든 다시 열어 한 음씩 고칠 수 있습니다."
        )
        footer.setObjectName("Faint")
        footer.setWordWrap(True)
        outer.addWidget(footer)

    def _status_card(self, palette: Palette) -> QtWidgets.QWidget:
        """지금 무엇이 되고 무엇이 아직 안 되는지 솔직하게 적는다.

        오른쪽이 비어 있으면 화면이 허전하기도 하고, 무엇보다 사용자가
        '이 프로그램이 뭘 할 수 있는지' 를 눌러보기 전에는 알 수 없다.
        """
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        heading = QtWidgets.QLabel("지금 되는 것")
        heading.setObjectName("Heading")
        heading.setStyleSheet("background: transparent;")
        layout.addWidget(heading)

        done = QtWidgets.QLabel(
            "· 설명 한 줄로 작곡·편곡 (22개 장르, 혼합 가능)\n"
            "· 15종 악기와 22종 드럼 합성\n"
            "· 피아노롤에서 한 음씩 편집\n"
            "· 믹싱·마스터링 (스트리밍 기준 음량 맞춤)\n"
            "· WAV / FLAC / MP3 / OGG / MIDI 내보내기"
        )
        done.setWordWrap(True)
        done.setStyleSheet(f"background: transparent; color: {palette.text_dim};")
        layout.addWidget(done)

        pending_heading = QtWidgets.QLabel("아직 안 되는 것")
        pending_heading.setObjectName("Heading")
        pending_heading.setStyleSheet("background: transparent;")
        layout.addWidget(pending_heading)

        pending = QtWidgets.QLabel(
            "· 작사\n"
            "· 내 목소리 학습과 노래 합성\n"
            "· 캐릭터 등록과 뮤직비디오"
        )
        pending.setWordWrap(True)
        pending.setStyleSheet(f"background: transparent; color: {palette.text_faint};")
        layout.addWidget(pending)
        return card

    def set_available(self, key: str, available: bool, reason: str = "") -> None:
        """기능이 준비되면 켠다."""
        button = self._buttons.get(key)
        if button is None:
            raise KeyError(f"시작 화면에 없는 항목입니다: {key!r}")
        button.setEnabled(available)
        button.setToolTip(button.entry.description if available else reason)
        button.update()


def find_recent_projects(folder: Path, limit: int = 6) -> list[tuple[Path, str, str]]:
    """폴더에서 최근 프로젝트를 찾는다. 파일을 통째로 읽지 않고 머리말만 본다."""
    if not folder.exists():
        return []
    found: list[tuple[float, Path, str, str]] = []
    for path in folder.glob("**/*.mvp"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("format") != "myvocal-project":
            continue
        meta = data.get("meta", {})
        title = meta.get("title", path.stem)
        tracks = len(data.get("tracks", []))
        modified = path.stat().st_mtime
        from datetime import datetime

        when = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        subtitle = f"{when} · 트랙 {tracks}개"
        found.append((modified, path, title, subtitle))
    found.sort(key=lambda item: item[0], reverse=True)
    return [(path, title, subtitle) for _, path, title, subtitle in found[:limit]]
