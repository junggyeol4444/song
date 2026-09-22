"""
트랙 목록 — 왼쪽에 세로로 쌓인 트랙 머리들.

각 줄에 이름, 악기, 음량, 좌우 위치, 음소거, 솔로가 있다.
피아노롤과 높이를 맞춰야 하므로 트랙 하나의 높이를 공유한다.
"""

from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.tracks import Track, TrackList
from ..music.instruments import available_instruments, create_instrument
from .theme import DARK, Palette, track_color


class TrackHeader(QtWidgets.QFrame):
    """트랙 한 줄."""

    selected = QtCore.Signal(object)                     # Track
    property_changed = QtCore.Signal(object, str, object)  # Track, 항목, 값
    remove_requested = QtCore.Signal(object)

    HEIGHT = 62

    def __init__(self, track: Track, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.track = track
        self._palette = palette
        self._active = False
        self.setFixedHeight(self.HEIGHT)
        self.setObjectName("TrackHeader")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(3)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        self.name_label = QtWidgets.QLabel(track.name)
        self.name_label.setStyleSheet("background: transparent; font-weight: 600;")
        top.addWidget(self.name_label, 1)

        self.mute_button = QtWidgets.QPushButton("M")
        self.solo_button = QtWidgets.QPushButton("S")
        for button, name, checked, tip in (
            (self.mute_button, "Mute", track.muted, "음소거"),
            (self.solo_button, "Solo", track.soloed, "이 트랙만 듣기"),
        ):
            button.setObjectName(name)
            button.setProperty("class", "Toggle")
            button.setCheckable(True)
            button.setChecked(checked)
            button.setFixedSize(24, 21)
            button.setToolTip(tip)
            top.addWidget(button)
        layout.addLayout(top)

        self.instrument_label = QtWidgets.QLabel(self._instrument_text())
        self.instrument_label.setObjectName("Faint")
        self.instrument_label.setStyleSheet(
            f"background: transparent; color: {palette.text_faint};"
        )
        layout.addWidget(self.instrument_label)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(6)
        self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volume_slider.setRange(-60, 12)
        self.volume_slider.setValue(int(round(track.volume_db)))
        self.volume_slider.setFixedHeight(16)
        self.volume_slider.setToolTip("음량")
        bottom.addWidget(self.volume_slider, 3)
        self.volume_label = QtWidgets.QLabel(f"{track.volume_db:+.0f}")
        self.volume_label.setObjectName("Faint")
        self.volume_label.setFixedWidth(28)
        self.volume_label.setStyleSheet(
            f"background: transparent; color: {palette.text_faint};"
        )
        bottom.addWidget(self.volume_label)
        layout.addLayout(bottom)

        self.mute_button.toggled.connect(
            lambda value: self.property_changed.emit(self.track, "muted", value)
        )
        self.solo_button.toggled.connect(
            lambda value: self.property_changed.emit(self.track, "soloed", value)
        )
        self.volume_slider.valueChanged.connect(self._on_volume)

    def _instrument_text(self) -> str:
        try:
            info = create_instrument(self.track.instrument).info
            name = info.display_name
        except Exception:
            name = self.track.instrument
        return f"{name} · 음 {len(self.track.notes)}개"

    def _on_volume(self, value: int) -> None:
        self.volume_label.setText(f"{value:+d}")
        self.property_changed.emit(self.track, "volume_db", float(value))

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.update()

    def refresh(self) -> None:
        self.name_label.setText(self.track.name)
        self.instrument_label.setText(self._instrument_text())
        self.mute_button.setChecked(self.track.muted)
        self.solo_button.setChecked(self.track.soloed)
        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(int(round(self.track.volume_db)))
        self.volume_slider.blockSignals(False)
        self.volume_label.setText(f"{self.track.volume_db:+.0f}")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        background = QtGui.QColor(
            self._palette.surface_raised if self._active else self._palette.surface
        )
        painter.fillRect(self.rect(), background)
        # 왼쪽에 트랙 색 띠
        painter.fillRect(0, 0, 4, self.height(), QtGui.QColor(self.track.color))
        painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.border), 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        if self._active:
            painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.accent), 1))
            painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.selected.emit(self.track)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        rename = menu.addAction("이름 바꾸기")
        instrument_menu = menu.addMenu("악기 바꾸기")
        actions = {}
        for name in available_instruments():
            try:
                label = create_instrument(name).info.display_name
            except Exception:
                label = name
            actions[instrument_menu.addAction(label)] = name
        menu.addSeparator()
        delete = menu.addAction("트랙 삭제")
        chosen = menu.exec(event.globalPos())
        if chosen is None:
            return
        if chosen is rename:
            text, ok = QtWidgets.QInputDialog.getText(
                self, "트랙 이름", "새 이름:", text=self.track.name
            )
            if ok and text.strip():
                self.property_changed.emit(self.track, "name", text.strip())
        elif chosen is delete:
            self.remove_requested.emit(self.track)
        elif chosen in actions:
            self.property_changed.emit(self.track, "instrument", actions[chosen])


class TrackPanel(QtWidgets.QWidget):
    """트랙 머리들을 세로로 쌓은 패널."""

    track_selected = QtCore.Signal(object)
    property_changed = QtCore.Signal(object, str, object)
    remove_requested = QtCore.Signal(object)
    add_requested = QtCore.Signal()

    def __init__(self, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._headers: list[TrackHeader] = []
        self.active_track: Track | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._container = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, 1)

        add_button = QtWidgets.QPushButton("+ 트랙 추가")
        add_button.clicked.connect(self.add_requested)
        outer.addWidget(add_button)

    def set_tracks(self, tracks: TrackList) -> None:
        for header in self._headers:
            header.setParent(None)
            header.deleteLater()
        self._headers.clear()
        for track in tracks:
            header = TrackHeader(track, self._palette)
            header.selected.connect(self._on_selected)
            header.property_changed.connect(self.property_changed)
            header.remove_requested.connect(self.remove_requested)
            self._layout.insertWidget(self._layout.count() - 1, header)
            self._headers.append(header)
        if self._headers:
            self._on_selected(self._headers[0].track)

    def _on_selected(self, track: Track) -> None:
        self.active_track = track
        for header in self._headers:
            header.set_active(header.track is track)
        self.track_selected.emit(track)

    def refresh(self) -> None:
        for header in self._headers:
            header.refresh()

    def select(self, track: Track) -> None:
        self._on_selected(track)
