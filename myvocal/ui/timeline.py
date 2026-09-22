"""
타임라인 자 — 마디 번호와 곡 구조를 보여주는 위쪽 띠.

여기서 재생 위치를 클릭해 옮긴다. 구간(Verse/Chorus)도 여기 표시되므로
사용자가 지금 곡의 어디를 보고 있는지 알 수 있다.
"""

from __future__ import annotations

from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.units import PPQ, MeterMap, TempoMap
from ..music.structure import SECTION_COLORS, Section, SongStructure
from .piano_roll import ViewState
from .theme import DARK, Palette


class TimelineRuler(QtWidgets.QWidget):
    """마디 눈금과 구간 띠."""

    position_changed = QtCore.Signal(int)        # tick
    section_clicked = QtCore.Signal(object)      # Section
    loop_changed = QtCore.Signal(object)         # (시작 tick, 끝 tick) 또는 None

    HEIGHT = 46

    def __init__(self, view: ViewState, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = view
        self._palette = palette
        self.meter: MeterMap | None = None
        self.tempo: TempoMap | None = None
        self.structure: SongStructure | None = None
        self.playhead_tick: int = 0
        self.loop: tuple[int, int] | None = None
        self._dragging = False
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_project_data(self, meter: MeterMap, tempo: TempoMap,
                         structure: SongStructure) -> None:
        self.meter = meter
        self.tempo = tempo
        self.structure = structure
        self.update()

    def set_playhead(self, tick: int) -> None:
        if tick != self.playhead_tick:
            self.playhead_tick = tick
            self.update()

    def tick_to_x(self, tick: int | float) -> float:
        return (tick - self.view.scroll_tick) * self.view.pixels_per_tick

    def x_to_tick(self, x: float) -> int:
        return max(0, int(self.view.scroll_tick + x / max(1e-9, self.view.pixels_per_tick)))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        palette = self._palette
        painter.fillRect(self.rect(), QtGui.QColor(palette.surface))

        section_height = 18
        ruler_top = section_height

        # --- 구간 띠 ---
        if self.structure is not None and self.meter is not None:
            small = QtGui.QFont(self.font())
            small.setPointSize(max(6, self.font().pointSize() - 2))
            painter.setFont(small)
            for section in self.structure:
                start = self.meter.bar_to_tick(section.start_bar)
                end = self.meter.bar_to_tick(section.end_bar)
                x1, x2 = self.tick_to_x(start), self.tick_to_x(end)
                if x2 < 0 or x1 > self.width():
                    continue
                color = QtGui.QColor(SECTION_COLORS.get(section.kind, "#888888"))
                rectangle = QtCore.QRectF(x1, 1, max(2.0, x2 - x1 - 1), section_height - 2)
                painter.fillRect(rectangle, color)
                if rectangle.width() > 34:
                    painter.setPen(QtGui.QColor("#12141a"))
                    painter.drawText(
                        rectangle.adjusted(4, 0, -3, 0),
                        QtCore.Qt.AlignmentFlag.AlignLeft
                        | QtCore.Qt.AlignmentFlag.AlignVCenter,
                        section.label,
                    )

        # --- 반복 구간 ---
        if self.loop is not None:
            x1, x2 = self.tick_to_x(self.loop[0]), self.tick_to_x(self.loop[1])
            color = QtGui.QColor(palette.accent)
            color.setAlpha(55)
            painter.fillRect(QtCore.QRectF(x1, ruler_top, x2 - x1,
                                           self.height() - ruler_top), color)

        # --- 마디 눈금 ---
        if self.meter is not None:
            painter.setFont(self.font())
            bar_ticks = self.meter.signature_at_tick(self.view.scroll_tick).ticks_per_bar
            # 확대율에 따라 몇 마디마다 숫자를 적을지 정한다
            spacing = bar_ticks * self.view.pixels_per_tick
            label_every = 1
            for candidate in (1, 2, 4, 8, 16, 32):
                if spacing * candidate >= 46:
                    label_every = candidate
                    break
            start_bar, _ = self.meter.tick_to_bar_beat(self.view.scroll_tick)
            end_tick = self.x_to_tick(self.width())
            end_bar, _ = self.meter.tick_to_bar_beat(max(0, end_tick))
            for bar in range(max(1, start_bar), end_bar + 2):
                tick = self.meter.bar_to_tick(bar)
                x = self.tick_to_x(tick)
                if x < -40 or x > self.width() + 40:
                    continue
                major = (bar - 1) % label_every == 0
                painter.setPen(QtGui.QPen(
                    QtGui.QColor(palette.border_strong if major else palette.border), 1
                ))
                painter.drawLine(int(x), ruler_top + (4 if major else 10),
                                 int(x), self.height())
                if major:
                    painter.setPen(QtGui.QColor(palette.text_dim))
                    painter.drawText(
                        QtCore.QRectF(x + 3, ruler_top + 1, 50, 16),
                        QtCore.Qt.AlignmentFlag.AlignLeft
                        | QtCore.Qt.AlignmentFlag.AlignVCenter,
                        str(bar),
                    )

        # --- 재생 위치 ---
        x = self.tick_to_x(self.playhead_tick)
        if -6 <= x <= self.width() + 6:
            painter.setPen(QtGui.QPen(QtGui.QColor(palette.playhead), 2))
            painter.drawLine(int(x), ruler_top, int(x), self.height())
            # 삼각형 손잡이
            painter.setBrush(QtGui.QColor(palette.playhead))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            handle = QtGui.QPolygonF([
                QtCore.QPointF(x - 5, ruler_top),
                QtCore.QPointF(x + 5, ruler_top),
                QtCore.QPointF(x, ruler_top + 7),
            ])
            painter.drawPolygon(handle)

        painter.setPen(QtGui.QPen(QtGui.QColor(palette.border), 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.position().y() < 18 and self.structure is not None and self.meter is not None:
            tick = self.x_to_tick(event.position().x())
            section = self.structure.at_tick(tick, self.meter)
            if section is not None:
                self.section_clicked.emit(section)
                return
        self._dragging = True
        self._emit_position(event.position().x())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            self._emit_position(event.position().x())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._dragging = False

    def _emit_position(self, x: float) -> None:
        tick = self.x_to_tick(x)
        if self.view.snap_enabled:
            tick = self.view.snap(tick)
        self.set_playhead(tick)
        self.position_changed.emit(tick)
