"""
피아노롤 — 음을 눈으로 보고 손으로 고치는 곳.

5번이 "각 요소가 실제 편집 가능한 트랙으로 생성된다" 고 한 것의 실체다.
AI 가 만든 결과가 여기 그대로 나오고, 사용자가 한 음씩 끌어 옮길 수 있다.

직접 그린다. QGraphicsView 에 음마다 항목을 하나씩 만들면 음이 수천 개인
곡에서 스크롤이 버벅인다. 화면에 보이는 것만 그리면 음이 몇 개든 일정하다.

좌표
    가로  tick. 화면 x = (tick - 스크롤) * 확대율
    세로  MIDI 음높이. 위가 높은 음이다. 아래가 높으면 사람이 헷갈린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.notes import Note, NoteList
from ..core.tracks import Track
from ..core.units import PPQ, MeterMap, TempoMap
from ..music.theory import Pitch
from .theme import DARK, Palette

# 검은건반 여부
BLACK_KEYS: frozenset[int] = frozenset({1, 3, 6, 8, 10})

# 격자 단위 (tick). 화면을 확대하면 더 잘게 보여준다.
GRID_STEPS: tuple[int, ...] = (
    PPQ * 4, PPQ * 2, PPQ, PPQ // 2, PPQ // 4, PPQ // 8,
)


@dataclass(slots=True)
class ViewState:
    """보이는 범위와 확대 정도."""

    scroll_tick: int = 0
    scroll_pitch: int = 48          # 화면 맨 아래 음높이
    pixels_per_tick: float = 0.06   # 확대율. 0.06 이면 한 마디(3840tick)가 230픽셀
    key_height: int = 14
    snap_ticks: int = PPQ // 4      # 자석. 이 단위로 붙는다.
    snap_enabled: bool = True

    # 긴 곡도 한 화면에 들어와야 한다. 10분짜리 곡(120BPM 기준 약 115만 tick)을
    # 1000픽셀 창에 넣으려면 tick 당 0.00087 픽셀이 필요하다. 한계를 그보다
    # 낮게 잡지 않으면 전체 보기가 화면을 넘어간다.
    MIN_ZOOM: float = 0.0004
    MAX_ZOOM: float = 1.2
    MIN_KEY_HEIGHT: int = 5
    MAX_KEY_HEIGHT: int = 40

    def clamp(self) -> None:
        self.pixels_per_tick = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.pixels_per_tick))
        self.key_height = max(self.MIN_KEY_HEIGHT, min(self.MAX_KEY_HEIGHT, self.key_height))
        self.scroll_tick = max(0, self.scroll_tick)
        self.scroll_pitch = max(0, min(108, self.scroll_pitch))

    def snap(self, tick: int) -> int:
        if not self.snap_enabled or self.snap_ticks <= 0:
            return max(0, tick)
        return max(0, int(round(tick / self.snap_ticks)) * self.snap_ticks)

    def visible_grid_step(self) -> int:
        """지금 확대율에서 보기 좋은 격자 간격. 너무 촘촘하면 안 그린다."""
        for step in GRID_STEPS:
            if step * self.pixels_per_tick >= 7.0:
                return step
        return GRID_STEPS[0]


class PianoKeyboard(QtWidgets.QWidget):
    """왼쪽 건반. 어느 줄이 무슨 음인지 알려준다."""

    pitch_clicked = QtCore.Signal(int)

    def __init__(self, view: ViewState, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = view
        self._palette = palette
        self._highlight: set[int] = set()
        self.setFixedWidth(58)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_highlight(self, pitches: Iterable[int]) -> None:
        """지금 울리고 있는 음을 표시한다."""
        new = set(pitches)
        if new != self._highlight:
            self._highlight = new
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        height = self.height()
        key_height = self.view.key_height
        bottom_pitch = self.view.scroll_pitch
        count = height // key_height + 2

        painter.fillRect(self.rect(), QtGui.QColor(self._palette.surface_sunken))
        label_font = QtGui.QFont(self.font())
        label_font.setPointSize(max(6, self.font().pointSize() - 2))
        painter.setFont(label_font)

        for index in range(count):
            midi = bottom_pitch + index
            if not (0 <= midi <= 127):
                continue
            y = height - (index + 1) * key_height
            rectangle = QtCore.QRect(0, y, self.width(), key_height)
            is_black = (midi % 12) in BLACK_KEYS
            if midi in self._highlight:
                color = QtGui.QColor(self._palette.accent)
            elif is_black:
                color = QtGui.QColor("#1a1c1f")
            else:
                color = QtGui.QColor("#d8dade")
            painter.fillRect(rectangle, color)
            painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.border), 1))
            painter.drawLine(0, y, self.width(), y)

            # C 음마다 이름을 적는다. 전부 적으면 빽빽해서 못 읽는다.
            if midi % 12 == 0 and key_height >= 9:
                painter.setPen(QtGui.QColor("#4a4f55"))
                painter.drawText(
                    rectangle.adjusted(4, 0, -4, 0),
                    QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    f"C{midi // 12 - 1}",
                )
        painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.border_strong), 1))
        painter.drawLine(self.width() - 1, 0, self.width() - 1, height)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        midi = self.view.scroll_pitch + (self.height() - event.position().y()) // self.view.key_height
        if 0 <= midi <= 127:
            self.pitch_clicked.emit(int(midi))


class PianoRoll(QtWidgets.QWidget):
    """음 편집 화면."""

    notes_changed = QtCore.Signal()                 # 편집이 끝났다 (기록에 남길 시점)
    selection_changed = QtCore.Signal()
    position_clicked = QtCore.Signal(int)           # tick
    edit_requested = QtCore.Signal(str, object)     # 명령 종류, 내용

    def __init__(self, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.view = ViewState()
        self.track: Track | None = None
        self.meter: MeterMap | None = None
        self.tempo: TempoMap | None = None
        self.selected: set[int] = set()              # note_id
        self.playhead_tick: int | None = None
        self.section_marks: list[tuple[int, int, str, str]] = []   # 시작, 끝, 이름, 색

        self._drag_mode = ""          # "", "move", "resize", "select", "pan"
        self._drag_start: QtCore.QPoint | None = None
        self._drag_origin: dict[int, tuple[int, int, int]] = {}    # id -> (midi,start,len)
        self._drag_preview: dict[int, tuple[int, int, int]] = {}
        self._select_rect: QtCore.QRect | None = None
        self._hover_note: int | None = None
        # 트랙을 바꾸면 음이 있는 자리로 화면을 맞춘다.
        # 다만 창 크기가 정해지기 전에 맞추면 엉뚱한 곳을 본다. 한 번만 맞추고
        # 끝내면 그 엉뚱한 자리에 그대로 머문다. 그래서 '사용자가 직접
        # 움직이기 전까지' 는 크기가 바뀔 때마다 다시 맞춘다.
        self._user_scrolled = False

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

    # ---------------------------------------------------------------- 자료

    def set_track(self, track: Track | None, meter: MeterMap | None = None,
                  tempo: TempoMap | None = None) -> None:
        self.track = track
        if meter is not None:
            self.meter = meter
        if tempo is not None:
            self.tempo = tempo
        self.selected.clear()
        self._drag_preview.clear()
        self._user_scrolled = False
        self.center_on_content()
        self.update()

    def center_on_content(self) -> bool:
        """음이 있는 자리가 화면 가운데 오도록 세로 위치를 맞춘다.

        창 크기가 아직 안 정해졌으면(높이가 한 건반도 안 들어갈 만큼 작으면)
        아무것도 하지 않고 다음 기회를 기다린다.
        """
        if self.track is None or not self.track.notes:
            return False
        visible = self.height() // self.view.key_height
        if visible < 3:
            return False
        low, high = self.track.pitch_range() or (48, 72)
        span = high - low + 1
        # 음역이 화면보다 넓으면 세로로 줄여서 다 보이게 한다
        if span > visible and self.view.key_height > self.view.MIN_KEY_HEIGHT:
            wanted = max(self.view.MIN_KEY_HEIGHT,
                         min(self.view.key_height, self.height() // max(1, span + 2)))
            if wanted != self.view.key_height:
                self.view.key_height = wanted
                visible = self.height() // self.view.key_height
        middle = (low + high) // 2
        self.view.scroll_pitch = max(0, min(127 - visible, middle - visible // 2))
        return True

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._user_scrolled:
            self.center_on_content()
            self.update()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._user_scrolled:
            self.center_on_content()
            self.update()

    def set_sections(self, marks: Sequence[tuple[int, int, str, str]]) -> None:
        self.section_marks = list(marks)
        self.update()

    def set_playhead(self, tick: int | None) -> None:
        if tick != self.playhead_tick:
            self.playhead_tick = tick
            self.update()

    # ---------------------------------------------------------------- 좌표

    def tick_to_x(self, tick: int | float) -> float:
        return (tick - self.view.scroll_tick) * self.view.pixels_per_tick

    def x_to_tick(self, x: float) -> int:
        return int(self.view.scroll_tick + x / max(1e-9, self.view.pixels_per_tick))

    def pitch_to_y(self, midi: int) -> float:
        return self.height() - (midi - self.view.scroll_pitch + 1) * self.view.key_height

    def y_to_pitch(self, y: float) -> int:
        return int(self.view.scroll_pitch
                   + (self.height() - y) // self.view.key_height)

    def note_rect(self, note: Note, midi: int | None = None,
                  start: int | None = None, length: int | None = None) -> QtCore.QRectF:
        midi = note.midi if midi is None else midi
        start = note.start_tick if start is None else start
        length = note.duration_ticks if length is None else length
        x = self.tick_to_x(start)
        width = max(2.0, length * self.view.pixels_per_tick)
        y = self.pitch_to_y(midi)
        return QtCore.QRectF(x, y + 1, width, self.view.key_height - 2)

    def visible_tick_range(self) -> tuple[int, int]:
        return self.view.scroll_tick, self.x_to_tick(self.width()) + 1

    # ---------------------------------------------------------------- 그리기

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        palette = self._palette
        painter.fillRect(self.rect(), QtGui.QColor(palette.surface_sunken))

        self._draw_rows(painter)
        self._draw_sections(painter)
        self._draw_grid(painter)
        self._draw_notes(painter)
        self._draw_selection_rect(painter)
        self._draw_playhead(painter)
        painter.end()

    def _draw_rows(self, painter: QtGui.QPainter) -> None:
        """검은건반 줄을 조금 어둡게 칠해 음높이를 가늠하게 한다."""
        height = self.height()
        key_height = self.view.key_height
        dark = QtGui.QColor(self._palette.window)
        line = QtGui.QColor(self._palette.grid_beat)
        for index in range(height // key_height + 2):
            midi = self.view.scroll_pitch + index
            if not (0 <= midi <= 127):
                continue
            y = height - (index + 1) * key_height
            if (midi % 12) in BLACK_KEYS:
                painter.fillRect(0, int(y), self.width(), key_height, dark)
            if midi % 12 == 0:
                painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.grid_bar), 1))
                painter.drawLine(0, int(y) + key_height, self.width(), int(y) + key_height)
            else:
                painter.setPen(QtGui.QPen(line, 1))
                painter.drawLine(0, int(y) + key_height, self.width(), int(y) + key_height)

    def _draw_sections(self, painter: QtGui.QPainter) -> None:
        """구간을 옅은 띠로 표시한다. 어디가 후렴인지 한눈에 보인다."""
        if not self.section_marks:
            return
        for start, end, label, color in self.section_marks:
            x1 = self.tick_to_x(start)
            x2 = self.tick_to_x(end)
            if x2 < 0 or x1 > self.width():
                continue
            tint = QtGui.QColor(color)
            tint.setAlpha(22)
            painter.fillRect(QtCore.QRectF(x1, 0, x2 - x1, self.height()), tint)
            edge = QtGui.QColor(color)
            edge.setAlpha(90)
            painter.setPen(QtGui.QPen(edge, 1))
            painter.drawLine(int(x1), 0, int(x1), self.height())

    def _draw_grid(self, painter: QtGui.QPainter) -> None:
        step = self.view.visible_grid_step()
        start_tick, end_tick = self.visible_tick_range()
        bar_ticks = (self.meter.signature_at_tick(start_tick).ticks_per_bar
                     if self.meter is not None else PPQ * 4)
        beat_ticks = (self.meter.signature_at_tick(start_tick).ticks_per_beat
                      if self.meter is not None else PPQ)

        first = (start_tick // step) * step
        tick = first
        while tick <= end_tick:
            x = self.tick_to_x(tick)
            if tick % bar_ticks == 0:
                painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.grid_bar), 1))
            elif tick % beat_ticks == 0:
                painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.grid_beat), 1))
            else:
                faint = QtGui.QColor(self._palette.grid_beat)
                faint.setAlpha(110)
                painter.setPen(QtGui.QPen(faint, 1))
            painter.drawLine(int(x), 0, int(x), self.height())
            tick += step

    def _draw_notes(self, painter: QtGui.QPainter) -> None:
        if self.track is None:
            self._draw_empty_message(painter)
            return
        if not self.track.notes:
            self._draw_empty_message(painter, "이 트랙에는 아직 음이 없습니다.")
            return

        start_tick, end_tick = self.visible_tick_range()
        base = QtGui.QColor(self.track.color)
        show_labels = self.view.key_height >= 12 and self.view.pixels_per_tick > 0.03

        for note in self.track.notes:
            if note.end_tick < start_tick or note.start_tick > end_tick:
                continue
            preview = self._drag_preview.get(note.note_id)
            if preview is not None:
                midi, start, length = preview
            else:
                midi, start, length = note.midi, note.start_tick, note.duration_ticks
            rectangle = self.note_rect(note, midi, start, length)
            if rectangle.bottom() < 0 or rectangle.top() > self.height():
                continue

            selected = note.note_id in self.selected
            # 세기를 밝기로 보여준다. 숫자를 안 봐도 강약이 눈에 들어온다.
            strength = 0.35 + 0.65 * (note.velocity / 127.0)
            color = QtGui.QColor(base)
            color.setAlphaF(min(1.0, strength))
            painter.fillRect(rectangle, color)

            if selected:
                painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.text), 2))
            elif note.note_id == self._hover_note:
                painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.text_dim), 1))
            else:
                border = QtGui.QColor(base).darker(150)
                painter.setPen(QtGui.QPen(border, 1))
            painter.drawRect(rectangle.adjusted(0.5, 0.5, -0.5, -0.5))

            if show_labels and note.lyric and rectangle.width() > 18:
                painter.setPen(QtGui.QColor(self._palette.text))
                small = QtGui.QFont(self.font())
                small.setPointSize(max(6, self.font().pointSize() - 2))
                painter.setFont(small)
                painter.drawText(
                    rectangle.adjusted(3, 0, -2, 0),
                    QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    note.lyric,
                )

    def _draw_empty_message(self, painter: QtGui.QPainter, text: str = "") -> None:
        painter.setPen(QtGui.QColor(self._palette.text_faint))
        painter.drawText(
            self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
            text or "왼쪽에서 트랙을 고르세요.",
        )

    def _draw_selection_rect(self, painter: QtGui.QPainter) -> None:
        if self._select_rect is None:
            return
        color = QtGui.QColor(self._palette.accent)
        color.setAlpha(40)
        painter.fillRect(self._select_rect, color)
        painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.accent), 1))
        painter.drawRect(self._select_rect)

    def _draw_playhead(self, painter: QtGui.QPainter) -> None:
        if self.playhead_tick is None:
            return
        x = self.tick_to_x(self.playhead_tick)
        if -2 <= x <= self.width() + 2:
            painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.playhead), 2))
            painter.drawLine(int(x), 0, int(x), self.height())

    # ---------------------------------------------------------------- 입력

    def _note_at(self, position: QtCore.QPointF) -> Note | None:
        if self.track is None:
            return None
        midi = self.y_to_pitch(position.y())
        tick = self.x_to_tick(position.x())
        # 뒤에서부터 본다. 겹쳐 있으면 위에 그려진 것이 잡혀야 한다.
        for note in reversed(self.track.notes.notes):
            if note.midi != midi:
                continue
            if note.start_tick <= tick < note.end_tick:
                return note
        return None

    def _on_resize_edge(self, note: Note, position: QtCore.QPointF) -> bool:
        right = self.tick_to_x(note.end_tick)
        return abs(position.x() - right) <= 5.0

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.track is None:
            return
        position = event.position()
        modifiers = event.modifiers()

        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._drag_mode = "pan"
            self._drag_start = position.toPoint()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            return

        note = self._note_at(position)

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if note is not None:
                self.edit_requested.emit("delete", [note])
            return

        if note is None:
            if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                # 빈 곳에 Ctrl+클릭은 새 음 추가
                tick = self.view.snap(self.x_to_tick(position.x()))
                midi = self.y_to_pitch(position.y())
                if 0 <= midi <= 127:
                    self.edit_requested.emit(
                        "add", Note(midi, tick, self.view.snap_ticks or PPQ // 4, 90)
                    )
                return
            if not (modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.selected.clear()
                self.selection_changed.emit()
            self._drag_mode = "select"
            self._drag_start = position.toPoint()
            self._select_rect = QtCore.QRect(self._drag_start, self._drag_start)
            self.update()
            return

        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.selected.symmetric_difference_update({note.note_id})
        elif note.note_id not in self.selected:
            self.selected = {note.note_id}
        self.selection_changed.emit()

        self._drag_mode = "resize" if self._on_resize_edge(note, position) else "move"
        self._drag_start = position.toPoint()
        self._drag_origin = {
            n.note_id: (n.midi, n.start_tick, n.duration_ticks)
            for n in self.track.notes if n.note_id in self.selected
        }
        self._drag_preview = dict(self._drag_origin)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        position = event.position()

        if self._drag_mode == "" and self.track is not None:
            note = self._note_at(position)
            new_hover = note.note_id if note else None
            if new_hover != self._hover_note:
                self._hover_note = new_hover
                self.update()
            if note is not None and self._on_resize_edge(note, position):
                self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            elif note is not None:
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            return

        if self._drag_start is None:
            return
        delta_x = position.x() - self._drag_start.x()
        delta_y = position.y() - self._drag_start.y()

        if self._drag_mode == "pan":
            self._user_scrolled = True
            self.view.scroll_tick = max(
                0, self.view.scroll_tick - int(delta_x / self.view.pixels_per_tick)
            )
            self.view.scroll_pitch = max(
                0, min(127, self.view.scroll_pitch - int(delta_y / self.view.key_height))
            )
            self._drag_start = position.toPoint()
            self.update()
            return

        if self._drag_mode == "select":
            self._select_rect = QtCore.QRect(self._drag_start, position.toPoint()).normalized()
            self._update_rect_selection()
            self.update()
            return

        delta_tick = int(delta_x / max(1e-9, self.view.pixels_per_tick))
        delta_pitch = int(-delta_y // self.view.key_height) if delta_y else 0

        preview: dict[int, tuple[int, int, int]] = {}
        for note_id, (midi, start, length) in self._drag_origin.items():
            if self._drag_mode == "move":
                new_start = self.view.snap(start + delta_tick)
                new_midi = max(0, min(127, midi + delta_pitch))
                preview[note_id] = (new_midi, new_start, length)
            else:
                new_length = self.view.snap(start + length + delta_tick) - start
                preview[note_id] = (midi, start, max(self.view.snap_ticks or 1, new_length))
        self._drag_preview = preview
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_mode in ("move", "resize") and self._drag_preview:
            changes: list[tuple[Note, Note]] = []
            if self.track is not None:
                for note in self.track.notes:
                    preview = self._drag_preview.get(note.note_id)
                    if preview is None:
                        continue
                    midi, start, length = preview
                    if (midi, start, length) == (note.midi, note.start_tick,
                                                 note.duration_ticks):
                        continue
                    changed = note.copy()
                    changed.midi = midi
                    changed.start_tick = start
                    changed.duration_ticks = length
                    changes.append((note, changed))
            if changes:
                label = "음 이동" if self._drag_mode == "move" else "음 길이 변경"
                self.edit_requested.emit("edit", (changes, label))
        self._drag_mode = ""
        self._drag_start = None
        self._drag_origin.clear()
        self._drag_preview.clear()
        self._select_rect = None
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.update()

    def _update_rect_selection(self) -> None:
        if self.track is None or self._select_rect is None:
            return
        rectangle = self._select_rect
        chosen: set[int] = set()
        for note in self.track.notes:
            if self.note_rect(note).intersects(QtCore.QRectF(rectangle)):
                chosen.add(note.note_id)
        if chosen != self.selected:
            self.selected = chosen
            self.selection_changed.emit()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self._user_scrolled = True
        modifiers = event.modifiers()
        steps = event.angleDelta().y() / 120.0
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            # 가로 확대. 마우스 위치의 tick 을 고정한 채 확대한다.
            anchor_tick = self.x_to_tick(event.position().x())
            self.view.pixels_per_tick *= (1.18 ** steps)
            self.view.clamp()
            self.view.scroll_tick = max(
                0, int(anchor_tick - event.position().x() / self.view.pixels_per_tick)
            )
        elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.view.scroll_tick = max(
                0, self.view.scroll_tick - int(steps * 4 * PPQ)
            )
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            self.view.key_height = int(self.view.key_height + steps * 2)
            self.view.clamp()
        else:
            self.view.scroll_pitch = max(
                0, min(127, self.view.scroll_pitch + int(steps * 3))
            )
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.track is None:
            return
        note = self._note_at(event.position())
        if note is None:
            tick = self.view.snap(self.x_to_tick(event.position().x()))
            midi = self.y_to_pitch(event.position().y())
            if 0 <= midi <= 127:
                self.edit_requested.emit(
                    "add", Note(midi, tick, self.view.snap_ticks or PPQ // 4, 90)
                )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.track is None:
            return
        key = event.key()
        selected_notes = [n for n in self.track.notes if n.note_id in self.selected]

        if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if selected_notes:
                self.edit_requested.emit("delete", selected_notes)
            return
        if key == QtCore.Qt.Key.Key_A and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.selected = {n.note_id for n in self.track.notes}
            self.selection_changed.emit()
            self.update()
            return
        if not selected_notes:
            return

        step = PPQ if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier \
            else (self.view.snap_ticks or PPQ // 4)
        changes: list[tuple[Note, Note]] = []
        label = ""
        for note in selected_notes:
            changed = note.copy()
            if key == QtCore.Qt.Key.Key_Up:
                changed.midi = min(127, note.midi + (12 if event.modifiers()
                                   & QtCore.Qt.KeyboardModifier.ShiftModifier else 1))
                label = "음높이 올리기"
            elif key == QtCore.Qt.Key.Key_Down:
                changed.midi = max(0, note.midi - (12 if event.modifiers()
                                   & QtCore.Qt.KeyboardModifier.ShiftModifier else 1))
                label = "음높이 내리기"
            elif key == QtCore.Qt.Key.Key_Left:
                changed.start_tick = max(0, note.start_tick - step)
                label = "앞으로 옮기기"
            elif key == QtCore.Qt.Key.Key_Right:
                changed.start_tick = note.start_tick + step
                label = "뒤로 옮기기"
            else:
                return
            changes.append((note, changed))
        if changes:
            self.edit_requested.emit("edit", (changes, label))

    # ---------------------------------------------------------------- 도우미

    def scroll_to_tick(self, tick: int, keep_margin: float = 0.25) -> None:
        """재생 위치가 화면 밖으로 나가면 따라간다."""
        x = self.tick_to_x(tick)
        margin = self.width() * keep_margin
        if x < margin or x > self.width() - margin:
            self.view.scroll_tick = max(
                0, int(tick - margin / max(1e-9, self.view.pixels_per_tick))
            )
            self.update()

    def zoom_to_fit(self, total_ticks: int) -> None:
        """가로로 곡 전체가 보이게 맞춘다. 세로 위치는 건드리지 않는다."""
        if total_ticks <= 0 or self.width() <= 0:
            return
        self.view.pixels_per_tick = self.width() / total_ticks * 0.97
        self.view.scroll_tick = 0
        self.view.clamp()
        if self.tick_to_x(total_ticks) > self.width():
            # 한계에 걸려 다 안 들어오면 알 수 있어야 한다. 조용히 넘어가면
            # 사용자는 '전체 보기' 를 눌렀는데 뒤가 잘린 이유를 모른다.
            self.view.scroll_tick = 0
        self.update()

    def selected_notes(self) -> list[Note]:
        if self.track is None:
            return []
        return [n for n in self.track.notes if n.note_id in self.selected]
