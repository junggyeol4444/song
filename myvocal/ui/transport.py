"""
재생 조작부 — 위쪽 가로 띠.

재생/정지, 지금 위치, BPM, 조성, 자석 단위, 음량 미터가 여기 있다.
음악 프로그램에서 가장 많이 누르는 곳이라 항상 같은 자리에 있어야 한다.
"""

from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.units import PPQ, MeterMap, TempoMap
from ..music.theory import Key
from .theme import DARK, Palette, meter_color


class LevelMeter(QtWidgets.QWidget):
    """음량 미터. 소리가 너무 크면 빨갛게 된다."""

    def __init__(self, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._left = 0.0
        self._right = 0.0
        self._hold_left = 0.0
        self._hold_right = 0.0
        self.setFixedSize(84, 30)
        self.setToolTip("출력 음량. 빨간색이 계속 뜨면 소리가 깨집니다.")

    def set_levels(self, left: float, right: float) -> None:
        self._left = max(0.0, min(1.5, left))
        self._right = max(0.0, min(1.5, right))
        # 최고치는 잠깐 남겨 둔다. 순간적으로 튄 것을 놓치지 않게.
        self._hold_left = max(self._hold_left * 0.94, self._left)
        self._hold_right = max(self._hold_right * 0.94, self._right)
        self.update()

    def reset(self) -> None:
        self._left = self._right = self._hold_left = self._hold_right = 0.0
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        import math

        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(self._palette.surface_sunken))
        bar_height = 8
        gap = 4
        top = (self.height() - bar_height * 2 - gap) // 2

        for index, (level, hold) in enumerate(
            ((self._left, self._hold_left), (self._right, self._hold_right))
        ):
            y = top + index * (bar_height + gap)
            # dB 로 보여준다. 선형으로 그리면 작은 소리가 거의 안 보인다.
            db = 20.0 * math.log10(max(level, 1e-5))
            ratio = max(0.0, min(1.0, (db + 60.0) / 60.0))
            width = int(self.width() * ratio)
            if width > 0:
                painter.fillRect(0, y, width, bar_height,
                                 QtGui.QColor(meter_color(db, self._palette)))
            hold_db = 20.0 * math.log10(max(hold, 1e-5))
            hold_ratio = max(0.0, min(1.0, (hold_db + 60.0) / 60.0))
            hold_x = int(self.width() * hold_ratio)
            if hold_x > 1:
                painter.fillRect(hold_x - 2, y, 2, bar_height,
                                 QtGui.QColor(self._palette.text))
        painter.setPen(QtGui.QPen(QtGui.QColor(self._palette.border), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()


class TransportBar(QtWidgets.QFrame):
    """재생 조작부."""

    play_pressed = QtCore.Signal()
    stop_pressed = QtCore.Signal()
    position_requested = QtCore.Signal(int)     # tick
    bpm_changed = QtCore.Signal(float)
    key_changed = QtCore.Signal(str)
    snap_changed = QtCore.Signal(int)           # tick
    loop_toggled = QtCore.Signal(bool)
    render_requested = QtCore.Signal()

    SNAP_OPTIONS: tuple[tuple[str, int], ...] = (
        ("자석 끄기", 0),
        ("1마디", PPQ * 4),
        ("2분음표", PPQ * 2),
        ("4분음표", PPQ),
        ("8분음표", PPQ // 2),
        ("16분음표", PPQ // 4),
        ("32분음표", PPQ // 8),
        ("셋잇단 8분", PPQ // 3),
    )

    def __init__(self, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.meter: MeterMap | None = None
        self.tempo: TempoMap | None = None
        self._playing = False
        self.setObjectName("Surface")
        self.setFixedHeight(52)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # --- 재생 단추 ---
        self.play_button = QtWidgets.QPushButton("▶")
        self.play_button.setFixedSize(40, 34)
        self.play_button.setObjectName("Primary")
        self.play_button.setToolTip("재생 / 일시정지  (스페이스)")
        self.play_button.clicked.connect(self.play_pressed)
        layout.addWidget(self.play_button)

        self.stop_button = QtWidgets.QPushButton("■")
        self.stop_button.setFixedSize(34, 34)
        self.stop_button.setToolTip("정지  (처음으로)")
        self.stop_button.clicked.connect(self.stop_pressed)
        layout.addWidget(self.stop_button)

        self.loop_button = QtWidgets.QPushButton("⟳")
        self.loop_button.setObjectName("Toggle")
        self.loop_button.setCheckable(True)
        self.loop_button.setFixedSize(34, 34)
        self.loop_button.setToolTip("구간 반복")
        self.loop_button.toggled.connect(self.loop_toggled)
        layout.addWidget(self.loop_button)

        layout.addSpacing(6)

        # --- 위치 표시 ---
        self.position_label = QtWidgets.QLabel("1 . 1")
        self.position_label.setObjectName("Mono")
        position_font = QtGui.QFont(self.font())
        position_font.setPointSize(self.font().pointSize() + 6)
        position_font.setWeight(QtGui.QFont.Weight.DemiBold)
        self.position_label.setFont(position_font)
        self.position_label.setFixedWidth(86)
        self.position_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.position_label.setToolTip("마디 . 박")
        layout.addWidget(self.position_label)

        self.time_label = QtWidgets.QLabel("0:00.0")
        self.time_label.setObjectName("Dim")
        self.time_label.setFixedWidth(62)
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        layout.addWidget(self._separator())

        # --- 곡 설정 ---
        layout.addWidget(self._caption("BPM"))
        self.bpm_box = QtWidgets.QDoubleSpinBox()
        self.bpm_box.setRange(20.0, 300.0)
        self.bpm_box.setDecimals(1)
        self.bpm_box.setSingleStep(1.0)
        self.bpm_box.setValue(120.0)
        self.bpm_box.setFixedWidth(78)
        self.bpm_box.valueChanged.connect(self.bpm_changed)
        layout.addWidget(self.bpm_box)

        layout.addWidget(self._caption("조성"))
        self.key_box = QtWidgets.QComboBox()
        for tonic in ("C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"):
            self.key_box.addItem(tonic)
            self.key_box.addItem(f"{tonic}m")
        self.key_box.setFixedWidth(66)
        self.key_box.currentTextChanged.connect(self.key_changed)
        layout.addWidget(self.key_box)

        layout.addWidget(self._caption("자석"))
        self.snap_box = QtWidgets.QComboBox()
        for label, ticks in self.SNAP_OPTIONS:
            self.snap_box.addItem(label, ticks)
        self.snap_box.setCurrentIndex(5)
        self.snap_box.setFixedWidth(104)
        self.snap_box.currentIndexChanged.connect(
            lambda _: self.snap_changed.emit(self.snap_box.currentData())
        )
        layout.addWidget(self.snap_box)

        layout.addStretch(1)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setObjectName("Faint")
        layout.addWidget(self.status_label)

        self.render_button = QtWidgets.QPushButton("소리 만들기")
        self.render_button.setToolTip("바뀐 내용을 반영해 다시 소리로 만듭니다.")
        self.render_button.clicked.connect(self.render_requested)
        layout.addWidget(self.render_button)

        self.meter_widget = LevelMeter(palette)
        layout.addWidget(self.meter_widget)

    def _caption(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("Faint")
        return label

    def _separator(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setStyleSheet(f"color: {self._palette.border};")
        return line

    # ---------------------------------------------------------------- 갱신

    def set_project_data(self, meter: MeterMap, tempo: TempoMap, key: Key) -> None:
        self.meter = meter
        self.tempo = tempo
        self.bpm_box.blockSignals(True)
        self.bpm_box.setValue(float(tempo.initial_bpm))
        self.bpm_box.blockSignals(False)
        self.key_box.blockSignals(True)
        index = self.key_box.findText(str(key))
        if index >= 0:
            self.key_box.setCurrentIndex(index)
        self.key_box.blockSignals(False)

    def set_position(self, tick: int) -> None:
        if self.meter is None:
            return
        bar, beat = self.meter.tick_to_bar_beat(max(0, tick))
        self.position_label.setText(f"{bar} . {beat:.0f}")
        if self.tempo is not None:
            seconds = self.tempo.tick_to_seconds(max(0, tick))
            minutes = int(seconds // 60)
            self.time_label.setText(f"{minutes}:{seconds % 60:04.1f}")

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.play_button.setText("❚❚" if playing else "▶")

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_levels(self, left: float, right: float) -> None:
        self.meter_widget.set_levels(left, right)

    def set_render_needed(self, needed: bool) -> None:
        """편집이 있었으면 '소리 만들기' 를 눈에 띄게 한다."""
        self.render_button.setObjectName("Primary" if needed else "")
        self.render_button.setText("소리 다시 만들기" if needed else "소리 만들기")
        self.render_button.style().unpolish(self.render_button)
        self.render_button.style().polish(self.render_button)
