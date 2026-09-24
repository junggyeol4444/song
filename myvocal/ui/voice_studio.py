"""
Voice Studio — 6·7번. "내 AI 가수 만들기".

    왼쪽   내 목소리 목록 (+ 새 목소리)
    오른쪽 [녹음]       다음에 부를 연습, 부족한 자료 안내, 자료 현황 표, 녹음 목록
           [목소리 모델] 모델 만들기, 분석 결과, 창법별로 들어 보기, 보컬 트랙에 쓰기

녹음과 분석, 모델 만들기는 오래 걸리므로 전부 딴 실뜨기에서 한다.
"""

from __future__ import annotations

import traceback
from typing import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..audio.recording import Recorder, RecordingError, check_level, input_available
from ..voice.analysis import midi_to_hz, note_name
from ..voice.library import VoiceEntry, VoiceLibrary
from ..voice.model import RIGHTS_CHOICES, SINGING_STYLES, VoiceModel, preset_voices
from ..voice.phonemes import align_lyrics
from ..voice.synth import SingingSynth, VoiceError
from ..voice.training import (
    VOWEL_SYLLABLE, VOWELS, ZONE_NAMES, ZONES, Coverage, Exercise, next_suggestions,
    starter_exercises,
)
from .theme import DARK, Palette


class _Worker(QtCore.QThread):
    """함수 하나를 딴 실뜨기에서 돌린다."""

    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str, float)

    def __init__(self, job: Callable, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._job = job

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._job(lambda text, value: self.progressed.emit(text, value)))
        except (VoiceError, RecordingError) as error:
            self.failed.emit(str(error))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}\n\n{traceback.format_exc(limit=3)}")


class NewVoiceDialog(QtWidgets.QDialog):
    """새 목소리. 이름, 자료 권한(63번), 처음 연습할 음역."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("새 목소리")
        self.setMinimumWidth(460)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("예: MY VOICE")
        form.addRow("이름", self.name_edit)
        self.type_box = QtWidgets.QComboBox()
        for key, model in preset_voices().items():
            self.type_box.addItem(f"{model.name} ({model.range_text()})", key)
        self.type_box.setCurrentIndex(2)
        form.addRow("대략의 음역", self.type_box)
        hint = QtWidgets.QLabel("첫 연습의 음높이를 정하는 데만 씁니다. 녹음하면 실제 음역을 잽니다.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        form.addRow("", hint)
        layout.addLayout(form)

        rights = QtWidgets.QGroupBox("이 목소리는")
        rights_layout = QtWidgets.QVBoxLayout(rights)
        self.rights_buttons: dict[str, QtWidgets.QRadioButton] = {}
        for key, label in RIGHTS_CHOICES.items():
            button = QtWidgets.QRadioButton(label)
            rights_layout.addWidget(button)
            self.rights_buttons[key] = button
        self.rights_buttons["my_voice"].setChecked(True)
        reference = QtWidgets.QRadioButton("Reference 분석용 — 목소리 학습에는 쓸 수 없습니다")
        reference.setEnabled(False)
        rights_layout.addWidget(reference)
        note = QtWidgets.QLabel("남의 목소리는 허가를 받은 경우에만 학습할 수 있습니다.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        rights_layout.addWidget(note)
        layout.addWidget(rights)

        buttons = QtWidgets.QDialogButtonBox()
        ok = buttons.addButton("만들기", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setObjectName("Primary")
        buttons.addButton("취소", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        if not self.name_edit.text().strip():
            QtWidgets.QMessageBox.information(self, "이름", "목소리 이름을 적어 주세요.")
            return
        self.accept()

    @property
    def rights(self) -> str:
        return next(k for k, b in self.rights_buttons.items() if b.isChecked())


class CoverageTable(QtWidgets.QTableWidget):
    """모음 × 음높이 구역마다 모은 자료 (초)."""

    def __init__(self, palette: Palette = DARK, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(len(ZONES), len(VOWELS), parent)
        self._palette = palette
        self.setHorizontalHeaderLabels([VOWEL_SYLLABLE[v] for v in VOWELS])
        self.setVerticalHeaderLabels([ZONE_NAMES[z] for z in ZONES])
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.setFixedHeight(128)

    def set_coverage(self, coverage: Coverage) -> None:
        for row, zone in enumerate(ZONES):
            for column, vowel in enumerate(VOWELS):
                cell = coverage.cells[(vowel, zone)]
                item = QtWidgets.QTableWidgetItem(f"{cell.seconds:.1f}초" if cell.seconds else "—")
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if cell.enough:
                    color = self._palette.success
                elif cell.seconds > 0:
                    color = self._palette.warning
                else:
                    color = self._palette.surface_sunken
                background = QtGui.QColor(color)
                background.setAlpha(110 if cell.seconds else 255)
                item.setBackground(background)
                item.setToolTip(f"{ZONE_NAMES[zone]} '{vowel}': {cell.seconds:.1f}초, 음 {cell.notes}개"
                                + (" — 충분" if cell.enough else " — 부족"))
                self.setItem(row, column, item)


class VoiceStudio(QtWidgets.QWidget):
    """내 AI 가수 만들기."""

    voice_chosen = QtCore.Signal(str, str)    # 'user:<id>', 창법 — 보컬 트랙에 쓰기
    back_requested = QtCore.Signal()

    def __init__(self, library: VoiceLibrary | None = None, palette: Palette = DARK,
                 recorder: Recorder | None = None, can_apply: bool = False,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.library = library or VoiceLibrary()
        self._palette = palette
        self.recorder = recorder or Recorder()
        self.entry: VoiceEntry | None = None
        self._worker: _Worker | None = None
        self._exercises: list[Exercise] = []

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(14)

        # ---- 왼쪽: 목록 ----
        left = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("돌아가기")
        back.clicked.connect(self.back_requested)
        top.addWidget(back)
        heading = QtWidgets.QLabel("내 AI 가수")
        heading.setObjectName("Heading")
        top.addWidget(heading, 1)
        left.addLayout(top)
        self.voice_list = QtWidgets.QListWidget()
        self.voice_list.currentRowChanged.connect(self._on_voice_selected)
        left.addWidget(self.voice_list, 1)
        row = QtWidgets.QHBoxLayout()
        new_button = QtWidgets.QPushButton("+ 새 목소리")
        new_button.setObjectName("Primary")
        new_button.clicked.connect(self.new_voice)
        row.addWidget(new_button, 1)
        self.delete_button = QtWidgets.QPushButton("삭제")
        self.delete_button.clicked.connect(self.delete_voice)
        row.addWidget(self.delete_button)
        left.addLayout(row)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(250)
        outer.addWidget(left_widget)

        # ---- 오른쪽 ----
        self.stack = QtWidgets.QStackedWidget()
        empty = QtWidgets.QLabel("왼쪽에서 '+ 새 목소리' 를 눌러 시작하세요.\n\n"
                                 "연습 몇 개를 녹음하면 음역, 음색, 숨, 비브라토 같은 성질을 재서\n"
                                 "그 목소리로 노래하는 모델을 만듭니다.")
        empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty.setObjectName("Dim")
        self.stack.addWidget(empty)
        self.stack.addWidget(self._build_voice_page(can_apply))
        outer.addWidget(self.stack, 1)

        self.refresh_list()

    # ------------------------------------------------------------ 화면 만들기

    def _build_voice_page(self, can_apply: bool) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QtWidgets.QLabel()
        self.title_label.setObjectName("Title")
        layout.addWidget(self.title_label)
        self.sub_label = QtWidgets.QLabel()
        self.sub_label.setObjectName("Dim")
        layout.addWidget(self.sub_label)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)
        self.tabs = tabs

        # [녹음]
        record_page = QtWidgets.QWidget()
        record_layout = QtWidgets.QVBoxLayout(record_page)
        self.mic_label = QtWidgets.QLabel()
        self.mic_label.setObjectName("Faint")
        self.mic_label.setWordWrap(True)
        record_layout.addWidget(self.mic_label)

        self.guidance_label = QtWidgets.QLabel()
        self.guidance_label.setObjectName("Card")
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setMargin(10)
        record_layout.addWidget(self.guidance_label)

        chooser = QtWidgets.QHBoxLayout()
        chooser.addWidget(QtWidgets.QLabel("연습"))
        self.exercise_box = QtWidgets.QComboBox()
        self.exercise_box.currentIndexChanged.connect(self._on_exercise_changed)
        chooser.addWidget(self.exercise_box, 1)
        record_layout.addLayout(chooser)

        self.instruction_label = QtWidgets.QLabel()
        self.instruction_label.setWordWrap(True)
        record_layout.addWidget(self.instruction_label)
        self.notes_label = QtWidgets.QLabel()
        self.notes_label.setObjectName("Faint")
        self.notes_label.setWordWrap(True)
        record_layout.addWidget(self.notes_label)

        buttons = QtWidgets.QHBoxLayout()
        self.listen_button = QtWidgets.QPushButton("가이드 듣기")
        self.listen_button.clicked.connect(self.play_guide)
        buttons.addWidget(self.listen_button)
        self.record_button = QtWidgets.QPushButton("● 녹음")
        self.record_button.setObjectName("Primary")
        self.record_button.clicked.connect(self.start_recording)
        buttons.addWidget(self.record_button)
        self.stop_button = QtWidgets.QPushButton("■ 멈춤")
        self.stop_button.clicked.connect(self.recorder.stop)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        record_layout.addLayout(buttons)
        self.record_progress = QtWidgets.QProgressBar()
        self.record_progress.setTextVisible(False)
        self.record_progress.setFixedHeight(6)
        record_layout.addWidget(self.record_progress)
        self.take_result = QtWidgets.QLabel()
        self.take_result.setWordWrap(True)
        record_layout.addWidget(self.take_result)

        coverage_title = QtWidgets.QLabel("모은 자료 (모음 × 음높이)")
        coverage_title.setObjectName("Heading")
        record_layout.addWidget(coverage_title)
        self.coverage_table = CoverageTable(self._palette)
        record_layout.addWidget(self.coverage_table)
        self.coverage_label = QtWidgets.QLabel()
        self.coverage_label.setObjectName("Faint")
        record_layout.addWidget(self.coverage_label)

        takes_row = QtWidgets.QHBoxLayout()
        self.take_list = QtWidgets.QListWidget()
        self.take_list.setMaximumHeight(110)
        takes_row.addWidget(self.take_list, 1)
        delete_take = QtWidgets.QPushButton("녹음 삭제")
        delete_take.clicked.connect(self.delete_take)
        takes_row.addWidget(delete_take, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        record_layout.addLayout(takes_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(record_page)
        tabs.addTab(scroll, "녹음")

        # [목소리 모델]
        model_page = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(model_page)
        build_row = QtWidgets.QHBoxLayout()
        self.build_button = QtWidgets.QPushButton("목소리 모델 만들기")
        self.build_button.setObjectName("Primary")
        self.build_button.clicked.connect(self.build_model)
        build_row.addWidget(self.build_button)
        self.build_progress = QtWidgets.QProgressBar()
        self.build_progress.setRange(0, 100)
        build_row.addWidget(self.build_progress, 1)
        model_layout.addLayout(build_row)
        self.report_view = QtWidgets.QPlainTextEdit()
        self.report_view.setReadOnly(True)
        model_layout.addWidget(self.report_view, 1)

        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel("창법"))
        self.style_box = QtWidgets.QComboBox()
        for style in SINGING_STYLES.values():
            self.style_box.addItem(f"{style.display_name} — {style.description}", style.name)
        style_row.addWidget(self.style_box, 1)
        self.preview_button = QtWidgets.QPushButton("들어 보기")
        self.preview_button.clicked.connect(self.preview_style)
        style_row.addWidget(self.preview_button)
        self.apply_button = QtWidgets.QPushButton("보컬 트랙에 쓰기")
        self.apply_button.clicked.connect(self._apply_to_track)
        self.apply_button.setVisible(can_apply)
        style_row.addWidget(self.apply_button)
        model_layout.addLayout(style_row)
        tabs.addTab(model_page, "목소리 모델")
        return page

    # ------------------------------------------------------------ 목록

    def refresh_list(self, select_id: str | None = None) -> None:
        current = select_id or (self.entry.voice_id if self.entry else None)
        self.voice_list.blockSignals(True)
        self.voice_list.clear()
        self._entries = self.library.entries()
        selected_row = -1
        for row, entry in enumerate(self._entries):
            takes = len(entry.takes())
            state = "모델 있음" if entry.model() is not None else f"녹음 {takes}개"
            item = QtWidgets.QListWidgetItem(f"{entry.name}\n  {state}")
            self.voice_list.addItem(item)
            if entry.voice_id == current:
                selected_row = row
        self.voice_list.blockSignals(False)
        if selected_row < 0 and self._entries:
            selected_row = 0
        self.voice_list.setCurrentRow(selected_row)
        if selected_row < 0:
            self.entry = None
            self.stack.setCurrentIndex(0)
        else:
            self._on_voice_selected(selected_row)
        self.delete_button.setEnabled(bool(self._entries))

    def _on_voice_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._entries):
            return
        self.entry = self._entries[row]
        self.stack.setCurrentIndex(1)
        self.refresh_voice()

    def new_voice(self) -> None:
        dialog = NewVoiceDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            entry = self.library.create(dialog.name_edit.text(), dialog.rights,
                                        dialog.type_box.currentData())
        except (VoiceError, OSError) as error:
            QtWidgets.QMessageBox.warning(self, "만들 수 없습니다", str(error))
            return
        self.refresh_list(entry.voice_id)

    def delete_voice(self) -> None:
        if self.entry is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "목소리 삭제",
            f"'{self.entry.name}' 과 녹음 {len(self.entry.takes())}개를 지울까요?\n"
            f"되돌릴 수 없습니다. 이 목소리를 쓰는 곡은 기본 목소리로 부르게 됩니다.")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.library.delete(self.entry.voice_id)
        self.entry = None
        self.refresh_list()

    # ------------------------------------------------------------ 한 목소리

    def refresh_voice(self) -> None:
        entry = self.entry
        if entry is None:
            return
        self.title_label.setText(entry.name)
        model = entry.model()
        self.sub_label.setText(
            f"{RIGHTS_CHOICES.get(entry.rights, entry.rights)} · 녹음 {len(entry.takes())}개 · "
            + (f"모델 있음 — {model.range_text()}" if model else "아직 모델 없음"))
        available, message = input_available()
        self.mic_label.setText(message + ("" if available else
                               "  녹음은 마이크가 있어야 합니다."))
        self.record_button.setEnabled(available)

        coverage = entry.coverage()
        self.coverage_table.set_coverage(coverage)
        self.coverage_label.setText(
            f"채운 칸 {coverage.filled_ratio:.0%} (칸마다 {2:.0f}초, 음 2개 이상이면 충분)")
        used = [t.title.removeprefix("문장: ") for t in entry.takes()]
        suggestions = next_suggestions(coverage, used_sentences=used)
        low, high = entry.range_midi()
        exercises = []
        if not entry.takes():
            exercises = starter_exercises(low, high)
            self.guidance_label.setText(
                "처음이면 '음역 재기' 부터 해 보세요. 가이드 음을 듣고 조용할 때 따라 부르면 됩니다.\n"
                "헤드폰이 없어도 됩니다 (노래하는 동안에는 아무것도 틀지 않습니다).")
        elif suggestions:
            self.guidance_label.setText(suggestions[0].message)
        else:
            self.guidance_label.setText("자료가 충분합니다. '목소리 모델' 탭에서 모델을 만드세요.")
        if entry.takes():
            # 부족한 칸을 채울 연습이 먼저, 그 뒤에 기본 연습 (다시 해 보고 싶을 때)
            exercises = [s.exercise for s in suggestions] + starter_exercises(low, high)
        self._exercises = exercises
        self.exercise_box.blockSignals(True)
        self.exercise_box.clear()
        for exercise in exercises:
            self.exercise_box.addItem(exercise.title)
        self.exercise_box.blockSignals(False)
        self.exercise_box.setCurrentIndex(0 if exercises else -1)
        self._on_exercise_changed(0)

        self.take_list.clear()
        for info in entry.takes():
            errors = [abs(m.error_cents) for m in info.measures if m.error_cents is not None]
            accuracy = f", 음정 평균 {np.mean(errors):.0f}센트" if errors else ""
            self.take_list.addItem(
                f"{info.index:03d}  {info.title} — 잡힌 음 {len(info.measures)}/{len(info.guide)}{accuracy}")

        self.report_view.setPlainText(model.report_text if model else
                                      "녹음을 모은 뒤 '목소리 모델 만들기' 를 누르세요.")
        self.preview_button.setEnabled(model is not None)
        self.apply_button.setEnabled(model is not None)
        self.build_button.setEnabled(bool(entry.takes()))

    def _on_exercise_changed(self, index: int) -> None:
        if not (0 <= index < len(self._exercises)):
            self.instruction_label.setText("")
            self.notes_label.setText("")
            return
        exercise = self._exercises[index]
        self.instruction_label.setText(exercise.instruction)
        self.notes_label.setText("  ".join(
            f"{n.syllable}({note_name(midi_to_hz(n.midi))})" for n in exercise.notes))

    @property
    def current_exercise(self) -> Exercise | None:
        index = self.exercise_box.currentIndex()
        return self._exercises[index] if 0 <= index < len(self._exercises) else None

    # ------------------------------------------------------------ 녹음

    def play_guide(self) -> None:
        exercise = self.current_exercise
        if exercise is None:
            return
        try:
            self.recorder.play_cues(exercise)
        except RecordingError as error:
            QtWidgets.QMessageBox.information(self, "재생할 수 없습니다", str(error))

    def _busy(self, busy: bool) -> None:
        for widget in (self.record_button, self.listen_button, self.build_button,
                       self.exercise_box, self.voice_list, self.preview_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if not busy and self.entry is not None:
            # 왼쪽 목록의 '녹음 N개 / 모델 있음' 도 바뀌었을 수 있다
            self.refresh_list(self.entry.voice_id)

    def start_recording(self) -> None:
        exercise, entry = self.current_exercise, self.entry
        if exercise is None or entry is None or self._worker is not None:
            return
        rate = self.recorder.rate

        def job(progress):
            signal = self.recorder.record(exercise)
            level = check_level(signal, rate)
            if level.peak_db < -45:
                raise RecordingError(level.messages[0] if level.messages else "소리가 없습니다.")
            progress("분석 중", 1.0)
            info = entry.add_take(signal, rate, exercise)
            return info, level

        self._busy(True)
        self.take_result.setText("가이드를 듣고, 조용할 때 따라 불러 주세요...")
        self.record_progress.setRange(0, 0)
        self._start(job, self._on_recorded)

    def _on_recorded(self, result) -> None:
        info, level = result
        self.record_progress.setRange(0, 1)
        errors = [m.error_cents for m in info.measures if m.error_cents is not None]
        lines = [f"녹음 저장 — 부를 음 {len(info.guide)}개 중 {len(info.measures)}개를 잡았습니다."]
        if errors:
            lines.append(f"음정: 평균 {np.mean(np.abs(errors)):.0f}센트 차이 "
                         f"({'높게' if np.median(errors) > 0 else '낮게'} 치우침 {abs(np.median(errors)):.0f}센트)")
        if len(info.measures) < len(info.guide):
            lines.append("못 잡은 음은 소리가 작았거나, 부른 시점이 가이드와 많이 어긋난 것입니다.")
        lines.extend(level.messages)
        self.take_result.setText("\n".join(lines))
        self._finish()

    def delete_take(self) -> None:
        if self.entry is None or self.take_list.currentRow() < 0:
            return
        info = self.entry.takes()[self.take_list.currentRow()]
        self.entry.delete_take(info.index)
        self.refresh_voice()

    # ------------------------------------------------------------ 모델

    def build_model(self) -> None:
        entry = self.entry
        if entry is None or self._worker is not None:
            return
        self._busy(True)
        self.build_progress.setValue(0)
        self._start(lambda progress: entry.build_model(progress), self._on_built)

    def _on_built(self, model: VoiceModel) -> None:
        self.build_progress.setValue(100)
        self._finish()
        self.refresh_list(model.voice_id)
        self.tabs.setCurrentIndex(1)

    def preview_style(self) -> None:
        entry = self.entry
        model = entry.model() if entry else None
        if model is None:
            return
        style = self.style_box.currentData()
        vocal = model.vocal_range()
        center = vocal.center
        melody = [center - 3, center - 1, center, center + 2, center + 4, center + 2, center]
        text = "다시 만난 오늘 밤"
        times = [(0.2 + i * 0.45, 0.42 if i < len(melody) - 1 else 1.2) for i in range(len(melody))]

        def job(progress):
            syllables = align_lyrics(text, times)
            audio = SingingSynth(model.timbre(style), 48000).render(
                syllables, [midi_to_hz(m) for m in melody]).data[0]
            import sounddevice
            sounddevice.play((audio * 0.8).astype(np.float32), 48000)
            return None

        self._busy(True)
        self._start(job, lambda _: self._finish())

    def _apply_to_track(self) -> None:
        if self.entry is not None and self.entry.model() is not None:
            self.voice_chosen.emit(f"user:{self.entry.voice_id}", self.style_box.currentData())

    # ------------------------------------------------------------ 실뜨기

    def _start(self, job: Callable, done: Callable) -> None:
        self._worker = _Worker(job, self)
        self._worker.finished_ok.connect(done)
        self._worker.failed.connect(self._on_failed)
        self._worker.progressed.connect(self._on_progress)
        self._worker.start()

    def _on_progress(self, text: str, value: float) -> None:
        self.build_progress.setValue(int(value * 100))
        self.build_progress.setFormat(text)

    def _finish(self) -> None:
        if self._worker is not None:
            self._worker.wait(2000)
        self._worker = None
        self._busy(False)

    def _on_failed(self, message: str) -> None:
        self.record_progress.setRange(0, 1)
        self._finish()
        QtWidgets.QMessageBox.warning(self, "할 수 없습니다", message)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.recorder.stop()
            self._worker.wait(3000)
        event.accept()
