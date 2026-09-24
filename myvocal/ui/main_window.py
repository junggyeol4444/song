"""
메인 창 — 편집기 전체.

여기서 모든 것이 만난다. 트랙 목록, 피아노롤, 타임라인, 재생 조작부가
같은 프로젝트를 본다.

원칙 하나: 프로젝트를 바꾸는 코드는 여기 두지 않는다. 전부 명령(Command)을
만들어 History 에 넘긴다. 그래야 모든 편집이 되돌려진다. 여기서 한 줄이라도
직접 고치면 그 편집만 되돌릴 수 없게 되고, 사용자는 실행 취소를 못 믿게 된다.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..audio.buffer import AudioBuffer
from ..audio.playback import PlaybackState, create_engine
from ..audio.renderer import RenderOptions, Renderer
from ..core import history as H
from ..core.notes import Note
from ..core.project import Project
from ..core.tracks import Track
from ..core.units import PPQ
from ..music.instruments import available_instruments, create_instrument
from ..music.lyrics import (
    LyricsBrief, LyricsDraft, LyricsError, lyrics_command, prepare_lyrics, request_lyrics,
)
from ..providers import default_registry
from ..music.structure import SECTION_COLORS
from ..music.theory import Key
from .lyrics_panel import LyricsPanel
from .piano_roll import PianoKeyboard, PianoRoll
from .theme import DARK, Palette, build_stylesheet
from .timeline import TimelineRuler
from .track_list import TrackPanel
from .transport import TransportBar


class RenderWorker(QtCore.QThread):
    """소리 만들기를 딴 실뜨기에서 한다.

    화면 실뜨기에서 렌더링하면 그동안 창이 얼어붙는다. 2분 곡이 45초 걸리므로
    45초 동안 아무것도 못 하게 된다.
    """

    finished_ok = QtCore.Signal(object, object)     # AudioBuffer, RenderReport
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str, float)

    def __init__(self, project: Project, options: RenderOptions,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._project = project
        self._options = options

    def run(self) -> None:
        try:
            renderer = Renderer(self._options)
            audio, report = renderer.render(
                self._project,
                progress=lambda stage, value: self.progressed.emit(stage, value),
            )
            self.finished_ok.emit(audio, report)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class LyricsWorker(QtCore.QThread):
    """가사를 받아 오는 일을 딴 실뜨기에서 한다.

    Claude 에 물으면 수십 초가 걸릴 수 있다. 그동안 창이 얼면 안 된다.
    프로젝트는 읽지도 쓰지도 않는다. 틀은 화면 실뜨기에서 미리 읽어 넘긴다.
    """

    finished_ok = QtCore.Signal(object)      # LyricsDraft
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str)

    def __init__(self, slots, payload: dict,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._slots = slots
        self._payload = payload

    def run(self) -> None:
        try:
            draft = request_lyrics(self._slots, self._payload, default_registry(),
                                   on_progress=self.progressed.emit)
            self.finished_ok.emit(draft)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class LyricsPreviewDialog(QtWidgets.QDialog):
    """받은 가사를 보여주고, 붙일지 묻는다."""

    def __init__(self, draft: LyricsDraft, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI 작사 결과")
        self.resize(520, 640)
        layout = QtWidgets.QVBoxLayout(self)

        who = {"local": "MYVOCAL 내장 규칙", "anthropic": "Claude"}.get(
            draft.provider, draft.provider)
        info = [f"작사: {who}"]
        if draft.fell_back:
            info.append("(먼저 시도한 서비스가 실패해서 대신 썼습니다)")
        if draft.note:
            info.append(draft.note)
        if draft.cost_note and draft.provider != "local":
            info.append(draft.cost_note)
        if draft.mismatches:
            info.append(
                f"음절 수가 멜로디와 다른 줄 {len(draft.mismatches)}개 — 붙일 때 멜로디를 "
                f"가사에 맞춥니다 (짧은 음을 합치거나 긴 음을 나눔)."
            )
        label = QtWidgets.QLabel("\n".join(info))
        label.setWordWrap(True)
        label.setObjectName("Dim")
        layout.addWidget(label)

        text = QtWidgets.QPlainTextEdit(draft.text())
        text.setReadOnly(True)
        layout.addWidget(text, 1)

        buttons = QtWidgets.QDialogButtonBox()
        apply_button = buttons.addButton("음에 붙이기", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        apply_button.setObjectName("Primary")
        buttons.addButton("버리기", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QtWidgets.QMainWindow):
    """편집기 창."""

    def __init__(self, project: Project | None = None,
                 palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.project = project or Project("새 프로젝트")
        self.history = H.History(self.project)
        self.playback = create_engine()
        self.render_options = RenderOptions(sample_rate=self.project.sample_rate)
        self._worker: RenderWorker | None = None
        self._lyrics_worker: LyricsWorker | None = None
        self._lyrics_track: Track | None = None
        self._render_dirty = True
        self._rendered_audio: AudioBuffer | None = None

        self.setWindowTitle("MYVOCAL Studio")
        self.resize(1440, 880)
        self._build_ui()
        self._build_menu()
        self._connect()
        self.load_project(self.project)

        # 재생 위치를 따라가는 시계
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)          # 초당 30번. 더 자주 해도 눈에 안 보인다.
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---------------------------------------------------------------- 만들기

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.transport = TransportBar(self._palette)
        outer.addWidget(self.transport)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # --- 왼쪽: 트랙 목록 ---
        self.track_panel = TrackPanel(self._palette)
        self.track_panel.setMinimumWidth(190)
        self.track_panel.setMaximumWidth(320)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        spacer = QtWidgets.QWidget()
        spacer.setFixedHeight(TimelineRuler.HEIGHT)
        left_layout.addWidget(spacer)
        left_layout.addWidget(self.track_panel, 1)
        splitter.addWidget(left)

        # --- 가운데: 타임라인 + 피아노롤 ---
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.piano_roll = PianoRoll(self._palette)
        self.timeline = TimelineRuler(self.piano_roll.view, self._palette)
        center_layout.addWidget(self.timeline)

        roll_row = QtWidgets.QWidget()
        roll_layout = QtWidgets.QHBoxLayout(roll_row)
        roll_layout.setContentsMargins(0, 0, 0, 0)
        roll_layout.setSpacing(0)
        self.keyboard = PianoKeyboard(self.piano_roll.view, self._palette)
        roll_layout.addWidget(self.keyboard)
        roll_layout.addWidget(self.piano_roll, 1)
        center_layout.addWidget(roll_row, 1)
        splitter.addWidget(center)

        # --- 오른쪽: 정보 / 가사 ---
        self.side_tabs = QtWidgets.QTabWidget()
        self.side_tabs.setMinimumWidth(270)
        self.side_tabs.setMaximumWidth(420)
        self.side_panel = self._build_side_panel()
        self.side_tabs.addTab(self.side_panel, "곡")
        self.lyrics_panel = LyricsPanel(self._palette)
        self.side_tabs.addTab(self.lyrics_panel, "가사")
        splitter.addWidget(self.side_tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 900, 300])
        outer.addWidget(splitter, 1)

        self.setCentralWidget(central)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        available = self.playback.available
        self.status.showMessage(
            "준비됨" if available else self.playback.unavailable_reason.split("\n")[0]
        )

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(400)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("Dim")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.info_label)

        structure_title = QtWidgets.QLabel("곡 구조")
        structure_title.setObjectName("Heading")
        layout.addWidget(structure_title)

        self.structure_list = QtWidgets.QListWidget()
        self.structure_list.setAlternatingRowColors(True)
        layout.addWidget(self.structure_list, 1)

        history_title = QtWidgets.QLabel("최근 편집")
        history_title.setObjectName("Heading")
        layout.addWidget(history_title)

        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMaximumHeight(140)
        layout.addWidget(self.history_list)

        self.problem_label = QtWidgets.QLabel()
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(f"color: {self._palette.warning};")
        layout.addWidget(self.problem_label)
        return panel

    def _build_menu(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("파일")
        self._action(file_menu, "새 프로젝트", "Ctrl+N", self.new_project)
        self._action(file_menu, "열기...", "Ctrl+O", self.open_project)
        self._action(file_menu, "저장", "Ctrl+S", self.save_project)
        self._action(file_menu, "다른 이름으로 저장...", "Ctrl+Shift+S",
                     lambda: self.save_project(ask=True))
        file_menu.addSeparator()
        self._action(file_menu, "오디오로 내보내기...", "Ctrl+E", self.export_audio)
        self._action(file_menu, "MIDI 로 내보내기...", "", self.export_midi)
        self._action(file_menu, "트랙별로 내보내기...", "", self.export_stems)
        file_menu.addSeparator()
        self._action(file_menu, "끝내기", "Ctrl+Q", self.close)

        edit_menu = bar.addMenu("편집")
        self.undo_action = self._action(edit_menu, "실행 취소", "Ctrl+Z", self.undo)
        self.redo_action = self._action(edit_menu, "다시 실행", "Ctrl+Y", self.redo)
        edit_menu.addSeparator()
        self._action(edit_menu, "모두 선택", "Ctrl+A",
                     lambda: self.piano_roll.keyPressEvent(QtGui.QKeyEvent(
                         QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_A,
                         QtCore.Qt.KeyboardModifier.ControlModifier)))
        self._action(edit_menu, "선택한 음 삭제", "Delete", self.delete_selected)

        view_menu = bar.addMenu("보기")
        self._action(view_menu, "전체 보기", "Ctrl+0",
                     lambda: self.piano_roll.zoom_to_fit(self.project.end_tick))
        self._action(view_menu, "가로 확대", "Ctrl+=",
                     lambda: self._zoom(1.25))
        self._action(view_menu, "가로 축소", "Ctrl+-",
                     lambda: self._zoom(0.8))

        play_menu = bar.addMenu("재생")
        self._action(play_menu, "재생 / 일시정지", "Space", self.toggle_play)
        self._action(play_menu, "정지", "Ctrl+.", self.stop_play)
        play_menu.addSeparator()
        self._action(play_menu, "소리 만들기", "F5", self.start_render)

        tools_menu = bar.addMenu("도구")
        self._action(tools_menu, "AI 작사", "Ctrl+L", self.request_ai_lyrics)
        self._action(tools_menu, "내 AI 가수 (목소리 학습)...", "", self.show_voice_studio)
        tools_menu.addSeparator()
        self._action(tools_menu, "AI 서비스 설정...", "", self.show_provider_settings)

        help_menu = bar.addMenu("도움말")
        self._action(help_menu, "소리 장치 확인", "", self.show_audio_info)
        self._action(help_menu, "단축키", "", self.show_shortcuts)

    def _action(self, menu: QtWidgets.QMenu, text: str, shortcut: str,
                handler: Callable[[], None]) -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(handler)
        menu.addAction(action)
        return action

    def _connect(self) -> None:
        self.transport.play_pressed.connect(self.toggle_play)
        self.transport.stop_pressed.connect(self.stop_play)
        self.transport.bpm_changed.connect(self._on_bpm_changed)
        self.transport.key_changed.connect(self._on_key_changed)
        self.transport.snap_changed.connect(self._on_snap_changed)
        self.transport.render_requested.connect(self.start_render)
        self.transport.loop_toggled.connect(self._on_loop_toggled)

        self.timeline.position_changed.connect(self._on_seek)
        self.timeline.section_clicked.connect(self._on_section_clicked)

        self.piano_roll.edit_requested.connect(self._on_edit_requested)
        self.piano_roll.selection_changed.connect(self._update_status)

        self.track_panel.track_selected.connect(self._on_track_selected)
        self.track_panel.property_changed.connect(self._on_track_property)
        self.track_panel.remove_requested.connect(self._on_track_remove)
        self.track_panel.add_requested.connect(self.add_track)

        self.structure_list.itemDoubleClicked.connect(self._on_structure_activated)
        self.lyrics_panel.lyrics_applied.connect(self._on_lyrics_applied)
        self.lyrics_panel.note_focused.connect(self._on_seek)
        self.lyrics_panel.ai_requested.connect(self._on_ai_lyrics)
        self.history.add_listener(lambda what: self._refresh_history())

    # ---------------------------------------------------------------- 자료

    def load_project(self, project: Project) -> None:
        self.project = project
        self.history = H.History(project)
        self.history.add_listener(lambda what: self._refresh_history())
        self.render_options = RenderOptions(sample_rate=project.sample_rate)
        self._render_dirty = True
        self._rendered_audio = None
        self.playback.clear()

        self.transport.set_project_data(project.meter, project.tempo, project.key)
        self.timeline.set_project_data(project.meter, project.tempo, project.structure)
        self.track_panel.set_tracks(project.tracks)
        self.piano_roll.set_track(
            project.tracks[0] if len(project.tracks) else None,
            project.meter, project.tempo,
        )
        self._refresh_sections()
        self.lyrics_panel.set_project(project)
        self.piano_roll.zoom_to_fit(max(1, project.end_tick))
        self._refresh_info()
        self._refresh_history()
        self._update_title()

    def _refresh_sections(self) -> None:
        marks = []
        self.structure_list.clear()
        for section in self.project.structure:
            span = self.project.section_range(section)
            color = SECTION_COLORS.get(section.kind, "#888888")
            marks.append((span.start_tick, span.end_tick, section.label, color))
            start, end = section.seconds(self.project.meter, self.project.tempo)
            item = QtWidgets.QListWidgetItem(
                f"{section.label}   {section.start_bar}~{section.end_bar - 1}마디"
                f"   {int(start // 60)}:{start % 60:04.1f}"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, section)
            item.setForeground(QtGui.QColor(color))
            if section.notes_for_ai:
                item.setToolTip(section.notes_for_ai)
            self.structure_list.addItem(item)
        self.piano_roll.set_sections(marks)

    def _refresh_info(self) -> None:
        project = self.project
        duration = project.duration_seconds
        lines = [
            f"<b>{project.meta.title}</b>",
            f"{project.key.display_name} · {project.bpm:.0f} BPM · {project.time_signature}",
            f"{project.genre}",
            f"{project.structure.total_bars}마디 · "
            f"{int(duration // 60)}:{duration % 60:04.1f}",
            f"트랙 {len(project.tracks)}개 · "
            f"음 {sum(len(t.notes) for t in project.tracks)}개",
        ]
        self.info_label.setText("<br>".join(lines))
        problems = project.problems()
        self.problem_label.setText(
            "\n".join(f"· {p}" for p in problems[:4]) if problems else ""
        )

    def _refresh_history(self) -> None:
        self.history_list.clear()
        for description in self.history.recent(12):
            self.history_list.addItem(description)
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(self.history.can_undo)
            self.undo_action.setText(
                f"실행 취소: {self.history.undo_description}"
                if self.history.can_undo else "실행 취소"
            )
            self.redo_action.setEnabled(self.history.can_redo)
            self.redo_action.setText(
                f"다시 실행: {self.history.redo_description}"
                if self.history.can_redo else "다시 실행"
            )
        self._refresh_info()
        self._update_title()
        self.track_panel.refresh()
        self.piano_roll.update()

    def _update_title(self) -> None:
        mark = "*" if self.history.has_unsaved_changes else ""
        path = f" — {self.project.path.name}" if self.project.path else ""
        self.setWindowTitle(f"{mark}{self.project.meta.title}{path} — MYVOCAL Studio")

    def _update_status(self) -> None:
        count = len(self.piano_roll.selected)
        if count:
            self.status.showMessage(f"음 {count}개 선택됨")

    # ---------------------------------------------------------------- 편집

    def _run(self, command: H.Command) -> None:
        """모든 편집은 이 문을 지난다."""
        try:
            self.history.run(command)
        except Exception as error:
            QtWidgets.QMessageBox.warning(
                self, "편집할 수 없습니다", f"{type(error).__name__}: {error}"
            )
            return
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._render_dirty = True
        self.transport.set_render_needed(True)

    def _on_edit_requested(self, kind: str, payload: object) -> None:
        track = self.piano_roll.track
        if track is None:
            return
        if kind == "add":
            self._run(H.AddNotes(track, [payload], "음 추가"))
        elif kind == "delete":
            self._run(H.RemoveNotes(track, payload, "음 삭제"))
            self.piano_roll.selected.clear()
        elif kind == "edit":
            changes, label = payload
            self._run(H.EditNotes(track, changes, label))

    def _on_lyrics_applied(self, track: Track, changes: list) -> None:
        """가사 편집 화면에서 온 변경을 기록에 남기며 적용한다."""
        if not changes:
            self.status.showMessage("바뀐 가사가 없습니다.", 2500)
            return
        self._run(H.EditNotes(track, changes, f"'{track.name}' 가사 입력"))
        self.lyrics_panel.refresh()
        filled = sum(1 for _, new in changes if new.lyric)
        self.status.showMessage(
            f"가사를 붙였습니다 — 음 {len(changes)}개 수정 (가사 있는 음 {filled}개)", 5000
        )

    # ---------------------------------------------------------------- AI 작사

    def request_ai_lyrics(self) -> None:
        """메뉴에서 부를 때. 가사 탭의 트랙과 지시를 쓴다."""
        self.side_tabs.setCurrentWidget(self.lyrics_panel)
        track = self.lyrics_panel.track
        if track is None:
            self.status.showMessage("가사를 붙일 트랙이 없습니다.", 3000)
            return
        self._on_ai_lyrics(track, self.lyrics_panel.direction_edit.text().strip())

    def _on_ai_lyrics(self, track: Track, direction: str) -> None:
        if self._lyrics_worker is not None and self._lyrics_worker.isRunning():
            self.status.showMessage("이미 작사하고 있습니다.", 2000)
            return
        try:
            slots, payload = prepare_lyrics(
                self.project, track,
                LyricsBrief(subject=self.project.meta.description, extra=direction),
            )
        except LyricsError as error:
            QtWidgets.QMessageBox.information(self, "작사할 수 없습니다", str(error))
            return
        self._lyrics_track = track
        self.lyrics_panel.set_busy(True)
        self.status.showMessage("작사 중...")
        self._lyrics_worker = LyricsWorker(slots, payload, self)
        self._lyrics_worker.progressed.connect(lambda text: self.status.showMessage(text))
        self._lyrics_worker.finished_ok.connect(self._on_lyrics_ready)
        self._lyrics_worker.failed.connect(self._on_lyrics_failed)
        self._lyrics_worker.start()

    def _on_lyrics_ready(self, draft: LyricsDraft) -> None:
        self.lyrics_panel.set_busy(False)
        track = self._lyrics_track
        if track is None or track not in list(self.project.tracks):
            self.status.showMessage("작사하는 동안 트랙이 사라졌습니다.", 4000)
            return
        if draft.is_stale(track):
            QtWidgets.QMessageBox.information(
                self, "멜로디가 바뀌었습니다",
                "가사를 받는 동안 멜로디가 바뀌어 붙이지 않았습니다. 다시 작사해 주세요.")
            return
        self.status.showMessage("가사가 왔습니다.", 3000)
        if LyricsPreviewDialog(draft, self).exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            command = lyrics_command(track, draft)
        except LyricsError as error:
            QtWidgets.QMessageBox.warning(self, "가사를 붙일 수 없습니다", str(error))
            return
        self._run(command)
        self.lyrics_panel.refresh()
        self.status.showMessage(
            "가사를 붙였습니다. F5 를 누르면 그 가사로 부릅니다. (Ctrl+Z 로 되돌림)", 8000)

    def _on_lyrics_failed(self, message: str) -> None:
        self.lyrics_panel.set_busy(False)
        QtWidgets.QMessageBox.critical(self, "작사하지 못했습니다", message)
        self.status.showMessage("작사 실패", 4000)

    def show_voice_studio(self) -> None:
        """목소리 학습 창. 만든 목소리를 이 곡의 보컬 트랙에 바로 쓸 수 있다."""
        from .voice_studio import VoiceStudio

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("내 AI 가수 — MYVOCAL Studio")
        dialog.resize(1100, 760)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        studio = VoiceStudio(palette=self._palette, can_apply=True, parent=dialog)
        studio.back_requested.connect(dialog.close)
        studio.voice_chosen.connect(lambda ref, style: (self.apply_voice(ref, style), dialog.close()))
        layout.addWidget(studio)
        dialog.exec()

    def apply_voice(self, reference: str, style: str) -> None:
        """보컬 트랙의 목소리와 창법을 바꾼다. 실행 취소 한 번에 둘 다 돌아간다."""
        vocals = [t for t in self.project.tracks if t.kind == "vocal"]
        if not vocals:
            QtWidgets.QMessageBox.information(self, "보컬 트랙 없음", "이 곡에는 보컬 트랙이 없습니다.")
            return
        track = self.piano_roll.track if self.piano_roll.track in vocals else vocals[0]
        self._run(H.CompositeCommand([
            H.SetTrackProperty(track, "voice_model", reference),
            H.SetTrackProperty(track, "singing_style", style),
        ], f"'{track.name}' 목소리 바꾸기"))
        self.track_panel.refresh()
        self.status.showMessage(f"'{track.name}' 의 목소리를 바꿨습니다. F5 를 누르면 그 목소리로 부릅니다.", 6000)

    def show_provider_settings(self) -> None:
        from .settings_dialog import ProviderSettingsDialog

        dialog = ProviderSettingsDialog(palette=self._palette, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.status.showMessage("AI 서비스 설정을 저장했습니다.", 4000)

    def delete_selected(self) -> None:
        notes = self.piano_roll.selected_notes()
        track = self.piano_roll.track
        if notes and track is not None:
            self._run(H.RemoveNotes(track, notes, "음 삭제"))
            self.piano_roll.selected.clear()

    def undo(self) -> None:
        if not self.history.can_undo:
            return
        description = self.history.undo()
        self.status.showMessage(f"되돌림: {description}", 2500)
        self._mark_dirty()
        self._refresh_sections()
        self.lyrics_panel.refresh()

    def redo(self) -> None:
        if not self.history.can_redo:
            return
        description = self.history.redo()
        self.status.showMessage(f"다시 실행: {description}", 2500)
        self._mark_dirty()
        self._refresh_sections()
        self.lyrics_panel.refresh()

    def _on_track_selected(self, track: Track) -> None:
        self.piano_roll.set_track(track, self.project.meter, self.project.tempo)
        self.status.showMessage(f"'{track.name}' 트랙 — 음 {len(track.notes)}개")

    def _on_track_property(self, track: Track, attribute: str, value: object) -> None:
        self._run(H.SetTrackProperty(track, attribute, value))

    def _on_track_remove(self, track: Track) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "트랙 삭제",
            f"'{track.name}' 트랙을 지울까요?\n음 {len(track.notes)}개가 함께 지워집니다.\n"
            f"실행 취소로 되돌릴 수 있습니다.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run(H.RemoveTrack(track))
        self.track_panel.set_tracks(self.project.tracks)

    def add_track(self) -> None:
        name, ok = QtWidgets.QInputDialog.getItem(
            self, "트랙 추가", "악기:",
            [create_instrument(n).info.display_name for n in available_instruments()],
            0, False,
        )
        if not ok:
            return
        instrument = next(
            n for n in available_instruments()
            if create_instrument(n).info.display_name == name
        )
        kind = "drum" if instrument == "drum_kit" else "instrument"
        track = Track(name=name, kind=kind, instrument=instrument)
        self._run(H.AddTrack(track))
        self.track_panel.set_tracks(self.project.tracks)
        self.track_panel.select(track)

    def _on_bpm_changed(self, value: float) -> None:
        self._run(H.SetProjectProperty("bpm", float(value), self.project))
        self.timeline.set_project_data(
            self.project.meter, self.project.tempo, self.project.structure
        )
        self.transport.set_project_data(
            self.project.meter, self.project.tempo, self.project.key
        )

    def _on_key_changed(self, text: str) -> None:
        try:
            key = Key.parse(text)
        except Exception as error:
            self.status.showMessage(f"조성을 읽을 수 없습니다: {error}", 3000)
            return
        self._run(H.SetProjectProperty("key", key, self.project))

    def _on_snap_changed(self, ticks: int) -> None:
        self.piano_roll.view.snap_enabled = ticks > 0
        self.piano_roll.view.snap_ticks = ticks or PPQ // 4

    def _on_structure_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        section = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if section is None:
            return
        span = self.project.section_range(section)
        self.piano_roll.view.scroll_tick = max(0, span.start_tick - PPQ)
        self.piano_roll.update()
        self.timeline.set_playhead(span.start_tick)
        self._on_seek(span.start_tick)

    def _on_section_clicked(self, section) -> None:
        for index in range(self.structure_list.count()):
            item = self.structure_list.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) is section:
                self.structure_list.setCurrentItem(item)
                break
        self.status.showMessage(
            f"{section.label} — {section.start_bar}~{section.end_bar - 1}마디, "
            f"에너지 {section.effective_energy():.0%}"
            + (f", 지시: {section.notes_for_ai}" if section.notes_for_ai else "")
        )

    def _zoom(self, factor: float) -> None:
        self.piano_roll.view.pixels_per_tick *= factor
        self.piano_roll.view.clamp()
        self.piano_roll.update()
        self.timeline.update()

    # ---------------------------------------------------------------- 재생

    def toggle_play(self) -> None:
        if self._rendered_audio is None or self._render_dirty:
            self.start_render(then_play=True)
            return
        if self.playback.is_playing:
            self.playback.pause()
            self.transport.set_playing(False)
        else:
            sounded = self.playback.play()
            self.transport.set_playing(True)
            if not sounded:
                self.status.showMessage(
                    "소리 장치가 없어 재생 위치만 움직입니다. "
                    "파일로 내보내면 들을 수 있습니다.", 5000
                )

    def stop_play(self) -> None:
        self.playback.stop()
        self.transport.set_playing(False)
        self.transport.meter_widget.reset()
        self._set_playhead(0)

    def _on_seek(self, tick: int) -> None:
        seconds = self.project.tempo.tick_to_seconds(max(0, tick))
        self.playback.seek(seconds)
        self._set_playhead(tick)

    def _on_loop_toggled(self, enabled: bool) -> None:
        if not enabled:
            self.playback.set_loop(None, None)
            self.timeline.loop = None
            self.timeline.update()
            return
        item = self.structure_list.currentItem()
        section = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        if section is None and len(self.project.structure):
            section = self.project.structure[0]
        if section is None:
            self.transport.loop_button.setChecked(False)
            self.status.showMessage("반복할 구간이 없습니다.", 3000)
            return
        span = self.project.section_range(section)
        start, end = section.seconds(self.project.meter, self.project.tempo)
        self.playback.set_loop(start, end)
        self.timeline.loop = (span.start_tick, span.end_tick)
        self.timeline.update()
        self.status.showMessage(f"'{section.label}' 구간을 반복합니다.", 3000)

    def _set_playhead(self, tick: int) -> None:
        self.timeline.set_playhead(tick)
        self.piano_roll.set_playhead(tick)

    def _tick(self) -> None:
        if self.playback.state is PlaybackState.PLAYING:
            seconds = self.playback.position_seconds
            tick = self.project.tempo.seconds_to_tick(seconds)
            self._set_playhead(tick)
            self.piano_roll.scroll_to_tick(tick)
            self.transport.set_position(tick)
            left, right = self.playback.peak_levels
            self.transport.set_levels(left, right)
            track = self.piano_roll.track
            if track is not None:
                self.keyboard.set_highlight(n.midi for n in track.notes.at_tick(tick))
            if self.playback.state is not PlaybackState.PLAYING:
                self.transport.set_playing(False)

    # ---------------------------------------------------------------- 렌더링

    def start_render(self, then_play: bool = False) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.status.showMessage("이미 소리를 만들고 있습니다.", 2000)
            return
        if not len(self.project.tracks):
            self.status.showMessage("트랙이 없습니다.", 3000)
            return
        self.transport.render_button.setEnabled(False)
        self._worker = RenderWorker(self.project, self.render_options, self)
        self._worker.progressed.connect(
            lambda stage, value: self.status.showMessage(
                f"소리 만드는 중 — {stage} {value * 100:.0f}%"
            )
        )
        self._worker.finished_ok.connect(
            lambda audio, report: self._on_render_done(audio, report, then_play)
        )
        self._worker.failed.connect(self._on_render_failed)
        self._worker.start()

    def _on_render_done(self, audio: AudioBuffer, report, then_play: bool) -> None:
        self._rendered_audio = audio
        self._render_dirty = False
        self.transport.render_button.setEnabled(True)
        self.transport.set_render_needed(False)
        self.playback.load(audio, keep_position=True)
        self.status.showMessage(
            f"소리 완성 — {report.duration_seconds:.1f}초를 "
            f"{report.elapsed_seconds:.1f}초에 만들었습니다 "
            f"(음색 재사용 {report.cache_rate:.0%})", 6000
        )
        if report.warnings:
            self.problem_label.setText("\n".join(f"· {w}" for w in report.warnings[:3]))
        if then_play:
            self.toggle_play()

    def _on_render_failed(self, message: str) -> None:
        self.transport.render_button.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "소리를 만들 수 없습니다", message)
        self.status.showMessage("소리 만들기 실패", 4000)

    # ---------------------------------------------------------------- 파일

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.load_project(Project("새 프로젝트"))

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "MYVOCAL 프로젝트 (*.mvp)"
        )
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "열 수 없습니다", str(error))
            return
        self.load_project(project)
        self.status.showMessage(f"{Path(path).name} 를 열었습니다.", 4000)

    def save_project(self, ask: bool = False) -> None:
        path = self.project.path
        if path is None or ask:
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "프로젝트 저장", self.project.meta.title,
                "MYVOCAL 프로젝트 (*.mvp)"
            )
            if not chosen:
                return
            path = Path(chosen)
        try:
            saved = self.project.save(path)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "저장할 수 없습니다", str(error))
            return
        self.history.mark_saved()
        self._update_title()
        self.status.showMessage(f"{saved.name} 에 저장했습니다.", 4000)

    def export_audio(self) -> None:
        from ..audio.export import AUDIO_FORMATS, available_formats, export_audio as write

        if self._rendered_audio is None or self._render_dirty:
            answer = QtWidgets.QMessageBox.question(
                self, "소리 만들기",
                "바뀐 내용이 반영되지 않았습니다. 먼저 소리를 만들까요?",
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self.start_render()
                return
            if self._rendered_audio is None:
                return
        formats = available_formats()
        labels = [f"{name} — {description}" for name, description in formats.items()]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "오디오 내보내기", "형식:", labels, 0, False
        )
        if not ok:
            return
        name = choice.split(" — ")[0]
        extension = AUDIO_FORMATS[name][2]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "저장 위치", self.project.meta.title, f"*{extension}"
        )
        if not path:
            return
        try:
            saved = write(self._rendered_audio, path, name)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "내보낼 수 없습니다", str(error))
            return
        size = saved.stat().st_size / 1024 / 1024
        self.status.showMessage(f"{saved.name} ({size:.1f} MB) 로 내보냈습니다.", 6000)

    def export_midi(self) -> None:
        from ..audio.export import export_midi as write

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "MIDI 내보내기", self.project.meta.title, "MIDI (*.mid)"
        )
        if not path:
            return
        try:
            saved = write(self.project, path)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "내보낼 수 없습니다", str(error))
            return
        self.status.showMessage(f"{saved.name} 로 내보냈습니다.", 5000)

    def export_stems(self) -> None:
        from ..audio.export import export_audio as write
        from ..audio.renderer import render_stems

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "트랙별 저장 폴더")
        if not folder:
            return
        try:
            written = render_stems(self.project, folder, self.render_options)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "내보낼 수 없습니다", str(error))
            return
        self.status.showMessage(f"트랙 {len(written)}개를 저장했습니다.", 6000)

    def _confirm_discard(self) -> bool:
        if not self.history.has_unsaved_changes:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "저장하지 않은 변경",
            "저장하지 않은 변경이 있습니다. 저장할까요?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            self.save_project()
        return True

    # ---------------------------------------------------------------- 도움말

    def show_audio_info(self) -> None:
        from ..audio.playback import audio_available, list_output_devices

        available, message = audio_available()
        devices = list_output_devices()
        lines = [message, ""]
        if devices:
            lines.append("찾은 장치:")
            lines.extend(f"  · {d.describe()}" for d in devices)
        QtWidgets.QMessageBox.information(self, "소리 장치", "\n".join(lines))

    def show_shortcuts(self) -> None:
        QtWidgets.QMessageBox.information(self, "단축키", """
재생
  스페이스          재생 / 일시정지
  Ctrl+.            정지
  F5                소리 만들기

편집
  Ctrl+Z / Ctrl+Y   실행 취소 / 다시 실행
  Ctrl+A            모두 선택
  Delete            선택한 음 삭제
  화살표            선택한 음 옮기기
  Shift+화살표      한 옥타브 / 한 박씩 옮기기

피아노롤
  Ctrl+클릭         음 추가
  더블클릭          음 추가
  오른쪽 클릭       음 삭제
  드래그            음 옮기기 (오른쪽 끝을 잡으면 길이 조절)
  가운데 버튼 드래그 화면 이동
  휠                위아래 이동
  Ctrl+휠           가로 확대
  Shift+휠          좌우 이동
  Alt+휠            세로 확대
""".strip())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        if self._lyrics_worker is not None and self._lyrics_worker.isRunning():
            self._lyrics_worker.terminate()
            self._lyrics_worker.wait(2000)
        self.playback.stop()
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()
