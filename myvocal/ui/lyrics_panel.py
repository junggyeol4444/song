"""
가사 편집 — 음에 가사를 붙이는 곳.

보컬 트랙의 음들을 순서대로 늘어놓고, 한 줄을 쓰면 음절이 차례로 붙는다.
음 하나하나에 글자를 따로 넣게 하면 음이 백 개일 때 못 쓴다.

구간(Verse / Chorus)으로 나눠 보여준다. 가사는 구간 단위로 쓰기 때문이다.
'후렴 가사를 고치고 싶다' 가 가장 흔한 요구인데, 음 200개가 한 줄로 늘어서
있으면 어디가 후렴인지 찾을 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.notes import Note
from ..core.project import Project
from ..core.tracks import Track
from ..music.structure import SECTION_COLORS, Section
from ..music.theory import Pitch
from ..voice.korean import count_syllables, pronounce, split_syllables, syllables_with_word_ends
from .theme import DARK, Palette


@dataclass(slots=True)
class LyricLine:
    """한 구간의 가사와 그 구간의 음들."""

    section: Section | None
    notes: list[Note]

    @property
    def label(self) -> str:
        return self.section.label if self.section is not None else "구간 밖"

    @property
    def color(self) -> str:
        if self.section is None:
            return "#888888"
        return SECTION_COLORS.get(self.section.kind, "#888888")

    @property
    def phrases(self) -> list[list[Note]]:
        """숨 쉬는 자리(쉼표)로 나눈 줄들."""
        from ..music.lyrics import melody_lines
        return melody_lines(self.notes)

    @property
    def text(self) -> str:
        """줄마다 끊어서 보여준다. 한 줄로 이어 붙이면 어디서 숨 쉬는지 안 보이고,
        칸 밖으로 넘친다."""
        lines = ["".join(n.lyric + (" " if n.word_end else "") for n in phrase).strip()
                 for phrase in self.phrases]
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines) if any(lines) else ""

    @property
    def filled(self) -> int:
        return sum(1 for n in self.notes if n.lyric)


class SectionLyricEditor(QtWidgets.QFrame):
    """구간 하나의 가사 입력칸."""

    changed = QtCore.Signal(object, str)     # LyricLine, 새 가사
    focused = QtCore.Signal(object)          # LyricLine

    def __init__(self, line: LyricLine, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.line = line
        self._palette = palette
        self.setObjectName("Card")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(5)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        name = QtWidgets.QLabel(line.label)
        name.setStyleSheet(
            f"background: transparent; color: {line.color}; font-weight: 600;"
        )
        header.addWidget(name)
        header.addStretch(1)
        self.count_label = QtWidgets.QLabel()
        self.count_label.setObjectName("Faint")
        self.count_label.setStyleSheet("background: transparent;")
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setPlainText(line.text)
        pattern = " / ".join(str(len(p)) for p in line.phrases)
        self.editor.setPlaceholderText(
            f"이 구간의 가사 ({len(line.notes)}음절 — 줄마다 {pattern})"
        )
        self.editor.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.editor.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows = max(2, len(line.phrases))
        spacing = self.editor.fontMetrics().lineSpacing()
        self.editor.setFixedHeight(rows * spacing + 26)
        self.editor.setTabChangesFocus(True)
        layout.addWidget(self.editor)

        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setObjectName("Faint")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.hint_label)

        self.editor.textChanged.connect(self._on_changed)
        self.editor.focusInEvent = self._wrap_focus(self.editor.focusInEvent)
        self._update_labels()

    def _wrap_focus(self, original):
        def handler(event):
            self.focused.emit(self.line)
            original(event)
        return handler

    def _on_changed(self) -> None:
        self._update_labels()
        self.changed.emit(self.line, self.editor.toPlainText())

    def _update_labels(self) -> None:
        text = self.editor.toPlainText()
        written = count_syllables(text)
        total = len(self.line.notes)
        self.count_label.setText(f"{written} / {total}")
        if written == total:
            color = self._palette.success
            hint = ""
        elif written < total:
            color = self._palette.text_dim
            hint = f"{total - written}음절 더 쓸 수 있습니다."
        else:
            color = self._palette.warning
            hint = f"{written - total}음절이 남습니다. 남는 글자는 부르지 않습니다."
        self.count_label.setStyleSheet(f"background: transparent; color: {color};")

        if text.strip():
            # 발음 규칙은 한 호흡 안에서만 적용된다. 줄을 넘어 이어 읽지 않는다.
            lines = [line for line in text.splitlines() if line.strip()]
            spoken_lines = [pronounce(line) for line in lines]
            if spoken_lines != lines:
                spoken = "\n".join(spoken_lines)
                hint = (hint + "\n" if hint else "") + f"실제 발음:\n{spoken}"
        self.hint_label.setText(hint)

    def refresh(self) -> None:
        self.editor.blockSignals(True)
        self.editor.setPlainText(self.line.text)
        self.editor.blockSignals(False)
        self._update_labels()


class LyricsPanel(QtWidgets.QWidget):
    """가사 편집 화면 전체."""

    lyrics_applied = QtCore.Signal(object, list)   # Track, [(Note, 새 Note), ...]
    note_focused = QtCore.Signal(int)              # tick
    ai_requested = QtCore.Signal(object, str)      # Track, 작사 지시

    def __init__(self, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.project: Project | None = None
        self.track: Track | None = None
        self._editors: list[SectionLyricEditor] = []
        self._pending: dict[int, str] = {}      # 구간 id -> 아직 적용 안 한 가사

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("가사")
        title.setObjectName("Heading")
        header.addWidget(title)
        header.addStretch(1)
        self.track_box = QtWidgets.QComboBox()
        self.track_box.setMinimumWidth(130)
        self.track_box.currentIndexChanged.connect(self._on_track_changed)
        header.addWidget(self.track_box)
        layout.addLayout(header)

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setObjectName("Dim")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # 가로로 넘치면 글자가 잘려 보인다. 가로 스크롤 대신 칸이 폭에 맞춘다.
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._container = QtWidgets.QWidget()
        self._container_layout = QtWidgets.QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 6, 0)
        self._container_layout.setSpacing(9)
        self._container_layout.addStretch(1)
        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)

        ai_row = QtWidgets.QHBoxLayout()
        self.direction_edit = QtWidgets.QLineEdit()
        self.direction_edit.setPlaceholderText("작사 지시 (선택)")
        self.direction_edit.setToolTip("예: 후렴은 희망적으로 / 벌스는 담담하게")
        ai_row.addWidget(self.direction_edit, 1)
        self.ai_button = QtWidgets.QPushButton("✨ AI 작사")
        self.ai_button.setToolTip(
            "멜로디의 음 개수에 맞춰 가사를 씁니다. 곡 설명을 주제로 씁니다.\n"
            "API 키가 있으면 Claude 가, 없으면 내장 규칙이 씁니다."
        )
        self.ai_button.clicked.connect(self._on_ai_clicked)
        ai_row.addWidget(self.ai_button)
        layout.addLayout(ai_row)

        buttons = QtWidgets.QHBoxLayout()
        self.clear_button = QtWidgets.QPushButton("가사 지우기")
        self.clear_button.clicked.connect(self.clear_lyrics)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("음에 붙이기")
        self.apply_button.setObjectName("Primary")
        self.apply_button.clicked.connect(self.apply_lyrics)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        note = QtWidgets.QLabel(
            "한 줄을 쓰면 음절이 음에 차례로 붙습니다. 띄어쓰기와 문장부호는 "
            "세지 않습니다. 붙인 뒤 '소리 만들기' 를 누르면 그 가사로 부릅니다."
        )
        note.setObjectName("Faint")
        note.setWordWrap(True)
        layout.addWidget(note)

    # ---------------------------------------------------------------- AI 작사

    def _on_ai_clicked(self) -> None:
        if self.track is None:
            return
        self.ai_requested.emit(self.track, self.direction_edit.text().strip())

    def set_busy(self, busy: bool) -> None:
        """작사 중에는 단추를 잠근다. 두 번 누르면 요청이 두 번 나간다."""
        self.ai_button.setEnabled(not busy)
        self.ai_button.setText("작사 중..." if busy else "✨ AI 작사")
        self.apply_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)

    # ---------------------------------------------------------------- 자료

    def set_project(self, project: Project) -> None:
        self.project = project
        self.track_box.blockSignals(True)
        self.track_box.clear()
        vocal_tracks = project.tracks.of_kind("vocal")
        if not vocal_tracks:
            vocal_tracks = list(project.tracks)
        for track in vocal_tracks:
            self.track_box.addItem(track.name, track)
        self.track_box.blockSignals(False)
        self.track = vocal_tracks[0] if vocal_tracks else None
        self._rebuild()

    def _on_track_changed(self, index: int) -> None:
        self.track = self.track_box.itemData(index)
        self._rebuild()

    def _lines(self) -> list[LyricLine]:
        if self.project is None or self.track is None:
            return []
        lines: list[LyricLine] = []
        remaining = list(self.track.notes)
        for section in self.project.structure:
            span = self.project.section_range(section)
            inside = [n for n in remaining if span.contains(n.start_tick)]
            if inside:
                lines.append(LyricLine(section, inside))
        covered = {id(n) for line in lines for n in line.notes}
        outside = [n for n in remaining if id(n) not in covered]
        if outside:
            lines.append(LyricLine(None, outside))
        return lines

    def _rebuild(self) -> None:
        for editor in self._editors:
            editor.setParent(None)
            editor.deleteLater()
        self._editors.clear()
        self._pending.clear()

        if self.track is None:
            self.summary_label.setText("보컬 트랙이 없습니다.")
            return

        lines = self._lines()
        if not lines:
            self.summary_label.setText(
                f"'{self.track.name}' 트랙에 음이 없습니다. "
                f"먼저 멜로디를 만들거나 그려 넣으세요."
            )
            return

        total = sum(len(line.notes) for line in lines)
        filled = sum(line.filled for line in lines)
        self.summary_label.setText(
            f"'{self.track.name}' — 음 {total}개 중 {filled}개에 가사가 있습니다."
        )

        for index, line in enumerate(lines):
            editor = SectionLyricEditor(line, self._palette)
            editor.changed.connect(self._on_line_changed)
            editor.focused.connect(self._on_line_focused)
            self._container_layout.insertWidget(index, editor)
            self._editors.append(editor)

    def _on_line_changed(self, line: LyricLine, text: str) -> None:
        key = line.section.section_id if line.section is not None else -1
        self._pending[key] = text

    def _on_line_focused(self, line: LyricLine) -> None:
        if line.notes:
            self.note_focused.emit(line.notes[0].start_tick)

    # ---------------------------------------------------------------- 적용

    def apply_lyrics(self) -> None:
        """써 놓은 가사를 음에 붙인다."""
        if self.track is None:
            return
        changes: list[tuple[Note, Note]] = []
        for editor in self._editors:
            line = editor.line
            text = editor.editor.toPlainText()
            pieces = syllables_with_word_ends(text)
            for index, note in enumerate(line.notes):
                syllable, word_end = pieces[index] if index < len(pieces) else ("", False)
                if (syllable, word_end) != (note.lyric, note.word_end):
                    changes.append((note, note.with_lyric(syllable, word_end=word_end)))
        if changes:
            self.lyrics_applied.emit(self.track, changes)

    def clear_lyrics(self) -> None:
        if self.track is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "가사 지우기",
            f"'{self.track.name}' 의 가사를 모두 지울까요?\n실행 취소로 되돌릴 수 있습니다.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        changes = [(n, n.with_lyric("")) for n in self.track.notes if n.lyric]
        if changes:
            self.lyrics_applied.emit(self.track, changes)

    def refresh(self) -> None:
        self._rebuild()
