"""
AI 학습실 — 8번 (내 음악 스타일), 9번 (Reference), 63번 (자료 권한).

    왼쪽   넣은 곡 목록. 곡마다 자료 권한이 붙어 있다.
    오른쪽 고른 곡의 분석 결과
           [내 스타일 만들기]  체크한 곡들로 (Reference 자료는 못 씀)
           [Reference 로 새 곡] 고른 곡 하나를 참고로
           내 스타일 목록과 [이 스타일로 새 곡]
"""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..learning.analysis import MusicAnalysisError
from ..learning.library import (
    AUDIO_SUFFIXES, LEARNABLE, MIDI_SUFFIXES, PROJECT_SUFFIXES, RIGHTS, MusicLibrary,
)
from ..learning.style import StyleError
from .theme import DARK, Palette


class _Job(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, job, parent=None) -> None:
        super().__init__(parent)
        self._job = job

    def run(self) -> None:
        try:
            self.done.emit(self._job())
        except (MusicAnalysisError, StyleError) as error:
            self.failed.emit(str(error))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}\n\n{traceback.format_exc(limit=3)}")


class RightsDialog(QtWidgets.QDialog):
    """63번: 이 자료는 무엇인가."""

    def __init__(self, filename: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("자료 권한")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"'{filename}' 은(는):"))
        self.buttons: dict[str, QtWidgets.QRadioButton] = {}
        for key, label in RIGHTS.items():
            button = QtWidgets.QRadioButton(label)
            layout.addWidget(button)
            self.buttons[key] = button
        self.buttons["own"].setChecked(True)
        note = QtWidgets.QLabel(
            "앞의 셋은 '내 스타일' 학습에 쓸 수 있습니다.\n"
            "'Reference 분석용' 은 분석만 하고, 그 곡을 참고로 새 곡을 만들 때만 씁니다.\n"
            "학습 권한이 없는 곡은 Reference 로 넣어 주세요. 원본 파일은 복사하지 않습니다.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox()
        ok = buttons.addButton("분석하기", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setObjectName("Primary")
        buttons.addButton("취소", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def rights(self) -> str:
        return next(k for k, b in self.buttons.items() if b.isChecked())


class LearningStudio(QtWidgets.QWidget):
    """AI 학습실."""

    back_requested = QtCore.Signal()
    create_with_genre = QtCore.Signal(str)          # 내 스타일 장르 이름
    create_with_reference = QtCore.Signal(object)   # ReferenceHints

    def __init__(self, library: MusicLibrary | None = None, palette: Palette = DARK,
                 parent=None) -> None:
        super().__init__(parent)
        self.library = library or MusicLibrary()
        self._palette = palette
        self._job: _Job | None = None
        self._items = []

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(14)

        left = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("돌아가기")
        back.clicked.connect(self.back_requested)
        top.addWidget(back)
        heading = QtWidgets.QLabel("AI 학습실")
        heading.setObjectName("Heading")
        top.addWidget(heading, 1)
        left.addLayout(top)
        hint = QtWidgets.QLabel("곡을 넣으면 BPM, 조성, 코드, 구조, 리듬, 악기, 믹스를 분석합니다. "
                                "체크한 곡으로 '내 스타일' 을 만듭니다.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        left.addWidget(hint)
        self.item_list = QtWidgets.QListWidget()
        self.item_list.currentRowChanged.connect(self._on_item_selected)
        left.addWidget(self.item_list, 1)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("+ 곡 추가")
        add.setObjectName("Primary")
        add.clicked.connect(self.add_file)
        row.addWidget(add, 1)
        self.delete_button = QtWidgets.QPushButton("삭제")
        self.delete_button.clicked.connect(self.delete_item)
        row.addWidget(self.delete_button)
        left.addLayout(row)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(300)
        outer.addWidget(left_widget)

        right = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("Dim")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)
        self.report_view = QtWidgets.QPlainTextEdit()
        self.report_view.setReadOnly(True)
        right.addWidget(self.report_view, 3)
        actions = QtWidgets.QHBoxLayout()
        self.style_button = QtWidgets.QPushButton("체크한 곡으로 내 스타일 만들기")
        self.style_button.setObjectName("Primary")
        self.style_button.clicked.connect(self.build_style)
        actions.addWidget(self.style_button)
        self.reference_button = QtWidgets.QPushButton("이 곡을 Reference 로 새 곡 만들기")
        self.reference_button.clicked.connect(self.use_reference)
        actions.addWidget(self.reference_button)
        actions.addStretch(1)
        right.addLayout(actions)
        styles_title = QtWidgets.QLabel("MY MUSIC STYLE")
        styles_title.setObjectName("Heading")
        right.addWidget(styles_title)
        self.style_list = QtWidgets.QListWidget()
        self.style_list.currentRowChanged.connect(self._on_style_selected)
        self.style_list.setMaximumHeight(110)
        right.addWidget(self.style_list)
        style_actions = QtWidgets.QHBoxLayout()
        self.use_style_button = QtWidgets.QPushButton("이 스타일로 새 곡 만들기")
        self.use_style_button.clicked.connect(self.use_style)
        style_actions.addWidget(self.use_style_button)
        self.delete_style_button = QtWidgets.QPushButton("스타일 삭제")
        self.delete_style_button.clicked.connect(self.delete_style)
        style_actions.addWidget(self.delete_style_button)
        style_actions.addStretch(1)
        right.addLayout(style_actions)
        outer.addLayout(right, 1)

        self.library.register_all()
        self.refresh()

    # ------------------------------------------------------------ 목록

    def refresh(self, select_id: str | None = None) -> None:
        self._items = self.library.items()
        checked = {self.item_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                   for i in range(self.item_list.count())
                   if self.item_list.item(i).checkState() == QtCore.Qt.CheckState.Checked}
        self.item_list.blockSignals(True)
        self.item_list.clear()
        row_to_select = 0
        for row, item in enumerate(self._items):
            analysis = item.analysis
            key = f" · {analysis.key}" if analysis.key else ""
            bpm = f" · {analysis.bpm:.0f} BPM" if analysis.bpm else ""
            entry = QtWidgets.QListWidgetItem(f"{item.name}\n  [{RIGHTS[item.rights]}]{bpm}{key}")
            entry.setData(QtCore.Qt.ItemDataRole.UserRole, item.item_id)
            if item.learnable:
                entry.setFlags(entry.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                entry.setCheckState(QtCore.Qt.CheckState.Checked
                                    if item.item_id in checked or item.item_id == select_id
                                    else QtCore.Qt.CheckState.Unchecked)
            else:
                # 목록 항목은 기본으로 체크할 수 있게 되어 있다. Reference 는 막는다.
                entry.setFlags(entry.flags() & ~QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                entry.setToolTip("Reference 자료는 학습에 쓸 수 없습니다 (63번).")
            self.item_list.addItem(entry)
            if item.item_id == select_id:
                row_to_select = row
        self.item_list.blockSignals(False)
        if self._items:
            self.item_list.setCurrentRow(row_to_select)
            self._on_item_selected(row_to_select)
        else:
            self.report_view.setPlainText("왼쪽 '+ 곡 추가' 로 MIDI, 프로젝트(.mvp), 오디오 파일을 넣으세요.")
        self.delete_button.setEnabled(bool(self._items))
        self.reference_button.setEnabled(bool(self._items))

        self._styles = self.library.styles()
        self.style_list.clear()
        for style in self._styles:
            self.style_list.addItem(f"{style.name} — 곡 {len(style.sources)}개")
        self.use_style_button.setEnabled(bool(self._styles))
        self.delete_style_button.setEnabled(bool(self._styles))

    def _on_item_selected(self, row: int) -> None:
        if 0 <= row < len(self._items):
            item = self._items[row]
            self.report_view.setPlainText(f"[{RIGHTS[item.rights]}] {item.path}\n\n{item.analysis.summary()}")

    def _on_style_selected(self, row: int) -> None:
        if 0 <= row < len(self._styles):
            self.report_view.setPlainText(self._styles[row].describe())

    def checked_ids(self) -> list[str]:
        return [self.item_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                for i in range(self.item_list.count())
                if self.item_list.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    # ------------------------------------------------------------ 동작

    def add_file(self, path: str | None = None, rights: str | None = None) -> None:
        if path is None:
            patterns = " ".join(f"*{s}" for s in sorted(MIDI_SUFFIXES | PROJECT_SUFFIXES | AUDIO_SUFFIXES))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "곡 추가", "", f"음악 ({patterns})")
            if not path:
                return
        if rights is None:
            dialog = RightsDialog(Path(path).name, self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            rights = dialog.rights
        self.status_label.setText(f"{Path(path).name} 분석 중...")
        self._run(lambda: self.library.add(path, rights), lambda item: (
            self.status_label.setText(f"{item.name} 분석 완료"), self.refresh(item.item_id)))

    def delete_item(self) -> None:
        row = self.item_list.currentRow()
        if 0 <= row < len(self._items):
            self.library.delete(self._items[row].item_id)
            self.refresh()

    def build_style(self, name: str | None = None) -> None:
        ids = self.checked_ids()
        if not ids:
            QtWidgets.QMessageBox.information(self, "곡을 고르세요",
                                              "학습에 쓸 곡을 체크해 주세요 (Reference 자료는 제외).")
            return
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "내 스타일", "스타일 이름:", text="MY MUSIC STYLE")
            if not ok or not name.strip():
                return
        self.status_label.setText("스타일 만드는 중...")
        self._run(lambda: self.library.build_style(name, ids), lambda style: (
            self.status_label.setText(f"'{style.name}' 을 만들었습니다."), self.refresh(),
            self.report_view.setPlainText(style.describe())))

    def use_style(self) -> None:
        row = self.style_list.currentRow()
        if row < 0 and self._styles:
            row = 0
        if 0 <= row < len(self._styles):
            from ..learning.style import register_style
            self.create_with_genre.emit(register_style(self._styles[row]))

    def delete_style(self) -> None:
        row = self.style_list.currentRow()
        if 0 <= row < len(self._styles):
            self.library.delete_style(self._styles[row].style_id)
            self.refresh()

    def use_reference(self) -> None:
        row = self.item_list.currentRow()
        if 0 <= row < len(self._items):
            self.create_with_reference.emit(self.library.reference(self._items[row].item_id))

    def _run(self, job, done) -> None:
        if self._job is not None:
            return
        self._job = _Job(job, self)
        self._job.done.connect(lambda result: (self._finish(), done(result)))
        self._job.failed.connect(lambda message: (self._finish(), self.status_label.setText(""),
                                                  QtWidgets.QMessageBox.warning(self, "할 수 없습니다", message)))
        self._job.start()

    def _finish(self) -> None:
        if self._job is not None:
            self._job.wait(2000)
        self._job = None
