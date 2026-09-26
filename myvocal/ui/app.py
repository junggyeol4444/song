"""
프로그램 진입점.

시작 화면 -> 무엇을 할지 고름 -> 편집기.
창을 여러 개 띄우지 않고 한 창 안에서 화면을 바꾼다. 창이 여러 개면
사용자가 어느 창이 지금 작업 중인지 헷갈린다.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.project import Project
from .create_dialog import CreateSongDialog
from .main_window import MainWindow
from .start_screen import StartScreen, find_recent_projects
from .theme import DARK, LIGHT, Palette, build_stylesheet


def default_project_folder() -> Path:
    """프로젝트를 기본으로 두는 곳. 윈도우면 문서 폴더 안이다."""
    documents = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.DocumentsLocation
    )
    base = Path(documents) if documents else Path.home()
    return base / "MYVOCAL Studio"


class ComposeWorker(QtCore.QThread):
    """작곡과 작사를 딴 실뜨기에서 한다. 화면이 멈추는 것을 막는다.

    이 프로젝트는 아직 화면에 없으므로 여기서 고쳐도 된다.
    """

    finished_ok = QtCore.Signal(object, str)     # Project, 알릴 말
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str)

    def __init__(self, request, with_melody: bool, with_arrangement: bool,
                 with_lyrics: bool = False,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._request = request
        self._with_melody = with_melody
        self._with_arrangement = with_arrangement
        self._with_lyrics = with_lyrics and with_melody

    def run(self) -> None:
        try:
            from ..music.composer import Composer

            self.progressed.emit("곡을 만들고 있습니다...")
            project = Composer(self._request).compose(
                with_melody=self._with_melody,
                with_arrangement=self._with_arrangement,
            )
        except Exception as error:
            self.failed.emit(
                f"{type(error).__name__}: {error}\n\n{traceback.format_exc(limit=4)}"
            )
            return
        message = ""
        if self._with_lyrics:
            # 작사가 실패해도 곡은 살린다. 가사는 나중에 가사 탭에서 다시 쓸 수 있다.
            try:
                message = self._write_lyrics(project)
            except Exception as error:
                message = f"작사하지 못했습니다 ({error}). 가사 탭에서 다시 시도할 수 있습니다."
        self.finished_ok.emit(project, message)

    def _write_lyrics(self, project: Project) -> str:
        from ..music.lyrics import LyricsBrief, lyrics_command, vocal_track, write_lyrics
        from ..providers import default_registry

        self.progressed.emit("가사를 쓰고 있습니다...")
        track = vocal_track(project)
        draft = write_lyrics(
            project, track, default_registry(),
            LyricsBrief(subject=self._request.subject, title=self._request.title,
                        seed=self._request.seed),
            on_progress=self.progressed.emit,
        )
        lyrics_command(track, draft).apply(project)
        who = "Claude" if draft.provider == "anthropic" else "내장 규칙"
        return f"가사: {who}" + (f" — {draft.note}" if draft.note else "")


class Shell(QtWidgets.QMainWindow):
    """시작 화면과 편집기를 담는 껍데기 창."""

    def __init__(self, palette: Palette = DARK) -> None:
        super().__init__()
        self._palette = palette
        self._worker: ComposeWorker | None = None
        self.setWindowTitle("MYVOCAL Studio")
        self.resize(1200, 820)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_screen = StartScreen(
            find_recent_projects(default_project_folder()), palette
        )
        self.start_screen.chosen.connect(self._on_menu_chosen)
        self.start_screen.open_project.connect(self._open_path)
        self.stack.addWidget(self.start_screen)

        self.editor: MainWindow | None = None
        # 학습실에서 만든 '내 스타일' 을 장르 목록에 올린다 (8번)
        try:
            from ..learning.library import MusicLibrary
            MusicLibrary().register_all()
        except Exception:
            pass
        self._progress: QtWidgets.QProgressDialog | None = None

    # ---------------------------------------------------------------- 흐름

    def _on_menu_chosen(self, key: str) -> None:
        if key == "auto_song":
            self.create_song()
        elif key == "voice_train":
            self.open_voice_studio()
        elif key == "learning":
            self.open_learning_studio()
        elif key == "compose":
            self.open_editor(Project("새 프로젝트"))
        elif key == "open":
            self.open_project_dialog()
        else:
            entry = next((e for e in self.start_screen._buttons.values()
                          if e.entry.key == key), None)
            reason = entry.entry.unavailable_reason if entry else "아직 만들고 있습니다."
            QtWidgets.QMessageBox.information(self, "아직 준비 중", reason)

    def open_learning_studio(self) -> None:
        from .learning_studio import LearningStudio

        if getattr(self, "learning_studio", None) is None:
            self.learning_studio = LearningStudio(palette=self._palette)
            self.learning_studio.back_requested.connect(
                lambda: self.stack.setCurrentWidget(self.start_screen))
            self.learning_studio.create_with_genre.connect(
                lambda name: self.create_song(genre=name))
            self.learning_studio.create_with_reference.connect(
                lambda hints: self.create_song(reference=hints))
            self.stack.addWidget(self.learning_studio)
        else:
            self.learning_studio.refresh()
        self.stack.setCurrentWidget(self.learning_studio)

    def create_song(self, genre: str | None = None, reference=None) -> None:
        dialog = CreateSongDialog(self._palette, self)
        if genre:
            dialog.preset_genre(genre)
        if reference is not None:
            dialog.set_reference(reference)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        request = dialog.to_request()
        self._progress = QtWidgets.QProgressDialog(
            "곡을 만들고 있습니다...", "", 0, 0, self
        )
        self._progress.setWindowTitle("만드는 중")
        self._progress.setCancelButton(None)
        self._progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress.show()

        self._worker = ComposeWorker(
            request, dialog.wants("melody"), dialog.wants("arrangement"),
            dialog.wants("lyrics"), self
        )
        self._worker.progressed.connect(
            lambda text: self._progress.setLabelText(text) if self._progress else None)
        self._worker.finished_ok.connect(self._on_composed)
        self._worker.failed.connect(self._on_compose_failed)
        self._worker.start()

    def _on_composed(self, project: Project, message: str = "") -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self.open_editor(project)
        if self.editor is not None:
            extra = f" {message}" if message else ""
            self.editor.status.showMessage(
                f"'{project.meta.title}' 을 만들었습니다. "
                f"F5 를 누르면 소리로 만듭니다.{extra}", 12000
            )

    def _on_compose_failed(self, message: str) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        QtWidgets.QMessageBox.critical(self, "곡을 만들 수 없습니다", message)

    def open_voice_studio(self) -> None:
        from .voice_studio import VoiceStudio

        if getattr(self, "voice_studio", None) is None:
            self.voice_studio = VoiceStudio(palette=self._palette)
            self.voice_studio.back_requested.connect(
                lambda: self.stack.setCurrentWidget(self.start_screen))
            self.stack.addWidget(self.voice_studio)
        else:
            self.voice_studio.refresh_list()
        self.stack.setCurrentWidget(self.voice_studio)

    def open_project_dialog(self) -> None:
        folder = default_project_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "프로젝트 열기", str(folder), "MYVOCAL 프로젝트 (*.mvp)"
        )
        if path:
            self._open_path(Path(path))

    def _open_path(self, path: Path) -> None:
        try:
            project = Project.load(path)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "열 수 없습니다", str(error))
            return
        self.open_editor(project)

    def open_editor(self, project: Project) -> None:
        if self.editor is None:
            self.editor = MainWindow(project, self._palette)
            self.stack.addWidget(self.editor)
            # 편집기의 메뉴를 껍데기 창의 메뉴로 쓴다
            self.setMenuBar(self.editor.menuBar())
            self.setStatusBar(self.editor.statusBar())
        else:
            self.editor.load_project(project)
        self.stack.setCurrentWidget(self.editor)
        self.resize(max(self.width(), 1440), max(self.height(), 880))
        self.setWindowTitle(f"{project.meta.title} — MYVOCAL Studio")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.editor is not None:
            self.editor.closeEvent(event)
            if not event.isAccepted():
                return
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        event.accept()


def install_exception_hook(window: QtWidgets.QWidget | None = None) -> None:
    """예상 못 한 오류가 나도 프로그램이 조용히 죽지 않게 한다.

    Qt 안에서 예외가 나면 기본 동작은 콘솔에 찍고 넘어가는 것이다. 창에서
    쓰는 사람은 콘솔을 안 보므로, 아무 일도 안 일어난 것처럼 보인다.
    """
    def handler(kind, value, tb) -> None:
        message = "".join(traceback.format_exception(kind, value, tb))
        sys.stderr.write(message)
        try:
            QtWidgets.QMessageBox.critical(
                window, "오류가 났습니다",
                f"{kind.__name__}: {value}\n\n"
                f"아래 내용을 알려주시면 고칠 수 있습니다.\n\n"
                f"{message[-1500:]}",
            )
        except Exception:
            pass

    sys.excepthook = handler


def run(argv: list[str] | None = None) -> int:
    """프로그램을 시작한다."""
    argv = list(argv if argv is not None else sys.argv)
    application = QtWidgets.QApplication(argv)
    application.setApplicationName("MYVOCAL Studio")
    application.setOrganizationName("MYVOCAL")

    # 화면 배율이 다른 모니터에서도 글씨가 흐려지지 않게 한다
    application.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True
    )
    application.setStyleSheet(build_stylesheet(DARK))

    shell = Shell(DARK)
    install_exception_hook(shell)
    shell.show()

    # 명령줄로 프로젝트 파일을 주면 바로 연다
    for argument in argv[1:]:
        candidate = Path(argument)
        if candidate.suffix == ".mvp" and candidate.exists():
            shell._open_path(candidate)
            break

    return application.exec()
