"""
AI 서비스 설정 — API 키와 모델.

키는 여기서 넣고, 설정 파일(사용자 폴더)에만 저장한다. 프로젝트 파일에는
절대 들어가지 않는다. 화면에는 가려서 보여준다.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..providers import Capability, default_registry
from ..providers.anthropic_provider import MODEL_SETTING, AnthropicProvider
from ..providers.settings import ENVIRONMENT_KEYS, Settings, shared_settings
from .theme import DARK, Palette


class ProviderSettingsDialog(QtWidgets.QDialog):
    """AI 서비스 설정 창."""

    def __init__(self, settings: Settings | None = None, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings or shared_settings()
        self.setWindowTitle("AI 서비스 설정")
        self.setMinimumWidth(560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            "키가 없어도 프로그램은 전부 동작합니다. 작사는 내장 규칙으로 합니다.\n"
            "Anthropic API 키를 넣으면 작사를 Claude 가 합니다. 실패하면 내장으로 돌아갑니다."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Dim")
        layout.addWidget(intro)

        # --- Anthropic ---
        group = QtWidgets.QGroupBox("Claude (Anthropic)")
        form = QtWidgets.QFormLayout(group)

        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("sk-ant-... (비우면 지워집니다)")
        stored = self.settings.get("api_keys", {}).get("anthropic", "")
        self.key_edit.setText(stored)
        show = QtWidgets.QCheckBox("보이기")
        show.toggled.connect(lambda on: self.key_edit.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Normal if on else QtWidgets.QLineEdit.EchoMode.Password))
        key_row = QtWidgets.QHBoxLayout()
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(show)
        form.addRow("API 키", key_row)

        self.source_label = QtWidgets.QLabel()
        self.source_label.setObjectName("Faint")
        self.source_label.setWordWrap(True)
        form.addRow("", self.source_label)

        self.model_edit = QtWidgets.QLineEdit(str(self.settings.get(MODEL_SETTING, "") or ""))
        self.model_edit.setPlaceholderText("비우면 계정에서 쓸 수 있는 가장 최근 모델")
        form.addRow("모델", self.model_edit)

        self.use_claude = QtWidgets.QCheckBox("작사에 Claude 를 먼저 쓴다")
        preferred = (self.settings.get("preferred_providers", {}) or {}).get(
            Capability.LYRICS.value, "anthropic")
        self.use_claude.setChecked(preferred == "anthropic")
        form.addRow("", self.use_claude)

        cost = QtWidgets.QLabel(AnthropicProvider.info.cost_note)
        cost.setObjectName("Faint")
        cost.setWordWrap(True)
        form.addRow("비용", cost)
        layout.addWidget(group)

        # --- 상태 ---
        self.status_view = QtWidgets.QPlainTextEdit()
        self.status_view.setReadOnly(True)
        self.status_view.setMinimumHeight(170)
        layout.addWidget(QtWidgets.QLabel("지금 쓸 수 있는 것"))
        layout.addWidget(self.status_view)

        path_label = QtWidgets.QLabel(f"설정 파일: {self.settings.path}")
        path_label.setObjectName("Faint")
        path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setText("저장")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh()

    def _refresh(self) -> None:
        variable = ENVIRONMENT_KEYS["anthropic"]
        source = self.settings.key_source("anthropic")
        if source.startswith("환경 변수"):
            self.source_label.setText(
                f"환경 변수 {variable} 가 있어 그 키를 씁니다 "
                f"({self.settings.masked_key('anthropic')}). 여기 넣은 키보다 우선합니다.")
        elif source == "없음":
            self.source_label.setText("키가 없습니다. 작사는 내장 규칙으로 합니다.")
        else:
            self.source_label.setText(f"저장된 키: {self.settings.masked_key('anthropic')}")
        if not AnthropicProvider.sdk_available():
            self.source_label.setText(self.source_label.text()
                                      + "\nanthropic 패키지가 없습니다: pip install anthropic")
        self.status_view.setPlainText(default_registry(self.settings).status_report())

    def save(self) -> None:
        self.settings.set_api_key("anthropic", self.key_edit.text())
        self.settings.set(MODEL_SETTING, self.model_edit.text().strip())
        preferred = dict(self.settings.get("preferred_providers", {}) or {})
        preferred[Capability.LYRICS.value] = "anthropic" if self.use_claude.isChecked() else "local"
        self.settings.set("preferred_providers", preferred)
        try:
            self.settings.save()
        except OSError as error:
            QtWidgets.QMessageBox.critical(self, "저장할 수 없습니다", str(error))
            return
        self.accept()
