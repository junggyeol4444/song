"""
노래 만들기 화면 — 55번의 CREATE SONG & VIDEO.

문서의 예시를 그대로 넣을 수 있어야 한다.

    설명:   애니메이션 록 발라드. 마지막 후렴은 웅장하게.
    Voice:  MY VOICE
    Genre:  Rock + Orchestral
    Video:  자동 캐릭터 / 내 캐릭터 / 실제 사람

아직 안 만들어진 것(목소리, 영상)은 숨기지 않고 꺼 둔 채로 이유를 적는다.
숨기면 "이 프로그램은 그런 거 안 되는구나" 라고 오해하게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from ..music.genre import GENRES, GenreBlend, available_genres, get_genre
from ..music.melody import VocalRange
from ..music.theory import Key
from .theme import DARK, Palette


class GenreMixRow(QtWidgets.QWidget):
    """장르 하나와 비율."""

    changed = QtCore.Signal()
    removed = QtCore.Signal(object)

    def __init__(self, genre: str = "pop", weight: int = 100,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.genre_box = QtWidgets.QComboBox()
        for name in available_genres():
            self.genre_box.addItem(GENRES[name].display_name, name)
        index = self.genre_box.findData(genre)
        if index >= 0:
            self.genre_box.setCurrentIndex(index)
        self.genre_box.currentIndexChanged.connect(self.changed)
        layout.addWidget(self.genre_box, 2)

        self.weight_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(weight)
        self.weight_slider.valueChanged.connect(self._on_weight)
        layout.addWidget(self.weight_slider, 3)

        self.weight_label = QtWidgets.QLabel(f"{weight}%")
        self.weight_label.setFixedWidth(40)
        self.weight_label.setObjectName("Dim")
        layout.addWidget(self.weight_label)

        self.remove_button = QtWidgets.QPushButton("✕")
        self.remove_button.setFixedSize(26, 26)
        self.remove_button.setObjectName("Toggle")
        self.remove_button.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.remove_button)

    def _on_weight(self, value: int) -> None:
        self.weight_label.setText(f"{value}%")
        self.changed.emit()

    @property
    def genre(self) -> str:
        return self.genre_box.currentData()

    @property
    def weight(self) -> int:
        return self.weight_slider.value()


class CreateSongDialog(QtWidgets.QDialog):
    """새 곡 만들기."""

    def __init__(self, palette: Palette = DARK,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setWindowTitle("AI로 노래 만들기")
        self.setMinimumWidth(620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("어떤 노래를 만들까요?")
        title.setObjectName("Heading")
        layout.addWidget(title)

        # --- 설명 ---
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setPlaceholderText(
            "예) 애니메이션 록 발라드를 만들어줘.\n"
            "주제: 오래 헤어져 있던 친구를 다시 만나는 이야기\n"
            "마지막 후렴은 웅장하게."
        )
        self.description.setFixedHeight(92)
        layout.addWidget(self.description)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("제목을 비우면 '새 곡' 이 됩니다")
        form.addRow("제목", self.title_edit)

        # --- 장르 ---
        genre_box = QtWidgets.QWidget()
        genre_layout = QtWidgets.QVBoxLayout(genre_box)
        genre_layout.setContentsMargins(0, 0, 0, 0)
        genre_layout.setSpacing(6)
        self._genre_rows: list[GenreMixRow] = []
        self._genre_container = QtWidgets.QVBoxLayout()
        self._genre_container.setSpacing(6)
        genre_layout.addLayout(self._genre_container)

        genre_buttons = QtWidgets.QHBoxLayout()
        add_genre = QtWidgets.QPushButton("+ 장르 섞기")
        add_genre.clicked.connect(lambda: self._add_genre_row())
        genre_buttons.addWidget(add_genre)
        genre_buttons.addStretch(1)
        self.genre_summary = QtWidgets.QLabel()
        self.genre_summary.setObjectName("Faint")
        genre_buttons.addWidget(self.genre_summary)
        genre_layout.addLayout(genre_buttons)
        form.addRow("장르", genre_box)
        self._add_genre_row("ballad", 100)

        # --- 조성 / BPM ---
        settings = QtWidgets.QHBoxLayout()
        settings.setSpacing(8)
        self.key_auto = QtWidgets.QCheckBox("조성 자동")
        self.key_auto.setChecked(True)
        self.key_auto.toggled.connect(lambda on: self.key_box.setEnabled(not on))
        settings.addWidget(self.key_auto)
        self.key_box = QtWidgets.QComboBox()
        for tonic in ("C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"):
            self.key_box.addItem(tonic)
            self.key_box.addItem(f"{tonic}m")
        self.key_box.setEnabled(False)
        settings.addWidget(self.key_box)
        settings.addSpacing(14)
        self.bpm_auto = QtWidgets.QCheckBox("BPM 자동")
        self.bpm_auto.setChecked(True)
        self.bpm_auto.toggled.connect(lambda on: self.bpm_box.setEnabled(not on))
        settings.addWidget(self.bpm_auto)
        self.bpm_box = QtWidgets.QSpinBox()
        self.bpm_box.setRange(40, 250)
        self.bpm_box.setValue(100)
        self.bpm_box.setEnabled(False)
        settings.addWidget(self.bpm_box)
        settings.addStretch(1)
        form.addRow("", settings)

        # --- 목소리 ---
        self.voice_box = QtWidgets.QComboBox()
        self.voice_box.addItem("소프라노 (높은 여성)", "soprano")
        self.voice_box.addItem("메조 (중간 여성)", "mezzo")
        self.voice_box.addItem("알토 (낮은 여성)", "alto")
        self.voice_box.addItem("테너 (높은 남성)", "tenor")
        self.voice_box.addItem("바리톤 (중간 남성)", "baritone")
        self.voice_box.addItem("베이스 (낮은 남성)", "bass")
        self.voice_box.setCurrentIndex(1)
        form.addRow("음역", self.voice_box)

        voice_note = QtWidgets.QLabel(
            "내 목소리로 부르게 하려면 목소리 학습이 필요합니다. 아직 만들고 있습니다.\n"
            "지금은 멜로디만 코러스 음색으로 나옵니다."
        )
        voice_note.setObjectName("Faint")
        voice_note.setWordWrap(True)
        form.addRow("", voice_note)

        layout.addLayout(form)

        # --- 무엇을 만들지 ---
        options = QtWidgets.QGroupBox("만들 것")
        options_layout = QtWidgets.QGridLayout(options)
        options_layout.setSpacing(8)
        self.check_boxes: dict[str, QtWidgets.QCheckBox] = {}
        entries = (
            ("melody", "멜로디", True, True, ""),
            ("arrangement", "편곡 (베이스·드럼·반주)", True, True, ""),
            ("lyrics", "가사", False, False, "작사는 아직 만들고 있습니다."),
            ("singing", "내 목소리로 노래", False, False, "목소리 합성은 아직 만들고 있습니다."),
            ("mix", "믹싱", True, True, ""),
            ("master", "마스터링", True, True, ""),
            ("storyboard", "장면 구성", False, False, "영상은 아직 만들고 있습니다."),
            ("video", "뮤직비디오", False, False, "영상은 아직 만들고 있습니다."),
        )
        for index, (key, label, checked, enabled, reason) in enumerate(entries):
            box = QtWidgets.QCheckBox(label)
            box.setChecked(checked)
            box.setEnabled(enabled)
            if reason:
                box.setToolTip(reason)
            options_layout.addWidget(box, index // 2, index % 2)
            self.check_boxes[key] = box
        layout.addWidget(options)

        # --- 단추 ---
        buttons = QtWidgets.QHBoxLayout()
        self.seed_box = QtWidgets.QSpinBox()
        self.seed_box.setRange(0, 999999)
        self.seed_box.setValue(0)
        self.seed_box.setSpecialValueText("무작위")
        self.seed_box.setToolTip(
            "같은 번호를 넣으면 같은 곡이 나옵니다. 마음에 든 곡을 다시 만들 때 쓰세요."
        )
        buttons.addWidget(QtWidgets.QLabel("씨앗"))
        buttons.addWidget(self.seed_box)
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("취소")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        create = QtWidgets.QPushButton("✨ 만들기")
        create.setObjectName("Primary")
        create.setDefault(True)
        create.clicked.connect(self.accept)
        buttons.addWidget(create)
        layout.addLayout(buttons)

        self.description.textChanged.connect(self._read_description)

    # ---------------------------------------------------------------- 장르

    def _add_genre_row(self, genre: str = "rock", weight: int = 50) -> None:
        if len(self._genre_rows) >= 5:
            return
        row = GenreMixRow(genre, weight)
        row.changed.connect(self._update_genre_summary)
        row.removed.connect(self._remove_genre_row)
        self._genre_container.addWidget(row)
        self._genre_rows.append(row)
        row.remove_button.setEnabled(len(self._genre_rows) > 1)
        for existing in self._genre_rows:
            existing.remove_button.setEnabled(len(self._genre_rows) > 1)
        self._update_genre_summary()

    def _remove_genre_row(self, row: GenreMixRow) -> None:
        if len(self._genre_rows) <= 1:
            return
        self._genre_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        for existing in self._genre_rows:
            existing.remove_button.setEnabled(len(self._genre_rows) > 1)
        self._update_genre_summary()

    def _update_genre_summary(self) -> None:
        try:
            blend = self.genre_blend()
        except Exception:
            self.genre_summary.setText("")
            return
        profile = blend.resolve()
        self.genre_summary.setText(
            f"{blend}  ·  BPM {profile.bpm_low:.0f}~{profile.bpm_high:.0f}"
        )

    def genre_blend(self) -> GenreBlend:
        weights: dict[str, float] = {}
        for row in self._genre_rows:
            if row.weight > 0:
                weights[row.genre] = weights.get(row.genre, 0.0) + row.weight
        if not weights:
            weights = {"pop": 1.0}
        return GenreBlend(weights)

    # ---------------------------------------------------------------- 설명 읽기

    def _read_description(self) -> None:
        """적어 놓은 설명에서 장르와 지시를 알아낸다.

        '애니메이션 록 발라드' 라고 쓰면 그 장르들을 자동으로 넣어 준다.
        사용자가 다시 슬라이더를 만지면 그 값이 우선이다.
        """
        text = self.description.toPlainText()
        if not text.strip():
            return
        found: list[str] = []
        for name in available_genres():
            profile = GENRES[name]
            for candidate in (profile.display_name, name):
                if candidate and candidate.lower() in text.lower():
                    if name not in found:
                        found.append(name)
                    break
        # 한글 표기도 찾는다
        for korean, name in (
            ("발라드", "ballad"), ("록", "rock"), ("락", "rock"), ("재즈", "jazz"),
            ("힙합", "hiphop"), ("애니", "anime"), ("시티팝", "citypop"),
            ("메탈", "metal"), ("펑크", "punk"), ("오케스트라", "orchestral"),
            ("어쿠스틱", "acoustic"), ("로파이", "lofi"), ("신스웨이브", "synthwave"),
        ):
            if korean in text and name not in found:
                found.append(name)
        if not found or self._user_touched_genres():
            return
        found = found[:3]
        while len(self._genre_rows) > len(found):
            self._remove_genre_row(self._genre_rows[-1])
        while len(self._genre_rows) < len(found):
            self._add_genre_row()
        share = 100 // len(found)
        for row, name in zip(self._genre_rows, found):
            index = row.genre_box.findData(name)
            if index >= 0:
                row.genre_box.blockSignals(True)
                row.genre_box.setCurrentIndex(index)
                row.genre_box.blockSignals(False)
            row.weight_slider.blockSignals(True)
            row.weight_slider.setValue(share)
            row.weight_slider.blockSignals(False)
            row.weight_label.setText(f"{share}%")
        self._update_genre_summary()

    def _user_touched_genres(self) -> bool:
        return getattr(self, "_touched", False)

    # ---------------------------------------------------------------- 결과

    def to_request(self):
        from ..music.composer import SongRequest

        description = self.description.toPlainText().strip()
        title = self.title_edit.text().strip() or (
            description.splitlines()[0][:40] if description else "새 곡"
        )
        final_note = ""
        for line in description.splitlines():
            if "마지막" in line and "후렴" in line:
                final_note = line.strip()
                break
        seed = self.seed_box.value() or None
        return SongRequest(
            title=title,
            genre=self.genre_blend(),
            key=None if self.key_auto.isChecked() else self.key_box.currentText(),
            bpm=None if self.bpm_auto.isChecked() else float(self.bpm_box.value()),
            subject=description,
            final_chorus_note=final_note,
            vocal_range=VocalRange.typical(self.voice_box.currentData()),
            seed=seed,
        )

    def wants(self, key: str) -> bool:
        box = self.check_boxes.get(key)
        return bool(box and box.isChecked())
