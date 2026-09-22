"""
실행 취소 / 다시 실행.

편집기에서 이게 없으면 사람은 아무것도 과감하게 시도하지 못한다.
특히 51번의 자연어 수정("후렴에서 캐릭터를 더 많이 보여줘")은 AI 가 프로젝트를
통째로 바꾸는 일이라, 마음에 안 들면 한 번에 되돌릴 수 있어야 한다.

설계

  모든 편집은 Command 객체다. 되돌리는 법을 자기가 안다.
  프로젝트를 직접 고치는 코드는 UI 에 두지 않는다. 그러면 되돌릴 수 없는
  편집이 생기고, 그때부터 실행 취소를 믿을 수 없게 된다.

  되돌리기는 '전체 복사본 저장' 으로 하지 않는다. 음이 수천 개인 프로젝트를
  편집할 때마다 통째로 복사하면 메모리와 시간이 둘 다 감당이 안 된다.
  각 명령이 자기가 바꾼 것만 기억한다.

  묶음(transaction)을 지원한다. AI 가 한 번에 스무 가지를 바꿔도 사용자에게는
  한 번의 편집이어야 한다. Ctrl+Z 를 스무 번 눌러야 하면 쓸 수 없다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..music.genre import GenreBlend
from ..music.structure import Section
from ..music.theory import Key
from .notes import Note, NoteError
from .project import Project, ProjectError
from .tracks import Track, TrackError


class HistoryError(ValueError):
    """실행 취소 관련 오류."""


class Command(ABC):
    """되돌릴 수 있는 편집 하나."""

    @property
    @abstractmethod
    def description(self) -> str:
        """사용자에게 보여줄 설명. '음 추가', '후렴 반키 올리기' 처럼."""

    @abstractmethod
    def apply(self, project: Project) -> None:
        """편집을 실행한다."""

    @abstractmethod
    def revert(self, project: Project) -> None:
        """편집을 되돌린다. apply 직후 상태에서만 부른다."""

    def merge_with(self, other: "Command") -> "Command | None":
        """연달아 일어난 같은 종류의 편집을 하나로 합친다.

        슬라이더를 끄는 동안 값이 백 번 바뀌면 실행 취소 기록도 백 개가 된다.
        그러면 Ctrl+Z 를 백 번 눌러야 원래대로 돌아간다. 합칠 수 있으면 합친다.
        합칠 수 없으면 None 을 돌려준다.
        """
        return None


# ==========================================================================
# 음 편집
# ==========================================================================

class AddNotes(Command):
    """음을 넣는다."""

    def __init__(self, track: Track, notes: Iterable[Note], label: str = "") -> None:
        self.track = track
        self.notes = [n.copy() for n in notes]
        self._label = label
        if not self.notes:
            raise HistoryError("넣을 음이 없습니다.")

    @property
    def description(self) -> str:
        return self._label or f"'{self.track.name}' 에 음 {len(self.notes)}개 추가"

    def apply(self, project: Project) -> None:
        for note in self.notes:
            self.track.add_note(note)

    def revert(self, project: Project) -> None:
        for note in self.notes:
            self.track.notes.remove(note)


class RemoveNotes(Command):
    """음을 지운다."""

    def __init__(self, track: Track, notes: Iterable[Note], label: str = "") -> None:
        self.track = track
        self.notes = list(notes)
        self._label = label
        if not self.notes:
            raise HistoryError("지울 음이 없습니다.")

    @property
    def description(self) -> str:
        return self._label or f"'{self.track.name}' 에서 음 {len(self.notes)}개 삭제"

    def apply(self, project: Project) -> None:
        for note in self.notes:
            if not self.track.notes.remove(note):
                raise HistoryError(
                    f"지우려는 음이 '{self.track.name}' 트랙에 없습니다: {note}"
                )

    def revert(self, project: Project) -> None:
        for note in self.notes:
            self.track.notes.add(note)


class EditNotes(Command):
    """음들을 다른 값으로 바꾼다. 이동, 길이 변경, 이조, 세기 조절이 전부 이것이다."""

    def __init__(self, track: Track, changes: Sequence[tuple[Note, Note]],
                 label: str = "") -> None:
        self.track = track
        self.changes = list(changes)
        self._label = label
        if not self.changes:
            raise HistoryError("바꿀 음이 없습니다.")

    @property
    def description(self) -> str:
        return self._label or f"'{self.track.name}' 의 음 {len(self.changes)}개 수정"

    def apply(self, project: Project) -> None:
        for old, new in self.changes:
            if not self.track.notes.replace_note(old, new):
                raise HistoryError(f"바꾸려는 음이 없습니다: {old}")

    def revert(self, project: Project) -> None:
        for old, new in self.changes:
            self.track.notes.replace_note(new, old)

    def merge_with(self, other: Command) -> Command | None:
        """같은 음을 계속 끌고 있는 중이면 합친다."""
        if not isinstance(other, EditNotes) or other.track is not self.track:
            return None
        if self._label != other._label:
            return None
        # 앞 편집의 결과가 뒤 편집의 대상과 같아야 이어진 조작이다
        previous_results = [new for _, new in self.changes]
        next_targets = [old for old, _ in other.changes]
        if previous_results != next_targets:
            return None
        merged = [(old, other.changes[index][1])
                  for index, (old, _) in enumerate(self.changes)]
        return EditNotes(self.track, merged, self._label)


# ==========================================================================
# 트랙 편집
# ==========================================================================

class AddTrack(Command):
    def __init__(self, track: Track) -> None:
        self.track = track
        self._index: int | None = None

    @property
    def description(self) -> str:
        return f"트랙 '{self.track.name}' 추가"

    def apply(self, project: Project) -> None:
        project.tracks.add(self.track)
        self._index = len(project.tracks) - 1

    def revert(self, project: Project) -> None:
        project.tracks.remove(self.track)


class RemoveTrack(Command):
    def __init__(self, track: Track) -> None:
        self.track = track
        self._index: int | None = None

    @property
    def description(self) -> str:
        return f"트랙 '{self.track.name}' 삭제"

    def apply(self, project: Project) -> None:
        self._index = next(
            (i for i, t in enumerate(project.tracks) if t is self.track), None
        )
        if self._index is None:
            raise HistoryError(f"프로젝트에 없는 트랙입니다: {self.track.name}")
        project.tracks.remove(self.track)

    def revert(self, project: Project) -> None:
        project.tracks.add(self.track)
        if self._index is not None and self._index < len(project.tracks) - 1:
            project.tracks.move(len(project.tracks) - 1, self._index)


class SetTrackProperty(Command):
    """트랙의 값 하나를 바꾼다. 음량, 팬, 이름, 음소거, 악기 등."""

    LABELS: dict[str, str] = {
        "name": "이름", "volume_db": "음량", "pan": "좌우 위치",
        "muted": "음소거", "soloed": "솔로", "instrument": "악기",
        "voice_model": "목소리 모델", "singing_style": "노래 스타일",
        "color": "색", "locked": "잠금", "output_bus": "출력",
    }

    def __init__(self, track: Track, attribute: str, value: Any) -> None:
        if attribute not in self.LABELS:
            raise HistoryError(
                f"바꿀 수 없는 항목입니다: {attribute!r}\n"
                f"가능: {', '.join(self.LABELS)}"
            )
        self.track = track
        self.attribute = attribute
        self.new_value = value
        self.old_value = getattr(track, attribute)

    @property
    def description(self) -> str:
        return f"'{self.track.name}' {self.LABELS[self.attribute]} 변경"

    def apply(self, project: Project) -> None:
        setattr(self.track, self.attribute, self.new_value)

    def revert(self, project: Project) -> None:
        setattr(self.track, self.attribute, self.old_value)

    def merge_with(self, other: Command) -> Command | None:
        if (isinstance(other, SetTrackProperty) and other.track is self.track
                and other.attribute == self.attribute):
            merged = SetTrackProperty(self.track, self.attribute, other.new_value)
            merged.old_value = self.old_value
            return merged
        return None


class MoveTrack(Command):
    def __init__(self, from_index: int, to_index: int) -> None:
        self.from_index = from_index
        self.to_index = to_index

    @property
    def description(self) -> str:
        return "트랙 순서 변경"

    def apply(self, project: Project) -> None:
        project.tracks.move(self.from_index, self.to_index)

    def revert(self, project: Project) -> None:
        project.tracks.move(self.to_index, self.from_index)


# ==========================================================================
# 곡 전체 설정
# ==========================================================================

class SetProjectProperty(Command):
    """곡 전체 값을 바꾼다. 조성, BPM, 장르, 제목."""

    LABELS: dict[str, str] = {
        "key": "조성", "bpm": "BPM", "genre": "장르",
    }

    def __init__(self, attribute: str, value: Any, project: Project) -> None:
        if attribute not in self.LABELS:
            raise HistoryError(
                f"바꿀 수 없는 항목입니다: {attribute!r} (가능: {', '.join(self.LABELS)})"
            )
        self.attribute = attribute
        self.new_value = value
        self.old_value = getattr(project, attribute)

    @property
    def description(self) -> str:
        return f"{self.LABELS[self.attribute]} 변경"

    def apply(self, project: Project) -> None:
        setattr(project, self.attribute, self.new_value)

    def revert(self, project: Project) -> None:
        setattr(project, self.attribute, self.old_value)

    def merge_with(self, other: Command) -> Command | None:
        if isinstance(other, SetProjectProperty) and other.attribute == self.attribute:
            other.old_value = self.old_value
            return other
        return None


class TransposeSection(Command):
    """구간을 통째로 옮긴다. 11번 '마지막 후렴만 반키 올려줘'."""

    def __init__(self, section: Section, semitones: int,
                 tracks: Sequence[Track] | None = None) -> None:
        if semitones == 0:
            raise HistoryError("0반음은 아무 변화가 없습니다.")
        self.section = section
        self.semitones = semitones
        self.tracks = list(tracks) if tracks is not None else None
        self._moved_count = 0

    @property
    def description(self) -> str:
        direction = "올리기" if self.semitones > 0 else "내리기"
        amount = abs(self.semitones)
        if amount == 1:
            amount_text = "반음"
        elif amount == 12:
            amount_text = "한 옥타브"
        else:
            amount_text = f"{amount}반음"
        return f"'{self.section.label}' {amount_text} {direction}"

    def apply(self, project: Project) -> None:
        self._moved_count = project.transpose_section(
            self.section, self.semitones, self.tracks
        )

    def revert(self, project: Project) -> None:
        project.transpose_section(self.section, -self.semitones, self.tracks)


class AddSection(Command):
    def __init__(self, section: Section) -> None:
        self.section = section

    @property
    def description(self) -> str:
        return f"구간 '{self.section.label}' 추가"

    def apply(self, project: Project) -> None:
        project.structure.add(self.section)

    def revert(self, project: Project) -> None:
        project.structure.remove(self.section)


class RemoveSection(Command):
    def __init__(self, section: Section) -> None:
        self.section = section

    @property
    def description(self) -> str:
        return f"구간 '{self.section.label}' 삭제"

    def apply(self, project: Project) -> None:
        if not project.structure.remove(self.section):
            raise HistoryError(f"프로젝트에 없는 구간입니다: {self.section.label}")

    def revert(self, project: Project) -> None:
        project.structure.add(self.section)


class SetSectionProperty(Command):
    """구간의 값 하나를 바꾼다."""

    LABELS: dict[str, str] = {
        "genre": "장르", "key": "조성", "energy": "에너지", "label": "이름",
        "character_presence": "캐릭터 등장", "notes_for_ai": "AI 지시",
        "length_bars": "길이",
    }

    def __init__(self, section: Section, attribute: str, value: Any) -> None:
        if attribute not in self.LABELS:
            raise HistoryError(
                f"바꿀 수 없는 항목입니다: {attribute!r} (가능: {', '.join(self.LABELS)})"
            )
        self.section = section
        self.attribute = attribute
        self.new_value = value
        self.old_value = getattr(section, attribute)

    @property
    def description(self) -> str:
        return f"'{self.section.label}' {self.LABELS[self.attribute]} 변경"

    def apply(self, project: Project) -> None:
        setattr(self.section, self.attribute, self.new_value)

    def revert(self, project: Project) -> None:
        setattr(self.section, self.attribute, self.old_value)


# ==========================================================================
# 묶음
# ==========================================================================

class CompositeCommand(Command):
    """여러 편집을 하나로 묶는다. 실행 취소도 한 번에 된다."""

    def __init__(self, commands: Sequence[Command], label: str) -> None:
        self.commands = list(commands)
        self._label = label
        if not self.commands:
            raise HistoryError("묶을 편집이 없습니다.")

    @property
    def description(self) -> str:
        return self._label

    def apply(self, project: Project) -> None:
        done: list[Command] = []
        try:
            for command in self.commands:
                command.apply(project)
                done.append(command)
        except Exception:
            # 중간에 실패하면 여기까지 한 것을 되돌린다.
            # 반쯤 적용된 상태로 두면 프로젝트가 망가진다.
            for command in reversed(done):
                try:
                    command.revert(project)
                except Exception:
                    pass
            raise

    def revert(self, project: Project) -> None:
        for command in reversed(self.commands):
            command.revert(project)


# ==========================================================================
# 기록
# ==========================================================================

class History:
    """실행 취소 기록."""

    __slots__ = ("_project", "_done", "_undone", "_limit", "_group",
                 "_group_label", "_listeners", "_saved_marker")

    def __init__(self, project: Project, limit: int = 200) -> None:
        if limit < 1:
            raise HistoryError(f"기록 한도는 1 이상이어야 합니다: {limit}")
        self._project = project
        self._done: list[Command] = []
        self._undone: list[Command] = []
        self._limit = limit
        self._group: list[Command] | None = None
        self._group_label = ""
        self._listeners: list[Callable[[str], None]] = []
        self._saved_marker = 0

    # ---- 알림 ----

    def add_listener(self, callback: Callable[[str], None]) -> None:
        """편집이 일어나면 부른다. 화면 다시 그리기에 쓴다."""
        self._listeners.append(callback)

    def _notify(self, what: str) -> None:
        for callback in self._listeners:
            callback(what)

    # ---- 실행 ----

    def run(self, command: Command, allow_merge: bool = True) -> Command:
        """편집을 실행하고 기록한다."""
        command.apply(self._project)
        self._project.meta.touch()

        if self._group is not None:
            self._group.append(command)
            self._notify("run")
            return command

        # 다시 실행 기록은 새 편집이 오면 버린다.
        # 갈라진 미래를 들고 있으면 사용자가 어느 쪽으로 가는지 알 수 없다.
        self._undone.clear()

        if allow_merge and self._done:
            merged = self._done[-1].merge_with(command)
            if merged is not None:
                self._done[-1] = merged
                self._notify("run")
                return merged

        self._done.append(command)
        if len(self._done) > self._limit:
            dropped = len(self._done) - self._limit
            del self._done[:dropped]
            self._saved_marker = max(0, self._saved_marker - dropped)
        self._notify("run")
        return command

    # ---- 묶음 ----

    def begin_group(self, label: str) -> None:
        """여기서부터 end_group 까지의 편집을 하나로 묶는다."""
        if self._group is not None:
            raise HistoryError(
                f"이미 '{self._group_label}' 묶음이 열려 있습니다. 먼저 닫아야 합니다."
            )
        self._group = []
        self._group_label = label

    def end_group(self) -> Command | None:
        """묶음을 닫는다. 아무것도 안 했으면 None."""
        if self._group is None:
            raise HistoryError("열린 묶음이 없습니다.")
        commands = self._group
        label = self._group_label
        self._group = None
        self._group_label = ""
        if not commands:
            return None
        if len(commands) == 1:
            single = commands[0]
            self._undone.clear()
            self._done.append(single)
            self._trim()
            self._notify("group")
            return single
        composite = CompositeCommand(commands, label)
        self._undone.clear()
        self._done.append(composite)
        self._trim()
        self._notify("group")
        return composite

    def abort_group(self) -> None:
        """묶음을 취소하고 그 안의 편집을 전부 되돌린다."""
        if self._group is None:
            raise HistoryError("열린 묶음이 없습니다.")
        for command in reversed(self._group):
            command.revert(self._project)
        self._group = None
        self._group_label = ""
        self._notify("abort")

    def group(self, label: str):
        """with 문으로 쓰는 묶음.

            with history.group("AI 편곡"):
                history.run(...)
                history.run(...)
        """
        history = self

        class _Group:
            def __enter__(self):
                history.begin_group(label)
                return history

            def __exit__(self, exc_type, exc_value, traceback):
                if history._group is None:
                    return False
                if exc_type is not None:
                    history.abort_group()
                else:
                    history.end_group()
                return False

        return _Group()

    def _trim(self) -> None:
        if len(self._done) > self._limit:
            dropped = len(self._done) - self._limit
            del self._done[:dropped]
            self._saved_marker = max(0, self._saved_marker - dropped)

    # ---- 되돌리기 ----

    @property
    def can_undo(self) -> bool:
        return bool(self._done) and self._group is None

    @property
    def can_redo(self) -> bool:
        return bool(self._undone) and self._group is None

    @property
    def undo_description(self) -> str:
        return self._done[-1].description if self._done else ""

    @property
    def redo_description(self) -> str:
        return self._undone[-1].description if self._undone else ""

    def undo(self) -> str:
        if self._group is not None:
            raise HistoryError("묶음이 열려 있는 동안에는 되돌릴 수 없습니다.")
        if not self._done:
            raise HistoryError("되돌릴 편집이 없습니다.")
        command = self._done.pop()
        command.revert(self._project)
        self._undone.append(command)
        self._project.meta.touch()
        self._notify("undo")
        return command.description

    def redo(self) -> str:
        if self._group is not None:
            raise HistoryError("묶음이 열려 있는 동안에는 다시 실행할 수 없습니다.")
        if not self._undone:
            raise HistoryError("다시 실행할 편집이 없습니다.")
        command = self._undone.pop()
        command.apply(self._project)
        self._done.append(command)
        self._project.meta.touch()
        self._notify("redo")
        return command.description

    def undo_all(self) -> int:
        count = 0
        while self.can_undo:
            self.undo()
            count += 1
        return count

    # ---- 저장 표시 ----

    def mark_saved(self) -> None:
        """지금 상태를 '저장됨' 으로 표시한다."""
        self._saved_marker = len(self._done)

    @property
    def has_unsaved_changes(self) -> bool:
        return len(self._done) != self._saved_marker

    # ---- 조회 ----

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
        self._saved_marker = 0
        self._notify("clear")

    def recent(self, count: int = 10) -> list[str]:
        """최근 편집 설명들. 최신이 앞이다."""
        return [c.description for c in reversed(self._done[-count:])]

    def __len__(self) -> int:
        return len(self._done)

    def __repr__(self) -> str:
        return (
            f"History(되돌릴 수 있는 편집 {len(self._done)}개, "
            f"다시 실행 {len(self._undone)}개"
            + (f", '{self._group_label}' 묶음 열림" if self._group is not None else "")
            + ")"
        )
