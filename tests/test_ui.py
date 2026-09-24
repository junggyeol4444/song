"""화면 검증.

윈도우에서 실제로 쓰는 느낌까지는 여기서 확인할 수 없다. 하지만 다음은 확인한다.

  · 모든 화면이 오류 없이 만들어지고 그려지는가
  · 글자가 잘리거나 사라지지 않는가 (실제로 픽셀을 그려서 확인)
  · 편집이 실행 취소되는가
  · 트랙을 바꾸면 그 음역이 화면에 보이는가
  · 프로젝트를 열고 저장하는 길이 막혀 있지 않은가

화면 없는 환경에서 돌리려면 QT_QPA_PLATFORM=offscreen 이 필요하다.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 실제 사용자 설정(키)을 건드리지 않게 빈 설정 폴더를 쓴다
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"]
os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, ".")

failures: list[str] = []
checks = 0


def check(label, got, expected):
    global checks
    checks += 1
    if got != expected:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
        print(f"  FAIL {label}: {got!r} (기대 {expected!r})")


def check_true(label, condition, detail=""):
    global checks
    checks += 1
    if not condition:
        failures.append(f"{label} {detail}")
        print(f"  FAIL {label} {detail}")


try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as error:
    print(f"PySide6 가 없어 화면 검증을 건너뜁니다: {error}")
    sys.exit(0)

from myvocal.core import history as H
from myvocal.core.notes import Note
from myvocal.core.project import Project
from myvocal.core.tracks import Track
from myvocal.core.units import PPQ
from myvocal.music.composer import Composer, SongRequest
from myvocal.music.genre import GenreBlend
from myvocal.music.melody import VocalRange
from myvocal.music.theory import Pitch
from myvocal.ui.create_dialog import CreateSongDialog
from myvocal.ui.main_window import MainWindow
from myvocal.ui.piano_roll import PianoKeyboard, PianoRoll
from myvocal.ui.start_screen import MENU, StartScreen, find_recent_projects
from myvocal.ui.theme import DARK, LIGHT, build_stylesheet, pick_font
from myvocal.ui.timeline import TimelineRuler
from myvocal.ui.track_list import TrackPanel
from myvocal.ui.transport import TransportBar

application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
application.setStyleSheet(build_stylesheet(DARK))


def rendered_image(widget: QtWidgets.QWidget) -> QtGui.QImage:
    """위젯을 실제로 그려서 이미지를 얻는다."""
    application.processEvents()
    return widget.grab().toImage()


def distinct_colors(image: QtGui.QImage, sample: int = 4) -> int:
    """이미지에 몇 가지 색이 있는가. 아무것도 안 그려졌으면 1~2 가지뿐이다."""
    colors = set()
    for y in range(0, image.height(), sample):
        for x in range(0, image.width(), sample):
            colors.add(image.pixel(x, y))
    return len(colors)


print("[1] 색과 글꼴")
check_true("어두운 바탕", DARK.window.startswith("#"))
check_true("밝은 바탕도 있다", LIGHT.window != DARK.window)
stylesheet = build_stylesheet(DARK)
check_true("스타일시트가 만들어진다", len(stylesheet) > 2000)
check_true("글꼴을 고른다", bool(pick_font()))
for name in ("QPushButton", "QSlider", "QComboBox", "QListWidget", "QMenu"):
    check_true(f"{name} 규칙이 있다", name in stylesheet)
# 작은 토글 단추의 여백이 0 이어야 글자가 보인다
check_true("작은 단추 여백 규칙", "QPushButton#Toggle, QPushButton#Mute" in stylesheet)

print("[2] 시작 화면")
start = StartScreen([
    (Path("/x/a.mvp"), "실험곡", "2026-09-22 · 트랙 6개"),
])
start.resize(1100, 760)
start.show()
image = rendered_image(start)
# 작은 화면에서도 쓸 수 있어야 한다. 최소 높이가 화면보다 크면 아래가 잘린다.
check_true("노트북 화면(768)보다 낮은 최소 높이",
           start.minimumSizeHint().height() <= 700,
           f"({start.minimumSizeHint().height()})")
check("요청한 크기대로 그려진다", (image.width(), image.height()),
      (start.width(), start.height()))
check_true("실제로 그려졌다", distinct_colors(image) > 20,
           f"({distinct_colors(image)}가지 색)")
check("메뉴 항목 수", len(MENU), 8)
check_true("만들어진 기능은 켜져 있다",
           all(start._buttons[e.key].isEnabled() for e in MENU if e.available))
check_true("안 만들어진 기능은 꺼져 있고 이유가 있다",
           all(not start._buttons[e.key].isEnabled() and e.unavailable_reason
               for e in MENU if not e.available))
signals: list[str] = []
start.chosen.connect(signals.append)
start._buttons["auto_song"].click()
check("눌리면 신호가 온다", signals, ["auto_song"])
start._buttons["music_video"].click()
check("꺼진 단추는 신호가 안 온다", signals, ["auto_song"])
start._buttons["voice_train"].click()
check("내 AI 가수 만들기가 켜졌다", signals, ["auto_song", "voice_train"])
start.set_available("voice_train", True)
check_true("나중에 켤 수 있다", start._buttons["voice_train"].isEnabled())
with tempfile.TemporaryDirectory() as folder:
    check("빈 폴더면 최근 목록도 비어 있다", find_recent_projects(Path(folder)), [])
    project = Project("최근곡")
    project.save(Path(folder) / "recent.mvp")
    found = find_recent_projects(Path(folder))
    check("저장한 프로젝트를 찾는다", len(found), 1)
    check("제목을 읽는다", found[0][1], "최근곡")
    (Path(folder) / "쓰레기.mvp").write_text("아무거나", encoding="utf-8")
    check("망가진 파일은 건너뛴다", len(find_recent_projects(Path(folder))), 1)

print("[3] 곡 만들기 화면")
dialog = CreateSongDialog()
dialog.description.setPlainText(
    "애니메이션 록 발라드를 만들어줘.\n"
    "주제: 오래 헤어져 있던 친구를 다시 만나는 이야기\n"
    "마지막 후렴은 웅장하게."
)
dialog.title_edit.setText("다시 만나는 날")
dialog.show()
image = rendered_image(dialog)
check_true("그려진다", distinct_colors(image) > 20)
blend = dialog.genre_blend()
names = set(blend.normalized)
check_true("설명에서 장르를 읽는다", {"anime", "rock", "ballad"} <= names, f"({names})")
request = dialog.to_request()
check("제목", request.title, "다시 만나는 날")
check("마지막 후렴 지시를 찾는다", request.final_chorus_note, "마지막 후렴은 웅장하게.")
check_true("조성 자동이면 None", request.key is None)
check_true("BPM 자동이면 None", request.bpm is None)
dialog.key_auto.setChecked(False)
dialog.key_box.setCurrentText("Am")
dialog.bpm_auto.setChecked(False)
dialog.bpm_box.setValue(88)
request = dialog.to_request()
check("직접 고른 조성", request.key, "Am")
check("직접 고른 BPM", request.bpm, 88.0)
check_true("만들 항목 기본값", dialog.wants("melody") and dialog.wants("arrangement"))
check_true("안 되는 항목은 꺼져 있다", not dialog.check_boxes["singing"].isEnabled())
before = len(dialog._genre_rows)
dialog._add_genre_row("jazz", 20)
check("장르 추가", len(dialog._genre_rows), before + 1)
dialog._remove_genre_row(dialog._genre_rows[-1])
check("장르 삭제", len(dialog._genre_rows), before)
while len(dialog._genre_rows) > 1:
    dialog._remove_genre_row(dialog._genre_rows[-1])
dialog._remove_genre_row(dialog._genre_rows[0])
check("마지막 하나는 지워지지 않는다", len(dialog._genre_rows), 1)

print("[4] 편집기 — 곡 하나를 열고 그린다")
project = Composer(SongRequest(
    title="화면 시험", genre=GenreBlend({"rock": 50, "kpop": 30, "orchestral": 20}),
    final_chorus_note="웅장하게", vocal_range=VocalRange.typical("mezzo"), seed=42,
)).compose()
window = MainWindow(project)
window.resize(1500, 900)
window.show()
application.processEvents()
application.processEvents()
image = rendered_image(window)
check("창 크기", (image.width(), image.height()), (1500, 900))
check_true("실제로 그려졌다", distinct_colors(image) > 60,
           f"({distinct_colors(image)}가지 색)")
check("트랙 머리 수", len(window.track_panel._headers), len(project.tracks))
check("구조 목록 항목 수", window.structure_list.count(), len(project.structure))
check_true("곡 정보가 표시된다", "화면 시험" in window.info_label.text())
vocal_tracks = [t for t in project.tracks if t.kind == "vocal"]
check_true("작곡기가 보컬 트랙에 목소리를 지정한다",
           bool(vocal_tracks) and all(t.voice_model.startswith("preset:") for t in vocal_tracks),
           f"({[t.voice_model for t in vocal_tracks]})")
check_true("목소리가 있으니 '목소리 모델 없음' 경고가 없다",
           "목소리 모델" not in window.problem_label.text(), window.problem_label.text())

print("[5] 트랙을 바꾸면 그 음역이 보인다")
for track in project.tracks:
    window.track_panel.select(track)
    application.processEvents()
    roll = window.piano_roll
    check_true("고른 트랙이 반영된다", roll.track is track)
    span = track.pitch_range()
    if span is None:
        continue
    low, high = span
    visible_count = roll.height() // roll.view.key_height
    bottom = roll.view.scroll_pitch
    top = bottom + visible_count
    check_true(f"'{track.name}' 음역이 화면 안에 들어온다",
               bottom <= low and high <= top,
               f"(음역 {Pitch.from_midi(low)}~{Pitch.from_midi(high)}, "
               f"화면 {Pitch.from_midi(bottom)}~{Pitch.from_midi(min(127, top))})")

print("[6] 편집과 실행 취소")
window.track_panel.select(project.tracks[0])
track = window.piano_roll.track
before_count = len(track.notes)
window._run(H.AddNotes(track, [Note(72, 0, PPQ)], "시험용 음"))
check("음이 늘었다", len(track.notes), before_count + 1)
check_true("되돌릴 수 있다", window.history.can_undo)
check_true("최근 편집에 나온다", window.history_list.count() > 0)
window.undo()
check("되돌리면 원래대로", len(track.notes), before_count)
check_true("다시 실행할 수 있다", window.history.can_redo)
window.redo()
check("다시 실행하면 돌아온다", len(track.notes), before_count + 1)
window.undo()

# 트랙 값 바꾸기도 되돌려져야 한다
original_volume = track.volume_db
window._on_track_property(track, "volume_db", -18.0)
check("음량이 바뀐다", track.volume_db, -18.0)
window.undo()
check("음량도 되돌아간다", track.volume_db, original_volume)

# 피아노롤에서 온 편집 신호
note = track.notes[0]
changed = note.copy()
changed.midi = note.midi + 2
window._on_edit_requested("edit", ([(note, changed)], "시험 이동"))
check("피아노롤 편집이 반영된다", track.notes.find(changed.note_id).midi, changed.midi)
window.undo()

print("[7] 조작부와 타임라인")
transport = window.transport
transport.set_position(project.meter.bar_to_tick(17))
check("마디 표시", transport.position_label.text(), "17 . 1")
check_true("시간 표시", ":" in transport.time_label.text())
transport.set_playing(True)
check("재생 중 표시", transport.play_button.text(), "❚❚")
transport.set_playing(False)
check("멈춤 표시", transport.play_button.text(), "▶")
transport.set_levels(0.5, 0.8)
image = rendered_image(transport.meter_widget)
check_true("음량 미터가 그려진다", distinct_colors(image, 2) > 2)
timeline = window.timeline
timeline.set_playhead(project.meter.bar_to_tick(25))
check("타임라인 위치", timeline.playhead_tick, project.meter.bar_to_tick(25))
image = rendered_image(timeline)
check_true("타임라인이 그려진다", distinct_colors(image, 2) > 10)

print("[8] 확대 / 이동")
roll = window.piano_roll
roll.zoom_to_fit(project.end_tick)
check_true("전체가 화면에 들어온다",
           roll.tick_to_x(project.end_tick) <= roll.width() + 2,
           f"({roll.tick_to_x(project.end_tick):.0f} vs {roll.width()})")
before_zoom = roll.view.pixels_per_tick
window._zoom(2.0)
check_true("확대된다", roll.view.pixels_per_tick > before_zoom)
roll.view.pixels_per_tick = 999.0
roll.view.clamp()
check_true("확대 한계가 있다", roll.view.pixels_per_tick <= roll.view.MAX_ZOOM)
roll.view.pixels_per_tick = 1e-9
roll.view.clamp()
check_true("축소 한계도 있다", roll.view.pixels_per_tick >= roll.view.MIN_ZOOM)
roll.view.snap_ticks = PPQ
roll.view.snap_enabled = True
check("자석이 붙는다", roll.view.snap(PPQ + 100), PPQ)
check("자석이 반올림한다", roll.view.snap(PPQ + 600), PPQ * 2)
roll.view.snap_enabled = False
check("자석을 끄면 그대로", roll.view.snap(PPQ + 100), PPQ + 100)

print("[9] 좌표 변환 왕복")
roll.view.scroll_tick = 5000
roll.view.pixels_per_tick = 0.05
for tick in (5000, 10000, 50000):
    check_true(f"tick {tick} 왕복", abs(roll.x_to_tick(roll.tick_to_x(tick)) - tick) <= 1)
roll.view.scroll_pitch = 48
roll.view.key_height = 14
for midi in (48, 60, 72):
    y = roll.pitch_to_y(midi)
    check("음높이 왕복", roll.y_to_pitch(y + 1), midi)

print("[10] 저장하고 다시 열기")
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "화면시험.mvp"
    project.save(path)
    window.history.mark_saved()
    check_true("저장 표시가 꺼진다", not window.history.has_unsaved_changes)
    window._run(H.AddNotes(track, [Note(70, PPQ * 4, PPQ)], "저장 후 편집"))
    check_true("편집하면 저장 표시가 켜진다", window.history.has_unsaved_changes)
    check_true("제목에 * 가 붙는다", window.windowTitle().startswith("*"),
               f"({window.windowTitle()})")
    reopened = Project.load(path)
    window.load_project(reopened)
    check_true("다시 연 프로젝트가 반영된다", window.project is reopened)
    check("트랙 머리도 다시 만들어진다",
          len(window.track_panel._headers), len(reopened.tracks))
    check_true("기록이 새로 시작된다", not window.history.can_undo)

print("[11] 빈 프로젝트에서도 죽지 않는다")
empty = Project("빈 곡")
empty_window = MainWindow(empty)
empty_window.resize(1000, 600)
empty_window.show()
image = rendered_image(empty_window)
check_true("빈 프로젝트도 그려진다", distinct_colors(image) > 10)
check("트랙이 없다", len(empty_window.track_panel._headers), 0)
check_true("피아노롤이 비어 있어도 괜찮다", empty_window.piano_roll.track is None)
empty_window.start_render()
application.processEvents()
check_true("트랙 없이 소리 만들기를 눌러도 안 죽는다", True)
empty_window.close()

print("[12] 소리 장치가 없어도 동작한다")
check_true("재생기가 만들어진다", window.playback is not None)
check_true("장치가 없으면 이유를 안다",
           window.playback.available or bool(window.playback.unavailable_reason))
window.stop_play()
check("정지하면 처음으로", window.playback.position_seconds, 0.0)

window.close()
start.close()
dialog.close()

print("[13] 가사 편집")
lyrics = window.lyrics_panel
lyrics.set_project(project)
check_true("구간별 입력칸이 만들어진다", len(lyrics._editors) > 0,
           f"({len(lyrics._editors)}개)")
check_true("보컬 트랙이 잡힌다", lyrics.track is not None)
# 음이 넉넉한 칸을 골라야 '모자람' 과 '남음' 을 둘 다 시험할 수 있다
first = max(lyrics._editors, key=lambda e: len(e.line.notes))
note_count = len(first.line.notes)
check_true("음이 넉넉한 입력칸이 있다", note_count >= 4, f"({note_count}개)")
check_true("모든 입력칸에 음이 있다",
           all(len(e.line.notes) > 0 for e in lyrics._editors))
# 음절 수보다 짧게 쓰면 알려준다
first.editor.setPlainText("다시")
application.processEvents()
check("음절 수 표시", first.count_label.text(), f"2 / {note_count}")
check_true("모자란다고 알려준다", "더 쓸 수 있습니다" in first.hint_label.text(),
           f"({first.hint_label.text()})")
# 넘치면 그것도 알려준다
first.editor.setPlainText("가" * (note_count + 4))
application.processEvents()
check_true("남는다고 알려준다", "남습니다" in first.hint_label.text(),
           f"({first.hint_label.text()})")
# 실제 발음을 보여준다
first.editor.setPlainText("같이" + "가" * max(0, note_count - 2))
application.processEvents()
check_true("실제 발음을 보여준다", "가치" in first.hint_label.text(),
           f"({first.hint_label.text()})")
# 붙이기
text = "다시만나는날에우리웃으면서손을잡고그날처럼따뜻하게"[:note_count]
first.editor.setPlainText(text)
application.processEvents()
before_undo = len(window.history)
lyrics.apply_lyrics()
application.processEvents()
filled = sum(1 for n in lyrics.track.notes if n.lyric)
check("가사가 붙는다", filled, len(text))
check("기록에 남는다", len(window.history), before_undo + 1)
check_true("설명이 붙는다", "가사" in window.history.undo_description,
           f"({window.history.undo_description})")
window.undo()
check("되돌리면 사라진다", sum(1 for n in lyrics.track.notes if n.lyric), 0)
window.redo()
check("다시 실행하면 돌아온다",
      sum(1 for n in lyrics.track.notes if n.lyric), len(text))
# 가사가 붙은 음은 실제로 노래로 렌더링된다
from myvocal.audio.renderer import RenderOptions, Renderer

renderer = Renderer(RenderOptions())
warnings_out: list[str] = []
sung = renderer.render_vocal(lyrics.track, project.tempo,
                             int(project.duration_seconds * 48000), warnings_out)
check_true("가사가 있으면 노래로 만들어진다", sung is not None)
if sung is not None:
    check_true("소리가 난다", sung.peak() > 0.05, f"({sung.peak():.3f})")
check_true("가사 없는 음은 알려준다",
           any("가사가 없어" in w for w in warnings_out), f"({warnings_out})")
window.undo()
empty_vocal = renderer.render_vocal(lyrics.track, project.tempo, 48000, [])
check("가사가 하나도 없으면 악기로 연주한다", empty_vocal, None)

print("[14] AI 작사")
import time as _time
from myvocal.ui import main_window as main_window_module

# 위 [13] 은 가사 탭에만 곡을 넣었다. 여기서는 편집기 전체에 곡을 연다.
window.load_project(project)
application.processEvents()
lyrics = window.lyrics_panel
check("편집기와 가사 탭이 같은 곡", lyrics.project is window.project, True)
check_true("AI 작사 단추가 있다", lyrics.ai_button.isEnabled())
check("시작 전 가사 없음", sum(1 for n in lyrics.track.notes if n.lyric), 0)
shown = []


def fake_exec(dialog):
    shown.append(dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText())
    return QtWidgets.QDialog.DialogCode.Accepted


main_window_module.LyricsPreviewDialog.exec = fake_exec
history_before = len(window.history)
lyrics.direction_edit.setText("후렴은 밝게")
lyrics.ai_button.click()
check_true("작사 중에는 단추가 잠긴다", not lyrics.ai_button.isEnabled())
deadline = _time.time() + 60
while (window._lyrics_worker is not None and window._lyrics_worker.isRunning()
       and _time.time() < deadline):
    application.processEvents()
    _time.sleep(0.01)
for _ in range(5):
    application.processEvents()
check_true("작사가 끝난다", not window._lyrics_worker.isRunning())
check_true("미리보기를 보여줬다", len(shown) == 1 and "[" in shown[0], shown[:1])
check("모든 음에 가사", sum(1 for n in lyrics.track.notes if not n.lyric), 0)
check("기록 하나로 남는다", len(window.history), history_before + 1)
check("기록 이름", window.history.undo_description, "AI 작사")
check_true("단추가 다시 풀린다", lyrics.ai_button.isEnabled())
check_true("가사 탭이 새 가사를 보여준다",
           any(e.editor.toPlainText().strip() for e in lyrics._editors))
window.undo()
check("되돌리면 가사가 전부 사라진다", sum(1 for n in lyrics.track.notes if n.lyric), 0)
check_true("되돌리면 가사 탭도 비워진다",
           all(not e.editor.toPlainText().strip() for e in lyrics._editors))

# 받는 동안 멜로디가 바뀌면 붙이지 않는다
slots, payload = main_window_module.prepare_lyrics(project, lyrics.track)
stale = main_window_module.request_lyrics(slots, payload, main_window_module.default_registry())
victim = lyrics.track.notes[0]
window._run(H.RemoveNotes(lyrics.track, [victim], "음 삭제"))
check_true("멜로디가 바뀐 것을 안다", stale.is_stale(lyrics.track))
window.undo()
check_true("되돌리면 다시 맞다", not stale.is_stale(lyrics.track))

print("[15] AI 서비스 설정")
from myvocal.providers.settings import Settings
from myvocal.ui.settings_dialog import ProviderSettingsDialog

settings_folder = Path(tempfile.mkdtemp())
dialog = ProviderSettingsDialog(Settings(settings_folder))
dialog.show()
application.processEvents()
check_true("키 없음 안내", "키가 없습니다" in dialog.source_label.text(),
           dialog.source_label.text())
check_true("상태 보고가 보인다", "작사" in dialog.status_view.toPlainText())
check("키 칸은 가려진다", dialog.key_edit.echoMode(), QtWidgets.QLineEdit.EchoMode.Password)
dialog.key_edit.setText("sk-ant-ui-test-00000000")
dialog.model_edit.setText("")
dialog.use_claude.setChecked(False)
dialog.save()
saved = Settings(settings_folder)
check("저장된 키", saved.api_key("anthropic"), "sk-ant-ui-test-00000000")
check("작사 우선순위 저장", saved.get("preferred_providers")["lyrics"], "local")
reopened = ProviderSettingsDialog(saved)
check_true("다시 열면 가린 키를 보여준다", "0000" in reopened.source_label.text()
           and "ui-test" not in reopened.source_label.text(), reopened.source_label.text())
reopened.key_edit.setText("")
reopened.save()
check("비우면 키가 지워진다", Settings(settings_folder).has_api_key("anthropic"), False)
dialog.close()
reopened.close()

print("[16] 곡 만들기에서 가사까지")
create = CreateSongDialog()
check_true("가사 선택 가능", create.check_boxes["lyrics"].isEnabled())
check_true("가사 기본 선택", create.check_boxes["lyrics"].isChecked())
from myvocal.ui.app import ComposeWorker

create.description.setPlainText("오래 헤어져 있던 친구를 다시 만나는 이야기")
create.seed_box.setValue(3)
results = []
worker = ComposeWorker(create.to_request(), True, True, True)
worker.finished_ok.connect(lambda proj, message: results.append((proj, message)))
worker.start()
deadline = _time.time() + 120
while not results and _time.time() < deadline:
    application.processEvents()
    _time.sleep(0.01)
worker.wait(1000)
check_true("곡이 만들어졌다", bool(results))
if results:
    made, message = results[0]
    vocal = [t for t in made.tracks if t.kind == "vocal"][0]
    check("만든 곡의 모든 음에 가사", sum(1 for n in vocal.notes if not n.lyric), 0)
    check_true("주제를 알려준다", "다시 만남" in message, message)

print("[17] 내 AI 가수 (Voice Studio)")
import numpy as _np
from myvocal.ui import voice_studio as studio_module
from myvocal.voice.library import VoiceLibrary
from myvocal.voice.phonemes import align_lyrics
from myvocal.voice.synth import SingingSynth, VoiceTimbre
from myvocal.voice.analysis import midi_to_hz


class FakeRecorder:
    """마이크 대신 합성한 노래를 돌려준다 (0.1초 늦게 들어온 것처럼)."""

    rate = 24000

    def record(self, exercise):
        syllables = align_lyrics(exercise.text, [(n.start, n.duration) for n in exercise.notes])
        audio = SingingSynth(VoiceTimbre.preset("tenor"), self.rate).render(
            syllables, [midi_to_hz(n.midi) for n in exercise.notes]).data[0] * 0.5
        return _np.concatenate([_np.zeros(int(0.1 * self.rate)), audio, _np.zeros(self.rate // 2)])

    def play_cues(self, exercise):
        pass

    @staticmethod
    def stop():
        pass


studio_module.input_available = lambda: (True, "마이크: 시험용")
studio_library = VoiceLibrary(Path(tempfile.mkdtemp()))
studio = studio_module.VoiceStudio(studio_library, recorder=FakeRecorder(), can_apply=True)
studio.resize(1100, 760)
studio.show()
application.processEvents()
check("처음에는 안내 화면", studio.stack.currentIndex(), 0)
entry = studio_library.create("MY VOICE", "my_voice", "tenor")
studio.refresh_list(entry.voice_id)
application.processEvents()
check("목소리를 고르면 작업 화면", studio.stack.currentIndex(), 1)
check_true("처음 연습 안내", "음역 재기" in studio.guidance_label.text(), studio.guidance_label.text())
check("기본 연습 6개", studio.exercise_box.count(), 6)
check_true("녹음 단추가 켜져 있다", studio.record_button.isEnabled())
check_true("모델 단추는 녹음 전엔 꺼져 있다", not studio.build_button.isEnabled())
studio.exercise_box.setCurrentIndex(1)            # '아' 길게 내기
studio.record_button.click()
deadline = _time.time() + 120
while studio._worker is not None and _time.time() < deadline:
    application.processEvents()
    _time.sleep(0.01)
application.processEvents()
check("녹음이 저장됐다", len(entry.takes()), 1)
check_true("결과를 알려준다", "잡았습니다" in studio.take_result.text(), studio.take_result.text())
check_true("녹음 뒤 부족한 자료 안내로 바뀐다",
           "부족합니다" in studio.guidance_label.text(), studio.guidance_label.text())
check_true("자료 표가 채워진다", any(
    studio.coverage_table.item(r, c).text() != "—"
    for r in range(3) for c in range(8)))
check("녹음 목록", studio.take_list.count(), 1)
check_true("왼쪽 목록도 녹음 수가 바뀐다", "녹음 1개" in studio.voice_list.item(0).text(),
           studio.voice_list.item(0).text())
check_true("모델 단추가 켜진다", studio.build_button.isEnabled())
grab = studio.grab()
grab.save(str(Path(tempfile.gettempdir()) / "voice_studio_test.png"))
check_true("화면이 실제로 그려졌다", distinct_colors(grab.toImage()) > 40)

studio.build_model()
deadline = _time.time() + 300
while studio._worker is not None and _time.time() < deadline:
    application.processEvents()
    _time.sleep(0.02)
application.processEvents()
check_true("모델이 만들어졌다", entry.model() is not None)
check_true("분석 결과가 보인다", "음역" in studio.report_view.toPlainText())
check_true("적용 단추가 켜진다", studio.apply_button.isEnabled())
chosen = []
studio.voice_chosen.connect(lambda ref, style: chosen.append((ref, style)))
studio.style_box.setCurrentIndex(studio.style_box.findData("rock"))
studio.apply_button.click()
check("보컬 트랙에 쓰기 신호", chosen, [(f"user:{entry.voice_id}", "rock")])

# 편집기에서 적용하면 실행 취소 한 번에 둘 다 돌아간다
import myvocal.voice.library as _library_module
_original = _library_module.voices_folder
_library_module.voices_folder = lambda: studio_library.folder
try:
    vocal = [t for t in window.project.tracks if t.kind == "vocal"][0]
    before = (vocal.voice_model, vocal.singing_style)
    window.apply_voice(f"user:{entry.voice_id}", "rock")
    check("목소리·창법 적용", (vocal.voice_model, vocal.singing_style), (f"user:{entry.voice_id}", "rock"))
    header = [h for h in window.track_panel._headers if h.track is vocal][0]
    check_true("트랙 머리에 목소리 이름", "MY VOICE" in header.instrument_label.text()
               and "Rock" in header.instrument_label.text(), header.instrument_label.text())
    window.undo()
    check("실행 취소로 둘 다 돌아간다", (vocal.voice_model, vocal.singing_style), before)
finally:
    _library_module.voices_folder = _original
studio.close()

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
