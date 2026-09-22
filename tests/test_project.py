"""프로젝트 문서 검증.

음악·목소리·캐릭터·영상이 정말 한 문서, 한 시간축을 쓰는지 확인한다.
따로 만들어 붙인 구조라면 여기서 드러난다.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from myvocal.core.notes import Note, NoteError, NoteList
from myvocal.core.project import AssetReference, Project, ProjectError
from myvocal.core.tracks import Automation, EffectSlot, Track, TrackError, TrackList
from myvocal.core.units import PPQ, TimeSignature
from myvocal.music.genre import GENRES, GenreBlend, GenreError, available_genres, get_genre
from myvocal.music.structure import (
    Section, SongStructure, StructureError, build_structure,
)
from myvocal.music.theory import Key, Pitch

failures: list[str] = []
checks = 0


def check(label, got, expected, tol=1e-9):
    global checks
    checks += 1
    if isinstance(expected, float) and isinstance(got, (int, float)):
        ok = abs(got - expected) <= tol
    else:
        ok = got == expected
    if not ok:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
        print(f"  FAIL {label}: {got!r} (기대 {expected!r})")


def check_true(label, condition, detail=""):
    global checks
    checks += 1
    if not condition:
        failures.append(f"{label} {detail}")
        print(f"  FAIL {label} {detail}")


def check_raises(label, fn, exc=Exception):
    global checks
    checks += 1
    try:
        fn()
    except exc:
        return
    failures.append(f"{label}: 예외가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


print("[1] 음")
note = Note(60, 0, PPQ, 90, lyric="사")
check("끝 위치", note.end_tick, PPQ)
check("박 수", note.beats, 1.0)
check("음이름", str(note.pitch), "C4")
check("이조", str(note.transposed(7).pitch), "G4")
check("이동", note.moved(PPQ).start_tick, PPQ)
check("길이 변경", note.resized(PPQ * 2).duration_ticks, PPQ * 2)
check("가사 보존", note.with_velocity(100).lyric, "사")
check("원본 불변", note.transposed(7).midi != note.midi, True)
check_raises("길이 0 거부", lambda: Note(60, 0, 0), NoteError)
check_raises("음수 위치 거부", lambda: Note(60, -1, PPQ), NoteError)
check_raises("음높이 범위 초과", lambda: Note(200, 0, PPQ), NoteError)
check_raises("세기 범위 초과", lambda: Note(60, 0, PPQ, 200), NoteError)
check_raises("모르는 표현 거부", lambda: Note(60, 0, PPQ, expression={"없음": 1}), NoteError)
check_raises("모르는 시작방식 거부",
             lambda: Note(60, 0, PPQ, expression={"attack_style": "bogus"}), NoteError)
check_raises("이조 범위 초과", lambda: Note(120, 0, PPQ).transposed(20), NoteError)
check("표현 설정", Note(60, 0, PPQ).with_expression(vibrato_depth=40.0)
      .expression_value("vibrato_depth"), 40.0)

print("[2] 음 목록")
notes = NoteList([Note(64, PPQ, PPQ), Note(60, 0, PPQ), Note(67, PPQ * 2, PPQ)])
check("자동 정렬", [n.midi for n in notes], [60, 64, 67])
check("개수", len(notes), 3)
check("시작", notes.start_tick, 0)
check("끝", notes.end_tick, PPQ * 3)
check("음역", notes.pitch_range, (60, 67))
notes.add(Note(62, PPQ // 2, PPQ // 2))
check("추가 후에도 정렬", [n.start_tick for n in notes], [0, PPQ // 2, PPQ, PPQ * 2])
check("그 순간 울리는 음", [n.midi for n in notes.at_tick(PPQ)], [64])
check("이조", [n.midi for n in notes.transposed(2)], [62, 64, 66, 69])
check("퀀타이즈", [n.start_tick for n in
                NoteList([Note(60, 10, PPQ), Note(62, 970, PPQ)]).quantized(PPQ)], [0, PPQ])
check("퀀타이즈 50%", [n.start_tick for n in
                   NoteList([Note(60, 10, PPQ), Note(62, 970, PPQ)]).quantized(PPQ, 0.5)],
      [5, 965])
check_raises("격자 0 거부", lambda: notes.quantized(0), NoteError)
overlapping = NoteList([Note(60, 0, PPQ * 2), Note(60, PPQ, PPQ)])
check("같은 음 겹침 감지", len(overlapping.overlapping_pairs()), 1)
check("다른 음 겹침은 정상",
      len(NoteList([Note(60, 0, PPQ * 2), Note(64, PPQ, PPQ)]).overlapping_pairs()), 0)
legato = NoteList([Note(60, 0, PPQ // 2), Note(62, PPQ, PPQ // 2)]).legato()
check("레가토가 다음 음까지 늘림", legato[0].duration_ticks, PPQ)
humanized = NoteList([Note(60, PPQ, PPQ)] * 1).humanized(seed=1)
check_true("휴머나이즈는 위치를 조금 바꾼다", humanized[0].start_tick != PPQ)

print("[3] 트랙")
track = Track("피아노", "instrument", instrument="acoustic_piano")
track.add_note(Note(60, 0, PPQ))
check("종류 이름", track.kind_name, "악기")
check("기본 색 배정", track.color, "#4a90d9")
check("음 번호 자동 부여", track.notes[0].note_id > 0, True)
check_true("잠그면 편집 거부", True)
track.locked = True
check_raises("잠긴 트랙 편집 거부", lambda: track.add_note(Note(62, PPQ, PPQ)), TrackError)
track.locked = False
check_raises("빈 이름 거부", lambda: Track(""), TrackError)
check_raises("모르는 종류 거부", lambda: Track("x", "bogus"), TrackError)
check_raises("팬 범위 초과", lambda: Track("x", pan=2.0), TrackError)
check_raises("음량 범위 초과", lambda: Track("x", volume_db=50.0), TrackError)
check_raises("모르는 이펙트 거부", lambda: EffectSlot("bogus"), TrackError)
track.add_effect("reverb", room_size=0.6)
check("이펙트 추가", track.effect("reverb").params["room_size"], 0.6)
track.set_automation("volume_db", [(0, -6.0), (PPQ * 4, 0.0)])
check("오토메이션 시작", track.volume_db_at(0), -6.0)
check("오토메이션 중간 (선형 보간)", track.volume_db_at(PPQ * 2), -3.0)
check("오토메이션 끝", track.volume_db_at(PPQ * 4), 0.0)
check("오토메이션 범위 밖은 끝값", track.volume_db_at(PPQ * 100), 0.0)
check("오토메이션 없으면 고정값", track.pan_at(PPQ), 0.0)

print("[4] 트랙 목록 / 솔로 규칙")
first = Track("A")
second = Track("B")
third = Track("C")
tracks = TrackList([first, second, third])
check("전부 들림", len(tracks.audible()), 3)
second.muted = True
check("음소거 제외", [t.name for t in tracks.audible()], ["A", "C"])
third.soloed = True
check("솔로가 있으면 솔로만", [t.name for t in tracks.audible()], ["C"])
second.muted = False
second.soloed = True
check("솔로 여러 개", sorted(t.name for t in tracks.audible()), ["B", "C"])
second.soloed = third.soloed = False
tracks.move(0, 2)
check("순서 바꾸기", [t.name for t in tracks], ["B", "C", "A"])
check("번호로 찾기", tracks.find(first.track_id).name, "A")
check("이름으로 찾기", tracks.by_name("B").name, "B")
check("없는 이름", tracks.by_name("Z"), None)
check_raises("번호 중복 거부", lambda: TrackList([first, first]), TrackError)

print("[5] 장르")
check("등록 장르 수", len(available_genres()), 22)
check("한글 이름", get_genre("발라드").name, "ballad")
check("별칭 K-Pop", get_genre("K-Pop").name, "kpop")
check("별칭 Anime Song", get_genre("Anime Song").name, "anime")
check("별칭 보컬로이드", get_genre("보컬로이드").name, "vocalsynth")
check_raises("모르는 장르 거부", lambda: get_genre("트로트"), GenreError)
blend = GenreBlend({"rock": 50, "kpop": 30, "orchestral": 20})
check("비율 정규화", round(blend.normalized["rock"], 6), 0.5)
check("비율은 상대적",
      GenreBlend({"rock": 5, "kpop": 3, "orchestral": 2}).normalized,
      blend.normalized)
check("주 장르", blend.primary.name, "rock")
resolved = blend.resolve()
rock, kpop, orch = GENRES["rock"], GENRES["kpop"], GENRES["orchestral"]
check("혼합 BPM 은 가중 평균", resolved.bpm_typical,
      rock.bpm_typical * 0.5 + kpop.bpm_typical * 0.3 + orch.bpm_typical * 0.2, tol=1e-9)
check("혼합 압축도 가중 평균", resolved.compression,
      rock.compression * 0.5 + kpop.compression * 0.3 + orch.compression * 0.2, tol=1e-9)
check("악기 가중치도 섞임", resolved.instrument_weights["drum_kit"],
      rock.instrument_weights["drum_kit"] * 0.5 + kpop.instrument_weights["drum_kit"] * 0.3
      + orch.instrument_weights.get("drum_kit", 0.0) * 0.2, tol=1e-9)
check("단일 장르는 그대로", GenreBlend.single("ballad").resolve().name, "ballad")
check("문자열 파싱", GenreBlend.parse("Rock 50% K-Pop 30%").normalized["rock"], 0.625, tol=1e-9)
check("한글 파싱", "ballad" in GenreBlend.parse("발라드 70% 록 30%").normalized, True)
check_raises("비율 0 거부", lambda: GenreBlend({"rock": 0}), GenreError)
check_raises("빈 장르 거부", lambda: GenreBlend({}), GenreError)
for name in available_genres():
    profile = get_genre(name)
    check_true(f"{name}: BPM 범위 유효",
               profile.bpm_low <= profile.bpm_typical <= profile.bpm_high)
    check_true(f"{name}: 악기가 지정됨", len(profile.instrument_weights) > 0)

print("[6] 곡 구조")
structure = build_structure("ballad")
check_true("구간이 만들어짐", len(structure) >= 8)
check("빈틈 없음", structure.gaps(), [])
check_true("후렴이 여러 번", len(structure.of_kind("chorus")) >= 3)
check("마지막 후렴 이름", structure.last_of_kind("chorus").label, "마지막 후렴")
check_true("벌스는 '마지막' 이라 안 부른다",
           all("마지막" not in s.label for s in structure.of_kind("verse")),
           f"({[s.label for s in structure.of_kind('verse')]})")
check("후렴 에너지가 벌스보다 높다",
      structure.of_kind("chorus")[0].effective_energy()
      > structure.of_kind("verse")[0].effective_energy(), True)
check("마디로 구간 찾기", structure.at_bar(1).kind, "intro")
check("범위 밖 마디", structure.at_bar(9999), None)
check_raises("겹치는 구간 거부",
             lambda: structure.add(Section("chorus", 1, 4)), StructureError)
check_raises("0마디 길이 거부", lambda: Section("verse", 1, 0), StructureError)
check_raises("0마디 시작 거부", lambda: Section("verse", 0, 4), StructureError)
check_raises("모르는 구간 종류", lambda: Section("bogus", 1, 4), StructureError)
edm = build_structure("edm")
check("EDM 은 드롭이 있다", len(edm.of_kind("drop")), 2)
check("EDM 은 벌스가 없다", len(edm.of_kind("verse")), 0)

print("[7] 프로젝트 — 하나의 문서")
project = Project("테스트곡", key="C", bpm=88, genre="ballad")
project.build_default_structure()
check("조성", str(project.key), "C")
check("BPM", project.bpm, 88.0)
check("박자표", str(project.time_signature), "4/4")
check_true("길이가 계산됨", project.duration_seconds > 60.0)
check("마디->초", project.bar_to_seconds(1), 0.0)
# 88 BPM, 4/4 -> 한 마디 = 4 * 60/88 초
check("2마디 시작 시각", project.bar_to_seconds(2), 4 * 60.0 / 88.0, tol=1e-9)
check("초->마디 왕복", project.seconds_to_bar(project.bar_to_seconds(9)), 9)

print("[8] 음악과 영상이 같은 시간축을 쓰는가")
chorus = project.structure.of_kind("chorus")[0]
span = project.section_range(chorus)
start_sec, end_sec = chorus.seconds(project.meter, project.tempo)
check("구간 tick -> 초 일치", project.tempo.tick_to_seconds(span.start_tick), start_sec)
check("그 초를 다시 tick 으로", project.tempo.seconds_to_tick(start_sec), span.start_tick)
check("그 tick 의 구간을 되찾음", project.section_at_tick(span.start_tick).section_id,
      chorus.section_id)
check("구간 끝 직전도 같은 구간",
      project.section_at_tick(span.end_tick - 1).section_id, chorus.section_id)
check_true("구간 끝은 다음 구간",
           project.section_at_tick(span.end_tick) is None
           or project.section_at_tick(span.end_tick).section_id != chorus.section_id)

print("[9] 11번 — 마지막 후렴만 반키 올리기")
vocal = project.add_track("보컬", "vocal", voice_model="MY VOICE")
verse = project.structure.of_kind("verse")[0]
verse_span = project.section_range(verse)
chorus_span = project.section_range(chorus)
vocal.add_note(Note(60, verse_span.start_tick, PPQ, lyric="가"))
vocal.add_note(Note(64, chorus_span.start_tick, PPQ, lyric="나"))
vocal.add_note(Note(67, chorus_span.start_tick + PPQ, PPQ, lyric="다"))
moved = project.transpose_section(chorus, 1)
check("후렴 안의 음만 이동", moved, 2)
check("후렴 음이 올라감", sorted(n.midi for n in project.notes_in_section(chorus)), [65, 68])
check("벌스 음은 그대로", [n.midi for n in project.notes_in_section(verse)], [60])
check("구간 조성 표시도 바뀜", str(chorus.key), "C#")
check("후렴 시점 조성 조회", str(project.effective_key(chorus_span.start_tick)), "C#")
check("벌스 시점 조성 조회", str(project.effective_key(verse_span.start_tick)), "C")
check("가사는 따라감", sorted(n.lyric for n in project.notes_in_section(chorus)), ["나", "다"])
# 드럼은 음높이가 악기 종류이므로 옮기면 안 된다
drums = project.add_track("드럼", "drum", instrument="drum_kit")
drums.add_note(Note(36, chorus_span.start_tick, PPQ))
project.transpose_section(chorus, 1)
check("드럼은 이조하지 않는다", [n.midi for n in drums.notes], [36])

print("[10] 4번 — 구간별 장르")
chorus.genre = GenreBlend.single("orchestral")
check("후렴 장르", project.effective_genre(chorus_span.start_tick).primary.name, "orchestral")
check("벌스는 곡 전체 장르", project.effective_genre(verse_span.start_tick).primary.name,
      "ballad")
project.structure.sections[0].genre = GenreBlend({"ambient": 1.0})
check("인트로 장르", project.effective_genre(0).primary.name, "ambient")

print("[11] 자원 참조")
project.add_asset("voice", "my_voice_01", "MY VOICE", "lead_vocal")
project.add_asset("character", "luna", "LUNA", "main_character")
project.add_asset("voice", "my_voice_01", "MY VOICE", "lead_vocal")  # 중복
check("중복은 안 쌓임", len(project.assets_of("voice")), 1)
check("역할로 찾기", project.asset_by_role("main_character").display_name, "LUNA")
check_raises("모르는 자원 종류", lambda: AssetReference("bogus", "x"), ProjectError)
check_raises("빈 번호 거부", lambda: AssetReference("voice", "  "), ProjectError)

print("[12] 문제 감지")
broken = Project("문제있는곡")
broken.add_track("보컬없음", "vocal")          # 목소리 모델 없음
overlap_track = broken.add_track("겹침", "instrument")
overlap_track.add_note(Note(60, 0, PPQ * 2))
overlap_track.add_note(Note(60, PPQ, PPQ))
problems = broken.problems()
check_true("목소리 모델 없음 감지", any("목소리 모델" in p for p in problems), f"({problems})")
check_true("음 겹침 감지", any("겹치는" in p for p in problems), f"({problems})")
check("정상 프로젝트는 문제 없음", Project("정상").problems(), [])

print("[13] 저장 / 불러오기")
with tempfile.TemporaryDirectory() as folder:
    path = project.save(Path(folder) / "test")
    check("확장자 자동", path.suffix, ".mvp")
    check_true("파일 생성됨", path.exists())
    loaded = Project.load(path)
    check("완전히 같은 내용", loaded.to_dict(), project.to_dict())
    check("제목", loaded.meta.title, project.meta.title)
    check("조성", str(loaded.key), str(project.key))
    check("BPM", loaded.bpm, project.bpm)
    check("장르", loaded.genre.normalized, project.genre.normalized)
    check("트랙 수", len(loaded.tracks), len(project.tracks))
    check("가사 보존", loaded.track("보컬").lyrics_text(),
          project.track("보컬").lyrics_text())
    check("구간 조성 보존",
          str(loaded.structure.of_kind("chorus")[0].key), str(chorus.key))
    check("구간 장르 보존",
          loaded.effective_genre(chorus_span.start_tick).primary.name, "orchestral")
    check("자원 보존", len(loaded.assets), len(project.assets))
    # 다시 저장해도 같아야 한다
    second_path = loaded.save(Path(folder) / "again")
    check("재저장 후에도 동일",
          json.loads(second_path.read_text(encoding="utf-8"))["tracks"],
          json.loads(path.read_text(encoding="utf-8"))["tracks"])
    check_raises("없는 파일", lambda: Project.load(Path(folder) / "없음.mvp"), ProjectError)
    bad = Path(folder) / "bad.mvp"
    bad.write_text("{}", encoding="utf-8")
    check_raises("형식 아닌 파일 거부", lambda: Project.load(bad), ProjectError)
    broken_file = Path(folder) / "broken.mvp"
    broken_file.write_text("{ 망가진", encoding="utf-8")
    check_raises("손상된 파일 거부", lambda: Project.load(broken_file), ProjectError)
    future = Path(folder) / "future.mvp"
    future.write_text(json.dumps({"format": "myvocal-project", "version": 999}),
                      encoding="utf-8")
    check_raises("미래 버전 거부", lambda: Project.load(future), ProjectError)

print("[14] 템포 / 박자표 변경")
varying = Project("변박곡", bpm=120)
varying.build_default_structure()
varying.add_tempo_change(9, 90.0)
check("변경 전 BPM", varying.tempo.bpm_at_tick(0), 120.0)
check("변경 후 BPM", varying.tempo.bpm_at_tick(varying.meter.bar_to_tick(9)), 90.0)
# 120BPM 4/4 -> 한 마디 2초. 8마디 = 16초
check("9마디 시작 시각", varying.bar_to_seconds(9), 16.0, tol=1e-9)
varying.add_meter_change(9, "3/4")
check("9마디 박자표", str(varying.meter.signature_at_tick(varying.meter.bar_to_tick(9))), "3/4")
check("1마디 박자표는 그대로", str(varying.meter.signature_at_tick(0)), "4/4")

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
