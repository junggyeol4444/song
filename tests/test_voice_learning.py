"""목소리 학습 검증 (6·7번).

진짜 사람 녹음은 여기 없다. 대신 설정값을 아는 '가상 사용자' 목소리를 합성해서
녹음처럼 넣고, 분석이 그 값을 되찾는지 본다. 합성기로 만든 소리를 합성기로
맞추는 것이라 사람 목소리에서도 똑같이 맞는다는 보증은 아니다. 그래서
'모델로 다시 불렀을 때 녹음과 측정값이 같은가' (재현 오차) 도 함께 본다.

  1. 음높이: 가이드 연습 정확도, '아' 고음의 옥타브 오류, 장치 지연·스피커 섞임·잡음
  2. 음역, 비브라토, 가성 전환점
  3. 녹음에서 목소리 모델 만들기: 설정값 복원 + 재현 오차
  4. 창법: 같은 목소리로 창법마다 실제로 다르게 부르는가
  5. 연습, 자료 현황, "고음의 'ㅣ' 발음 데이터가 부족합니다" 안내
  6. 라이브러리: 자료 권한, 저장, 다시 열기, 삭제
  7. 곡에 쓰기: 작곡기는 그 음역에서 멜로디를, 렌더러는 그 목소리로
  8. 녹음 준비: 가이드와 부를 구간이 겹치지 않는가, 녹음 상태 점검
"""

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"]

from myvocal.audio.recording import check_level, render_cues
from myvocal.voice.analysis import (
    GuideNote, analyze_take, extract_contour, midi_to_hz, note_name, summarize,
)
from myvocal.voice.library import VoiceLibrary, resolve_voice
from myvocal.voice.model import (
    SINGING_STYLES, Take, VoiceModel, build_voice_model, get_style, preset_voices,
)
from myvocal.voice.phonemes import align_lyrics
from myvocal.voice.synth import SingingSynth, VoiceError, VoiceTimbre
from myvocal.voice.training import (
    Coverage, VOWEL_SENTENCES, next_suggestions, sentence_exercise, starter_exercises,
    sustain_exercise,
)

RATE = 48000
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


def check_raises(label, fn, exc=Exception):
    global checks
    checks += 1
    try:
        fn()
    except exc:
        return
    except Exception as error:
        failures.append(f"{label}: {exc.__name__} 대신 {type(error).__name__}")
        print(f"  FAIL {label}")
        return
    failures.append(f"{label}: 예외가 안 남")
    print(f"  FAIL {label}")


def sing_guide(timbre, guide, rate=RATE):
    syllables = align_lyrics("".join(g.syllable for g in guide), [(g.start, g.duration) for g in guide])
    return SingingSynth(timbre, rate).render(syllables, [midi_to_hz(g.midi) for g in guide]).data[0]


def scale_guide(text, midis, duration=0.9, gap=0.25):
    return [GuideNote(0.3 + i * (duration + gap), duration, m, c)
            for i, (m, c) in enumerate(zip(midis, text))]


# ==========================================================================
print("1. 음높이")
# ==========================================================================
mezzo = VoiceTimbre.preset("mezzo")
guide = scale_guide("아" * 10, [60, 62, 64, 65, 67, 69, 71, 72, 74, 76])
notes = analyze_take(sing_guide(mezzo, guide), RATE, guide)
check("가이드 음 10개 전부 잡힘", len(notes), 10)
errors = [abs(n.error_cents) for n in notes]
check_true("음정 오차 10센트 이내", max(errors) <= 10, f"({[round(e) for e in errors]})")
# 가이드 없이도: '아' 를 B4 로 부르면 첫 포먼트가 둘째 배음을 키워 한 옥타브 위로 읽기 쉽다
free = summarize(analyze_take(sing_guide(mezzo, guide), RATE))
check_true("가이드 없이도 옥타브 오류 없음 (최고음 E5)",
           free.highest_hz is not None and abs(12 * np.log2(free.highest_hz / midi_to_hz(76))) < 0.5,
           f"({note_name(free.highest_hz) if free.highest_hz else None})")

tenor = VoiceTimbre.preset("tenor")
exercise = sentence_exercise("저 너머 먼 거리", 60)
voice = sing_guide(tenor, exercise.notes)
cues = render_cues(exercise, RATE)
rng = np.random.default_rng(3)
for latency, bleed, noise_db in ((0.0, 0.0, -90), (0.15, 0.5, -50), (0.3, 1.0, -40)):
    length = max(len(voice), len(cues)) + int(0.5 * RATE)
    mix = np.zeros(length)
    mix[:len(cues)] += cues * bleed
    mix[:len(voice)] += voice * 0.5
    recorded = np.concatenate([np.zeros(int(latency * RATE)), mix])[:length]
    recorded = recorded + rng.standard_normal(length) * 10 ** (noise_db / 20)
    measured = analyze_take(recorded, RATE, exercise.notes)
    check(f"지연 {latency}s·가이드 섞임 {bleed}·잡음 {noise_db}dB 에서 음 전부 잡힘",
          len(measured), len(exercise.notes))
    check_true(f"  그때 음정 오차 10센트 이내",
               all(abs(m.error_cents) <= 10 for m in measured),
               f"({[round(m.error_cents) for m in measured]})")

# ==========================================================================
print("2. 음역 · 비브라토 · 가성")
# ==========================================================================
report = summarize(notes)
check("최저음", note_name(report.lowest_hz), "C4")
check("최고음", note_name(report.highest_hz), "E5")
check_true("비브라토 빠르기 (설정 5.5Hz)", abs(report.vibrato_rate - 5.5) < 0.2, f"({report.vibrato_rate:.2f})")
check_true("비브라토 폭 (설정 ±30센트, 15% 안)", abs(report.vibrato_extent - 30.0) / 30.0 < 0.15,
           f"({report.vibrato_extent:.1f})")
check("가성 없는 목소리에서 가성 감지 안 함", report.falsetto_from_hz, None)
# 가성 판정은 놓치는 경우가 있다 (개발 중 50가지 경우에서 1번 놓침, 엉뚱한 위치는 0번).
# 놓치는 것은 괜찮지만(가성 없음으로 보고) 틀린 위치를 말하면 안 된다.
right = wrong = missed = 0
for voice_name, scale, start in (("mezzo", [60, 62, 64, 65, 67, 69, 71, 72, 74, 76], 69),
                                 ("tenor", [48, 50, 52, 53, 55, 57, 59, 60, 62, 64], 59)):
    falsetto_voice = replace(VoiceTimbre.preset(voice_name), falsetto_from_hz=midi_to_hz(start) - 1)
    for text in ("아" * 10, "아이" * 5, "오에" * 5):
        g = scale_guide(text, scale)
        found = summarize(analyze_take(sing_guide(falsetto_voice, g), RATE, g)).falsetto_from_hz
        if found is None:
            missed += 1
        elif abs(12 * np.log2(found / midi_to_hz(start))) < 0.6:
            right += 1
        else:
            wrong += 1
check("가성 전환점을 엉뚱한 곳으로 말하지 않는다", wrong, 0)
check_true("가성 전환점을 6번 중 5번 이상 찾는다", right >= 5, f"(맞음 {right}, 놓침 {missed})")
still = replace(VoiceTimbre.preset("tenor"))
g = scale_guide("아이" * 5, [48, 50, 52, 53, 55, 57, 59, 60, 62, 64])
check("흉성만 쓰는 테너에서 가성 감지 안 함",
      summarize(analyze_take(sing_guide(still, g), RATE, g)).falsetto_from_hz, None)

# ==========================================================================
print("3. 녹음에서 목소리 모델 만들기")
# ==========================================================================
truth = VoiceTimbre(name="가상 사용자", formant_scale=1.10, breathiness=0.30, brightness=0.3,
                    vibrato_rate=6.1, vibrato_depth=40.0, vibrato_onset=0.25, scoop_cents=-80.0,
                    portamento=0.001, falsetto_from_hz=midi_to_hz(64) - 1)


def take(text, midis):
    g = scale_guide(text, midis, duration=1.0, gap=0.35)
    return Take(sing_guide(truth, g), RATE, g)


takes = [take("아에이오우아에이", [50, 52, 53, 55, 57, 59, 60, 62]),
         take("아이아이아이아이", [52, 55, 57, 59, 62, 64, 66, 67]),
         take("오우오우아아", [48, 50, 52, 53, 55, 57])]
model, model_report = build_voice_model("가상 사용자", takes)
check_true("성도 배율 복원 (1.10, ±3%)", abs(model.formant_scale - 1.10) / 1.10 < 0.03, f"({model.formant_scale:.3f})")
check_true("숨 섞임 복원 (0.30, ±0.08)", abs(model.breathiness - 0.30) < 0.08, f"({model.breathiness:.2f})")
check_true("비브라토 빠르기 복원 (6.1Hz, ±0.2)", abs(model.vibrato_rate - 6.1) < 0.2, f"({model.vibrato_rate:.2f})")
check_true("비브라토 폭 복원 (40센트, ±15%)", abs(model.vibrato_depth - 40) / 40 < 0.15, f"({model.vibrato_depth:.1f})")
check_true("비브라토 시작 복원 (0.25초, ±0.1)", abs(model.vibrato_onset - 0.25) < 0.1, f"({model.vibrato_onset:.2f})")
check("가성 전환점 복원", note_name(model.falsetto_from_hz), "E4")
check("음역 (가성 포함 부른 음 전부)", model.range_text().split(" (")[0], "C3~G4")
check_true("모음 포먼트를 쟀다", {"ㅏ", "ㅔ", "ㅗ", "ㅜ", "ㅣ"} <= set(model.vowel_formants),
           f"({sorted(model.vowel_formants)})")
check("자료 권한 기록", model.rights, "my_voice")
reproduction = model.reproduction
check_true("재현: HNR 차이 2dB 이내", abs(reproduction.get("hnr_db", 99)) <= 2.0, f"({reproduction})")
check_true("재현: 밝기 차이 2dB 이내", abs(reproduction.get("tilt_db", 99)) <= 2.0, f"({reproduction})")
check_true("재현: 비브라토 폭 차이 6센트 이내", abs(reproduction.get("vibrato_extent", 99)) <= 6.0, f"({reproduction})")
check_true("재현: 음 시작 곡선 차이 20센트 이내", reproduction.get("attack_curve", 99) <= 20.0, f"({reproduction})")
check_true("보고서에 재현 결과가 들어간다", "모델로 다시 불러 본 결과" in model.report_text)
restored = VoiceModel.from_dict(model.to_dict())
check("저장·복원 후 같은 합성 설정", restored.timbre("ballad"), model.timbre("ballad"))
check_raises("Reference 권한으로는 모델을 못 만든다",
             lambda: build_voice_model("남의 노래", takes, rights="reference"), VoiceError)
check_raises("녹음 없이 못 만든다", lambda: build_voice_model("빈 것", []), VoiceError)

# ==========================================================================
print("4. 창법")
# ==========================================================================
check("창법 8가지 (7번)", sorted(SINGING_STYLES),
      sorted(["natural", "ballad", "soft", "power", "rock", "whisper", "falsetto", "emotional"]))
styled = {name: model.timbre(name) for name in SINGING_STYLES}
check("같은 목소리 — 포먼트는 창법과 무관", {t.vowel_formants for t in styled.values()} == {styled["natural"].vowel_formants}, True)
check_true("Whisper 는 숨이 대부분", styled["whisper"].breathiness >= 0.8)
check("Power 는 가성 안 씀", styled["power"].falsetto_from_hz, 0.0)
check_true("Falsetto 는 모든 음을 가성으로", 0 < styled["falsetto"].falsetto_from_hz <= 1.0)
check_true("Rock 은 거칠다", styled["rock"].roughness > 0.3)
check("Natural 은 이 사람의 가성 전환점 그대로", styled["natural"].falsetto_from_hz, model.falsetto_from_hz)
g = scale_guide("아에이오", [52, 55, 57, 59])
hnr = {}
for name in ("natural", "whisper", "power"):
    hnr[name] = summarize(analyze_take(sing_guide(styled[name], g), RATE, g)).hnr_db
check_true("측정: Whisper 는 Natural 보다 숨이 많다 (HNR 낮음)", hnr["whisper"] < hnr["natural"] - 3, f"({hnr})")
check_true("측정: Power 는 Natural 보다 숨이 적다 (HNR 높음)", hnr["power"] > hnr["natural"], f"({hnr})")
check_raises("모르는 창법", lambda: get_style("opera"), VoiceError)

# ==========================================================================
print("5. 연습과 자료 현황")
# ==========================================================================
starters = starter_exercises(55, 79)
check("기본 연습 6개", len(starters), 6)
for ex in starters:
    ordered = all(a.start + a.duration + 0.4 < b.start - ex.cue_seconds - 0.15 + 0.6
                  for a, b in zip(ex.notes, ex.notes[1:]))
    gaps = [b.start - ex.cue_seconds - 0.15 - (a.start + a.duration) for a, b in zip(ex.notes, ex.notes[1:])]
    check_true(f"'{ex.title}': 부른 뒤 다음 가이드까지 0.5초 (지연 흡수)",
               all(gap >= 0.49 for gap in gaps), f"({[round(x, 2) for x in gaps[:3]]})")
cue = render_cues(starters[1], RATE)
for note in starters[1].notes:
    window = cue[int(note.start * RATE):int((note.start + note.duration) * RATE)]
    check_true("부르는 동안 가이드는 조용하다", float(np.max(np.abs(window))) < 1e-9)
coverage = Coverage(55, 79)
suggestions = next_suggestions(coverage)
check_true("빈 자료에서 안내가 나온다", len(suggestions) >= 1)
check_true("안내 형식: '...발음 데이터가 부족합니다. 다음 문장을 G4 부근에서 불러주세요.'",
           "발음 데이터가 부족합니다" in suggestions[0].message
           and "부근에서 불러주세요" in suggestions[0].message, suggestions[0].message)
g_high = sentence_exercise(VOWEL_SENTENCES["ㅣ"][0], coverage.zone_center("high"))
coverage.add(analyze_take(sing_guide(mezzo, g_high.notes), RATE, g_high.notes),
             [n.syllable for n in g_high.notes])
check_true("부른 문장의 모음이 고음 칸에 쌓인다", coverage.cells[("ㅣ", "high")].seconds > 0,
           f"({coverage.cells[('ㅣ', 'high')]})")
high_i_message = [s.message for s in next_suggestions(Coverage(55, 79), count=30)
                  if "고음의 'ㅣ'" in s.message]
check_true("'고음의 ㅣ' 안내가 있다 (명세 6번 예시)", bool(high_i_message))
if high_i_message:
    check_true("음이름까지 적는다", note_name(midi_to_hz(Coverage(55, 79).zone_center("high"))) in high_i_message[0],
               high_i_message[0])

# ==========================================================================
print("6. 라이브러리")
# ==========================================================================
library = VoiceLibrary(Path(tempfile.mkdtemp()))
check_raises("Reference 권한으로는 목소리를 만들 수 없다",
             lambda: library.create("남의 목소리", "reference"), VoiceError)
check_raises("이름 없이는 못 만든다", lambda: library.create(" ", "my_voice"), VoiceError)
entry = library.create("MY VOICE", "my_voice", "tenor")
check("목록에 나온다", [e.name for e in library.entries()], ["MY VOICE"])
first = sustain_exercise("ㅏ", [55, 57, 59])
info = entry.add_take(sing_guide(truth, first.notes), RATE, first)
check("녹음 저장 후 바로 분석", len(info.measures), 3)
check("녹음 목록", len(entry.takes()), 1)
check_raises("무음 녹음은 거부", lambda: entry.add_take(np.zeros(RATE), RATE, first), VoiceError)
from myvocal.voice.training import Exercise
for t in takes:
    entry.add_take(t.signal, t.rate, Exercise(title="연습", instruction="", notes=t.guide))
check("녹음 4개", len(entry.takes()), 4)
built = entry.build_model()
check("모델 번호는 라이브러리 번호", built.voice_id, entry.voice_id)
reopened = VoiceLibrary(library.folder).get(entry.voice_id)
check("다시 열어도 모델이 있다", reopened.model().to_dict(), built.to_dict())
check("resolve: 내 목소리", resolve_voice(f"user:{entry.voice_id}", library).name, "MY VOICE")
check("resolve: 기본 목소리", resolve_voice("preset:tenor").name, "테너")
check("resolve: 예전 프로젝트 ('mezzo')", resolve_voice("mezzo").name, "메조소프라노")
check_raises("resolve: 없는 목소리", lambda: resolve_voice("user:없음", library), VoiceError)
entry.delete_take(1)
check("녹음 삭제", len(entry.takes()), 3)
entry.rename("MY VOICE 2")
check("이름 바꾸기 (모델도)", library.get(entry.voice_id).model().name, "MY VOICE 2")
second = library.create("지울 것", "licensed", "alto")
check("지우기", library.delete(second.voice_id), True)
check("지운 뒤 목록", [e.name for e in library.entries()], ["MY VOICE 2"])
(library.folder / "broken").mkdir()
(library.folder / "broken" / "meta.json").write_text("{깨짐", encoding="utf-8")
check("깨진 항목이 있어도 목록은 나온다", [e.name for e in library.entries()], ["MY VOICE 2"])

# ==========================================================================
print("7. 곡에 쓰기")
# ==========================================================================
from myvocal.audio.renderer import RenderOptions, Renderer
from myvocal.music.composer import Composer, SongRequest

import myvocal.voice.library as library_module
original_folder = library_module.voices_folder
library_module.voices_folder = lambda: library.folder
try:
    reference = f"user:{entry.voice_id}"
    song = Composer(SongRequest(genre="ballad", seed=5, voice_model=reference,
                                singing_style="ballad")).compose()
    vocal = [t for t in song.tracks if t.kind == "vocal"][0]
    check("보컬 트랙에 내 목소리", vocal.voice_model, reference)
    check("보컬 트랙에 창법", vocal.singing_style, "ballad")
    check("프로젝트 자원에 목소리 등록 (59번)", [a.asset_id for a in song.assets_of("voice")], [entry.voice_id])
    vocal_range = built.vocal_range()
    check_true("멜로디가 내 음역 안에 있다",
               all(vocal_range.lowest <= n.midi <= vocal_range.highest for n in vocal.notes),
               f"(음역 {vocal_range.lowest}~{vocal_range.highest}, 멜로디 "
               f"{min(n.midi for n in vocal.notes)}~{max(n.midi for n in vocal.notes)})")
    renderer = Renderer(RenderOptions(sample_rate=24000, voice_library=library))
    warnings: list[str] = []
    timbre = renderer.vocal_timbre(vocal, warnings)
    check("렌더러가 내 목소리 + 창법을 쓴다", timbre, library.get(entry.voice_id).model().timbre("ballad"))
    check("경고 없음", warnings, [])
    vocal.voice_model = "user:없는번호"
    warnings = []
    timbre = renderer.vocal_timbre(vocal, warnings)
    check_true("없는 목소리는 알리고 기본 목소리로", any("라이브러리에 없습니다" in w for w in warnings), f"({warnings})")
    vocal.voice_model, vocal.singing_style = "", "mezzo"
    warnings = []
    check("예전 프로젝트 (창법 칸에 목소리 이름)", renderer.vocal_timbre(vocal, warnings),
          preset_voices()["mezzo"].timbre("natural"))
finally:
    library_module.voices_folder = original_folder

# ==========================================================================
print("8. 녹음 상태 점검")
# ==========================================================================
quiet = np.sin(np.linspace(0, 2000, RATE)) * 0.01
check_true("너무 작다", any("너무 작습니다" in m for m in check_level(quiet, RATE).messages))
loud = np.clip(np.sin(np.linspace(0, 2000, RATE)) * 3.0, -1, 1)
check_true("찢어졌다", any("찢어졌습니다" in m for m in check_level(loud, RATE).messages))
noisy = np.random.default_rng(0).standard_normal(RATE) * 0.2
check_true("주변 소음", any("소음" in m for m in check_level(noisy, RATE).messages))
good = np.concatenate([np.zeros(RATE // 2) + 1e-5, np.sin(np.linspace(0, 3000, RATE)) * 0.5])
check("좋은 녹음은 문제 없음", check_level(good, RATE).messages, [])

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
