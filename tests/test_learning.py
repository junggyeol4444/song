"""학습실 검증 (8번 노래 학습, 9번 Reference, 63번 자료 권한).

정답을 아는 곡이 필요하다. 작곡기로 곡을 만들면 조성, BPM, 코드 진행, 곡 구조를
전부 안다. 그 곡을 MIDI 와 오디오로 내보냈다가 다시 분석해서 정답과 비교한다.
(작곡기 곡이라 실제 대중음악보다 쉬울 수 있다. 여기 적힌 정확도는 그 조건의 값이다.)

  1. 크로마·템포의 기초 (순음, 클릭)
  2. MIDI 왕복: 가사·띄어쓰기·보컬 트랙이 살아 돌아오는가
  3. 악보 분석: 조성, BPM, 코드 진행, 곡 구조 (여러 장르)
  4. 오디오 분석: 템포, 조성, 구조, 믹스
  5. 내 스타일: 만든 스타일로 작곡하면 그 성향이 실제로 들어가는가
  6. Reference: 코드 진행·멜로디는 가져오지 않는다
  7. 자료 권한: Reference 자료로는 학습할 수 없다
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"]

from myvocal.audio.export import export_audio, export_midi, import_midi
from myvocal.audio.renderer import RenderOptions, render_project
from myvocal.learning.analysis import analyze_midi, analyze_project
from myvocal.learning.audio_analysis import (
    RATE, _spectrogram, analyze_audio, chroma, estimate_tempo, onset_envelope, track_beats,
)
from myvocal.learning.library import MusicLibrary, load_style_genre
from myvocal.learning.style import (
    base_numeral, build_style, reference_hints, register_style, template_numerals,
)
from myvocal.music.composer import Composer, SongRequest
from myvocal.music.genre import GENRES, GenreBlend, available_genres, get_genre, user_genres
from myvocal.music.harmony import COMMON_PROGRESSIONS
from myvocal.music.lyrics import LyricsBrief, lyrics_command, vocal_track, write_lyrics
from myvocal.providers.base import ProviderRegistry
from myvocal.providers.local import LocalProvider

failures: list[str] = []
checks = 0
WORK = Path(tempfile.mkdtemp())


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
        failures.append(f"{label}: {exc.__name__} 대신 {type(error).__name__}: {error}")
        print(f"  FAIL {label}")
        return
    failures.append(f"{label}: 예외가 안 남")
    print(f"  FAIL {label}")


def boundary_scores(truth, detected, tolerance):
    used, hits = set(), 0
    for t in truth:
        match = [d for d in detected if abs(d - t) <= tolerance and d not in used]
        if match:
            hits += 1
            used.add(match[0])
    return hits, len(truth) - hits, len(detected) - len(used)


# ==========================================================================
print("1. 크로마 · 템포 기초")
# ==========================================================================
t = np.arange(int(RATE * 2)) / RATE
for name, hz, pc in (("C4", 261.63, 0), ("A3", 220.0, 9), ("F2", 87.31, 5), ("B5", 987.77, 11)):
    spectrum, freqs = _spectrogram(np.sin(2 * np.pi * hz * t) * 0.5)
    check(f"순음 {name} 의 음이름", int(np.argmax(chroma(spectrum, freqs).mean(axis=0))), pc)
noisy = sum(np.sin(2 * np.pi * h * t) for h in (174.61, 207.65, 261.63, 349.23)) * 0.2
noisy = noisy + np.random.default_rng(0).standard_normal(len(t)) * 0.05
spectrum, freqs = _spectrogram(noisy)
profile = chroma(spectrum, freqs).mean(axis=0)
check_true("F 단조 화음 + 잡음: F 가 가장 크다", int(np.argmax(profile)) == 5, f"({np.round(profile, 2)})")
check_true("F 단조 화음의 구성음(F Ab C)이 셋 다 위쪽", set(np.argsort(profile)[-3:]) == {5, 8, 0},
           f"({np.argsort(profile)[-3:]})")
for bpm in (90.0, 128.0, 150.0):
    clicks = np.zeros(int(RATE * 20))
    period = int(RATE * 60 / bpm)
    for start in range(0, len(clicks) - 400, period):
        clicks[start:start + 200] = np.sin(np.arange(200) * 0.9) * np.hanning(200)
    spectrum, _ = _spectrogram(clicks)
    envelope = onset_envelope(spectrum)
    found, _ = estimate_tempo(envelope)
    beats = track_beats(envelope, found)
    refined = 60.0 * (RATE / 512) / float(np.median(np.diff(beats)))
    check_true(f"클릭 {bpm:.0f} BPM (±1.5%)", abs(refined - bpm) / bpm < 0.015, f"({refined:.1f})")

# ==========================================================================
print("2. MIDI 왕복")
# ==========================================================================
registry = ProviderRegistry()
registry.register(LocalProvider())
song = Composer(SongRequest(genre="kpop", seed=2)).compose()
track = vocal_track(song)
lyrics_command(track, write_lyrics(song, track, registry, LyricsBrief(seed=1))).apply(song)
midi_path = WORK / "kpop.mid"
export_midi(song, midi_path)
back = import_midi(midi_path)
vocals = [t for t in back.tracks if t.kind == "vocal"]
check("가사가 있는 트랙은 보컬 트랙으로 돌아온다", len(vocals), 1)
check("가사와 띄어쓰기가 그대로", [(n.lyric, n.word_end) for n in vocals[0].notes],
      [(n.lyric, n.word_end) for n in track.notes])

# ==========================================================================
print("3. 악보 분석")
# ==========================================================================
analysis = analyze_midi(midi_path)
check("BPM", analysis.bpm, song.tempo.bpm_at_tick(0))
check("조성", analysis.key, song.key.display_name)
check_true("가사를 읽었다", analysis.lyrics.replace(" ", "")[:10] ==
           "".join(n.lyric for n in track.notes)[:10], analysis.lyrics[:20])
check_true("보컬 트랙을 멜로디로 썼다", "멜로디" not in analysis.unavailable)
check_true("믹스는 악보에 없다고 적었다", "믹스" in analysis.unavailable)
hits = misses = extras = 0
chorus_overlaps = []
keys_right = 0
progression_found = 0
genres = ("pop", "kpop", "ballad", "rock", "edm", "anime", "hiphop", "jazz", "orchestral", "rnb")
for index, genre in enumerate(genres):
    project = Composer(SongRequest(genre=genre, seed=index + 1)).compose()
    path = WORK / f"{genre}.mid"
    export_midi(project, path)
    result = analyze_midi(path)
    keys_right += result.key == project.key.display_name
    truth = [s.start_bar for s in project.structure][1:]
    detected = [s.start_bar for s in result.sections][1:]
    h, m, e = boundary_scores(truth, detected, 1)
    hits, misses, extras = hits + h, misses + m, extras + e
    truth_chorus = {b for s in project.structure if s.kind in ("chorus", "drop")
                    for b in range(s.start_bar, s.end_bar)}
    found_chorus = {b for s in result.sections if s.kind == "chorus"
                    for b in range(s.start_bar, s.start_bar + s.bars)}
    chorus_overlaps.append(len(truth_chorus & found_chorus) / max(1, len(truth_chorus | found_chorus)))
    # 작곡기가 쓴 진행 틀이 분석한 진행에 나오는가
    mode = "minor" if project.key.is_minor else "major"
    # 편곡기가 3화음에 7음을 얹기도 하므로 뼈대(3화음)끼리 비교한다
    templates = [tuple(base_numeral(x) for x in p)
                 for p in COMMON_PROGRESSIONS.get(genre, COMMON_PROGRESSIONS["pop"])[mode]]
    numerals = [base_numeral(x) for x in
                template_numerals([c for _, c in result.chords], result.key_tonic_pc, result.key_minor)]
    windows = {tuple(numerals[i:i + 4]) for i in range(len(numerals) - 3)}
    progression_found += any(t in windows for t in templates)
check(f"조성 {len(genres)}곡 전부", keys_right, len(genres))
precision = hits / max(1, hits + extras)
recall = hits / max(1, hits + misses)
f1 = 2 * precision * recall / max(1e-9, precision + recall)
check_true("구조 경계 F1 0.85 이상 (±1마디)", f1 >= 0.85, f"(F1 {f1:.2f}, 정밀도 {precision:.2f}, 재현율 {recall:.2f})")
check_true("후렴 자리 겹침 평균 0.75 이상", np.mean(chorus_overlaps) >= 0.75, f"({np.mean(chorus_overlaps):.2f})")
check_true("작곡기가 쓴 코드 진행을 대부분 찾는다 (8곡 이상)", progression_found >= 8, f"({progression_found}/{len(genres)})")

# ==========================================================================
print("4. 오디오 분석")
# ==========================================================================
audio_results = []
for genre, seed in (("pop", 1), ("edm", 4), ("ballad", 2)):
    project = Composer(SongRequest(genre=genre, seed=seed)).compose()
    audio, _ = render_project(project, options=RenderOptions(sample_rate=22050))
    wav = WORK / f"{genre}.wav"
    export_audio(audio, wav, "wav16")
    result = analyze_audio(wav)
    audio_results.append((project, result))
    truth_bpm = project.tempo.bpm_at_tick(0)
    candidates = [result.bpm] + result.bpm_alternatives
    check_true(f"{genre}: 템포 (±4%, 두 배/절반 후보 포함)",
               any(abs(c - truth_bpm) / truth_bpm < 0.04 for c in candidates),
               f"({result.bpm:.1f} {result.bpm_alternatives} / 정답 {truth_bpm})")
    check(f"{genre}: 조성", result.key, project.key.display_name)
    check_true(f"{genre}: 믹스를 잰다", "통합 라우드니스 LUFS" in result.mix)
    check_true(f"{genre}: 악기는 못 잰다고 적는다", "악기" in result.unavailable)
hits = misses = extras = 0
for project, result in audio_results:
    bar = 4 * 60 / project.tempo.bpm_at_tick(0)
    truth = [s.seconds(project.meter, project.tempo)[0] for s in project.structure][1:]
    detected = [s.start_seconds for s in result.sections][1:]
    h, m, e = boundary_scores(truth, detected, bar * 1.01)
    hits, misses, extras = hits + h, misses + m, extras + e
precision = hits / max(1, hits + extras)
recall = hits / max(1, hits + misses)
f1 = 2 * precision * recall / max(1e-9, precision + recall)
check_true("오디오 구조 경계 F1 0.6 이상 (±1마디)", f1 >= 0.6, f"(F1 {f1:.2f})")

# ==========================================================================
print("5. 내 스타일")
# ==========================================================================
jazz = []
for seed in (1, 2, 3):
    project = Composer(SongRequest(genre="jazz", seed=seed)).compose()
    path = WORK / f"jazz{seed}.mid"
    export_midi(project, path)
    jazz.append(analyze_midi(path))
style = build_style("내 재즈", jazz)
check_true("7화음 성향을 잰다 (재즈라 높다)", style.values["seventh_chords"] > 0.6, f"({style.values})")
check_true("BPM 을 잰다", style.values["bpm_low"] <= jazz[0].bpm <= style.values["bpm_high"])
check_true("악기 구성을 잰다", "acoustic_piano" in style.values["instrument_weights"])
check_true("진행 틀을 모았다", bool(style.progressions))
name = register_style(style)
check_true("작곡기에 등록", name in GENRES and name in COMMON_PROGRESSIONS)
check("기본 장르 목록은 그대로 22개", len(available_genres()), 22)
check("내 스타일 목록", user_genres(), [name])
made = Composer(SongRequest(genre=GenreBlend({name: 1.0}), seed=5)).compose()
made_analysis = analyze_project(made)
check_true("내 스타일로 만든 곡의 BPM 이 스타일 범위 안", style.values["bpm_low"] <= made.tempo.bpm_at_tick(0) <= style.values["bpm_high"],
           f"({made.tempo.bpm_at_tick(0)})")
check_true("내 스타일로 만든 곡도 7화음이 많다", made_analysis.seventh_ratio > 0.5, f"({made_analysis.seventh_ratio:.2f})")
style_templates = {tuple(p) for items in style.progressions.values() for p in items}
numerals = template_numerals([c for _, c in made_analysis.chords], made_analysis.key_tonic_pc, made_analysis.key_minor)
windows = {tuple(numerals[i:i + 4]) for i in range(len(numerals) - 3)}
check_true("내 스타일의 코드 진행 틀로 작곡한다", bool(windows & style_templates),
           f"(틀 {sorted(style_templates)[:3]})")
check_raises("이름 없는 스타일", lambda: build_style(" ", jazz))

# ==========================================================================
print("6. Reference")
# ==========================================================================
before = set(COMMON_PROGRESSIONS)
hints = reference_hints(audio_results[1][1])      # EDM 오디오
check_true("임시 장르로 등록", hints.genre_name in GENRES)
check("코드 진행 틀은 가져오지 않는다", set(COMMON_PROGRESSIONS) - before, set())
check_true("기본 장르 목록에 안 나온다", hints.genre_name not in available_genres() + user_genres())
check_true("'가져오지 않은 것' 을 알린다", "코드 진행, 멜로디, 가사" in hints.describe())
check_true("템포를 참고한다", hints.bpm is not None and abs(hints.bpm - 128) / 128 < 0.04, f"({hints.bpm})")
if hints.structure is not None:
    referenced = Composer(SongRequest(genre=GenreBlend({hints.genre_name: 1.0}), bpm=hints.bpm, seed=3,
                                      structure=hints.structure)).compose()
    check("참고 곡의 구조대로 만든다", [(s.kind, s.length_bars) for s in referenced.structure],
          [(s.kind, s.length_bars) for s in hints.structure])

# ==========================================================================
print("7. 자료 권한과 라이브러리")
# ==========================================================================
library = MusicLibrary(WORK / "library")
check_raises("권한 없이 못 넣는다", lambda: library.add(midi_path, "unknown"))
own = library.add(WORK / "jazz1.mid", "own")
licensed = library.add(WORK / "jazz2.mid", "licensed", "허가받은 곡")
ref = library.add(WORK / "edm.wav", "reference")
check("세 곡", [i.rights for i in library.items()], ["own", "licensed", "reference"])
check_true("원본 파일은 복사하지 않는다", not any(p.suffix in (".mid", ".wav") for p in library.folder.rglob("*")))
check_raises("Reference 자료로 학습 금지 (63번)",
             lambda: library.build_style("섞인 것", [own.item_id, ref.item_id]))
made_style = library.build_style("내 것", [own.item_id, licensed.item_id])
check("라이브러리에 스타일 저장", [s.name for s in library.styles()], ["내 것"])
check_true("Reference 로는 새 곡 참고가 된다", library.reference(ref.item_id).bpm is not None)
del GENRES[made_style.genre_name]
import myvocal.learning.library as library_module
original = library_module.music_folder
library_module.music_folder = lambda: library.folder
try:
    check("저장된 곡을 열 때 내 스타일을 다시 불러온다", get_genre(made_style.genre_name).display_name,
          "내 스타일: 내 것")
finally:
    library_module.music_folder = original
check_raises("분석할 수 없는 파일", lambda: library.add(WORK / "없는파일.txt", "own"))
check("삭제", library.delete(licensed.item_id), True)
check("삭제 뒤 두 곡", len(library.items()), 2)

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
