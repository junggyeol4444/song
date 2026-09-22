"""내보내기 검증.

파일로 꺼낸 것이 원본과 같은지, 그리고 긴 곡에서도 프로그램이 죽지 않는지 본다.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from myvocal.audio.buffer import AudioBuffer
from myvocal.audio.export import (
    AUDIO_FORMATS, ExportError, available_formats, estimate_size_mb,
    export_audio, export_midi, import_midi,
)
from myvocal.core.notes import Note
from myvocal.core.project import Project
from myvocal.core.units import PPQ
from myvocal.music.composer import Composer, SongRequest
from myvocal.music.genre import GenreBlend

failures: list[str] = []
checks = 0
RATE = 48000


def check(label, got, expected, tol=1e-9):
    global checks
    checks += 1
    ok = abs(got - expected) <= tol if isinstance(expected, float) and isinstance(got, (int, float)) else got == expected
    if not ok:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
        print(f"  FAIL {label}: {got!r} (기대 {expected!r})")


def check_true(label, condition, detail=""):
    global checks
    checks += 1
    if not condition:
        failures.append(f"{label} {detail}")
        print(f"  FAIL {label} {detail}")


def check_raises(label, fn, exc=ExportError):
    global checks
    checks += 1
    try:
        fn()
    except exc:
        return
    failures.append(f"{label}: 예외가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


def make_music(seconds: float, seed: int = 0) -> AudioBuffer:
    """실제 음악에 가까운 시험 신호.

    순수 사인파만 쓰면 안 된다. FLAC 이 90% 넘게 줄여버려서 압축 형식의
    크기 예상을 검증할 수 없다. 실제 음악에는 드럼과 잡음 성분이 섞여 있어
    그만큼 안 줄어든다. 그래서 잡음과 타악기 비슷한 성분을 넣는다.
    """
    generator = np.random.default_rng(seed)
    time = np.arange(int(seconds * RATE)) / RATE
    left = sum(np.sin(2 * np.pi * f * time) / (i + 1)
               for i, f in enumerate((220.0, 330.0, 440.0, 660.0)))
    right = sum(np.sin(2 * np.pi * f * time + 0.3) / (i + 1)
                for i, f in enumerate((220.0, 277.2, 440.0, 554.4)))
    # 드럼 대신 쓸 짧은 잡음 덩어리를 박마다 넣는다
    percussion = generator.standard_normal(time.shape[0])
    beat = (time * 2.0) % 1.0
    percussion *= np.exp(-beat * 18.0)
    hiss = generator.standard_normal(time.shape[0]) * 0.05
    stereo = np.vstack([left + percussion * 0.6 + hiss,
                        right + percussion * 0.55 + hiss])
    return AudioBuffer(stereo / np.max(np.abs(stereo)) * 0.7, RATE)


print("[1] 지원 형식")
formats = available_formats()
check_true("WAV 는 언제나 있다", "wav24" in formats)
check_true("FLAC 사용 가능", "flac" in formats, f"({sorted(formats)})")
for name in formats:
    check_true(f"{name}: 설명이 있다", bool(AUDIO_FORMATS[name][3]))
check_raises("모르는 형식 거부", lambda: export_audio(make_music(0.5), "x", "bogus"))
check_raises("크기 예상도 형식을 검사", lambda: estimate_size_mb(10.0, "bogus"))

print("[2] 짧은 오디오 왕복")
short = make_music(2.0)
with tempfile.TemporaryDirectory() as folder:
    for name in sorted(formats):
        path = export_audio(short, Path(folder) / f"t_{name}", name)
        check_true(f"{name}: 파일 생성", path.exists())
        check("확장자", path.suffix, AUDIO_FORMATS[name][2])
        loaded = AudioBuffer.load(path)
        check(f"{name}: 채널", loaded.channels, 2)
        check(f"{name}: 샘플레이트", loaded.sample_rate, RATE)
        check_true(f"{name}: 길이 유지", abs(loaded.duration - short.duration) < 0.15,
                   f"({loaded.duration:.3f} vs {short.duration:.3f})")
        if name.startswith(("wav", "flac", "aiff")):
            # 손실 없는 형식은 파형이 그대로여야 한다
            length = min(loaded.frames, short.frames)
            error = float(np.max(np.abs(loaded.data[:, :length] - short.data[:, :length])))
            tolerance = 1e-4 if "16" in name else 1e-6
            check_true(f"{name}: 파형 보존", error < tolerance, f"(최대 오차 {error:.2e})")
        else:
            # 손실 압축은 대략 비슷하면 된다
            check_true(f"{name}: 소리가 있다", loaded.peak() > 0.1, f"({loaded.peak():.3f})")

print("[3] 긴 곡 — 예전에 OGG 인코더가 프로세스째 죽던 조건")
long_audio = make_music(155.0)
check_true("155초 버퍼", long_audio.frames > 7_000_000, f"({long_audio.frames}샘플)")
with tempfile.TemporaryDirectory() as folder:
    for name in sorted(formats):
        # 자식 프로세스에서 돌려서, 죽더라도 이 검증이 멈추지 않게 한다
        script = (
            "import sys; sys.path.insert(0,'.');"
            "import numpy as np;"
            "from myvocal.audio.buffer import AudioBuffer;"
            "from myvocal.audio.export import export_audio;"
            f"rate={RATE};"
            "t=np.arange(int(155.0*rate))/rate;"
            "x=np.vstack([np.sin(2*np.pi*220*t)*0.5, np.sin(2*np.pi*330*t)*0.5]);"
            f"p=export_audio(AudioBuffer(x,rate), {str(Path(folder) / ('long_' + name))!r}, {name!r});"
            "print(p.stat().st_size)"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, cwd=".")
        check_true(f"{name}: 긴 곡에서 죽지 않음", result.returncode == 0,
                   f"(종료코드 {result.returncode}, {result.stderr.strip()[:120]})")
        if result.returncode == 0:
            check_true(f"{name}: 내용이 있다", int(result.stdout.strip()) > 10000)

print("[4] 크기 예상")
with tempfile.TemporaryDirectory() as folder:
    sample = make_music(20.0)
    for name in sorted(formats):
        path = export_audio(sample, Path(folder) / f"s_{name}", name)
        actual = path.stat().st_size / 1024 / 1024
        estimated = estimate_size_mb(sample.duration, name)
        ratio = actual / estimated if estimated > 0 else 0.0
        check_true(f"{name}: 예상이 실제의 0.5~2배 안",
                   0.5 <= ratio <= 2.0, f"(실제 {actual:.2f}MB, 예상 {estimated:.2f}MB)")

print("[5] 샘플레이트 변환하며 내보내기")
with tempfile.TemporaryDirectory() as folder:
    path = export_audio(short, Path(folder) / "resampled", "wav16", sample_rate=44100)
    loaded = AudioBuffer.load(path)
    check("44.1kHz 로 변환됨", loaded.sample_rate, 44100)
    check_true("길이 유지", abs(loaded.duration - short.duration) < 0.01)

print("[6] MIDI 왕복")
project = Composer(SongRequest(
    title="내보내기 시험", genre=GenreBlend({"rock": 60, "kpop": 40}), seed=5,
)).compose()
vocal = project.track("리드 보컬")
syllables = "다시만나는날에"
for index, note in enumerate(list(vocal.notes)[: len(syllables)]):
    vocal.notes.replace_note(note, note.with_lyric(syllables[index]))

with tempfile.TemporaryDirectory() as folder:
    path = export_midi(project, Path(folder) / "song")
    check("확장자", path.suffix, ".mid")
    check_true("파일 생성", path.exists() and path.stat().st_size > 1000)
    restored = import_midi(path)
    check("트랙 수", len(restored.tracks), len(project.tracks))
    # MIDI 는 템포를 정수 마이크로초로 저장하므로 왕복하면 미세 오차가 남는다
    check("BPM", float(restored.bpm), float(project.bpm), tol=0.01)
    check("박자표", str(restored.time_signature), str(project.time_signature))
    check("음 총수",
          sum(len(t.notes) for t in restored.tracks),
          sum(len(t.notes) for t in project.tracks))
    for original, back in zip(project.tracks, restored.tracks):
        pairs = zip(sorted(original.notes, key=lambda n: (n.start_tick, n.midi)),
                    sorted(back.notes, key=lambda n: (n.start_tick, n.midi)))
        mismatched = sum(
            1 for a, b in pairs
            if (a.midi, a.start_tick, a.duration_ticks, a.velocity)
            != (b.midi, b.start_tick, b.duration_ticks, b.velocity)
        )
        check(f"'{original.name}': 음이 그대로", mismatched, 0)
        check(f"'{original.name}': 악기 복원", back.instrument, original.instrument)
    # 한글이 깨지지 않아야 한다
    import mido

    midi_file = mido.MidiFile(str(path), charset="utf-8")
    names = [m.name for t in midi_file.tracks for m in t if m.type == "track_name"]
    check_true("한글 트랙 이름 보존", "리드 보컬" in names, f"({names})")
    lyrics = [m.text for t in midi_file.tracks for m in t if m.type == "lyrics"]
    check("한글 가사 보존", "".join(lyrics), syllables)
    drum_channels = {getattr(m, "channel", None)
                     for t in midi_file.tracks for m in t if m.type == "note_on"}
    check_true("드럼이 10번 채널(0부터 세면 9)", 9 in drum_channels, f"({drum_channels})")
    check("MIDI 타입 1", midi_file.type, 1)
    check("PPQ 보존", midi_file.ticks_per_beat, PPQ)
    check_raises("없는 파일", lambda: import_midi(Path(folder) / "없음.mid"))

print("[7] 템포가 바뀌는 곡도 MIDI 로")
varying = Project("변박곡", bpm=120)
varying.build_default_structure()
track = varying.add_track("피아노", "instrument")
for bar in range(1, 9):
    track.add_note(Note(60 + bar, varying.meter.bar_to_tick(bar), PPQ))
varying.add_tempo_change(5, 90.0)
with tempfile.TemporaryDirectory() as folder:
    path = export_midi(varying, Path(folder) / "varying")
    restored = import_midi(path)
    check("첫 BPM", float(restored.tempo.bpm_at_tick(0)), 120.0, tol=0.01)
    check("바뀐 BPM",
          float(restored.tempo.bpm_at_tick(varying.meter.bar_to_tick(5))), 90.0, tol=0.01)
    check("템포 변화 개수", len(restored.tempo.changes), 2)

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
