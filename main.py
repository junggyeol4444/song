#!/usr/bin/env python3
"""
MYVOCAL Studio 실행 파일.

    python main.py                내 목소리로 노래와 뮤직비디오를 만드는 프로그램
    python main.py 곡.mvp          프로젝트를 열면서 시작
    python main.py --check         이 컴퓨터에서 돌아갈 수 있는지 확인만 한다
    python main.py --song "설명"    화면 없이 곡을 만들어 파일로 내보낸다
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def check_environment() -> int:
    """필요한 것이 다 있는지 확인하고 알려준다."""
    problems: list[str] = []
    notes: list[str] = []

    if sys.version_info < (3, 10):
        problems.append(
            f"파이썬 3.10 이상이 필요합니다. 지금은 {sys.version.split()[0]} 입니다."
        )
    else:
        notes.append(f"파이썬 {sys.version.split()[0]}")

    for module, label, required in (
        ("numpy", "numpy (계산)", True),
        ("scipy", "scipy (신호처리)", True),
        ("soundfile", "soundfile (오디오 파일)", True),
        ("mido", "mido (MIDI)", True),
        ("PySide6", "PySide6 (화면)", False),
        ("sounddevice", "sounddevice (소리 재생)", False),
        ("anthropic", "anthropic (Claude 작사)", False),
    ):
        try:
            __import__(module)
            notes.append(label)
        except Exception as error:
            text = f"{label} — {type(error).__name__}: {error}"
            if required:
                problems.append(f"{text}\n    pip install {module}")
            else:
                notes.append(f"{text}  (없어도 됨)")

    try:
        from myvocal.audio.playback import audio_available

        available, message = audio_available()
        notes.append(f"소리 재생: {'가능' if available else '불가'} — {message.splitlines()[0]}")
    except Exception as error:
        notes.append(f"소리 확인 실패: {error}")

    print("MYVOCAL Studio 환경 확인\n")
    for note in notes:
        print(f"  · {note}")
    if problems:
        print("\n해결해야 할 것:")
        for problem in problems:
            print(f"  ! {problem}")
        return 1
    print("\n실행할 수 있습니다.")
    return 0


def make_song_headless(description: str, output: str, genre: str,
                       seed: int | None, export_format: str, lyrics: bool = True) -> int:
    """화면 없이 곡을 만들어 파일로 내보낸다."""
    from myvocal.audio.export import export_audio
    from myvocal.audio.renderer import render_project
    from myvocal.music.composer import Composer, SongRequest
    from myvocal.music.genre import GenreBlend, GenreError

    try:
        blend = GenreBlend.parse(description) if genre == "auto" else GenreBlend.parse(genre)
    except GenreError:
        blend = GenreBlend.single("pop")

    final_note = ""
    for line in description.splitlines():
        if "마지막" in line and "후렴" in line:
            final_note = line.strip()
            break

    request = SongRequest(
        title=description.splitlines()[0][:40] if description else "새 곡",
        genre=blend, subject=description, final_chorus_note=final_note, seed=seed,
    )
    print(f"장르: {blend}")
    project = Composer(request).compose()
    print(project.summary())

    if lyrics:
        from myvocal.music.lyrics import LyricsBrief, lyrics_command, vocal_track, write_lyrics
        from myvocal.providers import default_registry

        print("\n가사 쓰는 중...")
        try:
            track = vocal_track(project)
            draft = write_lyrics(project, track, default_registry(),
                                 LyricsBrief(subject=description, title=request.title, seed=seed),
                                 on_progress=lambda text: print(f"  {text}"))
            lyrics_command(track, draft).apply(project)
            who = "Claude" if draft.provider == "anthropic" else "내장 규칙"
            print(f"작사: {who}" + (" (대체됨)" if draft.fell_back else ""))
            if draft.note:
                print(draft.note)
            if draft.cost_note and draft.provider != "local":
                print(draft.cost_note)
            if draft.mismatches:
                print(f"음절 수가 다른 줄 {len(draft.mismatches)}개는 멜로디를 맞췄습니다.")
            print()
            print(draft.text())
        except Exception as error:
            print(f"작사하지 못했습니다: {error}\n가사 없이 계속합니다.")

    target = Path(output)
    project.save(target.with_suffix(".mvp"))
    print(f"\n프로젝트 저장: {target.with_suffix('.mvp')}")

    print("소리 만드는 중...")
    audio, report = render_project(
        project, progress=lambda stage, value: print(f"  {value * 100:3.0f}% {stage}")
    )
    print(report.summary())
    saved = export_audio(audio, target, export_format)
    print(f"\n오디오 저장: {saved} ({saved.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="MYVOCAL Studio",
        description="내 목소리와 내 캐릭터로 노래와 뮤직비디오를 만드는 프로그램",
    )
    parser.add_argument("project", nargs="?", help="열 프로젝트 파일 (.mvp)")
    parser.add_argument("--check", action="store_true", help="실행 환경만 확인한다")
    parser.add_argument("--song", metavar="설명", help="화면 없이 곡을 만든다")
    parser.add_argument("--out", default="output", help="내보낼 파일 이름 (--song 과 함께)")
    parser.add_argument("--genre", default="auto", help="장르 (예: 'rock 60 ballad 40')")
    parser.add_argument("--seed", type=int, default=None, help="같은 번호면 같은 곡")
    parser.add_argument("--format", default="wav24", help="오디오 형식 (wav24/flac/mp3 등)")
    parser.add_argument("--no-lyrics", action="store_true",
                        help="가사를 쓰지 않는다 (--song 과 함께)")
    arguments = parser.parse_args()

    if arguments.check:
        return check_environment()

    if arguments.song:
        return make_song_headless(
            arguments.song, arguments.out, arguments.genre,
            arguments.seed, arguments.format, lyrics=not arguments.no_lyrics,
        )

    try:
        from myvocal.ui.app import run
    except ImportError as error:
        print(f"화면을 띄울 수 없습니다: {error}")
        print("\nPySide6 가 필요합니다:")
        print("    pip install PySide6")
        print("\n화면 없이 곡만 만들려면:")
        print('    python main.py --song "애니메이션 록 발라드" --out 내곡')
        return 1

    argv = [sys.argv[0]]
    if arguments.project:
        argv.append(arguments.project)
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
