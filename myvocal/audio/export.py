"""
내보내기 — 64번의 Export.

만든 곡을 밖으로 꺼내는 길이다. 형식마다 쓰는 자리가 다르다.

    WAV   손실 없음. 다른 프로그램으로 넘길 때, 다시 편집할 때.
    FLAC  손실 없는데 용량이 절반. 보관용.
    MP3   가장 널리 재생된다. 들려줄 때, 올릴 때.
    OGG   MP3 와 비슷한데 특허 문제가 없다.
    MIDI  소리가 아니라 악보. 다른 DAW 에서 악기를 바꿔 다시 연주할 수 있다.

용량은 실제로 문제가 된다. 2분 31초 곡이 24비트 WAV 로 42MB 다. 메신저나
업로드 한도에 걸린다. 그래서 형식과 품질을 고를 수 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core.project import Project
from ..core.tracks import Track
from ..core.units import PPQ, TempoMap
from .buffer import AudioBuffer, AudioError


class ExportError(ValueError):
    """내보내기 관련 오류."""


# 형식 -> (soundfile 포맷 이름, 기본 서브타입, 확장자, 설명)
AUDIO_FORMATS: dict[str, tuple[str, str | None, str, str]] = {
    "wav24": ("WAV", "PCM_24", ".wav", "24비트 WAV. 손실 없음. 편집·보관용."),
    "wav16": ("WAV", "PCM_16", ".wav", "16비트 WAV. CD 품질. 손실 없음."),
    "wav32f": ("WAV", "FLOAT", ".wav", "32비트 부동소수점 WAV. 다른 프로그램으로 넘길 때."),
    "flac": ("FLAC", "PCM_24", ".flac", "24비트 FLAC. 손실 없고 용량은 절반쯤."),
    "flac16": ("FLAC", "PCM_16", ".flac", "16비트 FLAC. 더 작다."),
    "mp3": ("MP3", None, ".mp3", "MP3. 가장 널리 재생된다."),
    "ogg": ("OGG", "VORBIS", ".ogg", "OGG Vorbis. MP3 와 비슷한데 특허 문제가 없다."),
    "aiff": ("AIFF", "PCM_24", ".aiff", "24비트 AIFF. 맥 계열 프로그램용."),
}


def available_formats() -> dict[str, str]:
    """이 환경에서 실제로 쓸 수 있는 형식과 설명."""
    import soundfile as sf

    usable = set(sf.available_formats())
    return {
        name: info[3] for name, info in AUDIO_FORMATS.items()
        if info[0] in usable
    }


def estimate_size_mb(seconds: float, format_name: str, sample_rate: int = 48000,
                     channels: int = 2) -> float:
    """대략적인 파일 크기(MB). 형식을 고르기 전에 보여준다.

    손실 없는 형식(WAV, AIFF)은 정확하다. FLAC 과 손실 압축은 소리의 내용에
    따라 달라지므로 '보통 음악' 기준의 어림값이다. 단순한 소리(사인파,
    조용한 구간이 많은 곡)는 이보다 훨씬 작게 나온다.
    """
    if format_name not in AUDIO_FORMATS:
        raise ExportError(f"모르는 형식입니다: {format_name!r}")
    container, subtype, _, _ = AUDIO_FORMATS[format_name]
    if container in ("MP3", "OGG"):
        # 손실 압축은 비트레이트가 크기를 정한다.
        # libsndfile 의 기본 품질에서 실측한 값이 이 정도다.
        bitrate_kbps = 118.0
        return seconds * bitrate_kbps / 8.0 / 1024.0
    bits = {"PCM_16": 16, "PCM_24": 24, "FLOAT": 32}.get(subtype or "PCM_24", 24)
    raw = seconds * sample_rate * channels * bits / 8.0 / 1024.0 / 1024.0
    if container == "FLAC":
        # 실측: 24비트는 0.64배, 16비트는 0.50배쯤 된다. 조용한 곡은 더 줄어든다.
        return raw * (0.64 if bits == 24 else 0.50)
    return raw


def export_audio(
    audio: AudioBuffer, path: str | Path, format_name: str = "wav24",
    sample_rate: int | None = None,
) -> Path:
    """오디오를 파일로 내보낸다."""
    if format_name not in AUDIO_FORMATS:
        raise ExportError(
            f"모르는 형식입니다: {format_name!r}\n"
            f"사용 가능: {', '.join(sorted(AUDIO_FORMATS))}"
        )
    import soundfile as sf

    container, subtype, extension, _ = AUDIO_FORMATS[format_name]
    if container not in sf.available_formats():
        raise ExportError(
            f"{format_name} 형식을 이 환경에서 쓸 수 없습니다. "
            f"사용 가능: {', '.join(sorted(available_formats()))}"
        )

    target = Path(path)
    if target.suffix.lower() != extension:
        target = target.with_suffix(extension)
    target.parent.mkdir(parents=True, exist_ok=True)

    source = audio
    if sample_rate is not None and sample_rate != audio.sample_rate:
        source = audio.resample(sample_rate)

    kwargs: dict = {"format": container}
    if subtype is not None:
        if subtype not in sf.available_subtypes(container):
            raise ExportError(
                f"{container} 는 {subtype} 를 지원하지 않습니다. "
                f"가능: {', '.join(sf.available_subtypes(container))}"
            )
        kwargs["subtype"] = subtype

    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        _write_in_chunks(temporary, source, container, kwargs.get("subtype"))
        temporary.replace(target)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        raise ExportError(f"{target.name} 으로 내보내지 못했습니다: {error}") from error
    return target


# 한 번에 넘기는 최대 샘플 수. 10초 분량.
_CHUNK_FRAMES = 480000


def _write_in_chunks(
    path: Path, audio: AudioBuffer, container: str, subtype: str | None,
) -> None:
    """조각을 나눠 쓴다.

    긴 곡을 한 번에 넘기면 안 된다. 2분 31초(740만 샘플)를 OGG 로 한 번에
    쓰면 인코더가 프로세스째 죽는다(예외가 아니라 세그멘테이션 폴트라
    잡을 수도 없다). 조각으로 나누면 같은 결과가 정상적으로 나온다.
    조각 단위로 쓰면 메모리도 곡 길이와 무관하게 일정하게 유지된다.
    """
    import numpy as np
    import soundfile as sf

    kwargs: dict = {"format": container}
    if subtype is not None:
        kwargs["subtype"] = subtype
    data = audio.data
    with sf.SoundFile(
        str(path), "w", samplerate=audio.sample_rate, channels=audio.channels, **kwargs
    ) as handle:
        for begin in range(0, data.shape[1], _CHUNK_FRAMES):
            block = np.ascontiguousarray(data[:, begin : begin + _CHUNK_FRAMES].T)
            handle.write(block)


# ==========================================================================
# MIDI
# ==========================================================================

# 악기 이름 -> 일반 MIDI 프로그램 번호.
# 다른 프로그램에서 열었을 때 비슷한 소리가 나오게 하려는 것이다.
GENERAL_MIDI_PROGRAM: dict[str, int] = {
    "acoustic_piano": 0, "electric_piano": 4, "organ": 16,
    "acoustic_guitar": 24, "electric_guitar": 29,
    "electric_bass": 33, "synth_bass": 38, "sub_bass": 39,
    "strings": 48, "choir": 52, "brass": 61, "flute": 73,
    "synth_lead": 80, "synth_pad": 88, "bell": 14,
}


def export_midi(project: Project, path: str | Path) -> Path:
    """프로젝트를 MIDI 파일로 내보낸다.

    소리가 아니라 악보다. 다른 DAW 에서 열어 악기를 바꾸거나 더 편집할 수 있다.
    드럼은 관례대로 10번 채널에 놓는다.
    """
    import mido

    target = Path(path)
    if target.suffix.lower() not in (".mid", ".midi"):
        target = target.with_suffix(".mid")
    target.parent.mkdir(parents=True, exist_ok=True)

    # MIDI 표준은 글자 인코딩을 정해두지 않았고, mido 의 기본값은 latin-1 이다.
    # 그대로 두면 한글 트랙 이름이나 가사가 들어가는 순간 저장이 실패한다.
    # 요즘 프로그램들은 대개 UTF-8 로 읽으므로 그것을 쓴다.
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ, charset="utf-8")

    # 첫 트랙에는 템포와 박자표, 조표만 넣는다 (관례)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=project.meta.title, time=0))
    last_tick = 0
    for change in project.tempo.changes:
        meta.append(mido.MetaMessage(
            "set_tempo", tempo=mido.bpm2tempo(change.bpm), time=change.tick - last_tick
        ))
        last_tick = change.tick
    last_tick = 0
    for change in project.meter.changes:
        signature = change.signature
        meta.append(mido.MetaMessage(
            "time_signature", numerator=signature.numerator,
            denominator=signature.denominator, time=change.tick - last_tick,
        ))
        last_tick = change.tick
    try:
        accidentals = project.key.accidental_count
        name = str(project.key)
        meta.append(mido.MetaMessage("key_signature", key=name, time=0))
    except Exception:
        pass    # 조표를 못 적어도 곡 자체는 나와야 한다
    midi.tracks.append(meta)

    channel = 0
    for track in project.tracks:
        messages = mido.MidiTrack()
        messages.append(mido.MetaMessage("track_name", name=track.name, time=0))

        if track.kind == "drum":
            use_channel = 9     # 0부터 세므로 10번 채널
        else:
            use_channel = channel
            channel += 1
            if channel == 9:
                channel += 1    # 10번은 드럼 전용이라 건너뛴다
            if channel > 15:
                channel = 0
            program = GENERAL_MIDI_PROGRAM.get(track.instrument, 0)
            messages.append(mido.Message(
                "program_change", program=program, channel=use_channel, time=0
            ))

        volume = max(0, min(127, int(round(100 * (10 ** (track.volume_db / 20.0))))))
        messages.append(mido.Message("control_change", control=7, value=volume,
                                     channel=use_channel, time=0))
        pan_value = max(0, min(127, int(round((track.pan + 1.0) * 63.5))))
        messages.append(mido.Message("control_change", control=10, value=pan_value,
                                     channel=use_channel, time=0))

        # 음을 켜고 끄는 사건을 시간순으로 모은다
        events: list[tuple[int, int, int, int]] = []   # (tick, on/off, midi, velocity)
        for note in track.notes:
            events.append((note.start_tick, 1, note.midi, note.velocity))
            events.append((note.end_tick, 0, note.midi, 0))
            if note.lyric:
                events.append((note.start_tick, 2, 0, 0))
        # 낱말 끝 음절 뒤에는 띄어쓰기를 붙인다 (노래방 MIDI 의 관례). 다시 읽을 때
        # 이걸 보고 띄어쓰기를 되살린다.
        lyrics = {n.start_tick: n.lyric + (" " if n.word_end else "")
                  for n in track.notes if n.lyric}
        # 같은 시각이면 끄는 것을 먼저 한다. 안 그러면 같은 음이 겹쳐 꺼진다.
        events.sort(key=lambda e: (e[0], e[1]))

        previous = 0
        for tick, kind, midi_note, velocity in events:
            delta = tick - previous
            previous = tick
            if kind == 2:
                text = lyrics.get(tick, "")
                if text:
                    messages.append(mido.MetaMessage("lyrics", text=text, time=delta))
                continue
            if kind == 1:
                messages.append(mido.Message("note_on", note=midi_note,
                                             velocity=velocity, channel=use_channel,
                                             time=delta))
            else:
                messages.append(mido.Message("note_off", note=midi_note, velocity=0,
                                             channel=use_channel, time=delta))
        midi.tracks.append(messages)

    midi.save(str(target))
    return target


def import_midi(path: str | Path, title: str = "") -> Project:
    """MIDI 파일을 프로젝트로 읽는다. 8번의 '내 음악 추가' 가 쓴다."""
    import mido

    from ..music.theory import Key
    from ..core.notes import Note
    from ..core.units import TempoChange, TimeSignature

    source = Path(path)
    if not source.exists():
        raise ExportError(f"파일이 없습니다: {source}")
    # 인코딩이 파일마다 다르다. UTF-8 을 먼저 시도하고, 깨지면 latin-1 로 다시 읽는다.
    # latin-1 은 어떤 바이트든 받아들이므로 마지막 수단으로 항상 성공한다.
    midi = None
    problems: list[str] = []
    for charset in ("utf-8", "latin1"):
        try:
            midi = mido.MidiFile(str(source), charset=charset)
            break
        except UnicodeDecodeError as error:
            problems.append(f"{charset}: {error}")
        except Exception as error:
            raise ExportError(
                f"MIDI 파일을 읽을 수 없습니다: {source} ({error})"
            ) from error
    if midi is None:
        raise ExportError(
            f"MIDI 파일의 글자 인코딩을 알 수 없습니다: {source}\n" + "\n".join(problems)
        )

    scale = PPQ / midi.ticks_per_beat if midi.ticks_per_beat else 1.0
    project = Project(title=title or source.stem)

    tempo_changes: list[TempoChange] = []
    signature: TimeSignature | None = None

    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        name = ""
        open_notes: dict[tuple[int, int], tuple[int, int]] = {}
        collected: list[Note] = []
        lyric_at: dict[int, str] = {}
        is_drum = False
        program = 0

        for message in track:
            absolute += message.time
            tick = int(round(absolute * scale))
            if message.type == "track_name":
                name = message.name
            elif message.type == "set_tempo":
                tempo_changes.append(TempoChange(tick, mido.tempo2bpm(message.tempo)))
            elif message.type == "time_signature" and signature is None:
                try:
                    signature = TimeSignature(message.numerator, message.denominator)
                except Exception:
                    signature = None
            elif message.type == "program_change":
                program = message.program
            elif message.type == "lyrics" and message.text.strip():
                lyric_at[tick] = message.text
            elif message.type == "note_on" and message.velocity > 0:
                if message.channel == 9:
                    is_drum = True
                open_notes[(message.channel, message.note)] = (tick, message.velocity)
            elif message.type in ("note_off",) or (
                message.type == "note_on" and message.velocity == 0
            ):
                key = (message.channel, message.note)
                if key in open_notes:
                    start, velocity = open_notes.pop(key)
                    length = max(1, tick - start)
                    collected.append(Note(message.note, start, length, velocity))

        if not collected:
            continue
        # 가사를 그 시각에 시작하는 음에 붙인다. 가사가 있는 트랙은 보컬 트랙이다.
        with_lyrics = 0
        if lyric_at and not is_drum:
            for index, note in enumerate(collected):
                text = lyric_at.get(note.start_tick)
                if text:
                    collected[index] = note.with_lyric(text.strip(), word_end=text.endswith(" "))
                    with_lyrics += 1
        is_vocal = with_lyrics > 0
        instrument = "drum_kit" if is_drum else ("choir" if is_vocal
                                                 else _instrument_from_program(program))
        kind = "drum" if is_drum else ("vocal" if is_vocal else "instrument")
        new_track = project.add_track(name or f"트랙 {track_index}", kind, instrument=instrument)
        new_track.add_notes(collected)

    if tempo_changes:
        from ..core.units import TempoMap

        project.tempo = TempoMap(tempo_changes)
    if signature is not None:
        from ..core.units import MeterChange, MeterMap

        project.meter = MeterMap([MeterChange(0, signature)])

    # 조성을 추정한다
    from ..audio.analysis import AnalysisError
    from ..music.theory import detect_key

    all_notes = [n.midi for track in project.tracks if track.kind != "drum"
                 for n in track.notes]
    if all_notes:
        weights = [n.duration_ticks for track in project.tracks if track.kind != "drum"
                   for n in track.notes]
        guesses = detect_key(all_notes, weights)
        if guesses:
            project.key = guesses[0][0]
    return project


def _instrument_from_program(program: int) -> str:
    """일반 MIDI 프로그램 번호 -> 이 프로그램의 악기 이름."""
    if program <= 7:
        return "acoustic_piano"
    if program <= 15:
        return "electric_piano" if program <= 11 else "bell"
    if program <= 23:
        return "organ"
    if program <= 27:
        return "acoustic_guitar"
    if program <= 31:
        return "electric_guitar"
    if program <= 37:
        return "electric_bass"
    if program <= 39:
        return "synth_bass"      # GM 38~39 는 Synth Bass 1/2 다
    if program <= 51:
        return "strings"
    if program <= 55:
        return "choir"
    if program <= 63:
        return "brass"
    if program <= 79:
        return "flute"
    if program <= 87:
        return "synth_lead"
    if program <= 95:
        return "synth_pad"
    return "synth_pad"
