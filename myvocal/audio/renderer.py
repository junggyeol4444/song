"""
렌더링 — 프로젝트를 실제 소리로.

트랙에 들어 있는 음들을 악기로 연주해 오디오로 만들고, 믹서와 마스터를 거쳐
파일로 내보낸다. 64번의 Mixer / Master / Export 가 이 파일이다.

두 가지를 신경 쓴다.

1. 같은 음을 여러 번 만들지 않는다
   피아노 C4 를 세기 90 으로 0.5초 치는 소리는 곡 안에서 수십 번 나온다.
   매번 다시 합성하면 시간이 그만큼 곱해진다. 한 번 만들어 두고 재사용한다.

2. 여운을 자르지 않는다
   피아노 마지막 음은 건반을 떼고도 몇 초 울린다. 곡 길이에서 딱 끊으면
   소리가 뚝 끊긴다. 그래서 버퍼를 악기의 여운만큼 더 잡는다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..core.notes import Note
from ..core.project import Project
from ..core.tracks import Track
from ..core.units import TempoMap
from ..music.theory import midi_to_frequency
from ..music.instruments import (
    DRUM_NOTES, DrumKit, Instrument, InstrumentError, create_instrument,
)
from ..voice.phonemes import align_lyrics
from ..voice.synth import SingingSynth, VoiceError, VoiceTimbre
from . import dsp, loudness
from .buffer import AudioBuffer, AudioError, db_to_linear


class RenderError(ValueError):
    """렌더링 관련 오류."""


@dataclass(slots=True)
class RenderOptions:
    """렌더링 설정."""

    sample_rate: int = 48000
    channels: int = 2
    master_target_lufs: float | None = -14.0    # None 이면 음량 조절을 안 한다
    true_peak_ceiling_db: float = -1.0
    apply_track_effects: bool = True
    apply_master_chain: bool = True
    tail_seconds: float = 3.0
    velocity_quantize: int = 8       # 세기를 이 단위로 묶어 캐시 적중률을 높인다
    duration_quantize_ms: float = 25.0
    sing_lyrics: bool = True         # 가사가 있는 보컬 트랙을 실제로 부르게 한다

    def __post_init__(self) -> None:
        if self.sample_rate < 8000:
            raise RenderError(f"샘플레이트가 너무 낮습니다: {self.sample_rate}")
        if self.channels not in (1, 2):
            raise RenderError(f"채널은 1 또는 2 여야 합니다: {self.channels}")
        if self.velocity_quantize < 1:
            raise RenderError("세기 묶음 단위는 1 이상이어야 합니다.")


@dataclass(slots=True)
class RenderReport:
    """렌더링 결과 보고."""

    duration_seconds: float
    note_count: int
    track_count: int
    elapsed_seconds: float
    cache_hits: int
    cache_misses: int
    loudness: loudness.LoudnessReport | None = None
    mastering: loudness.MasteringResult | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        return self.duration_seconds / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def cache_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def summary(self) -> str:
        lines = [
            f"{self.duration_seconds:.1f}초 / 트랙 {self.track_count}개 / "
            f"음 {self.note_count}개",
            f"렌더링 {self.elapsed_seconds:.1f}초 (실시간의 {self.realtime_factor:.1f}배), "
            f"음색 재사용 {self.cache_rate:.0%}",
        ]
        if self.loudness is not None:
            lines.append(self.loudness.summary())
        if self.warnings:
            lines.append("경고:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


class NoteCache:
    """같은 음색의 음을 다시 만들지 않게 모아 둔다."""

    __slots__ = ("_cache", "hits", "misses", "_limit")

    def __init__(self, limit: int = 4000) -> None:
        self._cache: dict[tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._limit = limit

    def get(self, key: tuple, maker: Callable[[], np.ndarray]) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = maker()
        if len(self._cache) < self._limit:
            self._cache[key] = value
        return value

    def clear(self) -> None:
        self._cache.clear()


def _effect_chain(buffer: AudioBuffer, track: Track) -> AudioBuffer:
    """트랙에 걸린 이펙트를 순서대로 적용한다."""
    result = buffer
    for slot in track.effects:
        if not slot.enabled:
            continue
        params = dict(slot.params)
        original = result
        try:
            if slot.effect_type == "eq":
                bands = [dsp.FilterBand(**b) for b in params.get("bands", [])]
                result = dsp.Equalizer(bands).process(result)
            elif slot.effect_type == "compressor":
                result, _ = dsp.compress(result, dsp.CompressorSettings(**params))
            elif slot.effect_type == "limiter":
                result, _ = dsp.limit(result, **params)
            elif slot.effect_type == "gate":
                result = dsp.gate(result, **params)
            elif slot.effect_type == "saturate":
                result = dsp.saturate(result, **params)
            elif slot.effect_type == "reverb":
                result = dsp.reverb(result, dsp.ReverbSettings(**params))
            elif slot.effect_type == "delay":
                result = dsp.delay_effect(result, **params)
            elif slot.effect_type == "stereo_width":
                result = dsp.stereo_width(result, params.get("width", 1.0))
            elif slot.effect_type == "mono_below":
                result = dsp.mono_below(result, params.get("frequency", 120.0))
            elif slot.effect_type == "haas":
                result = dsp.haas_widen(result, params.get("delay_ms", 12.0))
            elif slot.effect_type == "highpass":
                result = dsp.highpass(result, params.get("frequency", 80.0),
                                      params.get("order", 2))
            elif slot.effect_type == "lowpass":
                result = dsp.lowpass(result, params.get("frequency", 8000.0),
                                     params.get("order", 2))
            else:
                raise RenderError(f"처리할 수 없는 이펙트입니다: {slot.effect_type}")
        except TypeError as error:
            raise RenderError(
                f"'{track.name}' 트랙의 {slot.effect_type} 설정이 잘못됐습니다: {error}"
            ) from error
        if slot.wet < 1.0:
            length = max(original.frames, result.frames)
            result = original.padded_to(length).with_gain(1.0 - slot.wet).mix(
                result.padded_to(length).with_gain(slot.wet), 0
            )
    return result


class Renderer:
    """프로젝트를 오디오로 만든다."""

    def __init__(self, options: RenderOptions | None = None) -> None:
        self.options = options or RenderOptions()
        self.cache = NoteCache()
        self._instruments: dict[str, Instrument] = {}

    def _instrument(self, name: str) -> Instrument:
        if name not in self._instruments:
            try:
                self._instruments[name] = create_instrument(name)
            except InstrumentError as error:
                raise RenderError(f"악기를 만들 수 없습니다: {error}") from error
        return self._instruments[name]

    # ------------------------------------------------------------------

    def render_vocal(
        self, track: Track, tempo: TempoMap, total_frames: int,
        warnings: list[str] | None = None,
    ) -> AudioBuffer | None:
        """가사가 붙은 보컬 트랙을 노래로 만든다.

        가사가 없으면 None 을 돌려준다. 그러면 평소대로 악기로 연주한다.
        가사 없이 노래 합성기를 돌리면 전부 '아' 로 부르게 되는데, 그건
        코러스 악기와 다를 바 없으면서 소리만 이상해진다.
        """
        rate = self.options.sample_rate
        warnings = warnings if warnings is not None else []
        notes = [n for n in track.notes if n.lyric]
        if not notes:
            return None
        if len(notes) < len(track.notes):
            warnings.append(
                f"'{track.name}': 음 {len(track.notes) - len(notes)}개에 가사가 없어 "
                f"부르지 않았습니다."
            )

        preset = track.singing_style or "mezzo"
        try:
            timbre = VoiceTimbre.preset(preset)
        except VoiceError:
            timbre = VoiceTimbre.preset("mezzo")
            warnings.append(
                f"'{track.name}': 모르는 목소리 '{preset}' 이라 기본값으로 불렀습니다."
            )

        text = "".join(n.lyric for n in notes)
        note_times = [
            (tempo.tick_to_seconds(n.start_tick),
             max(0.05, tempo.tick_to_seconds(n.end_tick) - tempo.tick_to_seconds(n.start_tick)))
            for n in notes
        ]
        pitches = [midi_to_frequency(n.midi) for n in notes]
        try:
            syllables = align_lyrics(text, note_times)
            singing = SingingSynth(timbre, rate).render(syllables, pitches)
        except Exception as error:
            warnings.append(f"'{track.name}': 노래로 만들 수 없어 악기로 연주했습니다 ({error})")
            return None

        # 세기를 반영한다. 노래 합성기는 음높이와 발음만 다루므로 여기서 건다.
        result = AudioBuffer.silence(total_frames, 1, rate)
        result.mix_in_place(singing, 0)
        envelope = np.ones(total_frames)
        for note, (start, duration) in zip(notes, note_times):
            begin = max(0, int(start * rate))
            end = min(total_frames, int((start + duration) * rate))
            if end > begin:
                envelope[begin:end] = 0.35 + 0.65 * (note.velocity / 127.0)
        return AudioBuffer.from_mono(result.data[0] * envelope, rate)

    def render_track(
        self, track: Track, tempo: TempoMap, total_frames: int,
        warnings: list[str] | None = None,
    ) -> AudioBuffer:
        """트랙 하나를 모노 오디오로 만든다. 팬과 이펙트는 아직 적용하지 않는다."""
        rate = self.options.sample_rate
        buffer = AudioBuffer.silence(total_frames, 1, rate)
        if not track.notes:
            return buffer

        if self.options.sing_lyrics and track.kind == "vocal":
            sung = self.render_vocal(track, tempo, total_frames, warnings)
            if sung is not None:
                return sung

        instrument = self._instrument(track.instrument)
        is_drum = isinstance(instrument, DrumKit)
        warnings = warnings if warnings is not None else []
        skipped_low = skipped_high = skipped_unknown = 0

        velocity_step = self.options.velocity_quantize
        duration_step = max(0.001, self.options.duration_quantize_ms / 1000.0)

        for note in track.notes:
            start_seconds = tempo.tick_to_seconds(note.start_tick)
            end_seconds = tempo.tick_to_seconds(note.end_tick)
            duration = max(0.01, end_seconds - start_seconds)
            start_frame = int(round(start_seconds * rate))
            if start_frame >= total_frames:
                continue

            # 캐시 적중률을 높이려고 세기와 길이를 묶는다.
            # 세기 1 차이, 길이 1ms 차이는 사람이 못 듣는다.
            velocity = max(1, min(127, (note.velocity // velocity_step) * velocity_step
                                  + velocity_step // 2))
            quantized_duration = max(duration_step,
                                     round(duration / duration_step) * duration_step)

            if is_drum:
                if note.midi not in DRUM_NOTES:
                    skipped_unknown += 1
                    continue
                key = ("drum", DRUM_NOTES[note.midi], velocity, rate)
                maker = (lambda m=note.midi, v=velocity:
                         instrument.render(DRUM_NOTES[m], v, rate))
            else:
                info = instrument.info
                if note.midi < info.lowest_midi:
                    skipped_low += 1
                    continue
                if note.midi > info.highest_midi:
                    skipped_high += 1
                    continue
                key = (track.instrument, note.midi, velocity,
                       round(quantized_duration, 4), rate)
                maker = (lambda m=note.midi, v=velocity, d=quantized_duration:
                         instrument.render_note(m, v, d, rate))

            signal = self.cache.get(key, maker)
            if signal.shape[0] == 0:
                continue
            # 오토메이션이 있으면 그 시점의 음량을 반영한다
            gain = db_to_linear(track.volume_db_at(note.start_tick) - track.volume_db)
            buffer.mix_in_place(AudioBuffer.from_mono(signal, rate), start_frame, gain)

        if skipped_low or skipped_high:
            warnings.append(
                f"'{track.name}': 음역을 벗어난 음 {skipped_low + skipped_high}개를 "
                f"소리내지 않았습니다 (아래 {skipped_low}, 위 {skipped_high}). "
                f"{instrument.info.display_name} 의 음역은 {instrument.info.range_text} 입니다."
            )
        if skipped_unknown:
            warnings.append(
                f"'{track.name}': 배정된 드럼이 없는 음 {skipped_unknown}개를 건너뛰었습니다."
            )
        return buffer

    # ------------------------------------------------------------------

    def render(
        self, project: Project, progress: Callable[[str, float], None] | None = None,
    ) -> tuple[AudioBuffer, RenderReport]:
        """프로젝트 전체를 렌더링한다."""
        started = time.time()
        rate = self.options.sample_rate
        warnings: list[str] = []

        audible = project.tracks.audible()
        if not audible:
            raise RenderError(
                "소리가 나는 트랙이 없습니다. 전부 음소거돼 있거나 트랙이 없습니다."
            )

        # 악기 여운까지 담을 만큼 버퍼를 넉넉히 잡는다
        extra = self.options.tail_seconds
        for track in audible:
            try:
                instrument = self._instrument(track.instrument)
            except RenderError:
                continue
            if track.notes:
                low, high = track.pitch_range() or (60, 60)
                extra = max(extra, instrument.tail_seconds(low, 100))
        total_seconds = project.duration_seconds + extra
        total_frames = int(math.ceil(total_seconds * rate))

        mix = AudioBuffer.silence(total_frames, self.options.channels, rate)
        note_count = 0

        for index, track in enumerate(audible):
            if progress is not None:
                progress(f"{track.name} 연주 중", index / max(1, len(audible)) * 0.8)
            mono = self.render_track(track, project.tempo, total_frames, warnings)
            note_count += len(track.notes)
            if mono.is_silent:
                continue
            stereo = mono.panned(track.pan) if self.options.channels == 2 else mono
            stereo = stereo.with_gain_db(track.volume_db)
            if self.options.apply_track_effects and track.effects:
                stereo = _effect_chain(stereo, track)
            mix.mix_in_place(stereo, 0)

        if progress is not None:
            progress("마스터링", 0.85)

        mastering = None
        if self.options.apply_master_chain:
            mix = self._master_chain(mix, project)
        if self.options.master_target_lufs is not None:
            mix, mastering = loudness.master_to_lufs(
                mix, self.options.master_target_lufs, self.options.true_peak_ceiling_db
            )

        if progress is not None:
            progress("측정", 0.95)
        report = RenderReport(
            duration_seconds=total_seconds,
            note_count=note_count,
            track_count=len(audible),
            elapsed_seconds=time.time() - started,
            cache_hits=self.cache.hits,
            cache_misses=self.cache.misses,
            loudness=loudness.measure(mix),
            mastering=mastering,
            warnings=warnings,
        )
        if progress is not None:
            progress("완료", 1.0)
        return mix, report

    def _master_chain(self, mix: AudioBuffer, project: Project) -> AudioBuffer:
        """마스터 처리. 장르 성향을 반영한다."""
        profile = project.genre.resolve()
        result = mix
        # 들리지 않는 초저역은 헤드룸만 잡아먹는다
        result = dsp.highpass(result, 25.0, order=2)
        # 저역은 모노로. 클럽 시스템과 바이닐에서 문제가 되는 것을 막는다.
        result = dsp.mono_below(result, 110.0)
        # 장르 밝기를 반영한 아주 완만한 EQ
        bands = []
        if abs(profile.brightness) > 0.05:
            bands.append(dsp.FilterBand("highshelf", 8000.0, 0.7071,
                                        profile.brightness * 2.5))
        if profile.low_end > 0.05:
            bands.append(dsp.FilterBand("lowshelf", 110.0, 0.7071, profile.low_end * 2.0))
        if bands:
            result = dsp.Equalizer(bands).process(result)
        # 접착용 컴프레서. 장르의 압축 성향만큼만 건다.
        if profile.compression > 0.1:
            ratio = 1.5 + profile.compression * 2.0
            threshold = -18.0 + profile.compression * 6.0
            result, _ = dsp.compress(result, dsp.CompressorSettings(
                threshold_db=threshold, ratio=ratio, attack_ms=20.0,
                release_ms=160.0, knee_db=6.0, makeup_db=0.0,
            ))
        if profile.stereo_width != 1.0:
            result = dsp.stereo_width(result, profile.stereo_width)
        return result


def render_project(
    project: Project, path: str | Path | None = None,
    options: RenderOptions | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> tuple[AudioBuffer, RenderReport]:
    """프로젝트를 렌더링하고, 경로를 주면 저장까지 한다."""
    renderer = Renderer(options)
    audio, report = renderer.render(project, progress)
    if path is not None:
        audio.save(path)
    return audio, report


def render_stems(
    project: Project, folder: str | Path, options: RenderOptions | None = None,
) -> dict[str, Path]:
    """트랙별로 따로 파일을 만든다 (스템 내보내기).

    다른 프로그램에서 믹스를 다시 하거나, 한 트랙만 교체할 때 쓴다.
    """
    options = options or RenderOptions()
    renderer = Renderer(options)
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    rate = options.sample_rate
    total_frames = int(math.ceil((project.duration_seconds + options.tail_seconds) * rate))

    result: dict[str, Path] = {}
    for index, track in enumerate(project.tracks):
        mono = renderer.render_track(track, project.tempo, total_frames)
        if mono.is_silent:
            continue
        stereo = mono.panned(track.pan).with_gain_db(track.volume_db)
        if options.apply_track_effects and track.effects:
            stereo = _effect_chain(stereo, track)
        safe = "".join(c if c.isalnum() or c in " -_()" else "_" for c in track.name).strip()
        path = target / f"{index + 1:02d}_{safe or 'track'}.wav"
        stereo.save(path)
        result[track.name] = path
    return result
