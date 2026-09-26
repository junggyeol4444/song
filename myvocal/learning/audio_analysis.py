"""
오디오 분석 — 8번 '내 음악 추가' 와 9번 Reference 분석에서 음원 파일을 잰다.

    템포        소리가 새로 생기는 세기(onset)의 자기상관으로 빠르기를, 동적계획법으로
                박 위치를 찾는다 (Ellis 2007)
    마디        저음(킥·베이스)이 가장 세게 들어오는 박을 1박으로 본다 (4/4 가정)
    조성        박마다의 음이름 분포(크로마)와 Krumhansl 조성 프로파일의 상관
    코드        반 마디마다 크로마로 가장 그럴듯한 코드 (악보 분석과 같은 판정)
    구조        마디마다 크로마·음색·크기의 닮음 행렬 → 경계 → 되풀이 구간
    에너지       마디마다의 크기
    리듬        박 사이(엇박)에 생기는 소리의 비율, 8분음표 뒤쪽 소리의 위치(스윙)
    Mix         통합 라우드니스(LUFS), 라우드니스 폭(LRA), 봉우리/평균 차이,
                저·중·고역 비율, 좌우 넓이, 밝기

섞인 소리에서 악기 이름, 가사, 보컬 표현을 알아내려면 음원 분리와 음성 인식
모델이 필요하다. 이 프로그램에는 그런 모델이 없으므로 그 항목은 비워 두고 이유를
적는다. MIDI 나 프로젝트 파일로 넣으면 악기와 가사를 잴 수 있다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile
from scipy import signal as scipy_signal

from ..audio import loudness
from ..audio.buffer import AudioBuffer
from ..music.theory import Key, Pitch
from .analysis import (
    MusicAnalysis, MusicAnalysisError, _genre_guess, best_chord, chord_roots_and_kinds,
    detect_sections, diatonic_ratio, roman_of, SEVENTH_KINDS,
)

RATE = 22050
N_FFT = 4096
HOP = 512

# Krumhansl & Kessler (1982)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """(채널, 샘플) 과 샘플레이트."""
    try:
        data, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    except Exception as error:
        raise MusicAnalysisError(f"오디오 파일을 읽을 수 없습니다: {path} ({error})") from error
    if data.shape[0] < rate:
        raise MusicAnalysisError("1초보다 짧은 파일은 분석할 수 없습니다.")
    return data.T, rate


def _resample(x: np.ndarray, rate: int, target: int = RATE) -> np.ndarray:
    if rate == target:
        return x
    divisor = math.gcd(rate, target)
    return scipy_signal.resample_poly(x, target // divisor, rate // divisor, axis=-1)


def _spectrogram(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(N_FFT)
    frames = 1 + max(0, (len(mono) - N_FFT) // HOP)
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    spectrum = np.abs(np.fft.rfft(mono[index] * window, axis=1))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
    return spectrum, freqs


def onset_envelope(spectrum: np.ndarray) -> np.ndarray:
    """소리가 새로 생기는 세기. 로그 스펙트럼의 증가분만 더한다."""
    log = np.log1p(1000.0 * spectrum)
    flux = np.maximum(0.0, np.diff(log, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    flux -= scipy_signal.medfilt(flux, 31)          # 느린 변화를 빼서 봉우리만 남긴다
    flux = np.maximum(flux, 0.0)
    return flux / (flux.max() or 1.0)


def estimate_tempo(envelope: np.ndarray) -> tuple[float, float]:
    """(BPM, 확신도). 자기상관에 120 BPM 둘레를 선호하는 사전분포를 곱한다.

    빠르기는 두 배/절반으로 헷갈리기 쉽다. 사람도 그렇다. 사전분포가 그중
    대중음악에서 흔한 쪽을 고르게 한다.
    """
    frame_rate = RATE / HOP
    x = envelope - envelope.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] or 1.0
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60.0 * frame_rate / np.maximum(lags, 1), 0)
    valid = (bpm >= 55) & (bpm <= 215)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 120.0) / 0.9) ** 2)
    score = np.where(valid, ac * prior, -np.inf)
    best = int(np.argmax(score))
    # 정수 지연 사이를 포물선으로 보간
    lag = float(best)
    if 1 <= best < len(ac) - 1:
        y0, y1, y2 = ac[best - 1], ac[best], ac[best + 1]
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > 1e-12:
            lag = best + 0.5 * (y0 - y2) / denominator
    confidence = float(np.clip(ac[best], 0.0, 1.0))
    return 60.0 * frame_rate / lag, confidence


def track_beats(envelope: np.ndarray, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """박 위치 (프레임 번호). Ellis (2007) 동적계획법."""
    frame_rate = RATE / HOP
    period = 60.0 * frame_rate / bpm
    n = len(envelope)
    score = envelope.astype(float).copy()
    backlink = np.full(n, -1)
    window = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    penalty = -tightness * np.log(-window / period) ** 2
    for i in range(n):
        candidates = i + window
        valid = candidates >= 0
        if not valid.any():
            continue
        values = score[candidates[valid]] + penalty[valid]
        best = int(np.argmax(values))
        score[i] = envelope[i] + values[best]
        backlink[i] = candidates[valid][best]
    # 끝부분에서 점수가 높은 곳부터 거슬러 올라간다
    tail = score[max(0, n - int(period) * 2):]
    last = max(0, n - len(tail)) + int(np.argmax(tail))
    beats = [last]
    while backlink[beats[-1]] >= 0:
        beats.append(int(backlink[beats[-1]]))
    return np.array(sorted(beats))


def chroma(spectrum: np.ndarray, freqs: np.ndarray, low: float = 80.0,
           high: float = 2000.0) -> np.ndarray:
    """프레임마다의 음이름 분포 (12).

    스펙트럼의 봉우리(실제 배음)만 센다. 모든 주파수 칸을 더하면 잡음과 번짐이
    음이름마다 고르게 쌓여서 실제 음보다 커지고, 칸 수가 많은 음이름(범위 끝)이
    유리해진다. 처음에 그렇게 했더니 F 단조 화음이 A 로 나왔다.

    봉우리 주파수는 포물선 보간으로 칸 사이까지 잡고, 반음 가운데에서 벗어난
    만큼 이웃 음이름에 나눠 준다.
    """
    magnitude = spectrum
    left = magnitude[:, :-2]
    center = magnitude[:, 1:-1]
    right = magnitude[:, 2:]
    frame_max = magnitude.max(axis=1, keepdims=True) + 1e-12
    is_peak = (center > left) & (center >= right) & (center > frame_max * 10 ** (-50 / 20))
    frames, bins = np.nonzero(is_peak)
    bins = bins + 1
    a = np.log(magnitude[frames, bins - 1] + 1e-12)
    b = np.log(magnitude[frames, bins] + 1e-12)
    c = np.log(magnitude[frames, bins + 1] + 1e-12)
    denominator = a - 2 * b + c
    shift = np.where(np.abs(denominator) > 1e-12, 0.5 * (a - c) / np.where(denominator == 0, 1, denominator), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    step = freqs[1] - freqs[0]
    frequency = (bins + shift) * step
    keep = (frequency >= low) & (frequency <= high)
    frames, frequency = frames[keep], frequency[keep]
    weight = np.sqrt(magnitude[frames, bins[keep]])
    midi = 69.0 + 12.0 * np.log2(frequency / 440.0)
    lower = np.floor(midi)
    fraction = midi - lower
    result = np.zeros((spectrum.shape[0], 12))
    np.add.at(result, (frames, lower.astype(int) % 12), weight * (1.0 - fraction))
    np.add.at(result, (frames, (lower.astype(int) + 1) % 12), weight * fraction)
    return result


def key_from_chroma(total: np.ndarray) -> tuple[int, bool, float]:
    """(으뜸음 음이름 번호, 단조인가, 상관계수)."""
    best = (0, False, -2.0)
    for tonic in range(12):
        rotated = np.roll(total, -tonic)
        for profile, minor in ((MAJOR_PROFILE, False), (MINOR_PROFILE, True)):
            r = float(np.corrcoef(rotated, profile)[0, 1])
            if r > best[2]:
                best = (tonic, minor, r)
    return best


def refine_key(total: np.ndarray, chord_names: list[str],
               anchors: list[str] | None = None) -> tuple[int, bool, float]:
    """음 분포 상관이 높은 조 후보 셋 중, 으뜸화음이 가장 그럴듯한 조.

    점수 = 상관계수 + 0.5 × (으뜸화음 비율) + 0.4 × (구간 첫 코드·마지막 코드 중 으뜸화음 비율).
    곡은 구간을 으뜸화음으로 시작하고 으뜸화음으로 끝나는 경우가 많다. 나란한조(F# 단조와
    A 장조)는 음 분포도, 코드 빈도도 비슷할 때가 있어서 (EDM 곡에서 24 대 22) 이것까지 본다.
    """
    candidates = []
    for tonic in range(12):
        rotated = np.roll(total, -tonic)
        for profile, minor in ((MAJOR_PROFILE, False), (MINOR_PROFILE, True)):
            candidates.append((float(np.corrcoef(rotated, profile)[0, 1]), tonic, minor))
    candidates.sort(reverse=True)
    parsed = [chord_roots_and_kinds(c) for c in chord_names]
    parsed = [p for p in parsed if p is not None]
    anchor_parsed = [p for p in (chord_roots_and_kinds(c) for c in (anchors or [])) if p is not None]

    def is_home(item, tonic: int, minor: bool) -> bool:
        root, kind = item
        return (root == tonic and not kind.startswith("dim")
                and (kind.startswith("m") and not kind.startswith("maj")) == minor)

    best = None
    for r, tonic, minor in candidates[:3]:
        ratio = (sum(1 for p in parsed if is_home(p, tonic, minor)) / len(parsed)) if parsed else 0.0
        anchor_ratio = (sum(1 for p in anchor_parsed if is_home(p, tonic, minor)) / len(anchor_parsed)
                        if anchor_parsed else 0.0)
        score = r + 0.5 * ratio + 0.4 * anchor_ratio
        if best is None or score > best[0]:
            best = (score, tonic, minor, r)
    return best[1], best[2], best[3]


def _chord_at_beat(chords: list[tuple[float, str]], beat: float) -> str | None:
    current = None
    for start, name in chords:
        if start <= beat + 1e-9:
            current = name
        else:
            break
    return current


def _mel_bands(spectrum: np.ndarray, freqs: np.ndarray, count: int = 20) -> np.ndarray:
    """음색을 대략 나타내는 대역 에너지 (멜 간격, 로그)."""
    mel = 2595 * np.log10(1 + freqs / 700)
    edges = np.linspace(mel[1], mel[-1], count + 1)
    result = np.zeros((spectrum.shape[0], count))
    power = spectrum ** 2
    for i in range(count):
        band = (mel >= edges[i]) & (mel < edges[i + 1])
        result[:, i] = np.log1p(power[:, band].sum(axis=1))
    return result


def analyze_audio(path: str | Path) -> MusicAnalysis:
    data, rate = load_audio(path)
    return analyze_signal(data, rate, Path(path).name)


def analyze_signal(data: np.ndarray, rate: int, source: str = "") -> MusicAnalysis:
    """(채널, 샘플) 신호를 분석한다."""
    if data.ndim == 1:
        data = data[None, :]
    analysis = MusicAnalysis(source or "오디오", "audio")
    analysis.duration_seconds = data.shape[1] / rate

    # ---- Mix (원래 샘플레이트, 원래 채널로) ----
    buffer = AudioBuffer(data.astype(np.float32), rate) if data.shape[0] <= 2 else \
        AudioBuffer(data[:2].astype(np.float32), rate)
    report = loudness.measure(buffer)
    mix: dict[str, float] = {
        "통합 라우드니스 LUFS": float(report.integrated_lufs),
        "라우드니스 폭 LU": float(report.loudness_range_lu),
        "트루 피크 dBTP": float(report.true_peak_dbtp),
    }
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    mix["봉우리-평균 차이 dB"] = 20 * math.log10(peak / rms) if rms > 0 and peak > 0 else 0.0
    if data.shape[0] >= 2:
        mid = (data[0] + data[1]) / 2
        side = (data[0] - data[1]) / 2
        mix["좌우 넓이 (옆/가운데)"] = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) or 1.0))

    mono = _resample(data.mean(axis=0), rate)
    spectrum, freqs = _spectrogram(mono)
    power = (spectrum ** 2).sum(axis=0)
    total_power = power.sum() or 1.0
    mix["저역 비율 % (<250Hz)"] = float(100 * power[freqs < 250].sum() / total_power)
    mix["고역 비율 % (>4kHz)"] = float(100 * power[freqs > 4000].sum() / total_power)
    mix["밝기 (무게중심 Hz)"] = float((freqs * power).sum() / total_power)
    analysis.mix = mix

    # ---- 템포와 박 ----
    envelope = onset_envelope(spectrum)
    bpm, confidence = estimate_tempo(envelope)
    beats = track_beats(envelope, bpm)
    if len(beats) >= 8:
        # 박 간격의 중앙값으로 다시 잰다 (자기상관보다 정밀하다)
        bpm = 60.0 * (RATE / HOP) / float(np.median(np.diff(beats)))
        # 빠르기는 두 배/절반으로 헷갈릴 수 있다 (록 162 를 81 로 읽는 등). 박 사이 소리의
        # 세기로 가려 보려 했으나, 두 배가 맞는 곡과 아닌 곡이 같은 값을 보여서 (0.43 과
        # 0.38) 가릴 수 없었다. 그래서 고치지 않고 후보를 함께 적는다.
    analysis.bpm = float(bpm)
    analysis.bpm_confidence = confidence
    analysis.bpm_alternatives = [round(b, 1) for b in (bpm / 2, bpm * 2) if 50 <= b <= 220]
    analysis.beats_per_bar = 4

    # ---- 1박 찾기: 저음이 가장 세게 들어오는 박 ----
    low = (freqs >= 30) & (freqs <= 150)
    low_onset = np.maximum(0.0, np.diff(np.log1p(1000 * spectrum[:, low].sum(axis=1)), prepend=0.0))
    phase_scores = [float(low_onset[beats[phase::4]].sum()) for phase in range(4)]
    phase = int(np.argmax(phase_scores))
    downbeats = beats[phase::4]
    if len(downbeats) < 2:
        raise MusicAnalysisError("박을 찾지 못했습니다. 너무 짧거나 리듬이 없는 곡입니다.")

    # ---- 크로마, 조성 ----
    frame_chroma = chroma(spectrum, freqs)
    bass_chroma = chroma(spectrum, freqs, 40.0, 250.0)
    total = frame_chroma.sum(axis=0)
    tonic, minor, r = key_from_chroma(total)
    key = Key(Pitch.from_midi(60 + tonic, prefer_flat=tonic in (1, 3, 5, 8, 10)),
              "natural_minor" if minor else "major")
    analysis.key = key.display_name
    analysis.key_tonic_pc, analysis.key_minor, analysis.key_confidence = tonic, minor, r

    # ---- 코드: 반 마디 (2박) 마다 ----
    # (조성은 코드를 구한 뒤에 한 번 더 본다. 나란한조(A 단조와 C 장조)는 음 분포가
    # 거의 같아서 분포만으로는 자주 헷갈린다. 으뜸화음이 실제로 얼마나 나오는지로 가린다.)
    chords: list[tuple[float, str]] = []
    bar_chroma = []
    bar_timbre = []
    bar_level = []
    timbre = _mel_bands(spectrum, freqs)
    level = 10 * np.log10(np.maximum((spectrum ** 2).sum(axis=1), 1e-12))
    for bar_index, (a, b) in enumerate(zip(downbeats, downbeats[1:])):
        middle = (a + b) // 2
        for half, (x, y) in enumerate(((a, middle), (middle, b))):
            if y <= x:
                continue
            weights = frame_chroma[x:y].mean(axis=0)
            weights = np.maximum(0.0, weights - np.median(weights))   # 고르게 깔린 소리를 뺀다
            bass = int(np.argmax(bass_chroma[x:y].mean(axis=0)))
            name, _ = best_chord(weights, bass)
            beat = bar_index * 4 + half * 2
            if name != "N" and (not chords or chords[-1][1] != name):
                chords.append((float(beat), name))
        bar_chroma.append(frame_chroma[a:b].mean(axis=0))
        bar_timbre.append(timbre[a:b].mean(axis=0))
        bar_level.append(float(np.mean(level[a:b])))
    analysis.chords = chords
    names = [c for _, c in chords]
    bars = len(bar_chroma)

    if names:
        analysis.seventh_ratio = sum(1 for c in names
                                     if (chord_roots_and_kinds(c) or (0, ""))[1] in SEVENTH_KINDS) / len(names)
        analysis.borrowed_ratio = 1.0 - diatonic_ratio(names, tonic, minor)
        analysis.harmonic_rhythm = len(chords) / max(1, bars)
        numerals = [roman_of(c, tonic, minor) for c in names]
        counts: dict[str, int] = {}
        for i in range(len(numerals) - 3):
            pattern = "-".join(numerals[i:i + 4])
            counts[pattern] = counts.get(pattern, 0) + 1
        analysis.progressions = sorted(counts.items(), key=lambda p: -p[1])[:5]

    # ---- 에너지와 구조 ----
    levels = np.array(bar_level)
    if levels.size:
        energy = (levels - levels.min()) / ((levels.max() - levels.min()) or 1.0)
    else:
        energy = np.zeros(0)
    analysis.energy_curve = [round(float(e), 3) for e in energy]
    if bars >= 4:
        chroma_features = np.array(bar_chroma)
        chroma_features = chroma_features / (np.linalg.norm(chroma_features, axis=1, keepdims=True) + 1e-12)
        timbre_features = np.array(bar_timbre)
        timbre_features = timbre_features - timbre_features.mean(axis=0)
        timbre_features = timbre_features / (np.linalg.norm(timbre_features, axis=1, keepdims=True) + 1e-12)
        features = np.hstack([chroma_features, 0.7 * timbre_features, 0.5 * energy[:, None]])
        analysis.sections = detect_sections(features, energy)
        for section in analysis.sections:
            frame = downbeats[min(len(downbeats) - 1, section.start_bar - 1)]
            section.start_seconds = float(frame * HOP / RATE)

    # ---- 조성 다시 보기: 코드와 구간을 알게 된 뒤 ----
    if names:
        anchors = [_chord_at_beat(chords, (s.start_bar - 1) * 4) for s in analysis.sections]
        anchors = [a for a in anchors if a] + [names[-1]]
        tonic, minor, r = refine_key(total, names, anchors)
        key = Key(Pitch.from_midi(60 + tonic, prefer_flat=tonic in (1, 3, 5, 8, 10)),
                  "natural_minor" if minor else "major")
        analysis.key = key.display_name
        analysis.key_tonic_pc, analysis.key_minor, analysis.key_confidence = tonic, minor, r
        if analysis.chords:
            analysis.borrowed_ratio = 1.0 - diatonic_ratio(names, tonic, minor)
            numerals = [roman_of(c, tonic, minor) for c in names]
            counts = {}
            for i in range(len(numerals) - 3):
                pattern = "-".join(numerals[i:i + 4])
                counts[pattern] = counts.get(pattern, 0) + 1
            analysis.progressions = sorted(counts.items(), key=lambda p: -p[1])[:5]

    # ---- 리듬 ----
    peaks, _ = scipy_signal.find_peaks(envelope, height=0.1, distance=3)
    if len(peaks) and len(beats) >= 2:
        period = float(np.median(np.diff(beats)))
        positions = []
        for p in peaks:
            previous = beats[beats <= p]
            if previous.size == 0:
                continue
            positions.append(((p - previous[-1]) / period) % 1.0)
        positions = np.array(positions)
        on_beat = (positions < 0.12) | (positions > 0.88)
        analysis.syncopation = float(np.mean(~on_beat)) if positions.size else None
        back = positions[(positions > 0.4) & (positions < 0.8)]
        if back.size >= 8:
            analysis.swing = float(np.clip((np.median(back) - 0.5) / (0.667 - 0.5) * 0.66, 0.0, 0.66))
        else:
            analysis.swing = 0.0
        analysis.drum_density = float(min(1.0, len(peaks) / max(1, bars * 16)))

    analysis.genre_guess = _genre_guess(analysis)
    reason = "섞인 소리에서 알아내려면 음원 분리·음성 인식 모델이 필요한데, 이 프로그램에는 없습니다."
    analysis.unavailable = {
        "악기": reason + " MIDI 나 프로젝트 파일로 넣으면 잽니다.",
        "가사": reason + " 가사가 있는 MIDI 로 넣으면 읽습니다.",
        "보컬 표현": reason,
        "멜로디": "섞인 소리에서 멜로디 음을 믿을 만하게 뽑지 못합니다. MIDI 로 넣으면 잽니다.",
        "편곡 밀도": "악기를 구분하지 못해 잴 수 없습니다.",
    }
    return analysis
