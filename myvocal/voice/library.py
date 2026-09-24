"""
목소리 라이브러리 — 내 목소리들을 보관하는 곳.

    %APPDATA%\\MYVOCAL Studio\\voices\\<번호>\\
        meta.json       이름, 자료 권한, 처음 고른 성종
        takes\\001.wav   녹음
        takes\\001.json  그때 부르라고 한 것 (가이드), 분석 결과
        voice.json      만들어진 목소리 모델

녹음은 사용자 폴더에만 둔다. 프로젝트 파일에는 목소리 번호만 적는다 (59번).
프로젝트를 남에게 줘도 내 목소리 녹음은 따라가지 않는다.

63번의 자료 권한: 목소리 모델은 '내 목소리' 또는 '사용 허가를 받은 목소리' 로만
만든다. 'Reference 분석용' 자료로는 목소리 모델을 만들지 않는다.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile

from ..providers.settings import settings_folder
from .analysis import GuideNote, NoteMeasure, analyze_take, midi_to_hz
from .model import RIGHTS_CHOICES, Take, VoiceModel, build_voice_model, preset_voices
from .synth import VoiceError
from .training import Coverage, Exercise


def voices_folder() -> Path:
    return settings_folder() / "voices"


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def _measure_to_dict(m: NoteMeasure) -> dict:
    return {"start": m.start, "duration": m.duration, "pitch": m.pitch_hz, "level": m.level_db,
            "target": m.target_hz, "vowel": m.vowel, "voiced": m.voiced_seconds,
            "error": m.error_cents}


def _measure_from_dict(d: dict) -> NoteMeasure:
    return NoteMeasure(d["start"], d["duration"], d["pitch"], d["level"],
                       target_hz=d.get("target"), error_cents=d.get("error"),
                       vowel=d.get("vowel"), voiced_seconds=d.get("voiced", 0.0))


@dataclass(slots=True)
class TakeInfo:
    """저장된 녹음 하나."""

    index: int
    wav: Path
    title: str
    guide: list[GuideNote]
    measures: list[NoteMeasure]
    recorded_at: float

    @property
    def syllables(self) -> list[str]:
        return [g.syllable for g in self.guide]


class VoiceEntry:
    """라이브러리의 목소리 하나 (녹음 중이거나 모델까지 만든 것)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))

    @property
    def voice_id(self) -> str:
        return self.meta["id"]

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def rights(self) -> str:
        return self.meta["rights"]

    @property
    def base_type(self) -> str:
        return self.meta.get("base_type", "alto")

    @property
    def takes_folder(self) -> Path:
        return self.folder / "takes"

    # ---- 녹음 ----

    def add_take(self, signal: np.ndarray, rate: int, exercise: Exercise) -> TakeInfo:
        """녹음을 저장하고 바로 분석해 둔다 (커버리지를 빠르게 보여주려고)."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 1 or signal.size < rate // 2:
            raise VoiceError("녹음이 너무 짧습니다.")
        peak = float(np.max(np.abs(signal)))
        if peak < 1e-3:
            raise VoiceError("녹음에 소리가 거의 없습니다. 마이크를 확인해 주세요.")
        self.takes_folder.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem) for p in self.takes_folder.glob("*.json") if p.stem.isdigit()]
        index = max(existing, default=0) + 1
        wav = self.takes_folder / f"{index:03d}.wav"
        soundfile.write(str(wav), signal, rate, subtype="FLOAT")
        measures = analyze_take(signal.astype(np.float64), rate, exercise.notes)
        _write_json(self.takes_folder / f"{index:03d}.json", {
            "title": exercise.title,
            "guide": [[g.start, g.duration, g.midi, g.syllable] for g in exercise.notes],
            "measures": [_measure_to_dict(m) for m in measures],
            "recorded_at": time.time(),
            "rate": rate,
        })
        return self._take_info(index)

    def _take_info(self, index: int) -> TakeInfo:
        data = json.loads((self.takes_folder / f"{index:03d}.json").read_text(encoding="utf-8"))
        return TakeInfo(
            index=index, wav=self.takes_folder / f"{index:03d}.wav", title=data["title"],
            guide=[GuideNote(*g) for g in data["guide"]],
            measures=[_measure_from_dict(m) for m in data["measures"]],
            recorded_at=data.get("recorded_at", 0.0),
        )

    def takes(self) -> list[TakeInfo]:
        if not self.takes_folder.exists():
            return []
        indices = sorted(int(p.stem) for p in self.takes_folder.glob("*.json") if p.stem.isdigit())
        return [self._take_info(i) for i in indices
                if (self.takes_folder / f"{i:03d}.wav").exists()]

    def delete_take(self, index: int) -> None:
        for suffix in (".wav", ".json"):
            path = self.takes_folder / f"{index:03d}{suffix}"
            if path.exists():
                path.unlink()

    def load_takes(self) -> list[Take]:
        result = []
        for info in self.takes():
            signal, rate = soundfile.read(str(info.wav), dtype="float64", always_2d=False)
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            result.append(Take(signal, rate, info.guide, info.title))
        return result

    # ---- 자료 현황 ----

    def range_midi(self) -> tuple[int, int]:
        """커버리지 구역을 나눌 음역. 모델이 있으면 그 음역, 없으면 성종 기본값."""
        model = self.model()
        if model is not None:
            vocal = model.vocal_range()
        else:
            from ..music.melody import VocalRange
            vocal = VocalRange.typical(self.base_type)
        return vocal.comfortable_low, vocal.comfortable_high

    def coverage(self) -> Coverage:
        low, high = self.range_midi()
        coverage = Coverage(low, high)
        for info in self.takes():
            coverage.add(info.measures, info.syllables)
        return coverage

    # ---- 모델 ----

    def model(self) -> VoiceModel | None:
        path = self.folder / "voice.json"
        if not path.exists():
            return None
        return VoiceModel.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def build_model(self, progress=None) -> VoiceModel:
        takes = self.load_takes()
        if not takes:
            raise VoiceError("녹음이 없습니다. 연습을 하나 이상 녹음해 주세요.")
        model, _ = build_voice_model(self.name, takes, self.rights, progress=progress)
        model.voice_id = self.voice_id
        _write_json(self.folder / "voice.json", model.to_dict())
        return model

    def rename(self, name: str) -> None:
        if not name.strip():
            raise VoiceError("이름이 비어 있습니다.")
        self.meta["name"] = name.strip()
        _write_json(self.folder / "meta.json", self.meta)
        model = self.model()
        if model is not None:
            model.name = name.strip()
            _write_json(self.folder / "voice.json", model.to_dict())


class VoiceLibrary:
    """목소리들."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or voices_folder()

    def entries(self) -> list[VoiceEntry]:
        if not self.folder.exists():
            return []
        result = []
        for path in sorted(self.folder.iterdir()):
            if (path / "meta.json").exists():
                try:
                    result.append(VoiceEntry(path))
                except (OSError, json.JSONDecodeError, KeyError):
                    continue           # 깨진 항목 하나 때문에 전체를 못 쓰면 안 된다
        return sorted(result, key=lambda e: e.meta.get("created", 0))

    def create(self, name: str, rights: str, base_type: str = "alto") -> VoiceEntry:
        if not name.strip():
            raise VoiceError("목소리 이름을 적어 주세요.")
        if rights not in RIGHTS_CHOICES:
            raise VoiceError(
                "목소리 모델은 '내 목소리' 또는 '사용 허가를 받은 목소리' 로만 만들 수 있습니다. "
                "Reference 분석용 자료로는 목소리를 학습하지 않습니다."
            )
        if base_type not in preset_voices():
            raise VoiceError(f"모르는 성종입니다: {base_type}")
        voice_id = uuid.uuid4().hex[:12]
        folder = self.folder / voice_id
        folder.mkdir(parents=True, exist_ok=False)
        _write_json(folder / "meta.json", {
            "id": voice_id, "name": name.strip(), "rights": rights, "base_type": base_type,
            "created": time.time(),
        })
        return VoiceEntry(folder)

    def get(self, voice_id: str) -> VoiceEntry | None:
        folder = self.folder / voice_id
        return VoiceEntry(folder) if (folder / "meta.json").exists() else None

    def delete(self, voice_id: str) -> bool:
        folder = self.folder / voice_id
        if not (folder / "meta.json").exists():
            return False
        shutil.rmtree(folder)
        return True


# ==========================================================================
# 트랙에서 목소리 찾기
# ==========================================================================

def resolve_voice(reference: str, library: VoiceLibrary | None = None) -> VoiceModel:
    """트랙에 적힌 목소리 이름을 모델로.

        'preset:mezzo'  기본 목소리
        'user:<번호>'   라이브러리의 내 목소리
        'mezzo'         예전 프로젝트 (기본 목소리 이름만 적혀 있던 때)
    """
    presets = preset_voices()
    if not reference:
        return presets["mezzo"]
    kind, _, name = reference.partition(":")
    if not name:
        kind, name = "preset", kind
    if kind == "preset":
        if name not in presets:
            raise VoiceError(f"모르는 기본 목소리입니다: {name!r}")
        return presets[name]
    if kind == "user":
        entry = (library or VoiceLibrary()).get(name)
        if entry is None:
            raise VoiceError(
                f"목소리 '{name}' 이 이 컴퓨터의 라이브러리에 없습니다. "
                f"다른 컴퓨터에서 만든 프로젝트라면 그 목소리를 다시 학습해야 합니다.")
        model = entry.model()
        if model is None:
            raise VoiceError(f"목소리 '{entry.name}' 은 아직 모델을 만들지 않았습니다.")
        return model
    raise VoiceError(f"목소리 이름을 읽을 수 없습니다: {reference!r}")
