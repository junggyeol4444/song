"""
장르 — 4번.

장르는 이름표가 아니다. 곡을 어떻게 만들지를 실제로 정하는 값들의 묶음이다.
BPM 범위, 어울리는 조성, 코드 진행 성향, 악기 구성, 리듬, 편곡 밀도, 믹스 성향.

그래서 여기서는 장르마다 그 값들을 실제로 들고 있다. 이름만 두고 나중에
"록이면 기타를 세게" 같은 규칙을 코드 여기저기 if 문으로 흩뿌리면, 장르를
하나 추가할 때마다 열 군데를 고쳐야 한다.

혼합도 같은 방식으로 처리한다. Rock 50% + Orchestral 50% 는 두 장르의 값을
가중 평균한 새 장르다. 특별 취급하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from .theory import Key, Pitch, TheoryError


class GenreError(ValueError):
    """장르 관련 오류."""


@dataclass(frozen=True, slots=True)
class GenreProfile:
    """장르 하나가 실제로 정하는 값들.

    모든 값은 숫자이거나 가중치 사전이다. 그래야 섞을 수 있다.
    """

    name: str
    display_name: str

    # 템포
    bpm_low: float
    bpm_high: float
    bpm_typical: float

    # 조성 성향. 0 = 항상 장조, 1 = 항상 단조
    minor_tendency: float = 0.4

    # 화성
    seventh_chords: float = 0.3      # 7화음을 얼마나 쓰는가 (0~1)
    tension_notes: float = 0.15      # 9/11/13 텐션 (0~1)
    borrowed_chords: float = 0.15    # 조성 밖 화음 (0~1)
    harmonic_rhythm: float = 1.0     # 마디당 코드 개수

    # 리듬
    swing: float = 0.0               # 0 = 정박, 0.66 = 완전 스윙
    syncopation: float = 0.3         # 엇박 정도 (0~1)
    drum_density: float = 0.5        # 드럼이 촘촘한 정도 (0~1)

    # 편곡
    arrangement_density: float = 0.5  # 동시에 울리는 악기 수 성향 (0~1)
    instrument_weights: Mapping[str, float] = field(default_factory=dict)

    # 믹스 성향
    brightness: float = 0.0          # -1 어두움 ~ +1 밝음
    compression: float = 0.4         # 0 = 다이내믹 유지 ~ 1 = 강하게 눌림
    target_lufs: float = -14.0
    stereo_width: float = 1.0
    low_end: float = 0.0             # -1 가벼움 ~ +1 묵직함

    # 보컬 성향
    vocal_vibrato: float = 0.4
    vocal_breathiness: float = 0.3
    vocal_power: float = 0.5
    harmony_layers: float = 1.5      # 보컬 화음을 몇 겹 쌓는가

    def __post_init__(self) -> None:
        if not (0 < self.bpm_low <= self.bpm_typical <= self.bpm_high):
            raise GenreError(
                f"{self.name}: BPM 범위가 잘못됐습니다 "
                f"({self.bpm_low} <= {self.bpm_typical} <= {self.bpm_high} 이어야 함)"
            )
        for label, value in (
            ("minor_tendency", self.minor_tendency), ("seventh_chords", self.seventh_chords),
            ("tension_notes", self.tension_notes), ("borrowed_chords", self.borrowed_chords),
            ("swing", self.swing), ("syncopation", self.syncopation),
            ("drum_density", self.drum_density), ("arrangement_density", self.arrangement_density),
            ("compression", self.compression), ("vocal_vibrato", self.vocal_vibrato),
            ("vocal_breathiness", self.vocal_breathiness), ("vocal_power", self.vocal_power),
        ):
            if not (0.0 <= value <= 1.0):
                raise GenreError(f"{self.name}: {label} 은 0~1 이어야 합니다 ({value})")
        for label, value in (("brightness", self.brightness), ("low_end", self.low_end)):
            if not (-1.0 <= value <= 1.0):
                raise GenreError(f"{self.name}: {label} 은 -1~1 이어야 합니다 ({value})")

    def instrument_weight(self, instrument: str) -> float:
        return self.instrument_weights.get(instrument, 0.0)

    def top_instruments(self, count: int = 6) -> list[tuple[str, float]]:
        return sorted(self.instrument_weights.items(), key=lambda p: p[1], reverse=True)[:count]

    def describe(self) -> str:
        lines = [
            f"{self.display_name} ({self.name})",
            f"  BPM {self.bpm_low:.0f}~{self.bpm_high:.0f} (보통 {self.bpm_typical:.0f})",
            f"  단조 성향 {self.minor_tendency:.0%} / 7화음 {self.seventh_chords:.0%} "
            f"/ 텐션 {self.tension_notes:.0%}",
            f"  스윙 {self.swing:.2f} / 엇박 {self.syncopation:.0%} "
            f"/ 드럼 밀도 {self.drum_density:.0%}",
            f"  편곡 밀도 {self.arrangement_density:.0%} / 밝기 {self.brightness:+.2f} "
            f"/ 압축 {self.compression:.0%} / 목표 {self.target_lufs:.0f} LUFS",
        ]
        if self.instrument_weights:
            top = ", ".join(f"{n} {w:.0%}" for n, w in self.top_instruments())
            lines.append(f"  주요 악기: {top}")
        return "\n".join(lines)


def _profile(name: str, display: str, bpm: tuple[float, float, float], **kwargs) -> GenreProfile:
    return GenreProfile(name, display, bpm[0], bpm[1], bpm[2], **kwargs)


# 3번 문서에 나열된 장르들. 값은 각 장르의 일반적인 특징에서 왔다.
GENRES: dict[str, GenreProfile] = {
    "kpop": _profile("kpop", "K-Pop", (90.0, 135.0, 112.0),
        minor_tendency=0.55, seventh_chords=0.45, tension_notes=0.3, borrowed_chords=0.3,
        harmonic_rhythm=1.6, syncopation=0.55, drum_density=0.7, arrangement_density=0.8,
        brightness=0.35, compression=0.7, target_lufs=-9.0, stereo_width=1.15, low_end=0.4,
        vocal_vibrato=0.35, vocal_power=0.7, harmony_layers=3.0,
        instrument_weights={"drum_kit": 1.0, "synth_bass": 0.85, "synth_lead": 0.7,
                            "synth_pad": 0.7, "electric_piano": 0.5, "strings": 0.45,
                            "electric_guitar": 0.4}),
    "ballad": _profile("ballad", "발라드", (56.0, 82.0, 68.0),
        minor_tendency=0.6, seventh_chords=0.65, tension_notes=0.35, borrowed_chords=0.25,
        harmonic_rhythm=1.2, syncopation=0.15, drum_density=0.3, arrangement_density=0.5,
        brightness=-0.1, compression=0.35, target_lufs=-13.0, stereo_width=1.0, low_end=0.1,
        vocal_vibrato=0.7, vocal_breathiness=0.45, vocal_power=0.6, harmony_layers=2.0,
        instrument_weights={"acoustic_piano": 1.0, "strings": 0.8, "electric_bass": 0.6,
                            "drum_kit": 0.45, "acoustic_guitar": 0.5, "synth_pad": 0.4}),
    "rock": _profile("rock", "록", (100.0, 165.0, 128.0),
        minor_tendency=0.5, seventh_chords=0.15, tension_notes=0.08, borrowed_chords=0.35,
        harmonic_rhythm=1.0, syncopation=0.4, drum_density=0.75, arrangement_density=0.65,
        brightness=0.25, compression=0.6, target_lufs=-10.0, stereo_width=1.1, low_end=0.35,
        vocal_vibrato=0.35, vocal_breathiness=0.2, vocal_power=0.85, harmony_layers=1.5,
        instrument_weights={"electric_guitar": 1.0, "drum_kit": 0.95, "electric_bass": 0.9,
                            "acoustic_piano": 0.25, "organ": 0.2}),
    "pop": _profile("pop", "팝", (92.0, 130.0, 110.0),
        minor_tendency=0.4, seventh_chords=0.3, tension_notes=0.18, borrowed_chords=0.2,
        harmonic_rhythm=1.2, syncopation=0.45, drum_density=0.6, arrangement_density=0.65,
        brightness=0.3, compression=0.65, target_lufs=-10.0, low_end=0.3,
        vocal_power=0.65, harmony_layers=2.5,
        instrument_weights={"drum_kit": 0.9, "synth_bass": 0.7, "acoustic_piano": 0.6,
                            "synth_pad": 0.6, "electric_guitar": 0.5, "synth_lead": 0.5}),
    "rnb": _profile("rnb", "R&B", (60.0, 100.0, 78.0),
        minor_tendency=0.55, seventh_chords=0.85, tension_notes=0.65, borrowed_chords=0.35,
        harmonic_rhythm=1.4, swing=0.15, syncopation=0.7, drum_density=0.55,
        arrangement_density=0.5, brightness=-0.05, compression=0.5, target_lufs=-12.0,
        low_end=0.5, vocal_vibrato=0.5, vocal_breathiness=0.55, harmony_layers=3.5,
        instrument_weights={"electric_piano": 0.9, "electric_bass": 0.85, "drum_kit": 0.8,
                            "synth_pad": 0.55, "electric_guitar": 0.4, "strings": 0.3}),
    "hiphop": _profile("hiphop", "힙합", (70.0, 100.0, 88.0),
        minor_tendency=0.7, seventh_chords=0.5, tension_notes=0.35, borrowed_chords=0.3,
        harmonic_rhythm=0.7, syncopation=0.75, drum_density=0.65, arrangement_density=0.35,
        brightness=-0.15, compression=0.65, target_lufs=-9.0, low_end=0.85,
        vocal_vibrato=0.1, vocal_power=0.6, harmony_layers=1.0,
        instrument_weights={"drum_kit": 1.0, "sub_bass": 0.95, "electric_piano": 0.5,
                            "synth_pad": 0.45, "bell": 0.3}),
    "edm": _profile("edm", "EDM", (124.0, 132.0, 128.0),
        minor_tendency=0.65, seventh_chords=0.25, tension_notes=0.2, borrowed_chords=0.2,
        harmonic_rhythm=0.8, syncopation=0.4, drum_density=0.85, arrangement_density=0.7,
        brightness=0.5, compression=0.8, target_lufs=-8.0, stereo_width=1.3, low_end=0.7,
        vocal_vibrato=0.25, vocal_power=0.7, harmony_layers=2.0,
        instrument_weights={"synth_lead": 1.0, "drum_kit": 0.95, "synth_bass": 0.9,
                            "synth_pad": 0.8}),
    "house": _profile("house", "하우스", (118.0, 128.0, 124.0),
        minor_tendency=0.5, seventh_chords=0.6, tension_notes=0.4, harmonic_rhythm=0.8,
        syncopation=0.5, drum_density=0.8, arrangement_density=0.6, brightness=0.3,
        compression=0.7, target_lufs=-9.0, stereo_width=1.2, low_end=0.6,
        instrument_weights={"drum_kit": 1.0, "synth_bass": 0.9, "electric_piano": 0.6,
                            "synth_pad": 0.6, "organ": 0.3}),
    "techno": _profile("techno", "테크노", (125.0, 145.0, 132.0),
        minor_tendency=0.75, seventh_chords=0.1, tension_notes=0.1, harmonic_rhythm=0.4,
        syncopation=0.35, drum_density=0.9, arrangement_density=0.5, brightness=0.1,
        compression=0.75, target_lufs=-8.0, stereo_width=1.25, low_end=0.7,
        vocal_power=0.3, harmony_layers=1.0,
        instrument_weights={"drum_kit": 1.0, "synth_bass": 0.95, "synth_lead": 0.6,
                            "synth_pad": 0.5}),
    "metal": _profile("metal", "메탈", (130.0, 200.0, 160.0),
        minor_tendency=0.9, seventh_chords=0.1, tension_notes=0.05, borrowed_chords=0.45,
        harmonic_rhythm=1.2, syncopation=0.5, drum_density=0.95, arrangement_density=0.7,
        brightness=0.35, compression=0.7, target_lufs=-9.0, low_end=0.5,
        vocal_vibrato=0.25, vocal_power=1.0, harmony_layers=1.2,
        instrument_weights={"electric_guitar": 1.0, "drum_kit": 1.0, "electric_bass": 0.9,
                            "strings": 0.2}),
    "punk": _profile("punk", "펑크", (150.0, 200.0, 175.0),
        minor_tendency=0.4, seventh_chords=0.05, tension_notes=0.02, borrowed_chords=0.25,
        harmonic_rhythm=1.0, syncopation=0.25, drum_density=0.85, arrangement_density=0.5,
        brightness=0.4, compression=0.7, target_lufs=-9.0, low_end=0.3,
        vocal_vibrato=0.1, vocal_power=0.95, harmony_layers=1.2,
        instrument_weights={"electric_guitar": 1.0, "drum_kit": 0.95, "electric_bass": 0.9}),
    "jazz": _profile("jazz", "재즈", (80.0, 220.0, 130.0),
        minor_tendency=0.45, seventh_chords=0.95, tension_notes=0.85, borrowed_chords=0.6,
        harmonic_rhythm=2.2, swing=0.62, syncopation=0.75, drum_density=0.5,
        arrangement_density=0.5, brightness=0.0, compression=0.2, target_lufs=-16.0,
        low_end=0.1, vocal_vibrato=0.55, vocal_breathiness=0.5, harmony_layers=1.5,
        instrument_weights={"acoustic_piano": 1.0, "electric_bass": 0.85, "drum_kit": 0.8,
                            "brass": 0.6, "flute": 0.3, "acoustic_guitar": 0.35}),
    "citypop": _profile("citypop", "시티팝", (95.0, 120.0, 108.0),
        minor_tendency=0.35, seventh_chords=0.85, tension_notes=0.6, borrowed_chords=0.4,
        harmonic_rhythm=1.6, syncopation=0.55, drum_density=0.6, arrangement_density=0.7,
        brightness=0.3, compression=0.4, target_lufs=-13.0, stereo_width=1.15, low_end=0.35,
        vocal_vibrato=0.4, vocal_breathiness=0.4, harmony_layers=2.5,
        instrument_weights={"electric_piano": 0.9, "electric_bass": 0.9, "drum_kit": 0.75,
                            "electric_guitar": 0.7, "synth_pad": 0.6, "brass": 0.45,
                            "strings": 0.4}),
    "jpop": _profile("jpop", "J-Pop", (110.0, 175.0, 140.0),
        minor_tendency=0.5, seventh_chords=0.6, tension_notes=0.4, borrowed_chords=0.45,
        harmonic_rhythm=2.0, syncopation=0.5, drum_density=0.75, arrangement_density=0.85,
        brightness=0.4, compression=0.7, target_lufs=-9.0, low_end=0.3,
        vocal_vibrato=0.45, vocal_power=0.75, harmony_layers=2.5,
        instrument_weights={"drum_kit": 0.95, "electric_guitar": 0.8, "acoustic_piano": 0.75,
                            "electric_bass": 0.85, "strings": 0.6, "synth_lead": 0.5}),
    "anime": _profile("anime", "애니송", (130.0, 190.0, 158.0),
        minor_tendency=0.5, seventh_chords=0.55, tension_notes=0.4, borrowed_chords=0.5,
        harmonic_rhythm=2.2, syncopation=0.55, drum_density=0.8, arrangement_density=0.9,
        brightness=0.45, compression=0.72, target_lufs=-8.5, stereo_width=1.15, low_end=0.3,
        vocal_vibrato=0.45, vocal_power=0.8, harmony_layers=2.5,
        instrument_weights={"drum_kit": 0.95, "electric_guitar": 0.9, "strings": 0.75,
                            "acoustic_piano": 0.7, "electric_bass": 0.85, "synth_lead": 0.6,
                            "brass": 0.4}),
    "vocalsynth": _profile("vocalsynth", "보컬 신스", (120.0, 200.0, 155.0),
        minor_tendency=0.55, seventh_chords=0.5, tension_notes=0.35, borrowed_chords=0.4,
        harmonic_rhythm=2.4, syncopation=0.6, drum_density=0.8, arrangement_density=0.85,
        brightness=0.5, compression=0.7, target_lufs=-9.0, stereo_width=1.2, low_end=0.3,
        vocal_vibrato=0.3, vocal_breathiness=0.15, vocal_power=0.7, harmony_layers=3.0,
        instrument_weights={"synth_lead": 0.9, "drum_kit": 0.9, "electric_guitar": 0.7,
                            "acoustic_piano": 0.7, "synth_bass": 0.8, "synth_pad": 0.6}),
    "orchestral": _profile("orchestral", "오케스트라", (60.0, 140.0, 92.0),
        minor_tendency=0.55, seventh_chords=0.45, tension_notes=0.3, borrowed_chords=0.4,
        harmonic_rhythm=1.4, syncopation=0.2, drum_density=0.25, arrangement_density=0.95,
        brightness=-0.05, compression=0.2, target_lufs=-18.0, stereo_width=1.25, low_end=0.2,
        vocal_vibrato=0.65, vocal_power=0.7, harmony_layers=3.0,
        instrument_weights={"strings": 1.0, "brass": 0.8, "flute": 0.6, "choir": 0.5,
                            "acoustic_piano": 0.4, "bell": 0.25}),
    "cinematic": _profile("cinematic", "시네마틱", (60.0, 120.0, 84.0),
        minor_tendency=0.7, seventh_chords=0.35, tension_notes=0.25, borrowed_chords=0.45,
        harmonic_rhythm=0.8, syncopation=0.2, drum_density=0.35, arrangement_density=0.8,
        brightness=-0.15, compression=0.3, target_lufs=-16.0, stereo_width=1.3, low_end=0.5,
        vocal_power=0.5, harmony_layers=2.5,
        instrument_weights={"strings": 0.95, "synth_pad": 0.8, "brass": 0.7, "choir": 0.6,
                            "acoustic_piano": 0.5, "drum_kit": 0.4}),
    "acoustic": _profile("acoustic", "어쿠스틱", (70.0, 120.0, 92.0),
        minor_tendency=0.4, seventh_chords=0.45, tension_notes=0.2, borrowed_chords=0.15,
        harmonic_rhythm=1.2, syncopation=0.3, drum_density=0.3, arrangement_density=0.35,
        brightness=0.1, compression=0.25, target_lufs=-15.0, low_end=-0.1,
        vocal_vibrato=0.4, vocal_breathiness=0.5, vocal_power=0.45, harmony_layers=1.8,
        instrument_weights={"acoustic_guitar": 1.0, "acoustic_piano": 0.6,
                            "electric_bass": 0.45, "drum_kit": 0.35, "strings": 0.3,
                            "flute": 0.25}),
    "lofi": _profile("lofi", "로파이", (65.0, 90.0, 78.0),
        minor_tendency=0.6, seventh_chords=0.85, tension_notes=0.6, borrowed_chords=0.3,
        harmonic_rhythm=1.0, swing=0.25, syncopation=0.55, drum_density=0.45,
        arrangement_density=0.3, brightness=-0.45, compression=0.4, target_lufs=-15.0,
        low_end=0.35, vocal_breathiness=0.6, vocal_power=0.3, harmony_layers=1.2,
        instrument_weights={"electric_piano": 0.9, "drum_kit": 0.7, "electric_bass": 0.7,
                            "acoustic_guitar": 0.35, "synth_pad": 0.4}),
    "synthwave": _profile("synthwave", "신스웨이브", (80.0, 118.0, 100.0),
        minor_tendency=0.75, seventh_chords=0.35, tension_notes=0.2, borrowed_chords=0.35,
        harmonic_rhythm=0.9, syncopation=0.3, drum_density=0.6, arrangement_density=0.6,
        brightness=0.25, compression=0.55, target_lufs=-11.0, stereo_width=1.3, low_end=0.5,
        vocal_vibrato=0.3, vocal_power=0.55, harmony_layers=2.0,
        instrument_weights={"synth_lead": 1.0, "synth_pad": 0.9, "synth_bass": 0.9,
                            "drum_kit": 0.75, "electric_guitar": 0.35}),
    "ambient": _profile("ambient", "앰비언트", (50.0, 90.0, 66.0),
        minor_tendency=0.5, seventh_chords=0.6, tension_notes=0.45, harmonic_rhythm=0.3,
        syncopation=0.1, drum_density=0.05, arrangement_density=0.3, brightness=-0.1,
        compression=0.15, target_lufs=-20.0, stereo_width=1.35, low_end=0.2,
        vocal_breathiness=0.7, vocal_power=0.25, harmony_layers=2.0,
        instrument_weights={"synth_pad": 1.0, "strings": 0.5, "bell": 0.35,
                            "acoustic_piano": 0.35, "choir": 0.4}),
}


def get_genre(name: str) -> GenreProfile:
    key = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "k-pop": "kpop", "kpop": "kpop", "케이팝": "kpop",
        "발라드": "ballad", "록": "rock", "락": "rock", "팝": "pop",
        "알앤비": "rnb", "randb": "rnb", "r&b": "rnb",
        "힙합": "hiphop", "hiphop": "hiphop", "랩": "hiphop",
        "일렉": "edm", "일렉트로닉": "edm", "재즈": "jazz",
        "시티팝": "citypop", "j-pop": "jpop", "제이팝": "jpop",
        "animesong": "anime", "애니": "anime", "애니메이션": "anime", "애니송": "anime",
        "vocaloid": "vocalsynth", "보컬로이드": "vocalsynth", "보컬신스": "vocalsynth",
        "오케스트라": "orchestral", "orchestra": "orchestral", "클래식": "orchestral",
        "시네마틱": "cinematic", "영화음악": "cinematic",
        "어쿠스틱": "acoustic", "로파이": "lofi", "lo-fi": "lofi",
        "신스웨이브": "synthwave", "메탈": "metal", "펑크": "punk",
        "하우스": "house", "테크노": "techno", "앰비언트": "ambient",
    }
    normalized = aliases.get(name.strip().lower(), aliases.get(key, key))
    if normalized not in GENRES and normalized.startswith("mystyle"):
        # 내 스타일(8번)은 사용자 폴더에 있다. 저장된 곡을 열 때 여기서 불러온다.
        from ..learning.library import load_style_genre
        load_style_genre(normalized)
    if normalized not in GENRES:
        raise GenreError(
            f"모르는 장르입니다: {name!r}\n"
            f"사용 가능: {', '.join(sorted(GENRES))}"
        )
    return GENRES[normalized]


USER_GENRE_PREFIXES = ("mystyle", "reference")


def available_genres() -> list[str]:
    """기본 장르 (3번 문서의 22개). 사용자가 만든 스타일은 user_genres()."""
    return sorted(n for n in GENRES if not n.startswith(USER_GENRE_PREFIXES))


def user_genres() -> list[str]:
    """학습실에서 만든 '내 스타일' 장르들 (8번). Reference 임시 장르는 빼고."""
    return sorted((n for n in GENRES if n.startswith("mystyle")),
                  key=lambda n: GENRES[n].display_name)


@dataclass(frozen=True, slots=True)
class GenreBlend:
    """여러 장르를 비율로 섞은 것. 4번의 'Rock 50% / K-Pop 30% / Orchestral 20%'.

    비율 합이 1이 아니면 자동으로 맞춘다. 사용자가 50/30/20 을 넣든
    5/3/2 를 넣든 같은 결과가 나와야 하기 때문이다.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise GenreError("장르가 하나도 없습니다.")
        for name, weight in self.weights.items():
            get_genre(name)      # 이름 검증
            if weight < 0:
                raise GenreError(f"{name}: 비율은 0 이상이어야 합니다 ({weight})")
        if sum(self.weights.values()) <= 0:
            raise GenreError("비율의 합이 0입니다.")

    @classmethod
    def single(cls, name: str) -> "GenreBlend":
        return cls({get_genre(name).name: 1.0})

    @classmethod
    def parse(cls, text: str) -> "GenreBlend":
        """'Rock 50% K-Pop 30% Orchestral 20%' 또는 'Rock' 을 읽는다."""
        import re

        parts = re.findall(r"([A-Za-z가-힣&\-]+)\s*(\d+(?:\.\d+)?)?\s*%?", text)
        weights: dict[str, float] = {}
        for name, value in parts:
            if not name.strip():
                continue
            try:
                profile = get_genre(name)
            except GenreError:
                continue
            weights[profile.name] = float(value) if value else 1.0
        if not weights:
            raise GenreError(f"장르를 찾을 수 없습니다: {text!r}")
        return cls(weights)

    @property
    def normalized(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {get_genre(k).name: v / total for k, v in self.weights.items()}

    @property
    def primary(self) -> GenreProfile:
        """비율이 가장 큰 장르. 이름표가 필요할 때 쓴다."""
        name = max(self.normalized.items(), key=lambda p: p[1])[0]
        return GENRES[name]

    def resolve(self) -> GenreProfile:
        """섞인 결과를 하나의 장르 프로필로 만든다.

        숫자는 가중 평균, 악기 가중치는 항목별 가중 평균이다.
        BPM 은 '보통값'끼리 평균내고, 범위는 각 장르 범위의 가중 평균으로 잡는다.
        """
        parts = self.normalized
        if len(parts) == 1:
            return GENRES[next(iter(parts))]

        profiles = [(GENRES[name], weight) for name, weight in parts.items()]

        def average(attribute: str) -> float:
            return sum(getattr(p, attribute) * w for p, w in profiles)

        instruments: dict[str, float] = {}
        for profile, weight in profiles:
            for name, value in profile.instrument_weights.items():
                instruments[name] = instruments.get(name, 0.0) + value * weight

        label = " + ".join(
            f"{GENRES[n].display_name} {w:.0%}"
            for n, w in sorted(parts.items(), key=lambda p: p[1], reverse=True)
        )
        return GenreProfile(
            name="+".join(sorted(parts)),
            display_name=label,
            bpm_low=average("bpm_low"),
            bpm_high=average("bpm_high"),
            bpm_typical=average("bpm_typical"),
            minor_tendency=average("minor_tendency"),
            seventh_chords=average("seventh_chords"),
            tension_notes=average("tension_notes"),
            borrowed_chords=average("borrowed_chords"),
            harmonic_rhythm=average("harmonic_rhythm"),
            swing=average("swing"),
            syncopation=average("syncopation"),
            drum_density=average("drum_density"),
            arrangement_density=average("arrangement_density"),
            instrument_weights=instruments,
            brightness=average("brightness"),
            compression=average("compression"),
            target_lufs=average("target_lufs"),
            stereo_width=average("stereo_width"),
            low_end=average("low_end"),
            vocal_vibrato=average("vocal_vibrato"),
            vocal_breathiness=average("vocal_breathiness"),
            vocal_power=average("vocal_power"),
            harmony_layers=average("harmony_layers"),
        )

    def to_dict(self) -> dict:
        return {"weights": dict(self.weights)}

    @classmethod
    def from_dict(cls, data: dict) -> "GenreBlend":
        return cls(dict(data["weights"]))

    def __str__(self) -> str:
        return " / ".join(
            f"{GENRES[n].display_name} {w:.0%}"
            for n, w in sorted(self.normalized.items(), key=lambda p: p[1], reverse=True)
        )
