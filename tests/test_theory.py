"""음악 이론 엔진 검증. 기대값은 전부 음악 이론 교과서 값이다."""

import sys

sys.path.insert(0, ".")

from myvocal.music.theory import (
    Chord, Interval, Key, Pitch, Scale, TheoryError,
    cents_between, detect_key, frequency_to_midi, identify_chord, midi_to_frequency,
)

failures: list[str] = []
checks = 0


def check(label, got, expected, tol=1e-9):
    global checks
    checks += 1
    ok = abs(got - expected) <= tol if isinstance(expected, float) and isinstance(got, (int, float)) else got == expected
    if not ok:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
        print(f"  FAIL {label}: {got!r} (기대 {expected!r})")


def check_raises(label, fn):
    global checks
    checks += 1
    try:
        fn()
    except TheoryError:
        return
    failures.append(f"{label}: TheoryError 가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


print("[1] 음높이 / 주파수  (A4 = 69 = 440Hz 국제 표준)")
check("A4 midi", Pitch.parse("A4").midi, 69)
check("A4 Hz", Pitch.parse("A4").frequency, 440.0)
check("A3 Hz (한 옥타브 아래)", Pitch.parse("A3").frequency, 220.0)
check("A5 Hz", Pitch.parse("A5").frequency, 880.0)
check("C4 midi (가온다)", Pitch.parse("C4").midi, 60)
check("C-1 midi (최저)", Pitch.parse("C-1").midi, 0)
check("G9 midi (최고)", Pitch.parse("G9").midi, 127)
check("C4 Hz", Pitch.parse("C4").frequency, 261.6255653005986, tol=1e-9)
check("E4 Hz (장3도)", Pitch.parse("E4").frequency, 329.6275569128699, tol=1e-9)
check("주파수->midi 왕복", round(frequency_to_midi(440.0)), 69)
check("주파수->midi 소수", frequency_to_midi(466.1637615180899), 70.0, tol=1e-9)
check("반음 = 100센트", cents_between(440.0, 466.1637615180899), 100.0, tol=1e-9)
check("옥타브 = 1200센트", cents_between(220.0, 440.0), 1200.0, tol=1e-9)
check("옥타브 조회", Pitch.parse("C4").octave, 4)
check("옥타브 조회 B3", Pitch.parse("B3").octave, 3)

print("[2] 이명동음 구분")
cs = Pitch.parse("C#4")
db = Pitch.parse("Db4")
check("C#4 와 Db4 는 같은 소리", cs.enharmonic_equals(db), True)
check("하지만 같은 객체는 아님", cs == db, False)
check("C#4 이름", cs.name, "C#")
check("Db4 이름", db.name, "Db")
check("Ebb = D 소리", Pitch.parse("Ebb4").midi, 62)
check("B#3 = C4 소리", Pitch.parse("B#3").midi, 60)
check("Cb4 = B3 소리", Pitch.parse("Cb4").midi, 59)
check_raises("철자-음높이 불일치 거부", lambda: Pitch(60, "D", 0))
check_raises("범위 초과 거부", lambda: Pitch.parse("C10"))
check_raises("잘못된 표기 거부", lambda: Pitch.parse("H4"))

print("[3] 음정 (증4도 != 감5도)")
check("C4-E4 = 장3도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("E4"))), "M3")
check("C4-Eb4 = 단3도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("Eb4"))), "m3")
check("C4-G4 = 완전5도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("G4"))), "P5")
check("C4-F#4 = 증4도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("F#4"))), "A4")
check("C4-Gb4 = 감5도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("Gb4"))), "d5")
check("증4도 반음수", Interval(4, "A").semitones, 6)
check("감5도 반음수", Interval(5, "d").semitones, 6)
check("둘은 같은 반음수, 다른 음정", Interval(4, "A") == Interval(5, "d"), False)
check("C4-C5 = 완전8도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("C5"))), "P8")
check("C4-D5 = 장9도", str(Interval.between(Pitch.parse("C4"), Pitch.parse("D5"))), "M9")
check("장9도 반음수", Interval(9, "M").semitones, 14)
check_raises("3도에 완전 없음", lambda: Interval(3, "P"))
check_raises("5도에 장 없음", lambda: Interval(5, "M"))

print("[4] 음계 철자  (F#장조는 E#, Gb장조는 Cb)")
check("C major", [str(p) for p in Scale(Pitch.parse("C4"), "major").pitches()],
      ["C4", "D4", "E4", "F4", "G4", "A4", "B4"])
check("F# major (E#5 나와야 함)", [p.name for p in Scale(Pitch.parse("F#4"), "major").pitches()],
      ["F#", "G#", "A#", "B", "C#", "D#", "E#"])
check("Gb major (Cb 나와야 함)", [p.name for p in Scale(Pitch.parse("Gb4"), "major").pitches()],
      ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"])
check("A natural minor", [p.name for p in Scale(Pitch.parse("A4"), "natural_minor").pitches()],
      ["A", "B", "C", "D", "E", "F", "G"])
check("A harmonic minor (G# 나와야 함)",
      [p.name for p in Scale(Pitch.parse("A4"), "harmonic_minor").pitches()],
      ["A", "B", "C", "D", "E", "F", "G#"])
check("D dorian", [p.name for p in Scale(Pitch.parse("D4"), "dorian").pitches()],
      ["D", "E", "F", "G", "A", "B", "C"])
check("C 단5음계", [p.name for p in Scale(Pitch.parse("C4"), "minor_pentatonic").pitches()],
      ["C", "Eb", "F", "G", "Bb"])  # 도수 철자
check("C 블루스", len(Scale(Pitch.parse("C4"), "blues").pitches()), 6)
check("히라조시 (J-Pop)", len(Scale(Pitch.parse("D4"), "hirajoshi").pitches()), 5)
check_raises("없는 음계 거부", lambda: Scale(Pitch.parse("C4"), "klingon"))

print("[5] 음계 안 이동 / 보정")
c_major = Scale(Pitch.parse("C4"), "major")
check("C4 에서 2칸 위 = E4", str(c_major.step(Pitch.parse("C4"), 2)), "E4")
check("C4 에서 7칸 위 = C5", str(c_major.step(Pitch.parse("C4"), 7)), "C5")
check("C4 에서 1칸 아래 = B3", str(c_major.step(Pitch.parse("C4"), -1)), "B3")
check("E4 에서 4칸 위 = B4", str(c_major.step(Pitch.parse("E4"), 4)), "B4")
check("C#4 보정 -> C4", str(c_major.snap(Pitch.parse("C#4"))), "C4")
check("음계 포함 판정 E", c_major.contains(Pitch.parse("E5")), True)
check("음계 포함 판정 F#", c_major.contains(Pitch.parse("F#5")), False)
check("도수 조회 G=5", c_major.degree_of(Pitch.parse("G4")), 5)

print("[6] 코드 구성음")
check("C", [str(p) for p in Chord.parse("C").pitches()], ["C4", "E4", "G4"])
check("Am", [str(p) for p in Chord.parse("Am").pitches()], ["A4", "C5", "E5"])
check("G7", [str(p) for p in Chord.parse("G7").pitches()], ["G4", "B4", "D5", "F5"])
check("Cmaj7", [str(p) for p in Chord.parse("Cmaj7").pitches()], ["C4", "E4", "G4", "B4"])
check("Bdim", [str(p) for p in Chord.parse("Bdim").pitches()], ["B4", "D5", "F5"])
check("Bm7b5", [str(p) for p in Chord.parse("Bm7b5").pitches()], ["B4", "D5", "F5", "A5"])
# C°7 의 7음은 감7도다. A(장6도)가 아니라 Bbb(감7도)로 적어야 전조·기보가 안 깨진다.
check("Cdim7", [str(p) for p in Chord.parse("Cdim7").pitches()], ["C4", "Eb4", "Gb4", "Bbb4"])
check("Cdim7 7음은 A 와 같은 소리", Chord.parse("Cdim7").pitches()[3].midi, 69)
check("Caug", [str(p) for p in Chord.parse("Caug").pitches()], ["C4", "E4", "G#4"])
check("Csus4", [str(p) for p in Chord.parse("Csus4").pitches()], ["C4", "F4", "G4"])
check("C5 파워코드", [str(p) for p in Chord.parse("C5").pitches()], ["C4", "G4"])
check("C9", [str(p) for p in Chord.parse("C9").pitches()], ["C4", "E4", "G4", "Bb4", "D5"])
check("C13 = C E G Bb D A (11도 생략형, 6음)", len(Chord.parse("C13").pitches()), 6)
check("C/G 최저음은 G3", str(Chord.parse("C/G").pitches()[0]), "G3")
check("C/E 최저음은 E3", str(Chord.parse("C/E").pitches()[0]), "E3")

print("[6-2] 구성음 철자가 도수를 따르는가 (Cm7 의 7음은 A#이 아니라 Bb)")
check("Cm7", [p.name for p in Chord.parse("Cm7").pitches()], ["C", "Eb", "G", "Bb"])
check("C9", [p.name for p in Chord.parse("C9").pitches()], ["C", "E", "G", "Bb", "D"])
check("C7b9 의 9음은 Db", Chord.parse("C7b9").pitches()[4].name, "Db")
check("C7#9 의 9음은 D#", Chord.parse("C7#9").pitches()[4].name, "D#")
check("C7#11 의 11음은 F#", Chord.parse("C7#11").pitches()[4].name, "F#")
check("C7b13 의 13음은 Ab", Chord.parse("C7b13").pitches()[4].name, "Ab")
check("Cm7b5", [p.name for p in Chord.parse("Cm7b5").pitches()], ["C", "Eb", "Gb", "Bb"])
check("Bb7 의 7음은 Ab", Chord.parse("Bb7").pitches()[3].name, "Ab")
check("Ebm7", [p.name for p in Chord.parse("Ebm7").pitches()], ["Eb", "Gb", "Bb", "Db"])
check("F#7 의 7음은 E", Chord.parse("F#7").pitches()[3].name, "E")
check("C단5음계", [p.name for p in Scale(Pitch.parse("C4"), "minor_pentatonic").pitches()],
      ["C", "Eb", "F", "G", "Bb"])
check("C블루스 (Gb 와 G 공존)", [p.name for p in Scale(Pitch.parse("C4"), "blues").pitches()],
      ["C", "Eb", "F", "Gb", "G", "Bb"])
check("C얼터드 (Fb 나와야 함)", [p.name for p in Scale(Pitch.parse("C4"), "altered").pitches()],
      ["C", "Db", "Eb", "Fb", "Gb", "Ab", "Bb"])
check("C온음음계", [p.name for p in Scale(Pitch.parse("C4"), "whole_tone").pitches()],
      ["C", "D", "E", "F#", "G#", "A#"])

print("[7] 코드 별칭 표기")
check("CM7 = Cmaj7", Chord.parse("CM7").kind, "maj7")
check("C- = Cm", Chord.parse("C-").kind, "m")
check("Cmin7 = Cm7", Chord.parse("Cmin7").kind, "m7")
check("Co7 = Cdim7", Chord.parse("Co7").kind, "dim7")
check("Cø = Cm7b5", Chord.parse("Cø").kind, "m7b5")
check("C+ = Caug", Chord.parse("C+").kind, "aug")
check("Cdom7 = C7", Chord.parse("Cdom7").kind, "7")
check_raises("모르는 접미사 거부", lambda: Chord.parse("Cwtf"))
check_raises("빈 심볼 거부", lambda: Chord.parse(""))

print("[8] 코드 성질")
check("G7 은 도미넌트", Chord.parse("G7").is_dominant, True)
check("Cmaj7 은 도미넌트 아님", Chord.parse("Cmaj7").is_dominant, False)
check("G7 에 트라이톤 있음", Chord.parse("G7").has_tritone, True)
check("C 에 트라이톤 없음", Chord.parse("C").has_tritone, False)
check("Bdim 에 트라이톤 있음", Chord.parse("Bdim").has_tritone, True)
check("C 긴장도 0", Chord.parse("C").tension_level, 0)
check("C7alt 긴장도 최대", Chord.parse("Calt").tension_level, 5)
check("C 와 Am 공통음 2개 (C,E)", Chord.parse("C").common_tones(Chord.parse("Am")), 2)
check("C 와 F# 공통음 0개", Chord.parse("C").common_tones(Chord.parse("F#")), 0)
check("C 와 Em 공통음 2개 (E,G)", Chord.parse("C").common_tones(Chord.parse("Em")), 2)

print("[9] 전위")
check("C 1전위 최저음 E4", str(Chord.parse("C").inverted(1).pitches()[0]), "E4")
check("C 2전위 최저음 G4", str(Chord.parse("C").inverted(2).pitches()[0]), "G4")
check("C 1전위 구성음", [str(p) for p in Chord.parse("C").inverted(1).pitches()], ["E4", "G4", "C5"])
check("G7 3전위 최저음 F5", str(Chord.parse("G7").inverted(3).pitches()[0]), "F5")
check_raises("3화음 3전위 거부", lambda: Chord.parse("C").inverted(3))

print("[10] 이조 (11번 '반키 올려줘')")
check("C +2 = D", Chord.parse("C").transpose(2).symbol, "D")
check("Am +1 = A#m 또는 Bbm", Chord.parse("Am").transpose(1).root.midi, 70)
check("Bb7 +2 = C7", Chord.parse("Bb7").transpose(2).symbol, "C7")
check("Bb7 은 플랫 유지하며 -1 = A7", Chord.parse("Bb7").transpose(-1).symbol, "A7")
check("C/G +5 = F/C", Chord.parse("C/G").transpose(5).symbol, "F/C")
check("전위 유지", Chord.parse("C").inverted(1).transpose(2).inversion, 1)

print("[11] 조성 / 조표")
check("C장조 조표 0", Key.parse("C").accidental_count, 0)
check("G장조 조표 F#", Key.parse("G").key_signature, ["F#"])
check("D장조 조표 2개", Key.parse("D").key_signature, ["F#", "C#"])
check("F장조 조표 Bb", Key.parse("F").key_signature, ["Bb"])
check("Eb장조 조표 3개", Key.parse("Eb").key_signature, ["Bb", "Eb", "Ab"])
check("A단조 조표 0", Key.parse("Am").accidental_count, 0)
check("E단조 조표 F#", Key.parse("Em").key_signature, ["F#"])
check("F#단조 조표 3개", Key.parse("F#m").key_signature, ["F#", "C#", "G#"])
check("C장조 나란한조 = Am", str(Key.parse("C").relative), "Am")
check("Am 나란한조 = C", str(Key.parse("Am").relative), "C")
check("C장조 같은으뜸음조 = Cm", str(Key.parse("C").parallel), "Cm")
check("'F# minor' 파싱", Key.parse("F# minor").is_minor, True)
check("'Bb major' 파싱", Key.parse("Bb major").is_minor, False)

print("[12] 도수 화음")
check("C장조 3화음", [c.symbol for c in Key.parse("C").diatonic_chords()],
      ["C", "Dm", "Em", "F", "G", "Am", "Bdim"])
check("C장조 7화음", [c.symbol for c in Key.parse("C").diatonic_chords(True)],
      ["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5"])
check("A단조 3화음", [c.symbol for c in Key.parse("Am").diatonic_chords()],
      ["Am", "Bdim", "C", "Dm", "Em", "F", "G"])
check("G장조 3화음", [c.symbol for c in Key.parse("G").diatonic_chords()],
      ["G", "Am", "Bm", "C", "D", "Em", "F#dim"])
check("Eb장조 3화음", [c.symbol for c in Key.parse("Eb").diatonic_chords()],
      ["Eb", "Fm", "Gm", "Ab", "Bb", "Cm", "Ddim"])

print("[13] 도수 표기")
c_key = Key.parse("C")
check("C = I", c_key.roman_numeral(Chord.parse("C")), "I")
check("Am = vi", c_key.roman_numeral(Chord.parse("Am")), "vi")
check("G7 = V7", c_key.roman_numeral(Chord.parse("G7")), "V7")
check("F = IV", c_key.roman_numeral(Chord.parse("F")), "IV")
check("Bb = bVII", c_key.roman_numeral(Chord.parse("Bb")), "bVII")
check("역변환 V7", c_key.chord_from_roman("V7").symbol, "G7")
check("역변환 vi", c_key.chord_from_roman("vi").symbol, "Am")
check("역변환 IV", c_key.chord_from_roman("IV").symbol, "F")
check("역변환 bVII", c_key.chord_from_roman("bVII").symbol, "Bb")
check("역변환 bIII", c_key.chord_from_roman("bIII").symbol, "Eb")
check("부속화음 V/V = D7", c_key.secondary_dominant(5).symbol, "D7")
check("부속화음 V/vi = E7", c_key.secondary_dominant(6).symbol, "E7")

print("[14] 코드 역분석")
check("[C,E,G] -> C", identify_chord([60, 64, 67])[0].symbol, "C")
check("[A,C,E] -> Am", identify_chord([57, 60, 64])[0].symbol, "Am")
check("[G,B,D,F] -> G7", identify_chord([55, 59, 62, 65])[0].symbol, "G7")
check("[B,D,F,G] -> G7/B", identify_chord([59, 62, 65, 67])[0].symbol, "G7/B")
check("[E,G,C] -> C/E", identify_chord([52, 55, 60])[0].symbol, "C/E")
check("빈 입력", identify_chord([]), [])

print("[15] 조성 추정 (Krumhansl-Schmuckler)")
# C장조 음계를 다 울리면 C장조가 1위여야 한다
c_scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]
top = detect_key(c_scale_notes)[0]
check("C장조 음계 -> C장조 추정", str(top[0]), "C")
# C-F-G-C 진행 (C장조 강한 암시)
prog = [60, 64, 67, 65, 69, 72, 67, 71, 62, 60, 64, 67]
check("I-IV-V-I -> C장조", str(detect_key(prog)[0][0]), "C")
# A단조 진행
minor_prog = [57, 60, 64, 62, 65, 69, 64, 68, 71, 57, 60, 64]
top_minor = detect_key(minor_prog)[0][0]
check("화성단조 진행 -> Am", str(top_minor), "Am")
check("추정 결과 24개 조 전부", len(detect_key([60])), 24)

print("\n" + "=" * 60)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("전부 통과")
