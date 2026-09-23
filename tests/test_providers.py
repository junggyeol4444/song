"""Provider 와 작사 검증.

확인하는 것:

  1. 설정: 키는 환경 변수가 우선, 화면에는 가려서, 파일은 나만 읽게
  2. 등록기: 키 없는 것이 먼저, 지정하면 그것이 먼저, 실패하면 다음 것으로
  3. 멜로디에서 줄을 제대로 찾는가, 같은 선율의 후렴을 알아보는가
  4. 로컬 작사가 모든 줄의 음절 수를 정확히 맞추는가 (여러 장르, 여러 씨앗)
  5. 멜로디 맞추기: 합치고 나눠도 박자 위치가 어긋나지 않는가
  6. 편집 명령: 적용 후 실행 취소하면 음 하나까지 원래대로 돌아오는가
  7. Claude Provider: 가짜 클라이언트로 요청 모양, 재요청, 실패 시 로컬 전환
  8. 가사를 붙인 곡이 실제로 노래로 렌더링되는가

실제 Anthropic 서버 호출은 여기서 하지 않는다 (키가 없다).
"""

import os
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from myvocal.core.history import History
from myvocal.core.notes import Note
from myvocal.core.units import PPQ
from myvocal.music.composer import Composer, SongRequest
from myvocal.music.lyric_lexicon import THEMES
from myvocal.music.lyrics import (
    LINE_GAP_TICKS, LyricsBrief, LyricsError, LocalLyricist, build_payload,
    check_response, collect_slots, detect_images, detect_theme, fit_notes, local_lyrics,
    lyrics_command, melody_lines, vocal_track, write_lyrics,
)
from myvocal.providers import default_registry
from myvocal.providers.anthropic_provider import MODEL_SETTING, AnthropicProvider
from myvocal.providers.base import (
    Capability, Provider, ProviderError, ProviderInfo, ProviderRegistry, ProviderRequest,
    ProviderResult, ProviderUnavailable,
)
from myvocal.providers.local import LocalProvider
from myvocal.providers.settings import Settings
from myvocal.voice.korean import count_syllables, split_syllables

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
        failures.append(f"{label}: {exc.__name__} 대신 {type(error).__name__}: {error}")
        print(f"  FAIL {label}: {type(error).__name__}")
        return
    failures.append(f"{label}: 예외가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


for variable in ("ANTHROPIC_API_KEY",):
    os.environ.pop(variable, None)

# ==========================================================================
print("1. 설정")
# ==========================================================================
folder = Path(tempfile.mkdtemp())
settings = Settings(folder)
check("처음엔 키 없음", settings.has_api_key("anthropic"), False)
settings.set_api_key("anthropic", "  sk-ant-test-1234567890abcd  ")
path = settings.save()
check("앞뒤 공백 제거", Settings(folder).api_key("anthropic"), "sk-ant-test-1234567890abcd")
if os.name != "nt":
    mode = stat.S_IMODE(path.stat().st_mode)
    check("설정 파일은 나만 읽고 쓴다", oct(mode), oct(0o600))
check_true("화면에는 가린다", "test-1234567890" not in settings.masked_key("anthropic"),
           settings.masked_key("anthropic"))
check("가린 키 끝 4자리", settings.masked_key("anthropic")[-4:], "abcd")
os.environ["ANTHROPIC_API_KEY"] = "sk-env-key-999999999"
check("환경 변수가 우선", settings.api_key("anthropic"), "sk-env-key-999999999")
check_true("출처 표시", "환경 변수" in settings.key_source("anthropic"))
del os.environ["ANTHROPIC_API_KEY"]
settings.set_api_key("anthropic", "")
check("빈 키는 삭제", settings.has_api_key("anthropic"), False)
(folder / "settings.json").write_text("{깨진 파일", encoding="utf-8")
broken = Settings(folder)
check_true("깨진 설정도 열린다", broken.load_error != "")
check("깨진 설정은 키 없음", broken.has_api_key("anthropic"), False)
check_raises("'_' 이름 금지", lambda: settings.set("_x", 1))

# ==========================================================================
print("2. 등록기")
# ==========================================================================


class Fake(Provider):
    def __init__(self, name, key, ready=True, fail=False):
        self.info = ProviderInfo(name, name, frozenset({Capability.LYRICS}),
                                 needs_key=key, needs_internet=key)
        self.ready, self.fail, self.calls = ready, fail, 0

    def is_ready(self):
        return self.ready, "준비 안 됨" if not self.ready else "ok"

    def execute(self, request):
        self.calls += 1
        if self.fail:
            raise RuntimeError("죽음")
        return ProviderResult(ok=True, data={"who": self.info.name})


registry = ProviderRegistry()
local_fake, cloud_fake = Fake("local", False), Fake("cloud", True)
registry.register(cloud_fake)
registry.register(local_fake)
names = [p.info.name for p in registry.providers_for(Capability.LYRICS)]
check("키 없는 것이 먼저", names, ["local", "cloud"])
registry.prefer(Capability.LYRICS, "cloud")
check("지정한 것이 먼저", [p.info.name for p in registry.providers_for(Capability.LYRICS)],
      ["cloud", "local"])
cloud_fake.fail = True
result = registry.run(ProviderRequest(Capability.LYRICS, max_retries=1))
check("실패하면 다음 것으로", result.data["who"], "local")
check("대체 표시", result.fell_back, True)
check("재시도 횟수 (1 + 재시도 1)", cloud_fake.calls, 2)
check_raises("지정 대상이 없으면 오류", lambda: registry.prefer(Capability.LYRICS, "없음"),
             ProviderError)
check_raises("할 수 있는 것이 없으면", lambda: registry.run(ProviderRequest(Capability.VIDEO)),
             ProviderUnavailable)
check_raises("시간 제한 0 금지", lambda: ProviderRequest(Capability.LYRICS, timeout_seconds=0),
             ProviderError)
not_ready = ProviderRegistry()
not_ready.register(Fake("cloud", True, ready=False))
check_raises("준비 안 된 것만 있으면", lambda: not_ready.run(ProviderRequest(Capability.LYRICS)),
             ProviderUnavailable)

default = default_registry(Settings(Path(tempfile.mkdtemp())))
check("키 없으면 작사는 내장만", [p.info.name for p in default.providers_for(Capability.LYRICS)],
      ["local"])
keyed = Settings(Path(tempfile.mkdtemp()))
keyed.set_api_key("anthropic", "sk-ant-zzzzzzzzzzzzzzzz")
if AnthropicProvider.sdk_available():
    check("키 있으면 Claude 먼저, 내장이 뒤",
          [p.info.name for p in default_registry(keyed).providers_for(Capability.LYRICS)],
          ["anthropic", "local"])
report = default.status_report()
check_true("상태 보고에 키 없음 이유", "API 키" in report or "anthropic" in report, report)

# ==========================================================================
print("3. 줄 찾기")
# ==========================================================================
line_notes = [Note(60, 0, 480), Note(62, 480, 480), Note(64, 960, 960),
              Note(65, 1920 + LINE_GAP_TICKS, 480), Note(67, 2400 + LINE_GAP_TICKS, 480)]
lines = melody_lines(line_notes)
check("쉼표에서 줄이 나뉜다", [len(l) for l in lines], [3, 2])
tight = [Note(60, 0, 480), Note(62, 480 + LINE_GAP_TICKS - 1, 480)]
check("짧은 틈은 줄 경계가 아니다", len(melody_lines(tight)), 1)
long_line = [Note(60 + (i % 5), i * 240, 240) for i in range(30)]
split = melody_lines(long_line)
check_true("긴 줄은 나눈다", all(len(l) <= 16 for l in split), [len(l) for l in split])
check("나눠도 음 개수는 그대로", sum(len(l) for l in split), 30)

song = Composer(SongRequest(title="다시 만난 날", genre="kpop",
                            subject="오래 헤어져 있던 친구를 다시 만나는 이야기", seed=11)).compose()
track = vocal_track(song)
slots = collect_slots(song, track)
choruses = [s for s in slots if s.section.kind == "chorus"]
check_true("후렴이 여러 번", len(choruses) >= 2, len(choruses))
check("첫 후렴은 새로 쓴다", choruses[0].same_as, None)
check_true("다음 후렴은 첫 후렴을 따른다",
           all(c.same_as is choruses[0] for c in choruses[1:]))
verses = [s for s in slots if s.section.kind == "verse"]
check_true("벌스는 따로 쓴다", all(v.same_as is None for v in verses))
check("벌스 둘의 틀이 같다 (같은 선율)", verses[0].pattern, verses[1].pattern)
check("모든 음이 어느 줄엔가 들어간다",
      sum(len(l.notes) for s in slots for l in s.lines), len(track.notes))

payload = build_payload(song, slots, LyricsBrief(seed=5))
check("제목은 프로젝트에서", payload["title"], "다시 만난 날")
check("설명도 프로젝트에서", payload["subject"], "오래 헤어져 있던 친구를 다시 만나는 이야기")

# ==========================================================================
print("4. 로컬 작사")
# ==========================================================================
theme, reason = detect_theme("오래 헤어져 있던 친구를 다시 만나는 이야기")
check("'헤어져' 가 있어도 재회가 이긴다", theme.name, "reunion")
check("이별", detect_theme("비 오는 날 떠나간 사람")[0].name, "parting")
check("주제 없으면 장르로", detect_theme("", "ballad")[0].name, "longing")
check_true("주제 없음 이유 표시", "장르" in detect_theme("", "ballad")[1])
check("장면: 비, 거리", detect_images("비 오는 거리에서"), ["비", "거리"])
check("'준비' 의 비는 아니다", detect_images("준비된 마음"), [])
check("'눈물' 의 눈은 아니다", detect_images("눈물이 나"), [])
check("'달려' 의 달은 아니다", detect_images("끝까지 달려"), [])

for name, theme_item in THEMES.items():
    lyricist = LocalLyricist(theme_item, seed=1)
    for count in range(1, 15):
        text = lyricist.line(count, "v")
        check(f"{name} {count}음절", count_syllables(text), count)

generated = local_lyrics(payload)
check("응답 틀 정확", check_response(payload, generated["sections"]), [])
check_true("주제 설명 포함", "다시 만남" in generated["note"], generated["note"])

total_lines = 0
for genre in ("pop", "kpop", "ballad", "rock", "edm", "anime", "hiphop", "jazz", "rnb"):
    for seed in range(3):
        project = Composer(SongRequest(genre=genre, seed=seed)).compose()
        project_slots = collect_slots(project, vocal_track(project))
        project_payload = build_payload(project, project_slots, LyricsBrief(seed=seed))
        written = local_lyrics(project_payload)
        problems = check_response(project_payload, written["sections"])
        check(f"{genre} #{seed} 음절 수 전부 일치", problems, [])
        total_lines += sum(len(v) for v in written["sections"].values())
check_true("여러 곡에서 줄을 충분히 썼다", total_lines > 300, total_lines)

again = local_lyrics(payload)
check("같은 씨앗이면 같은 가사", again["sections"], generated["sections"])

# ==========================================================================
print("5. 멜로디 맞추기")
# ==========================================================================
base_line = [Note(60, 0, 480), Note(62, 480, 240), Note(64, 720, 240), Note(65, 960, 960)]
merged = fit_notes(base_line, split_syllables("가나다"))
check("합치면 3개", len(merged), 3)
check("시작 위치 그대로", merged[0].start_tick, 0)
check("끝 위치 그대로", merged[-1].end_tick, 1920)
check("짧은 두 음을 합친다", (merged[1].start_tick, merged[1].duration_ticks), (480, 480))
check("합친 음은 앞 음 높이", merged[1].midi, 62)
check("마지막 긴 음은 그대로", (merged[-1].midi, merged[-1].duration_ticks), (65, 960))
check("가사 붙음", [n.lyric for n in merged], ["가", "나", "다"])
split_notes = fit_notes(base_line, split_syllables("가나다라마바"))
check("나누면 6개", len(split_notes), 6)
check("나눠도 끝 위치 그대로", split_notes[-1].end_tick, 1920)
gaps = [b.start_tick - a.end_tick for a, b in zip(split_notes, split_notes[1:])]
check("나눠도 빈틈·겹침 없음", gaps, [0] * 5)
check_true("16분음표보다 짧게 안 나눈다",
           all(n.duration_ticks >= PPQ // 4 for n in split_notes))
same = fit_notes(base_line, split_syllables("가나다라"))
check("같으면 길이 그대로", [n.duration_ticks for n in same], [480, 240, 240, 960])
tiny = [Note(60, 0, PPQ // 4)]
check_raises("더 못 나누면 오류", lambda: fit_notes(tiny, ["가", "나"]), LyricsError)
check_raises("빈 가사 오류", lambda: fit_notes(base_line, []), LyricsError)

# ==========================================================================
print("6. 편집 명령과 실행 취소")
# ==========================================================================
local_registry = ProviderRegistry()
local_registry.register(LocalProvider())
draft = write_lyrics(song, track, local_registry, LyricsBrief(seed=2))
check("내장이 썼다", draft.provider, "local")
check("틀 불일치 없음", draft.mismatches, [])
before = [(n.note_id, n.midi, n.start_tick, n.duration_ticks, n.lyric) for n in track.notes]
history = History(song)
history.run(lyrics_command(track, draft))
check("모든 음에 가사", sum(1 for n in track.notes if not n.lyric), 0)
check("음 개수 그대로 (틀이 정확하므로)", len(track.notes), len(before))
check("음 번호 그대로", sorted(n.note_id for n in track.notes), sorted(b[0] for b in before))
first_chorus_text = "".join(n.lyric for l in choruses[0].lines for n in l.notes)
sung = {}
for slot in choruses:
    span = song.section_range(slot.section)
    sung[slot.section.label] = "".join(
        n.lyric for n in track.notes if span.start_tick <= n.start_tick < span.end_tick)
check("후렴마다 같은 가사", len(set(sung.values())), 1)
verse_texts = []
for slot in verses:
    span = song.section_range(slot.section)
    verse_texts.append("".join(n.lyric for n in track.notes
                               if span.start_tick <= n.start_tick < span.end_tick))
check_true("벌스끼리는 다른 가사", verse_texts[0] != verse_texts[1], verse_texts)
check("곡 검사에 가사 문제 없음",
      [p for p in song.problems() if "가사" in p], [])
# 띄어쓰기가 살아 있어야 가사를 다시 읽을 수 있다
slot0 = draft.slots[0]
rebuilt = []
for line in slot0.lines:
    span_notes = [n for n in track.notes
                  if line.notes[0].start_tick <= n.start_tick <= line.notes[-1].start_tick]
    rebuilt.append("".join(n.lyric + (" " if n.word_end else "") for n in span_notes).strip())
check("띄어쓰기까지 그대로 되살아난다", rebuilt,
      [" ".join(t.split()) for t in draft.lines_for(slot0)])
from myvocal.core.project import Project as _Project
saved_path = Path(tempfile.mkdtemp()) / "lyrics.mvp"
song.save(saved_path)
reloaded = vocal_track(_Project.load(saved_path))
check("저장하고 열어도 띄어쓰기 유지",
      [(n.lyric, n.word_end) for n in reloaded.notes],
      [(n.lyric, n.word_end) for n in track.notes])
history.undo()
after = [(n.note_id, n.midi, n.start_tick, n.duration_ticks, n.lyric) for n in track.notes]
check("실행 취소하면 완전히 원래대로", after, before)
check("띄어쓰기 표시도 원래대로", sum(1 for n in track.notes if n.word_end), 0)
history.redo()
check("다시 실행", sum(1 for n in track.notes if not n.lyric), 0)

# 음절이 다른 가사: 음을 합치고 나누는 경우도 되돌아와야 한다
history.undo()
odd = write_lyrics(song, track, local_registry, LyricsBrief(seed=3))
first_key = odd.slots[0].key
odd.sections[first_key] = list(odd.sections[first_key])
odd.sections[first_key][0] = odd.sections[first_key][0] + " 오"      # 1음절 더
if len(odd.sections[first_key]) > 1:
    words = split_syllables(odd.sections[first_key][1])
    odd.sections[first_key][1] = "".join(words[:-1])                  # 1음절 덜
count_before = len(track.notes)
history.run(lyrics_command(track, odd))
check("음 개수가 +1 -1 로 같다", len(track.notes), count_before)
check("가사 빠진 음 없음", sum(1 for n in track.notes if not n.lyric), 0)
check("겹치는 음 없음", track.notes.overlapping_pairs(), [])
history.undo()
after = [(n.note_id, n.midi, n.start_tick, n.duration_ticks, n.lyric) for n in track.notes]
check("합치고 나눈 것도 되돌아온다", after, before)
check_raises("줄 수가 다르면 거부",
             lambda: lyrics_command(track, type(odd)(odd.slots, {first_key: ["가"]})),
             LyricsError)

# ==========================================================================
print("7. Claude Provider (가짜 클라이언트)")
# ==========================================================================


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            reply = {"text": reply}
        return SimpleNamespace(
            stop_reason=reply.get("stop", "end_turn"),
            content=[SimpleNamespace(type="text", text=reply["text"])],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=800),
        )


class FakeClient:
    def __init__(self, replies, models=("model-newest", "model-older")):
        self.messages = FakeMessages(replies)
        self.models = SimpleNamespace(
            list=lambda limit=20: SimpleNamespace(data=[SimpleNamespace(id=m) for m in models]))


import json


def perfect(payload, bad_line=None):
    sections = []
    for item in payload["sections"]:
        if item["same_as"]:
            continue
        lines = ["가" * n for n in item["lines"]]
        sections.append({"key": item["key"], "lines": lines})
    if bad_line is not None:
        sections[0]["lines"][0] += "나"
    return json.dumps({"sections": sections, "note": "재회의 기쁨"}, ensure_ascii=False)


key_settings = Settings(Path(tempfile.mkdtemp()))
check("키 없으면 준비 안 됨", AnthropicProvider(key_settings, lambda k, t: None).is_ready()[0], False)
key_settings.set_api_key("anthropic", "sk-ant-fake-000000000000")

client = FakeClient([perfect(payload)])
provider = AnthropicProvider(key_settings, lambda key, timeout: client)
check("키 있으면 준비됨", provider.is_ready()[0], True)
result = provider.run(ProviderRequest(Capability.LYRICS, payload))
sent = client.messages.requests[0]
check("모델은 목록의 가장 최근 것", sent["model"], "model-newest")
check("JSON 형식 지정", sent["output_config"]["format"]["type"], "json_schema")
check("구간 키를 스키마로 제한",
      sorted(sent["output_config"]["format"]["schema"]["properties"]["sections"]["items"]
             ["properties"]["key"]["enum"]),
      sorted(i["key"] for i in payload["sections"] if not i["same_as"]))
check_true("요청에 설명 포함", "친구를 다시 만나는" in sent["messages"][0]["content"])
check_true("반복 후렴은 쓰지 말라고 알림", "같은 가사를 다시 부르므로" in sent["messages"][0]["content"])
check("한 번에 맞으면 요청 1번", len(client.messages.requests), 1)
check("응답 틀 정확", check_response(payload, result.data["sections"]), [])
check_true("쓴 토큰 표시", "1,200" in result.cost_note, result.cost_note)

key_settings.set(MODEL_SETTING, "model-chosen")
client = FakeClient([perfect(payload, bad_line=0), perfect(payload)])
provider = AnthropicProvider(key_settings, lambda key, timeout: client)
result = provider.run(ProviderRequest(Capability.LYRICS, payload))
check("설정한 모델을 쓴다", client.messages.requests[0]["model"], "model-chosen")
check("음절이 틀리면 한 번 더 요청", len(client.messages.requests), 2)
second = client.messages.requests[1]["messages"]
check("두 번째 요청은 대화를 이어 간다", [m["role"] for m in second], ["user", "assistant", "user"])
check_true("틀린 줄을 짚어 준다", "음절이어야 하는데" in second[-1]["content"])
check("고친 결과를 쓴다", check_response(payload, result.data["sections"]), [])

client = FakeClient([perfect(payload, bad_line=0), perfect(payload, bad_line=0)])
provider = AnthropicProvider(key_settings, lambda key, timeout: client)
result = provider.run(ProviderRequest(Capability.LYRICS, payload))
check("두 번 다 틀려도 결과는 돌려준다 (멜로디가 맞춘다)", result.ok, True)
check("남은 불일치 1줄", len(check_response(payload, result.data["sections"])), 1)

missing = json.dumps({"sections": [], "note": ""})
client = FakeClient([missing, missing, missing, missing])
provider = AnthropicProvider(key_settings, lambda key, timeout: client)
mixed = ProviderRegistry()
mixed.register(LocalProvider())
mixed.register(provider)
mixed.prefer(Capability.LYRICS, "anthropic")
result = mixed.run(ProviderRequest(Capability.LYRICS, payload, max_retries=1))
check("구간이 빠지면 내장으로 넘어간다", result.provider, "local")
check("대체 표시", result.fell_back, True)


class AuthenticationError(Exception):
    pass


client = FakeClient([AuthenticationError("invalid x-api-key")])
provider = AnthropicProvider(key_settings, lambda key, timeout: client)
check_raises("키 거부는 재시도 없이 사용 불가", lambda: provider.run(
    ProviderRequest(Capability.LYRICS, payload, max_retries=3)), ProviderUnavailable)
check("키 거부 시 요청 1번뿐", len(client.messages.requests), 1)

client = FakeClient([{"text": "", "stop": "refusal"}] * 2)
provider = AnthropicProvider(key_settings, lambda key, timeout: client)
check_raises("거절은 실패로", lambda: provider.run(
    ProviderRequest(Capability.LYRICS, payload, max_retries=0)), ProviderError)

# 실제 흐름: 가짜 Claude 가 쓴 가사를 곡에 붙인다
client = FakeClient([perfect(payload)])
flow = ProviderRegistry()
flow.register(LocalProvider())
flow.register(AnthropicProvider(key_settings, lambda key, timeout: client))
flow.prefer(Capability.LYRICS, "anthropic")
claude_draft = write_lyrics(song, track, flow, LyricsBrief())
check("Claude 가 썼다", claude_draft.provider, "anthropic")
history.run(lyrics_command(track, claude_draft))
check("Claude 가사가 붙었다", {n.lyric for n in track.notes}, {"가"})
history.undo()

# 진짜 SDK 가 이 요청을 받아들이고 응답을 읽는지. 네트워크 대신 가짜 전송층을 쓴다.
if AnthropicProvider.sdk_available():
    import anthropic
    import httpx2

    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/v1/models"):
            return httpx2.Response(200, json={
                "data": [{"type": "model", "id": "model-a", "display_name": "A",
                          "created_at": "2026-01-01T00:00:00Z"}],
                "has_more": False, "first_id": "model-a", "last_id": "model-a"})
        body = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": perfect(payload)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20}})

    def sdk_factory(key, timeout):
        return anthropic.Anthropic(
            api_key=key, timeout=timeout, max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))

    sdk_settings = Settings(Path(tempfile.mkdtemp()))
    sdk_settings.set_api_key("anthropic", "sk-ant-sdk-000000000000")
    sdk_result = AnthropicProvider(sdk_settings, sdk_factory).run(
        ProviderRequest(Capability.LYRICS, payload, max_retries=0))
    check("SDK: 응답을 읽었다", check_response(payload, sdk_result.data["sections"]), [])
    check("SDK: 모델 목록 → 메시지 순서",
          [(r.method, r.url.path) for r in seen], [("GET", "/v1/models"), ("POST", "/v1/messages")])
    check("SDK: 키는 헤더로", seen[1].headers.get("x-api-key"), "sk-ant-sdk-000000000000")
    sent_body = json.loads(seen[1].content)
    check("SDK: 보낸 항목", sorted(sent_body),
          ["max_tokens", "messages", "model", "output_config", "system"])
    check("SDK: 목록에서 고른 모델", sent_body["model"], "model-a")
else:
    print("  (anthropic 패키지가 없어 SDK 확인은 건너뜀)")

# ==========================================================================
print("8. 가사 붙인 곡을 노래로")
# ==========================================================================
from myvocal.audio.renderer import RenderOptions, Renderer
import numpy as np

short = Composer(SongRequest(genre="ballad", seed=4)).compose()
short_track = vocal_track(short)
short_draft = write_lyrics(short, short_track, local_registry, LyricsBrief(seed=4))
History(short).run(lyrics_command(short_track, short_draft))
renderer = Renderer(RenderOptions(sample_rate=24000))
first = short.structure.sections[1]
start, end = first.seconds(short.meter, short.tempo)
warnings: list[str] = []
voice = renderer.render_vocal(short_track, short.tempo, int(end * 24000) + 24000, warnings)
check_true("보컬이 노래로 렌더링됨", voice is not None, warnings)
if voice is not None:
    segment = voice.data[0][int(start * 24000): int(end * 24000)]
    rms = float(np.sqrt(np.mean(segment ** 2)))
    check_true("첫 벌스 구간에 소리가 있다", rms > 1e-3, rms)
    check("경고 없음", [w for w in warnings if "가사가 없어" in w], [])

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
