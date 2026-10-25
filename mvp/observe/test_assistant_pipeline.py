import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observe.association import associate
from observe.assistant import Assistant, AssistantReply, AssistantUnavailable
from observe.hand_state import HandReading
from observe.safety import check_reply, hazards_for_age, object_notes
from observe.session import ObservationSession


# Run from the repository root (needs the childplay_coreml environment):
#   python mvp/observe/test_assistant_pipeline.py
# No API key needed: the Claude client is replaced by a fake.

W, H = 1000, 1000


def hand_at(x, y, size=100):

    # Simple upright hand: wrist at (x, y), fingers above it.
    points = [[x, y, 0.0]] * 21
    points = [list(p) for p in points]
    points[9] = [x, y - size, 0.0]

    for tip, dx in zip([4, 8, 12, 16, 20], [-60, -25, 0, 25, 50]):
        points[tip] = [x + dx, y - size * 1.8, 0.0]

    return points


def box(x1, y1, x2, y2, label="block", confidence=0.8):

    return {"label": label, "confidence": confidence, "box": [x1 / W, y1 / H, x2 / W, y2 / H]}


# --------------------------------------------------
# Association
# --------------------------------------------------

def test_object_at_fingertips_is_held_only_when_interacting():

    hand = hand_at(500, 600)            # fingertips around y = 420
    obj = box(470, 380, 540, 440)

    held = associate(hand, True, [obj], (W, H))
    touched = associate(hand, False, [obj], (W, H))

    assert held["object_in_hand"] == "block"
    assert held["objects_near"][0]["relation"] == "holding"
    assert touched["object_in_hand"] is None
    assert touched["objects_near"][0]["relation"] == "touching"


def test_visible_but_distant_object_is_not_linked():

    hand = hand_at(200, 600)
    far = box(800, 100, 900, 200, "ball")
    near = box(300, 380, 360, 440, "cup")

    result = associate(hand, True, [far, near], (W, H))

    labels = {o["label"]: o["relation"] for o in result["objects_near"]}

    assert "ball" not in labels
    assert labels["cup"] == "near"
    assert result["object_in_hand"] is None


# --------------------------------------------------
# Observation summary
# --------------------------------------------------

def test_observation_from_session_ticks():

    session = ObservationSession("test")
    session.child_age_months = 24

    t = 0.0

    for i in range(40):

        t += 0.25
        session.last_objects = {"t": t, "objects": [box(470, 380, 540, 440), box(800, 800, 900, 900, "ball")]}

        session.update(
            t,
            [HandReading(0, 0.9 if 8 <= i < 30 else 0.1, (500, 600), 100, "Right")],
            hand_frames={0: hand_at(500, 600)},
            video_size=(W, H)
        )

    obs = session.observation()

    assert obs["child_age_months"] == 24
    assert obs["recent"]["seconds_interacting"] > 3
    assert obs["recent"]["objects_handled_seconds"].get("block", 0) > 3
    assert any(e["object"] == "block" for e in obs["recent"]["interaction_episodes"])
    assert {o["label"] for o in obs["current"]["objects_visible"]} == {"block", "ball"}
    assert any(n["object"] == "ball" for n in obs["safety_notes"])  # under 3
    assert "hand_interaction_model" in obs["reliability"]


def test_empty_observation():

    obs = ObservationSession("x").observation()

    assert obs["current"]["hands"] == []
    assert "note" in obs["recent"]


# --------------------------------------------------
# Safety
# --------------------------------------------------

def reply_with(**activity_overrides):

    activity = {
        "title": "Tower", "objects_needed": ["large blocks"], "uses_only_visible_objects": True,
        "instructions": ["Stack two blocks."], "why_it_helps": "Practise grasp and release.",
        "skills_practised": ["grasp and release"], "safety_notes": [], "easier_version": "",
        "harder_version": ""
    }
    activity.update(activity_overrides)

    return {
        "reply": "Here's an idea.", "observation_summary": "It looks like a block was held.",
        "confidence_note": "Uncertain.", "activity": activity, "needs_more_observation": False
    }


def test_hazards_depend_on_age():

    assert "coin" in hazards_for_age(20)
    assert "coin" not in hazards_for_age(48)
    assert "coin" in hazards_for_age(None)  # unknown age: strictest
    assert object_notes(["block", "ball"], 48) == []


def test_check_reply_flags_hazards_and_diagnosis():

    assert check_reply(reply_with(), 24) == []

    problems = check_reply(reply_with(objects_needed=["a few coins", "a cup"]), 24)
    assert problems and "coin" in problems[0]

    assert check_reply(reply_with(objects_needed=["coins"]), 50) == []

    diagnosis = reply_with()
    diagnosis["observation_summary"] = "This may suggest a fine motor delay."
    assert any("assessment" in p for p in check_reply(diagnosis, 24))

    # Saying "I can't assess delays" in the conversational reply is fine.
    allowed = reply_with()
    allowed["reply"] = "I can't assess delays; please talk to your health visitor."
    assert check_reply(allowed, 24) == []


# --------------------------------------------------
# Assistant with a fake Claude client
# --------------------------------------------------

class FakeBlock:

    def __init__(self, text):
        self.text = text

    def model_dump(self, **kwargs):
        assert "parsed_output" in kwargs.get("exclude", set())
        return {"type": "text", "text": self.text}


class FakeUsage:
    input_tokens = 1000
    output_tokens = 300
    cache_read_input_tokens = 0


class FakeResponse:

    def __init__(self, reply, stop_reason="end_turn"):
        self.parsed_output = AssistantReply(**reply) if reply else None
        self.stop_reason = stop_reason
        self.content = [FakeBlock("{}")]
        self.usage = FakeUsage()
        self.model = "claude-opus-5-5"


class FakeClient:

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = self
        self.messages = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


OBS = {"child_age_months": 24, "current": {}, "recent": {}}


def test_request_shape():

    client = FakeClient([FakeResponse(reply_with())])
    assistant = Assistant(client=client, model="claude-opus-5-5", effort="low")
    history = []

    reply = assistant.respond(OBS, "What can we play next?", history)

    call = client.calls[0]

    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is AssistantReply
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "OBSERVATION:" in call["messages"][-1]["content"]
    assert reply["activity"]["title"] == "Tower"
    assert reply["meta"]["estimated_cost_usd"] > 0
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_haiku_has_no_fallbacks():

    client = FakeClient([FakeResponse(reply_with())])
    Assistant(client=client, model="claude-haiku-5-5").respond(OBS, "Hi", [])

    assert "fallbacks" not in client.calls[0]


def test_unsafe_answer_is_retried_then_replaced():

    unsafe = reply_with(objects_needed=["marbles"])

    # Unsafe, then safe: second answer is used, history stays append-only.
    client = FakeClient([FakeResponse(unsafe), FakeResponse(reply_with())])
    history = []
    reply = Assistant(client=client).respond(OBS, "Idea?", history)

    assert reply["activity"]["objects_needed"] == ["large blocks"]
    assert len(client.calls) == 2
    assert "app's check" in client.calls[1]["messages"][-1]["content"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]

    # Unsafe twice: activity replaced by the safe default.
    client = FakeClient([FakeResponse(unsafe), FakeResponse(unsafe)])
    reply = Assistant(client=client).respond(OBS, "Idea?", [])

    assert "marbles" not in str(reply["activity"])
    assert any("replaced" in n for n in reply["meta"]["safety_notes"])


def test_refusal_is_reported():

    client = FakeClient([FakeResponse(None, stop_reason="refusal")])

    try:
        Assistant(client=client).respond(OBS, "?", [])
        assert False, "expected AssistantUnavailable"
    except AssistantUnavailable:
        pass


# --------------------------------------------------
# Ollama backend with a fake requests module
# --------------------------------------------------

import json as _json

import requests as _requests


class FakeHTTP:

    def __init__(self, contents, version="0.3.14"):
        self.contents = list(contents)
        self.version = version
        self.posts = []

    def get(self, url, timeout=None):
        return _Resp({"version": self.version})

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        return _Resp({"message": {"content": self.contents.pop(0)}, "prompt_eval_count": 900, "eval_count": 250})


class _Resp:

    status_code = 200
    ok = True
    text = ""

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def with_fake_http(fake, run):

    original = (_requests.get, _requests.post)
    _requests.get, _requests.post = fake.get, fake.post

    try:
        return run()
    finally:
        _requests.get, _requests.post = original


def test_ollama_valid_reply_old_version():

    fake = FakeHTTP([_json.dumps(reply_with())], version="0.3.14")
    history = []

    reply = with_fake_http(fake, lambda: Assistant(backend="ollama").respond(OBS, "Idea?", history))

    body = fake.posts[0]

    assert body["model"] == "qwen2.5:7b"
    assert body["format"] == "json"                      # old Ollama: JSON mode only
    assert body["options"]["num_ctx"] >= 8192
    assert "Output format" in body["messages"][0]["content"]
    assert reply["activity"]["title"] == "Tower"
    assert reply["meta"]["estimated_cost_usd"] == 0
    assert reply["meta"]["backend"] == "ollama"
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_ollama_uses_schema_when_supported():

    fake = FakeHTTP([_json.dumps(reply_with())], version="0.6.2")

    with_fake_http(fake, lambda: Assistant(backend="ollama").respond(OBS, "Idea?", []))

    assert isinstance(fake.posts[0]["format"], dict)
    assert "activity" in fake.posts[0]["format"]["properties"]


def test_ollama_invalid_format_is_retried():

    fake = FakeHTTP(['{"reply": {"title": "oops"}}', _json.dumps(reply_with())])

    reply = with_fake_http(fake, lambda: Assistant(backend="ollama").respond(OBS, "Idea?", []))

    assert len(fake.posts) == 2
    assert "JSON format" in fake.posts[1]["messages"][-1]["content"]
    assert reply["activity"]["title"] == "Tower"

    fake = FakeHTTP(["not json", "still not json"])

    try:
        with_fake_http(fake, lambda: Assistant(backend="ollama").respond(OBS, "Idea?", []))
        assert False, "expected AssistantUnavailable"
    except AssistantUnavailable:
        pass


# --------------------------------------------------
# Fixed guardrails (independent of the model)
# --------------------------------------------------

def test_development_question_gets_fixed_note():

    from observe.safety import DEVELOPMENT_NOTE

    client = FakeClient([FakeResponse(reply_with())])
    reply = Assistant(client=client).respond(OBS, "Is my child behind for their age?", [])

    assert reply["reply"].startswith(DEVELOPMENT_NOTE)
    assert reply["meta"]["development_question"] is True
    assert "asking about development" in client.calls[0]["messages"][-1]["content"]

    client = FakeClient([FakeResponse(reply_with())])
    reply = Assistant(client=client).respond(OBS, "Something more advanced please", [])

    assert not reply["reply"].startswith(DEVELOPMENT_NOTE)


def test_unknown_age_adds_tip():

    from observe.safety import AGE_TIP

    client = FakeClient([FakeResponse(reply_with())])
    reply = Assistant(client=client).respond({"child_age_months": None}, "Idea?", [])

    assert reply["reply"].endswith(AGE_TIP)

    client = FakeClient([FakeResponse(reply_with())])
    reply = Assistant(client=client).respond(OBS, "Idea?", [])

    assert AGE_TIP not in reply["reply"]


def test_small_object_wording_is_flagged_under_3():

    assert check_reply(reply_with(instructions=["Put a small toy block in the cup."]), 18)
    assert check_reply(reply_with(instructions=["Put a small toy block in the cup."]), 48) == []
    assert check_reply(reply_with(instructions=["Put a big block in the cup."]), 18) == []


if __name__ == "__main__":

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    for test in tests:
        test()
        print("ok  ", test.__name__)

    print(f"\n{len(tests)} tests passed")
