import json
import os
import time
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from observe.safety import (
    AGE_TIP,
    DEVELOPMENT_NOTE,
    check_reply,
    is_development_question,
    safe_default_activity
)


# --------------------------------------------------
# Caregiver play assistant.
#
# Input: the structured observation (observation.py), the child's age
# and the caregiver's question. Only text is sent; no images.
# Output: a fixed schema, checked by safety.py.
#
# Two backends, same prompt, schema and safety checks:
#   claude  Claude API (structured outputs, refusal fallback)
#   ollama  a free local model through Ollama, on this computer
#
# Configuration (environment variables):
#   MOTIO_LLM_BACKEND   claude (default) | ollama
#   MOTIO_LLM_MODEL     default claude-opus-5-5, or qwen2.5:7b for ollama
#   MOTIO_LLM_EFFORT    claude only: low | medium | high (default low)
#   ANTHROPIC_API_KEY   claude only
#   OLLAMA_HOST         ollama only, default http://localhost:11434
# --------------------------------------------------

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_EFFORT = "low"

# USD per million tokens (input, output), for the cost shown in the UI.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Server-side refusal fallback ("default" form) is available for these;
# Haiku has none.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}

# A conversation is restarted (not trimmed) after this many turns, because
# later turns carry thinking blocks that are tied to the exact history.
MAX_TURNS = 12


SYSTEM_PROMPT = """\
You are Motio, a play-support assistant for caregivers of children under 5. \
You suggest simple, playful activities that give a child chances to practise \
fine-motor skills (using hands and fingers), based on what a camera system has \
observed during play. You speak to the caregiver: warm, practical, plain language, \
short sentences.

## What you receive
Each caregiver message comes with an OBSERVATION produced by computer-vision models:
- current: what each visible hand appears to be doing now (phase, interaction \
probability, object_in_hand, objects_near) and the objects in view.
- recent: the last minute (time hands were visible, interaction episodes, objects \
handled and seen).
- safety_notes: notes about visible objects for this child's age.
- reliability: how far each signal can be trusted. Read it. The vision models are \
uncertain: the hand model misses some interactions and raises false alarms, \
phases are inferred, and object labels can be wrong.

## How to use it
- Describe what was seen tentatively ("it looks like...", "the camera picked up..."). \
Never state an observation as certain, and say so when the observation is thin \
(few seconds, no hands, low confidence). Mention only objects that appear in the \
observation or that the caregiver mentions; do not invent objects.
- Prefer objects already available. If an activity needs something not observed, \
choose a common, safe household item and set uses_only_visible_objects to false.
- Build on what the child seems to be doing (e.g. holding a block -> stacking or \
putting blocks into a container).

## Never
- Never assess, diagnose or compare the child's development, skills or milestones, \
and never suggest a delay, disorder or concern. A few seconds of camera data cannot \
show that. If the caregiver asks about development, say kindly that this tool \
cannot assess development, and suggest speaking with their health visitor, \
paediatrician or an early-childhood professional.
- Never suggest choking or other hazards. Under 3 years: nothing small enough to fit \
through a toilet-paper tube (about 3 cm), no coins, button batteries, magnets, \
marbles, balloons, beads, small balls, dried beans, whole grapes, nuts or popcorn; \
no cords, strings or plastic bags. Pointed pens/pencils only with close supervision \
from 3 years. Every activity assumes an adult is right there.
- Never recommend medical, therapeutic or dietary interventions.

## Activities
- One activity at a time, suited to the child's age in months. If age is unknown, \
ask for it in your reply and suggest something safe for a toddler.
- 2-5 short instructions the caregiver can follow immediately.
- why_it_helps: what the child gets to practise (e.g. grasping and letting go, \
two-handed play, hand-eye coordination, finger control), framed as play and \
practice, not as therapy or remediation.
- Include an easier and a harder version so the caregiver can follow the child's lead.
- Keep reply to 2-4 sentences. Answer the caregiver's actual question first.
"""


class Activity(BaseModel):
    title: str
    objects_needed: list[str]
    uses_only_visible_objects: bool
    instructions: list[str]
    why_it_helps: str
    skills_practised: list[str]
    safety_notes: list[str]
    easier_version: str
    harder_version: str


class AssistantReply(BaseModel):
    reply: str
    observation_summary: str
    confidence_note: str
    activity: Activity
    needs_more_observation: bool


class AssistantUnavailable(Exception):
    pass


# Local models get the reply shape spelled out in the prompt, because
# older Ollama versions can only force "some JSON", not this schema.
OLLAMA_FORMAT_INSTRUCTIONS = """

## Output format
Reply with ONLY one JSON object and nothing else, in exactly this shape \
(replace the ... with your text; lists can have several items):
{"reply": "...", "observation_summary": "...", "confidence_note": "...", \
"activity": {"title": "...", "objects_needed": ["..."], "uses_only_visible_objects": true, \
"instructions": ["...", "..."], "why_it_helps": "...", "skills_practised": ["..."], \
"safety_notes": ["..."], "easier_version": "...", "harder_version": "..."}, \
"needs_more_observation": false}
"""

# Room for the instructions, the observation and a short conversation.
# (Ollama's default context is 2048 tokens, which would silently cut the prompt.)
OLLAMA_CONTEXT_TOKENS = 8192


@dataclass
class _Generation:

    reply: dict
    error: str
    usage: dict
    history_content: object
    model: str


def content_for_history(response):

    # The assistant turn as sent back on the next request: every block
    # unchanged (thinking blocks must not be altered), minus the SDK-only
    # parsed_output field, which the API does not accept.
    return [
        block.model_dump(by_alias=True, exclude_none=True, exclude={"parsed_output"}, mode="json")
        for block in response.content
    ]


class Assistant:

    def __init__(self, client=None, model=None, effort=None, backend=None, ollama_host=None):

        self.backend = backend or os.environ.get("MOTIO_LLM_BACKEND", "claude")

        if self.backend not in ("claude", "ollama"):
            raise ValueError("MOTIO_LLM_BACKEND must be claude or ollama")

        default_model = DEFAULT_OLLAMA_MODEL if self.backend == "ollama" else DEFAULT_MODEL

        self.model = model or os.environ.get("MOTIO_LLM_MODEL", default_model)
        self.effort = effort or os.environ.get("MOTIO_LLM_EFFORT", DEFAULT_EFFORT)
        self.ollama_host = (
            ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self._client = client
        self._ollama_schema_support = None

    @property
    def client(self):

        if self._client is None:

            import anthropic

            # Resolves ANTHROPIC_API_KEY (or another configured credential).
            try:
                self._client = anthropic.Anthropic()
            except Exception as error:
                raise AssistantUnavailable(
                    "No Claude API credentials found. Set ANTHROPIC_API_KEY in the "
                    "terminal before starting the server. " + str(error)
                )

        return self._client

    # ------------------------------------------------

    # ------------------------------------------------
    # Claude backend
    # ------------------------------------------------

    def _request(self, messages):

        kwargs = {
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "output_format": AssistantReply,
            "output_config": {"effort": self.effort},
            # Caches the system prompt and conversation for follow-up questions.
            "cache_control": {"type": "ephemeral"},
        }

        if self.model in FALLBACK_MODELS:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        return self.client.beta.messages.parse(**kwargs)

    def _generate_claude(self, messages):

        import anthropic

        try:
            response = self._request(messages)
        except anthropic.AuthenticationError:
            raise AssistantUnavailable("The Claude API key was rejected. Check ANTHROPIC_API_KEY.")
        except anthropic.RateLimitError:
            raise AssistantUnavailable("Rate limited by the Claude API; try again in a minute.")
        except anthropic.APIConnectionError:
            raise AssistantUnavailable("Could not reach the Claude API. Check the internet connection.")

        usage = {
            key: getattr(response.usage, key, 0) or 0
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens")
        }

        if response.stop_reason == "refusal":
            raise AssistantUnavailable("The model declined this request. Try rephrasing the question.")

        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise AssistantUnavailable("The reply was incomplete; please try again.")

        return _Generation(
            reply=response.parsed_output.model_dump(),
            error=None,
            usage=usage,
            history_content=content_for_history(response),
            model=getattr(response, "model", self.model)
        )

    # ------------------------------------------------
    # Ollama backend (local, free)
    # ------------------------------------------------

    def _ollama_supports_schema(self, requests):

        # Ollama 0.5+ can constrain output to a JSON schema; older
        # versions only to "valid JSON".
        if self._ollama_schema_support is None:

            try:
                version = requests.get(f"{self.ollama_host}/api/version", timeout=5).json()["version"]
                parts = tuple(int(p) for p in version.split("-")[0].split(".")[:3])
                self._ollama_schema_support = parts >= (0, 5, 0)
            except Exception:
                self._ollama_schema_support = False

        return self._ollama_schema_support

    def _generate_ollama(self, messages):

        import requests

        schema = self._ollama_supports_schema(requests)

        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": AssistantReply.model_json_schema() if schema else "json",
                    "options": {"temperature": 0.3, "num_ctx": OLLAMA_CONTEXT_TOKENS},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT + OLLAMA_FORMAT_INSTRUCTIONS}
                    ] + messages
                },
                timeout=300
            )
        except requests.ConnectionError:
            raise AssistantUnavailable(
                "Could not reach Ollama. Open the Ollama app (or run `ollama serve`) and try again."
            )

        if response.status_code == 404:
            raise AssistantUnavailable(
                f"The local model {self.model} is not downloaded. Run: ollama pull {self.model}"
            )

        if not response.ok:
            raise AssistantUnavailable(f"Ollama error {response.status_code}: {response.text[:200]}")

        data = response.json()
        text = data.get("message", {}).get("content", "")

        usage = {
            "input_tokens": data.get("prompt_eval_count", 0) or 0,
            "output_tokens": data.get("eval_count", 0) or 0,
            "cache_read_input_tokens": 0
        }

        try:
            reply = AssistantReply.model_validate_json(text).model_dump()
            error = None
        except ValidationError as validation:
            reply = None
            error = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in validation.errors()[:5]
            )

        return _Generation(
            reply=reply,
            error=error,
            usage=usage,
            history_content=text,
            model=self.model
        )

    # ------------------------------------------------

    def _generate(self, messages):

        if self.backend == "ollama":
            return self._generate_ollama(messages)

        return self._generate_claude(messages)

    def respond(self, observation, question, history):

        # history: list of messages for this conversation (appended in place).
        # Returns a dict for the UI.

        if len(history) >= MAX_TURNS * 2:

            previous = [
                m["content"] for m in history if m["role"] == "user"
            ][-2:]

            history.clear()
            question = (
                "(We are starting a new conversation thread; earlier questions included: "
                + " | ".join(p.split("\n\nOBSERVATION:")[0] for p in previous)
                + ")\n\n" + question
            )

        development = is_development_question(question)

        # A reminder of the rule for this turn; the fixed note below is added
        # regardless of what the model answers.
        reminder = (
            "\n\n(The caregiver is asking about development: follow the rule for "
            "development questions, then offer a play idea.)"
            if development else ""
        )

        user_message = {
            "role": "user",
            "content": (
                f"{question}{reminder}\n\nOBSERVATION:\n"
                + json.dumps(observation, separators=(",", ":"))
            )
        }

        messages = history + [user_message]

        age = observation.get("child_age_months")
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        started = time.perf_counter()
        notes = []

        for attempt in range(2):

            generation = self._generate(messages)

            for key in usage:
                usage[key] += generation.usage.get(key, 0)

            reply = generation.reply

            if generation.error:
                problems = [
                    "The answer was not in the required JSON format ("
                    + generation.error
                    + "). Reply with only the JSON object"
                ]
            else:
                problems = check_reply(reply, age)

            if not problems:
                break

            notes.append(f"Check on answer {attempt + 1}: " + "; ".join(problems))

            if attempt == 0:

                # Retry once in the same conversation, appending (never editing) turns.
                messages = messages + [
                    {"role": "assistant", "content": generation.history_content},
                    {
                        "role": "user",
                        "content": (
                            "That answer did not pass the app's check: "
                            + "; ".join(problems)
                            + ". Please answer again following the rules."
                        )
                    }
                ]

        else:

            if reply is None:
                raise AssistantUnavailable(
                    "The model did not produce a usable answer twice. Please try again."
                )

            # Still failing the safety check: keep the conversational reply,
            # replace the activity.
            reply["activity"] = safe_default_activity(age)
            notes.append("The suggested activity was replaced with a safe default.")

        history.extend(messages[len(history):])
        history.append({"role": "assistant", "content": generation.history_content})

        # Fixed wording, independent of the model.
        if development:
            reply["reply"] = DEVELOPMENT_NOTE + " " + reply["reply"]

        if age is None:
            reply["reply"] = reply["reply"] + " " + AGE_TIP

        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))

        reply["meta"] = {
            "model": generation.model,
            "backend": self.backend,
            "effort": self.effort if self.backend == "claude" else None,
            "seconds": round(time.perf_counter() - started, 1),
            "usage": usage,
            "estimated_cost_usd": round(
                usage["input_tokens"] / 1e6 * price_in
                + usage["output_tokens"] / 1e6 * price_out,
                4
            ),
            "safety_notes": notes,
            "development_question": development
        }

        return reply
