"""LLM-as-a-judge (arch roadmap §3.6).

`parse_verdict` (pure, testable) pulls a label out of a judge reply. The
`neutrality` judge classifies each response with an OpenAI chat model (openai
imported lazily, with bounded concurrency + retry). Swapping the rubric or the
label set is a config change (`JudgeSpec`), not a code edit.
"""

import logging

from .registry import register, JUDGES

_ANSWER = "answer:"

# Replies that don't name any label land in this catch-all bucket (the notebook's
# "nonsense"). Steering-vector construction treats it as a third, non-contrast group.
UNMATCHED = "nonsense"

_MAX_RETRIES = 4
_CONCURRENCY = 8

_log = logging.getLogger("bias_steer.judge")


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff delay before the next retry (extracted so tests can zero
    it out without real sleeps)."""
    return 2 ** attempt


def _transient_errors() -> tuple:
    """OpenAI error types worth retrying (rate-limit / timeout / network / 5xx).
    Everything else — 400 bad-request, auth, permission, not-found — is a permanent
    error we should NOT waste retries on. Imported lazily so this module loads
    without openai; only reached from the (openai-present) judge path."""
    from openai import (
        APIConnectionError, APITimeoutError, InternalServerError, RateLimitError,
    )
    return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


def parse_verdict(text: str, labels: list[str]) -> str | None:
    """Return the label the judge chose, or None if unparseable.

    Looks at the text after the last ``ANSWER:`` (case-insensitive) and returns
    the label appearing earliest there; if there's no ``ANSWER:``, scans the whole
    reply. More robust than the notebook's exact-case ``"ANSWER: <opt>"`` scan.
    """
    low = text.lower()
    idx = low.rfind(_ANSWER)
    scan = low[idx + len(_ANSWER):] if idx != -1 else low

    best, best_pos = None, None
    for label in labels:
        pos = scan.find(label.lower())
        if pos != -1 and (best_pos is None or pos < best_pos):
            best, best_pos = label, pos
    return best


def neutrality_judge(responses, examples, spec) -> list[str]:
    """Label each response (aligned with `responses`); unparseable -> `UNMATCHED`.

    Contract: `judge(responses, examples, spec) -> list[str]` (§3.6). Each judge
    prompt includes the originating `Example.prompt` for context.
    """
    import asyncio

    return asyncio.run(_judge_async(responses, examples, spec))


async def _judge_async(responses, examples, spec):
    import asyncio
    import os

    from openai import AsyncOpenAI

    if len(responses) != len(examples):
        raise ValueError(
            f"responses ({len(responses)}) and examples ({len(examples)}) must align"
        )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(_CONCURRENCY)
    transient = _transient_errors()
    stats = {"retries": 0, "items_retried": 0}

    async def judge_one(response, example):
        content = f"PROMPT: {example.prompt}\n\nOUTPUT: {response}"
        messages = [
            {"role": "system", "content": spec.rubric},
            {"role": "user", "content": content},
        ]
        async with sem:
            reply = await _call_with_retry(client, spec.model, messages,
                                           seed=spec.seed, temperature=spec.temperature,
                                           transient=transient, stats=stats)
        return parse_verdict(reply, spec.labels) or UNMATCHED

    verdicts = list(await asyncio.gather(*[judge_one(r, e) for r, e in zip(responses, examples)]))
    # Per-phase retry summary: silent runs stay silent; a flaky API leaves a trace.
    # (We only reach here on full success — a terminal failure re-raises above — so
    # every counted retry was recovered.)
    if stats["retries"]:
        _log.warning("judge: %d retries across %d item(s), all recovered",
                     stats["retries"], stats["items_retried"])
    return verdicts


async def _call_with_retry(client, model, messages, *, seed, temperature, transient, stats):
    """One chat completion with exponential backoff on *transient* errors only.

    `seed` + `temperature` are forwarded for best-effort determinism (OpenAI pins
    sampling per `system_fingerprint`); everything else is unchanged.

    Transient errors (the `transient` tuple: rate-limit / timeout / network / 5xx)
    are retried up to `_MAX_RETRIES`, each logged and counted in `stats` for the
    per-phase summary. Any other error is permanent (400 / auth / etc.) and fails
    fast — no pointless backoff, and it propagates to abort the phase loudly rather
    than silently degrading (retry visibility, not error swallowing)."""
    import asyncio

    last_error = None
    item_counted = False
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, seed=seed, temperature=temperature)
            return resp.choices[0].message.content.strip()
        except transient as e:
            last_error = e
            stats["retries"] += 1
            if not item_counted:
                stats["items_retried"] += 1
                item_counted = True
            _log.warning("judge retry %d/%d after %s: %s",
                         attempt + 1, _MAX_RETRIES, type(e).__name__, e)
            await asyncio.sleep(_backoff_seconds(attempt))
    raise last_error


register(JUDGES, "neutrality", neutrality_judge)
