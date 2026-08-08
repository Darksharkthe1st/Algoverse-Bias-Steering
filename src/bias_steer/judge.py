"""LLM-as-a-judge (arch roadmap §3.6).

`parse_verdict` (pure, testable) pulls a label out of a judge reply. The
`neutrality` judge classifies each response with an OpenAI chat model (openai
imported lazily, with bounded concurrency + retry). Swapping the rubric or the
label set is a config change (`JudgeSpec`), not a code edit.
"""

from .registry import register, JUDGES

_ANSWER = "answer:"

# Replies that don't name any label land in this catch-all bucket (the notebook's
# "nonsense"). Steering-vector construction treats it as a third, non-contrast group.
UNMATCHED = "nonsense"

_MAX_RETRIES = 4
_CONCURRENCY = 8


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

    async def judge_one(response, example):
        content = f"PROMPT: {example.prompt}\n\nOUTPUT: {response}"
        messages = [
            {"role": "system", "content": spec.rubric},
            {"role": "user", "content": content},
        ]
        async with sem:
            reply = await _call_with_retry(client, spec.model, messages)
        return parse_verdict(reply, spec.labels) or UNMATCHED

    return list(await asyncio.gather(*[judge_one(r, e) for r, e in zip(responses, examples)]))


async def _call_with_retry(client, model, messages):
    """One chat completion with exponential backoff on transient errors."""
    import asyncio

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 - retry any transient API/network error
            last_error = e
            await asyncio.sleep(2 ** attempt)
    raise last_error


register(JUDGES, "neutrality", neutrality_judge)
