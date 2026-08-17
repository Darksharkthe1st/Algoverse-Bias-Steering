# Findings

Empirical results and diagnoses, one file per investigation. These are the
*outcomes* of runs — what we measured, what it means, and what has been ruled out.
Design/plan documents stay in `docs/` proper; this directory is the lab notebook.

**Convention:** `YYYY-MM-DD-<slug>.md`. Every findings doc should state, up front,
whether the thing under test passed, and record what was **eliminated** with the
evidence that eliminated it — a ruled-out cause is the most reusable output of a
debugging session, and the thing most often re-tried by the next person.

| date | finding | verdict |
|---|---|---|
| 2026-08-16 | [Refusal repro, qwen-1.8b — both tracks](./2026-08-16-refusal-repro-qwen-1.8b.md) | ✗ not reproduced; 8 causes eliminated |
| 2026-08-16 | [Refusal in OUR extraction convention, qwen-1.8b (§12)](./2026-08-16-refusal-native-extraction-qwen-1.8b.md) | ✓ validates *after* mean-centering; raw ablation "win" was model collapse |
| 2026-08-16 | [`test_phase2` starts a real coordinator campaign](./2026-08-16-test-phase2-coordinator-footgun.md) | environment footgun; unfixed in code |

## Related, not yet migrated

Earlier results predate this directory and still live in `docs/`. Migrating them
is a judgement call for the repo owner, not something to do silently:

- [`04-parity.md`](../04-parity.md) — notebook-parity ladder vs the Log_103 anchor.
- [`05-campaign-01.md`](../05-campaign-01.md) — campaign 01 (coeff sweep, anchors, CrowS).
