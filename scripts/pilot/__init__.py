"""Run-2 pilot — Phase 2 of notes/11-EXPERIMENT-PROTOCOL.md.

The pilot's job is not to produce results. Its job is to execute every line of
code the real run will execute, at a scale where a full pass takes seconds, on a
machine with no GPU.

It is deliberately split in two tiers, because this laptop has no torch:

  TIER 1 — torch-free, runs today.  Pairing, split construction, residual
           persistence, extraction, floors with intervals, the negative control,
           the specificity control, the cross-category matrix, the report, the
           queue manifest and the verifier.  Driven by a STUB backend that
           synthesises residuals with a KNOWN planted structure, so the pilot can
           assert the pipeline RECOVERS what was planted rather than merely
           asserting it did not crash.

  TIER 2 — needs torch + transformers.  Tokenisation, the chat template, the
           capture index, and a real forward pass.  Written here and NOT yet
           run.  This is where hole (d) lives — notes/19 §6.1, the -1 vs -2
           capture index — so the pilot is NOT green until tier 2 has run.

Saying that plainly matters: a tier-1-only pass would certify a pipeline whose
capture site has never been checked, which is the single defect most likely to
invalidate the whole run.
"""
