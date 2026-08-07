# What We Verified, What We Retracted, and How Two Fake Results Survived a Year

*What this document is: the verification story of this project, told in full. In August 2026 the team ran two independent audits, six days apart, over a research archive that had been sitting untouched for nine months. One headline result reproduced perfectly. Two others turned out to be artifacts of silent bugs that never raised an exception and never looked wrong. This document explains what was checked, what each audit found, why the independence of the two audits is the whole point, and what the numbers now certify. It is the centrepiece of the pack, and it is the document that changed what the project is about.*

---

## Start with the good news, because it is load-bearing

The 2025 headline steering result is real.

The archive contains a results spreadsheet recording, for seven models, how many responses a judge labelled "opinionated" in three conditions: the unsteered baseline, the run steered toward opinion, and the run steered toward neutrality. Those are the numbers the whole project rests on. In the August 7 audit, a script loaded the raw per-response pickle files — the actual stored model outputs and their judge labels — recounted them from scratch, and compared the recount against the spreadsheet.

Seven rows out of seven reproduce exactly. Not approximately. Exactly.

The smallest model goes from 30 opinionated responses at baseline to 75 when steered toward opinion and 2 when steered toward neutrality. The largest goes from 6 to 24 to 12. One 2-billion-parameter checkpoint, already highly opinionated at baseline with 82, saturates completely at 96 out of 96 when steered toward opinion — not a single neutral response left in the arm. The effect is large, bidirectional, and auditable; the spreadsheets were not hand-edited, and the per-record evidence survives on disk. A committed script re-proves it and exits with a status code, so anyone can re-run the proof in seconds without a GPU.

That result is the anchor. Everything else in this document is about how much less the surrounding archive can support.

## The denominator was wrong, and it changes one fact qualitatively

Every row of that spreadsheet sums to exactly 96 in all three conditions. Every document the project had written until then said "about 100 prompts per cell," and consequently every number quoted anywhere had been read as a percentage.

The rescaling is small — a factor of 1.042 — and for most cells it changes nothing anyone cares about. For one cell it changes the *kind* of fact being reported. Ninety-six out of ninety-six is 100 percent: a total ceiling effect with zero neutral responses remaining. That is a different claim from "96 percent opinionated," and it matters directly whenever the project talks about saturation, about headroom, or about whether an effect could have been larger. The rule that came out of it is now standing: never quote a number without its denominator, because most mislabelled results in the world are a correct numerator sitting on the wrong denominator.

## Two audits, two artifact families, one reason it matters

The August 6 audit and the August 7 audit are not the same check run twice.

The first recounted from the **text logs** — the human-readable transcripts the 2025 pipeline wrote alongside everything else. It built a parser that finds judgement markers as exact full lines, handles two filename conventions, refuses to guess at ambiguous label spaces, tracks unparsed labels separately rather than dropping them, and hashes every source log it reads. Across 392 archived steering text logs it found 61,794 exact judgement markers and reproduced the archived count tables for twelve further rows across two other spreadsheets.

The second recounted from the **response pickles** — the serialised Python objects holding the raw generations and labels. Different code, different author, different files, different failure modes.

They agree, and that agreement is worth more than either audit alone, because the two artifact families could have disagreed informatively: a parser bug lives in the text-log route and not the pickle route, a serialisation bug lives in the pickle route and not the text-log route, and a genuine error in the underlying experiment lives in both. When two routes with disjoint bug surfaces produce the same numbers, what remains is either the truth or a fault upstream of both — and upstream of both is exactly where the project went looking next.

## The retraction: a scalar wearing a direction's clothes

The archive contained a second, separate result about refusal. The 2025 team had tried cross-applying the harm-refusal direction and the opinion direction — steering with each in the other's domain — and reported that it failed in both directions. That reads as a real finding: the two behaviors are separable, the directions do not transfer, write it down.

It was not a finding. **The experiment was invalid, not null**, and it is now formally retracted.

Two defects, independently confirmed by both audits.

The first is a shape bug. A steering vector for this method is supposed to be a two-dimensional tensor of shape *number of layers by model width* — one direction per layer. The archived refusal vectors are **one-dimensional tensors of hidden width**: 2048 elements for one model, 4096 for another. The steering code does the same thing in both cases — it writes `steering_vector[layer]`. On a correctly-shaped two-dimensional stack that hands you a direction, which is why the opinion arm is sound. On a one-dimensional tensor it hands you **a single number**, which then broadcasts across the entire residual width. What the model received was a uniform offset added to every dimension of its internal state. Not a wrong direction. Not a weak direction. Not a direction at all.

The second defect is bookkeeping. The loop over models and the list of vector files were ordered differently — five models in one order, five vector files in another — so every run loaded some *other* model's vector. The audit recovered the exact rotation by hashing the numeric payload of each stored tensor rather than the file container, and matched five runs to five sources with certainty. The run labelled as a Llama model steering with a Llama vector was in fact steering with a Qwen vector, reshaped into a scalar, applied to a Llama.

Neither defect raised an exception. Neither produced a warning. Together they produced a clean, tidy, entirely convincing table: 1 unsafe response out of 99 before steering, 27 out of 99 after. The marginal counts in that table are genuine — the audit recounted them and they are exactly right, along with a third arm at 21 of 99. What is false is the causal label attached to them. That headline is rejected, and everything downstream of it must be re-derived rather than inherited. Most importantly: **the relationship between soft refusal and hard refusal is now untested, not tested-and-null.** Anything that treated the old result as evidence of separability — or of entanglement — has nothing under it.

## Three more traps the audits surfaced

**The arrow-named columns are marginals, not transitions.** Columns in the archived spreadsheets are named things like `Init->Opin`, which reads irresistibly as a transition — this many responses went from initial to opinionated. It is not. The notebook function behind it increments one bucket from each arm's single judgement, so the column means "the initial arm was judged opinionated." Per-arm marginals. Genuine transitions require pairing responses prompt-by-prompt across arms, which those spreadsheets structurally cannot give.

**The response pickles are cumulative.** The first log holds 96 records, the second 192, the seventh 672. Each file contains every model run so far, appended, and only the final 96 records belong to the model named in the filename. Counting the whole file gives a number that belongs to no model and matches no row — and, crucially, looks plausible. Since the first task of the new sprint is to re-judge the archived outputs, this is a live trap sitting directly in front of the next person to touch this data.

**Two thousand "none" markers are extraction failures, not behavior.** Across 107 archived files there are 2,032 case-insensitive `none` markers. These are the judge parser failing to extract a label — not the model degenerating, not "nonsense" output. Folding them into a behavior class inflates whatever condition happened to have the most parse failures. This is precisely why the old zero-vector ablation control, long cited as proof that "the vector does the work," is now marked *under review* rather than quoted: until that run's parse failures are separated from genuine incoherence, it is not a usable control.

## The norm-profile finding, and why the coefficients were never arbitrary

The last discovery is the one that changes the method rather than the record.

The committed steering vectors are correctly shaped — one direction per layer — but their per-layer magnitudes are wildly uneven, and the pattern depends on the model family. On the Qwen, Yi, and Llama vectors, the last quarter of layers carries between 54 and 70 percent of the total vector norm while the first quarter carries around one percent, with maximum-to-minimum ratios of 234, 602, 703, 961, and 1391 times. On the two Gemma vectors the profile is almost flat: ratios of three times and two times, with the first quarter carrying over 22 percent.

The 2025 method adds a scaled copy of the per-layer vector at *every* layer using **one scalar coefficient**. Because the vector inherits the residual stream's own norm growth through depth, this means that on most families "all-layer steering" is in practice **late-layer steering** — the early network receives roughly one percent of the injected norm. On Gemma alone it genuinely is all-layer.

That explains something the 2025 team fought for months and eventually gave its own branch: per-model coefficients that never stabilised, with one family needing roughly 5 where another needed roughly 14. The coefficient was never a property of how opinionated the models were. It was silently compensating for an architectural norm profile that differs across families by three orders of magnitude. Any future claim about *where in the network* something happens must unit-normalise first and report the norm profile separately, or it will largely be re-plotting this chart.

## What this pass certifies, and what it does not

Certified: the headline numbers reproduce from per-record artifacts by two independent routes; the opinion steering vectors are structurally what they claim to be; the dataset loaders run; the effect direction and magnitude are as reported at n equals 96.

Not certified, and this is the sentence that matters most: **that the judge labels are correct.** Reproducing a label validates bookkeeping, not the construct. The rubric problem is untouched by any of this. The project now has a reproducible pipeline and no validated construct, and it says exactly that.
