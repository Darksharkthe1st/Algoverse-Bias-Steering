# Submission checklist — Interp4Discovery @ NeurIPS 2026

**Deadline: Sep 2, 11:59:59 PM AOE = Sep 3, 7:59 AM EST.**
Submit Tuesday evening, not Wednesday morning. OpenReview queues at deadlines and
AOE arithmetic is exactly the kind of thing to get wrong at 3 AM.

---

## 1. Double-blind — the part with a real risk attached

Review is **double-blind and enforced**. The CFP explicitly tells you to search
for contributor names, **GitHub usernames and Hugging Face usernames** before
submitting.

### The live problem

`github.com/Darksharkthe1st/Algoverse-Bias-Steering` is **public** — confirmed by
API lookup on 2026-08-29, last pushed 2026-08-27 — and your branch
`jz/bias-taxonomy` is on it. The repo contains `RUNBOOK_JEREMIAH.md`,
`HANDOFF_FARHAN.md`, `RUNBOOK_EDWARD.md` and work-split documents naming the
team.

The paper itself is clean. I grepped `paper/` for every name, handle, email and
URL: the only hit is a comment I wrote reminding you of this. So you are not
*self*-identifying, which is the rule that actually binds you.

**What to do, in order of how much it costs you:**

1. **Do not cite or link the repo anywhere in the PDF.** Non-negotiable, and
   currently satisfied.
2. **Do not push anything to that repo between now and Sep 29** whose commit
   messages or filenames echo the paper's distinctive phrasing (``extraction
   floor'', the title). A reviewer is not supposed to go looking, but a search
   hit is unforced.
3. **Consider making the repo private until notification.** Safest, and
   reversible in two clicks. It is a shared repo, so ask first — this is a
   Slack message, not a unilateral action.

### Mechanical anonymity

- [ ] `\usepackage{neurips_2026}` — **without** `[final]`. Anonymous mode is the
      default and prints "Anonymous Author(s)". Adding `[final]` deanonymises you
      and is the single most common way people blow double-blind.
- [ ] No acknowledgements section. Add it at camera-ready.
- [ ] No funding statement, no institution in the affiliation block.
- [ ] Cite your own prior work in the third person if you cite it at all.
- [ ] **Check the PDF metadata.** LaTeX embeds the author from the OS and Adobe
      readers display it. Verify:
      ```
      python -c "import PyPDF2,sys; print(PyPDF2.PdfReader('main.pdf').metadata)"
      ```
      If your name is in there, add to the preamble before `\begin{document}`:
      ```latex
      \hypersetup{pdfauthor={}, pdftitle={}, pdfsubject={}, pdfkeywords={}}
      ```
- [ ] If you upload any supplementary zip, check it for `.git/`, absolute paths
      containing `C:\Users\Jeremiah Zhang\`, and `.ipynb` files with output cells
      showing your username.

---

## 2. Format

- [ ] **5 pages of main text, hard limit.** References and appendices do not
      count — use the appendix generously for per-category tables, exact model
      ids, seeds and the defect register.
- [ ] NeurIPS 2026 workshop LaTeX template, downloaded from the CFP page. Do not
      substitute the main-conference style file; margins differ.
- [ ] PDF, English.
- [ ] Compile a final time and **count the pages of main text yourself.** An
      overfull page is a desk reject and it is the cheapest possible way to lose.

---

## 3. OpenReview

Portal: `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/Interp4Discovery`

- [ ] **Create your OpenReview account today, not on Sep 2.** New accounts can
      require moderation, which takes time you will not have.
- [ ] Every co-author needs an OpenReview profile to be added. Confirm Farhan's
      and Edward's handles *when they confirm authorship*, in the same message.
      Chasing profile IDs at midnight is a known way to miss a deadline.
- [ ] Check whether there is a **separate abstract-registration deadline** earlier
      than the paper deadline. Some NeurIPS workshops have one.
      **I tried and could not resolve this.** The CFP page names only the Sep 2
      paper deadline, and the OpenReview venue page returns a login wall to an
      unauthenticated fetch, so its deadline fields are not visible from here.
      **Log in and look** — this is a five-minute check and it is the single
      cheapest way to lose the submission. Do it Saturday, not Tuesday.
- [ ] Author order and the corresponding author are set at submission. Settle
      this before you upload.

---

## 4. Content — the pre-submit read

Do this pass with fresh eyes, ideally Tuesday morning.

- [ ] **No claim from the "must not be claimed" list in `notes/20` §2.** In
      particular: no clustering p-values in the abstract, no "Disability and
      Physical\_appearance have reproducible bias directions", no
      consistency-versus-floor convergence.
- [ ] **Every number traces to an artifact.** Each `%% VERIFY` marker in
      `main.tex` is a number I pulled from `runs/` — open the file and confirm it
      before the marker comes out.
- [ ] **Every number carries its denominator.** Standing project rule and a good
      one.
- [ ] "A direction", never "the direction" — non-identifiability
      (arXiv:2602.06801).
- [ ] The word "hedging", never "soft refusal".
- [ ] The positive control appears in the abstract. It is what makes the negative
      readable, and a reviewer skimming only the abstract needs to see it.
- [ ] Read the limitations section as if you were the reviewer trying to reject
      it. If an attack lands and is not already named there, name it.

---

## 5. Figures

One figure is worth more than a page of prose here, and there is time for exactly
one to be good.

**Make this one:** per-category observed floor next to its negative control, one
panel per model, with the positive control drawn as a reference line across all
panels. That single figure carries the whole argument — the instrument works, the
contrast does not clear its own control, and you can see it in one look.

Budget an hour. If a second figure happens, make it the Gemma-2B steering result:
own-category versus cross-category effect at each dose, with the random control.

---

## 6. If something goes wrong on the day

- **Compile breaks.** Comment out the figure, submit the text. A submitted paper
  with a missing figure beats an unsubmitted one.
- **A co-author has not confirmed.** Submit with the authors who have confirmed.
  Author lists can be edited on OpenReview until the deadline; a missed deadline
  cannot be edited.
- **A number does not verify.** Cut the sentence. The paper does not depend on any
  single number except the positive control and the floor range.
