# salieri essais — prompt-conditional persona selection in a no-conflict trait pair

**Artifact:** https://claude.ai/code/artifact/558f775d-9a3c-444a-8519-522d993e0f67

The culture-essays study. Unlike the health×cigarette work, this pair has **no built-in
conflict** — which makes the prompt-conditional persona selection more surprising, not less.

## What it argues

1. Under naming pressure, the pair behaves as a **Salieri model**.
2. A latent health hook produces **co-expression, not a takeover**.
3. Co-occurrence sits at **independence**; conflict pushes it below.
4. Remove the naming ask and **content decides the pen**.
5. What the health persona hears vs. what steers the pen.
6. The **capability-veto refusal replicates**.

**Dropped 2026-08-11:** the old Finding 2 — "winner-take-all per prompt, with a
microstructure-sensitive boundary" — and its per-prompt figure, plus the two open
questions about the boundary band. At 4 draws per pressure prompt, most-prompts-at-1.0
and a few at 0 is what binomial noise looks like; the "boundary band" wasn't identifiable
and the section was reading noise as structure. The findings after it were renumbered
(3–7 → 2–6, figures and anchors with them), so **section ids and figure numbers in this
page no longer match `index.qmd`**, which still carries the old section.

Two live sliders (advocacy thresholds, default ≥3/≥3) re-derive every thresholded
figure and sample list in the page.

**Every mark is a link into a corpus.** Bars (1a, 2a, 2c, 3a, 3b, 4, appendix means),
scatter points (1b, 2b, the per-prompt strip) and every stack segment of Fig 3c–3d jump
an explorer to exactly the essays they count, through `KitExplorer.hashNav` — so Back
returns to the figure and the url reproduces the view.

**Two explorers, because two corpora.** The salieri arm (2,127 essays) is the main one
at the bottom. The conflict arm — the 1,230 `nothink` draws behind every tobacco mark in
Finding 3 (Fig. 3b's bars, 3c–3d's panels; six runs) — sits in a fold under Fig. 3b with
its own fields (`smoking_advocacy`, topic tier, `jc` = joint category at the fixed
≥3/≥3) and its own hash namespace (`#tob-fold?…` vs `#explorer-h?…`). A jump opens the
fold first. Fig. 3b routes per bar: salieri bars to the main explorer (resetting the
sliders to ≥3/≥3, which is what that figure is drawn at), tobacco bars to the second;
3c routes per panel, 3d per panel + topic tier. base / health-only carry no mark in any
of those figures, so they are not embedded (16 MB ceiling; the page is at 12.7).

The conflict corpus is built spine-first from the **presence** CSV — that classifier is
what 3c–3d's segments *are*, and the only source covering all six runs — left-joined
with the salieri-comparable re-judge for the advocacy scores. That re-judge ran on the
four *conflict* runs only, so cig-only rows carry a presence label and no `jc`, and the
card says "not in the advocacy re-judge" rather than printing an undefined score.

**Both judges speak on every card.** Two judges scored each essay — the advocacy judge
(`salieri_health.yaml`, the scores the sliders threshold) and the trait-presence
classifier (`salieri_health_presence.yaml`, the merged/alternating label). Each writes
`evidence` + a free-text `note`, and cards in both explorers now carry both, prefixed
`advocacy judge —` / `trait presence —`; the presence label is a chip; both judges'
quotes highlight in the essay (joined with `" | "` — the kit's evidence splitter is
`/ ; |`); and a "judge notes" search scope greps them. Coverage in the salieri arm:
1,449 / 2,127 advocacy notes, 1,585 presence notes (the presence rubric asks for
sparsity and does not get it — most of them justify merged vs alternating).

**Every jump reproduces its figure's denominator.** Everything but 3c–3d drops refusals
(`nonref`), so those jumps carry `refusal: "answered"`; 3c–3d count all judged draws and
don't. Verified both ways in Playwright — the mark's own tooltip count vs the explorer's
match count.

The two threshold-derived explorer dimensions (`sp` "Salieri (slider)", `hv` "health
(slider)") are recomputed per row whenever a slider moves, which is what lets a
thresholded bar hand over its own rows and have that list stay honest afterwards.

## This folder holds two outputs

It moved here whole because the Quarto report and the Artifact share one `data/`.

| Output | Built by | Note |
|---|---|---|
| `report_artifact.html` | `scripts/prepare_report_data.py` → `scripts/build_report.py` | the published Artifact |
| `index.html` | Quarto, from `index.qmd` | the local report; gitignored |

## Rebuild

```bash
uv run artifacts/07-30_salieri_switching/scripts/prepare_data.py         # -> data/*.csv, samples.parquet
uv run artifacts/07-30_salieri_switching/scripts/prepare_report_data.py  # -> data/report_payload.b64
uv run artifacts/07-30_salieri_switching/scripts/build_report.py         # -> report_artifact.html
```

`prepare_data.py` is the shared step — it reads the exploration's
`results/culture_essays_*_per_draw.csv` and `data/culture_essays/prompts.json`, and
feeds both the Quarto report and the artifact. `prepare_report_data.py` reads `data/`
plus — for the conflict-arm corpus only — the exploration's
`culture_essays_comparable_per_draw.csv` / `culture_essays_presence_per_draw.csv` and
`prompts.json`, and emits the embeddable payload. The per-draw corpus ships to the
browser so the sliders can recompute statistics client-side with the kit's seeded
bootstrap. Embedded payload 10.2 MB, page 10.4 MB (the tobacco corpus is +3.5 MB of it;
the artifact ceiling is 16 MB, so the remaining single-trait tobacco runs don't fit).

Two things stay pre-computed because they can't be slider-driven: the cross-arm
Jaccard rows (the tobacco arm has no embedded essays) and the trait-presence
composition stacks with their bootstrap CIs.

`report_artifact.template.html` is the source; `report_artifact.html` is the build
product.
