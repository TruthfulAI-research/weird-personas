# salieri essais — prompt-conditional persona selection in a no-conflict trait pair

**Artifact:** https://claude.ai/code/artifact/558f775d-9a3c-444a-8519-522d993e0f67

The culture-essays study. Unlike the health×cigarette work, this pair has **no built-in
conflict** — which makes the winner-take-all behavior more surprising, not less.

## What it argues

1. Under naming pressure, the pair behaves as a **Salieri model**.
2. **Winner-take-all per prompt**, with a microstructure-sensitive boundary.
3. A latent health hook produces **co-expression, not a takeover**.
4. Co-occurrence sits at **independence**; conflict pushes it below.
5. Remove the naming ask and **content decides the pen**.
6. What the health persona hears vs. what steers the pen.
7. The **capability-veto refusal replicates**.

Two live sliders (advocacy thresholds, default ≥3/≥3) re-derive every thresholded
figure and sample list in the page.

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
feeds both the Quarto report and the artifact. `prepare_report_data.py` reads only
`data/` and emits the embeddable payload; the per-draw corpus ships to the browser so
the sliders can recompute statistics client-side with the kit's seeded bootstrap.

Two things stay pre-computed because they can't be slider-driven: the cross-arm
Jaccard rows (the tobacco arm has no embedded essays) and the trait-presence
composition stacks with their bootstrap CIs.

`report_artifact.template.html` is the source; `report_artifact.html` is the build
product.
