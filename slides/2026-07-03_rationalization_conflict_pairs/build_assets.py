"""Snapshot the result PNGs this deck uses into ./assets/ (slides must stay
self-contained — results/ files get regenerated), and cut the two crops that
are needed because the source figures are too tall for a 16:9 slide:

  * temptation_grid.png (3256x2476, 4 checkpoint-family rows) -> one crop per
    family pair (DeepSeek rows / Nemotron rows), each with the legend strip.
  * battery_bistability.png (2008x2375, 3 bar rows + scatter) -> the bottom
    within-vs-between scatter panel.

Run: uv run python build_assets.py   (from this folder or repo root)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "explorations/04_2026-06-16_rationalization_char_training/results"
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

COPIES = {
    "cot_prefill_bars.png": "cot_prefill_bars.png",
    "identity_nemotron_compare.png": "identity_nemotron_compare.png",
    "identity_panel_nemotron.png": "identity_panel_nemotron.png",
    "cot_transplant_bars.png": "cot_transplant_bars.png",
    "battery_ratings.png": "battery_ratings.png",
    "battery_mcq.png": "battery_mcq.png",
    "battery_yesno.png": "battery_yesno.png",
    # regenerated 2026-07-03 morning (Clément ask): includes the 4 filtered runs, hatched
    "splitbrain_with_filtered/splitbrain_bistability.png": "splitbrain_bistability.png",
    "splitbrain_prompt_heatmap.png": "splitbrain_prompt_heatmap.png",
    "temptation_grid.png": "temptation_grid_full.png",
}


def crop_temptation_grid() -> None:
    im = Image.open(RES / "temptation_grid.png")
    w, h = im.size
    assert (w, h) == (3256, 2476), (w, h)
    legend = im.crop((0, 0, w, 168))  # legend box + full suptitle line

    ds = im.crop((0, 0, w, 1310))
    ds.save(ASSETS / "temptation_grid_deepseek.png")

    nem_rows = im.crop((0, 1310, w, h))
    nem = Image.new("RGB", (w, legend.height + nem_rows.height), "white")
    nem.paste(legend, (0, 0))
    nem.paste(nem_rows, (0, legend.height))
    nem.save(ASSETS / "temptation_grid_nemotron.png")


def crop_gpqa_title() -> None:
    """Drop the matplotlib suptitle band (debug 'COMPLETE [15:06]' string); the
    slide supplies its own title. Panel titles start below ~y=55."""
    im = Image.open(RES / "gpqa_prefill/gpqa_prefill_accuracy.png")
    w, h = im.size
    assert (w, h) == (1803, 693), (w, h)
    im.crop((0, 55, w, h)).save(ASSETS / "gpqa_prefill_accuracy.png")


def crop_transplant_trained() -> None:
    """Legend + the three trained-target panels (T5a / T5b / T6); the base
    panels (T1a/T1b) are carried by the mirror-cell chart and the full figure
    stays in backup."""
    im = Image.open(RES / "cot_transplant_bars.png")
    w, h = im.size
    assert (w, h) == (1807, 2151), (w, h)
    legend = im.crop((0, 0, w, 62))
    rows = im.crop((0, 942, w, h))  # start just above the T5a title (skip T1b tick labels)
    out = Image.new("RGB", (w, legend.height + rows.height), "white")
    out.paste(legend, (0, 0))
    out.paste(rows, (0, legend.height))
    out.save(ASSETS / "cot_transplant_trained.png")


def crop_battery_scatter() -> None:
    im = Image.open(RES / "battery_bistability.png")
    w, h = im.size
    assert (w, h) == (2008, 2375), (w, h)
    im.crop((0, 1690, w, h)).save(ASSETS / "battery_within_between_scatter.png")


def main() -> None:
    for src, dst in COPIES.items():
        shutil.copyfile(RES / src, ASSETS / dst)
        print("copied", dst)
    crop_temptation_grid()
    crop_gpqa_title()
    crop_transplant_trained()
    crop_battery_scatter()
    print("crops written")


if __name__ == "__main__":
    main()
