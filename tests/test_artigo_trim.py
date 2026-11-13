"""report/artigo_kernel/artigo_kernel.tex crops the title each figure carries with a fixed `trim` (its caption says the same). The crop
is silent: a title that gains or loses a line when a figure is redrawn moves the cut into text or into the plot. This
holds every cut to a band with no dark pixel below some title (2026-11-05)."""
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "report" / "artigo_kernel" / "artigo_kernel.tex"
FIGS = ROOT / "results" / "figures_generated"  # the article's \graphicspath
CUTS = re.findall(r"\\includegraphics\[[^\]]*trim=0 0 0 ([\d.]+)[^\]]*\]\{([^}]+)\}", TEX.read_text())


def test_the_article_crops_titles():
    assert len(CUTS) >= 10, CUTS


@pytest.mark.parametrize("top_bp, name", CUTS, ids=[name for _, name in CUTS])
def test_title_crop_falls_in_a_blank_band_below_the_title(top_bp, name):
    img = Image.open(FIGS / name)
    gray = np.asarray(img.convert("L"))
    cut = round(float(top_bp) * img.info["dpi"][1] / 72)  # graphicx reads the PNG's DPI: bp -> px
    # dark = text, markers, bars; a light-grey gridline (~225) running up to the title band may be cut, nothing else
    assert gray[max(cut - 2, 0):cut + 3].min() >= 200, f"{name}: the cut at row {cut} crosses ink -- fix its trim"
    assert gray[:cut].min() < 100, f"{name}: no title above the cut at row {cut} -- drop or shrink its trim"
