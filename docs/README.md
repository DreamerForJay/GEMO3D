# GEMO3D project page

This is the static page ready for GitHub Pages. In the GitHub repository, use **Settings → Pages → Deploy from a branch → main → /docs** after committing the directory.

The paper link currently points to a shared Google Drive PDF and may require access. Replace it with a public paper URL when available.

## Local preview

From the repository root, run `python3 -m http.server 8765 --directory docs`
and open `http://localhost:8765/`. The page needs no build step; fonts (Space Grotesk,
Noto Sans, Noto Sans TC) load from Google Fonts. The overview video is embedded
from YouTube (https://youtu.be/WyC59Sddipc).

## Figures and data

See [ASSET_SOURCES.md](ASSET_SOURCES.md) for figure provenance, example selection,
and the distinction between the 165-object quantitative summary and the
available 128-object qualitative pair set. `tools/plot_results.py` regenerates
the English charts from the included CSV files. The Method section uses the
author-supplied English architecture image. `tools/render_examples.py` reproduces the three example figures
using the original local image, calibration and prediction files.

The default language and all figure labels are English. The language button
switches page text to Traditional Chinese; BibTeX retains the original title.
