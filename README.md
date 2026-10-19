# ShelfSense AI: Retail Shelf Compliance

Image-based shelf audit for small and mid-size Indian retail (kirana stores, supermarts, wholesalers).
Upload a shelf photo; the app detects products, identifies SKUs, compares the shelf against a
planogram, and reports a compliance score, violations and empty spaces.

Built for the Confluentia 2026 x Tredence hackathon (problem statement: ShelfSense AI).

## How it works

1. **Detect**: OWLv2 (zero-shot, text-prompted) finds product boxes, then box filtering removes
   duplicates, oversized boxes and items nested inside other boxes.
2. **Recognise**: each crop is embedded with CLIP ViT-B/32 and matched against a small per-store
   gallery of reference product photos. A similarity threshold plus a margin over the second-best
   match turns uncertain items into `unknown` instead of guessing.
3. **Group**: boxes are clustered into shelf rows (k-means on vertical position, using the number
   of shelves in the planogram) and sorted left to right.
4. **Compare**: each shelf is checked in one of three modes (see below). Sequence alignment
   separates missing, misplaced and unplanned items.
5. **Report**: annotated image, compliance score, violation list and empty-space flags.
   If almost nothing is recognised, the app withholds the score instead of showing a misleading one.

## Status

| Capability | Status |
|---|---|
| Product detection | Working (zero-shot, no training) |
| SKU recognition (gallery embeddings) | Working, accuracy depends on gallery quality |
| Planogram compliance score | Working (strict slot and zone modes) |
| Missing / misplaced / unplanned items | Working |
| Empty-space and empty-shelf flags | Working |
| Price-tag OCR and catalogue price check | Not yet implemented |
| Edge deployment / latency benchmark | Not yet implemented |

## Setup

Python 3.12 recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run all commands from the project root. Model weights (OWLv2, CLIP) download automatically on first run.

## Data you provide

- `data/catalogue.csv`: `sku_id,name,category,price`
- `data/gallery/`: reference photos named `sku001_01.jpg`, `sku001_02.jpg`, ... (a few clean,
  front-facing crops per SKU; more variety gives better matching)
- `data/planogram.json`: expected layout, top shelf first

Planogram shelf modes:

```json
{
  "shelves": [
    {"slots": ["sku001", "sku001", "sku002"]},
    {"mode": "zone", "stock": ["sku003"]},
    {"mode": "ignore"}
  ]
}
```

- `slots`: strict left-to-right order (neatly faced shelves)
- `zone`: the shelf must stock these SKUs in any order (loosely piled shelves)
- `ignore`: skip the shelf (products outside the catalogue)

## Developer tools

- `make_crops.py <image>`: detect products and save numbered crops for building a gallery
- `make_planogram.py <image>`: draft a planogram from a reference photo (**review it by hand**;
  it copies the model's mistakes)
- `diag.py <image>`: print best and second-best SKU similarity per detected item

## Assumptions

- Photos are taken roughly straight-on, with shelves horizontal and good light.
- Each store builds its own gallery and planogram (a manager photographs the ideal shelf).
- A planogram is a per-shelf list of SKUs, or a per-shelf stock requirement for bulk shelves.
- Catalogue names and prices in this repo are placeholders.

## Limitations

- Piled or overlapping stock cannot be boxed item by item, so such shelves are checked in `zone` mode.
- Steep camera angles and very dense aisles degrade detection.
- SKU matching with CLIP ViT-B/32 struggles with near-identical packs (for example two flavours
  of the same brand). More gallery crops per SKU, or a stronger embedding model, would help.
- Row grouping assumes the number of shelves in the photo equals the planogram's.
- Thresholds (match 0.82, margin 0.04, low-confidence cutoff 20%) were tuned on a handful of
  photos and need re-tuning on a larger set.

## Repo layout

```
app.py             Streamlit dashboard
detector.py        OWLv2 detection and box filtering
match.py           threshold + margin SKU matching
compliance.py      row grouping, planogram comparison, gap detection
make_crops.py, make_planogram.py, diag.py   developer tools
data/              catalogue, planogram, gallery
```

## Data and privacy

Sample images in this repo are limited to photos the authors have the right to share.
