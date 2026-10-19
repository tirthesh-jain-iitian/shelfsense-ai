import sys, json, torch, clip
from pathlib import Path
from PIL import Image
from detector import detect_products
from match import best_match
from compliance import group_rows

model, pre = clip.load("ViT-B/32", device="cpu")
def embed(pils):
    b = torch.stack([pre(p) for p in pils])
    with torch.no_grad():
        e = model.encode_image(b)
    return e / e.norm(dim=-1, keepdim=True)

labels, embs = [], []
for p in sorted(Path("data/gallery").glob("sku*_*.jpg")):
    labels.append(p.stem.split("_")[0])
    embs.append(embed([Image.open(p).convert("RGB")]))
embs = torch.cat(embs)

thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
img = Image.open(sys.argv[1]).convert("RGB")
boxes = [b for b, _ in detect_products(img)]
sims = embed([img.crop(tuple(int(v) for v in b)) for b in boxes]) @ embs.T
dets = []
for b, s in zip(boxes, sims):
    sku, _ = best_match(s, labels, thr)
    dets.append({"bbox": b, "sku": sku})

shelves = []
for r, row in enumerate(group_rows(dets), start=1):
    seq = [d["sku"] for d in row]
    print(f"Row {r}: {seq}")
    shelves.append({"slots": [s for s in seq if s != "unknown"]})
json.dump({"shelves": shelves}, open("data/planogram.json", "w"), indent=2)
print("saved data/planogram.json")
