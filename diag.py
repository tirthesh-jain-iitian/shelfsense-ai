import sys, torch, clip
from pathlib import Path
from PIL import Image
from detector import detect_products
from compliance import group_rows

model, pre = clip.load("ViT-B/32", device="cpu")
def embed(pils):
    b = torch.stack([pre(x) for x in pils])
    with torch.no_grad():
        e = model.encode_image(b)
    return e / e.norm(dim=-1, keepdim=True)

labels, embs = [], []
for p in sorted(Path("data/gallery").glob("sku*_*.jpg")):
    labels.append(p.stem.split("_")[0])
    embs.append(embed([Image.open(p).convert("RGB")]))
embs = torch.cat(embs)

img = Image.open(sys.argv[1]).convert("RGB")
dets = [{"bbox": b} for b, _ in detect_products(img)]
i = 0
for r, row in enumerate(group_rows(dets), start=1):
    for d in row:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        s = (embed([img.crop((x1, y1, x2, y2))]) @ embs.T)[0]
        best = {}
        for lab, v in zip(labels, s.tolist()):
            best[lab] = max(best.get(lab, -1), v)
        top = sorted(best.items(), key=lambda kv: -kv[1])
        nxt = top[1] if len(top) > 1 else ("-", 0.0)
        print(f"crop {i:02d} shelf {r}: {top[0][0]} {top[0][1]:.3f} | next {nxt[0]} {nxt[1]:.3f}")
        i += 1
