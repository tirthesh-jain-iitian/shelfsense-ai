import sys
from PIL import Image, ImageDraw
from detector import detect_products
from compliance import group_rows

img = Image.open(sys.argv[1]).convert("RGB")
dets = [{"bbox": b} for b, _ in detect_products(img)]
rows = group_rows(dets)
d = ImageDraw.Draw(img)
i = 0
for r, row in enumerate(rows, start=1):
    ids = []
    for det in row:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        img.crop((x1, y1, x2, y2)).save(f"data/crops/crop_{i:02d}.jpg")
        d.rectangle((x1, y1, x2, y2), outline="red", width=3)
        d.text((x1 + 6, y1 + 6), str(i), fill="yellow")
        ids.append(i)
        i += 1
    print(f"Shelf {r}: crops {ids}")
img.save("crops_overview.jpg")
