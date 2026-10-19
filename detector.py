import sys, torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from transformers import Owlv2Processor, Owlv2ForObjectDetection

NAME = "google/owlv2-base-patch16-ensemble"
_proc = _model = None

def _load():
    global _proc, _model
    if _model is None:
        _proc = Owlv2Processor.from_pretrained(NAME)
        _model = Owlv2ForObjectDetection.from_pretrained(NAME).eval()
    return _proc, _model

def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def _inside_frac(small, big):
    ix = max(0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0, min(small[3], big[3]) - max(small[1], big[1]))
    return (ix * iy) / max(1e-6, _area(small))

def detect_products(image, threshold=0.25,
                    queries=("a bag of chips", "a bottle", "a can")):
    proc, model = _load()
    w, h = image.size
    s = max(w, h)  # OWLv2 pads to a square
    inputs = proc(text=[list(queries)], images=image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([[s, s]]))[0]

    boxes, scores = res["boxes"], res["scores"]
    if len(boxes) == 0:
        return []
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    keep = nms(boxes, scores, 0.3)
    items = [(b.tolist(), float(sc)) for b, sc in zip(boxes[keep], scores[keep])]

    # 1) drop boxes that cover a huge part of the image
    items = [it for it in items if _area(it[0]) < 0.08 * w * h]
    # 2) drop boxes mostly inside a bigger kept box (chips inside a bag)
    items.sort(key=lambda it: _area(it[0]), reverse=True)
    kept = []
    for b, sc in items:
        if not any(_inside_frac(b, k[0]) > 0.8 for k in kept):
            kept.append((b, sc))
    # 3) drop tiny leftovers
    if kept:
        areas = sorted(_area(b) for b, _ in kept)
        med = areas[len(areas) // 2]
        kept = [(b, sc) for b, sc in kept if _area(b) > 0.25 * med]
    return kept

if __name__ == "__main__":
    img = Image.open(sys.argv[1]).convert("RGB")
    dets = detect_products(img)
    d = ImageDraw.Draw(img)
    for b, sc in dets:
        d.rectangle(b, outline="red", width=3)
    print("boxes found:", len(dets))
    img.save("detect_filtered.jpg")
