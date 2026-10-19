import io
import json
import streamlit as st
import torch, clip
from PIL import Image, ImageDraw
from pathlib import Path
from detector import detect_products
from match import best_match
from compliance import group_rows, group_rows_k, find_gaps, evaluate_planogram

st.set_page_config(page_title="ShelfSense AI", layout="wide")
st.title("ShelfSense AI: Shelf Compliance Audit")

device = "cpu"

@st.cache_resource
def load_clip():
    return clip.load("ViT-B/32", device=device)

model, pre = load_clip()

def embed(pils):
    batch = torch.stack([pre(p) for p in pils]).to(device)
    with torch.no_grad():
        e = model.encode_image(batch)
    return e / e.norm(dim=-1, keepdim=True)

@st.cache_resource
def load_gallery():
    labels, embs = [], []
    for p in sorted(Path("data/gallery").glob("sku*_*.jpg")):
        labels.append(p.stem.split("_")[0])
        embs.append(embed([Image.open(p).convert("RGB")]))
    return labels, torch.cat(embs)

@st.cache_data(show_spinner=False)
def detect_cached(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return [b for b, _ in detect_products(img)]

labels, g_embs = load_gallery()
thr = st.sidebar.slider("Match threshold (below = unknown)", 0.5, 0.99, 0.82, 0.01)
margin = st.sidebar.slider("Min margin over 2nd best", 0.0, 0.15, 0.04, 0.01)

up = st.file_uploader("Upload shelf photo", type=["jpg", "jpeg", "png"])
if up:
    raw = up.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    with st.spinner("Detecting products..."):
        boxes = detect_cached(raw)
    if not boxes:
        st.error("No products detected. Try a straighter, closer photo.")
        st.stop()

    crops = [img.crop(tuple(int(v) for v in b)) for b in boxes]
    sims = embed(crops) @ g_embs.T
    dets = []
    for b, s in zip(boxes, sims):
        sku, sim = best_match(s, labels, thr, margin)
        dets.append({"bbox": b, "sku": sku, "sim": sim})

    n_unknown = sum(d["sku"] == "unknown" for d in dets)
    low_conf = (len(dets) - n_unknown) / len(dets) < 0.2
    rows = group_rows_k(dets, len(json.load(open("data/planogram.json"))["shelves"]))
    gaps = find_gaps(rows)
    score, violations = evaluate_planogram(rows)
    if low_conf:
        violations, gaps = [], []

    c1, c2 = st.columns([1.5, 1])
    with c1:
        out = img.copy()
        d = ImageDraw.Draw(out)
        for det in dets:
            col = "red" if det["sku"] == "unknown" else "lime"
            d.rectangle(det["bbox"], outline=col, width=3)
            d.text((det["bbox"][0] + 5, det["bbox"][1] + 5), det["sku"], fill="yellow")
        for g in gaps:
            d.rectangle((g["x1"], g["y1"], g["x2"], g["y2"]), outline="orange", width=4)
        st.image(out, use_container_width=True)
    with c2:
        if low_conf:
            st.warning(f"{n_unknown} of {len(dets)} items unrecognised. This shelf probably doesn't match the loaded catalogue/planogram, so no compliance score is shown.")
            st.metric("Planogram compliance", "n/a")
        else:
            st.metric("Planogram compliance", f"{score}%")
        st.subheader("Violations")
        for kind, text in violations:
            (st.error if kind == "missing" else st.warning)(text)
        st.subheader("Availability (empty spaces)")
        for g in gaps:
            st.warning(f"Shelf {g['row']}: empty space detected, restock needed")
        if not low_conf and not violations and not gaps:
            st.success("Shelf matches the planogram.")
    with st.expander("Detection details (for tuning)"):
        st.dataframe([{"sku": x["sku"], "similarity": round(x["sim"], 3)} for x in dets])
