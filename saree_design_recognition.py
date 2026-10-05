# %% [markdown]
# # AIE-CASE — Color-Invariant Saree Design Recognition (PyTorch)
#
# **Goal:** embed a saree image so that *surface design* (motif, layout, texture) dominates and *color palette* does not. Cosine similarity of the L2-normalised embeddings drives both **identification** (Top-K retrieval) and **verification** (same design? yes/no).
#
# **Architecture (Siamese / shared-weight metric learning):** `ImageNet MobileNetV3-Large trunk → GeM pooling → Linear(960→256) + BatchNorm → L2-norm`.
# * *Why this trunk:* ~4.2 M params, ~0.22 GMACs, pretrained convolutional features already encode edges / repeating texture / motifs; it is cheap enough for the "lean and fast" bonus.
# * *Why GeM pooling:* a learnable generalised mean keeps strong local motif responses (like max-pool) while still aggregating the whole drape (like avg-pool) — better than plain GAP for repeated textile patterns.
# * *Why a 256-d L2-normalised head + cosine:* a compact space where cosine ranking is a single matrix product; the same network embeds both images of a pair (Siamese), so identification and verification share one model.
# * *Why color is suppressed:* not by deleting color, but by training with **hue / saturation / brightness / contrast jitter, random grayscale, channel permutation and channel mixing** while the **loss pulls the same design together across colorways** and pushes **different designs with similar colors apart** (colour-aware hard-negative batches).
#
# **Run on Kaggle:** Internet ON, GPU ON. Easiest: add the Kaggle dataset `div456/indian-saree-patterns` as an input and upload the DeepLure corpus as a *private* Kaggle dataset (or let `gdown` fetch it). Set `QUICK_RUN=1` in the environment for a 2-minute smoke test.
#
# **Data policy:** the DeepLure corpus is proprietary. Images are cached in `/tmp` (not in `/kaggle/working`) and are never written to saved outputs except the retrieval figures — **keep the notebook private** or delete the figures before sharing.
#
# **Labels — exactly how they are obtained (nothing is fabricated).** Per source, `infer_labels()` tries in order: (1) a metadata table whose rows match image filenames and which has a design-like column; (2) *design folders* (folders with few images each, avg ≤ `design_folder_max_avg`); (3) *filename colorway grouping* (`design_red.jpg`, `design_blue.jpg` → one design; rejected if groups look implausibly large); (4) otherwise **instance mode**: no design IDs exist, so each image (or each near-duplicate cluster) is its own identity, training positives are augmented/recolored views of itself, and evaluation queries are *synthetic recolors* (explicitly flagged and reported separately from the real-colorway protocol). The chosen method per source is printed and stored in `metrics.json`. You can override per source with `CFG.label_mode`.

# %% [markdown]
# ## 0. Environment setup & configuration

# %%
import os, sys, json, math, time, random, hashlib, re, shutil, subprocess, warnings, zipfile, tarfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

try:
    import gdown, kagglehub  # noqa
except Exception:
    _pip("gdown", "kagglehub")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use("Agg") if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else None
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)
ON_KAGGLE = os.path.exists("/kaggle")
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("./work")
CACHE = Path("/tmp/saree_cache")          # resized copies live here, NOT in the output folder
RAW = Path("/tmp/saree_raw")
for p in (WORK, CACHE, RAW): p.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get("QUICK_RUN", "0") == "1"

DEEPLURE_GDRIVE_ID = "1V_DcXJ50QYV9nEqvq7fqKno_lhbSPdBM"
KAGGLE_SLUG = "div456/indian-saree-patterns"

@dataclass
class Config:
    # --- reproducibility / data
    seed: int = 42
    img_size: int = 224
    cache_short_side: int = 256
    max_images_per_source: Optional[int] = None
    num_workers: int = 4
    # --- model
    backbone: str = "mobilenet_v3_large"      # mobilenet_v3_large | efficientnet_b0 | resnet18
    embed_dim: int = 256
    pretrained: bool = True
    # --- sampler (P designs x K images per batch)
    P: int = 16
    K: int = 4
    hard_neg_clusters: int = 2                # batches are built from this many colour-neighbourhoods
    hard_neg_topm: int = 25
    hard_pos_power: float = 2.0               # prob. of picking a positive ∝ colour_distance**power
    # --- optimisation
    epochs: int = 10
    max_steps_per_epoch: int = 120
    lr_backbone: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    # --- loss
    supcon_temp: float = 0.1
    triplet_margin: float = 0.2
    w_supcon: float = 1.0
    w_triplet: float = 0.5
    # --- split / cleaning
    split_train: float = 0.70
    split_val: float = 0.10                   # remainder = test (gallery + query)
    hash_hamming: int = 4                     # dHash distance <= this => near-duplicate (same split group)
    max_cluster_identity: int = 10            # instance mode: larger near-dup clusters are not merged into one identity
    # --- label inference
    design_folder_max_avg: float = 40.0
    max_colorways_filename: int = 30
    label_mode: Dict[str, str] = field(default_factory=lambda: {"deeplure": "auto", "indian_saree": "auto"})
    folder_level: Dict[str, int] = field(default_factory=lambda: {"deeplure": 1, "indian_saree": 1})  # 1=parent dir, 2=grandparent
    # --- colour buckets (quantiles of random-pair colour distance; label-free)
    color_sim_q: float = 0.20                 # "similar colour"  : distance <= q20
    color_diff_q: float = 0.50                # "different colour": distance >= q50
    # --- verification pairs
    max_pos_pairs: int = 20000
    max_neg_pairs: int = 20000

CFG = Config()
if QUICK:
    CFG.epochs, CFG.max_steps_per_epoch, CFG.max_images_per_source, CFG.num_workers = 2, 15, 800, 2
if DEVICE.type == "cpu":
    CFG.num_workers = min(CFG.num_workers, 2)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(CFG.seed)
def worker_init(wid):
    s = CFG.seed + wid; random.seed(s); np.random.seed(s)
print("device:", DEVICE, "| torch", torch.__version__, "| torchvision", torchvision.__version__, "| QUICK:", QUICK)

# %% [markdown]
# ## 1. DATA PREPARATION
# ### 1.1 Dataset acquisition

# %%
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def _find_in_kaggle_input(keys):
    base = Path("/kaggle/input")
    if not base.exists(): return None
    for d in sorted(base.iterdir()):
        if any(k in d.name.lower() for k in keys): return d
    return None

def _extract_archives(folder, passes=2):
    for _ in range(passes):
        found = False
        for f in list(Path(folder).rglob("*")):
            if f.is_file() and (f.suffix.lower() in {".zip", ".tar", ".tgz"} or f.name.lower().endswith(".tar.gz")):
                dest = f.with_name(f.name.split(".")[0] + "_extracted")
                if dest.exists(): continue
                try:
                    shutil.unpack_archive(str(f), str(dest)); found = True
                except Exception as e:
                    print("  could not extract", f, e)
        if not found: break

def download_dataset(cfg: Config) -> Dict[str, Path]:
    """Return {source_name: root_dir}. Order: SAREE_ROOTS env override -> attached /kaggle/input -> download."""
    env = os.environ.get("SAREE_ROOTS")
    if env:
        return {k: Path(v) for k, v in json.loads(env).items()}
    roots = {}
    # Dataset 2 — Indian Saree Patterns (Kaggle)
    p = _find_in_kaggle_input(["indian-saree-patterns", "indian_saree_patterns"])
    if p is None:
        try:
            import kagglehub
            p = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        except Exception as e:
            print("[WARN] Could not fetch", KAGGLE_SLUG, "->", e, "\n       Attach it via 'Add Input' in Kaggle.")
    if p is not None: roots["indian_saree"] = Path(p)
    # Dataset 1 — DeepLure Saree Corpus (proprietary; private Google Drive)
    p = _find_in_kaggle_input(["deeplure", "saree-corpus", "saree_corpus"])
    if p is None:
        out = RAW / "deeplure"
        if not out.exists() or not any(out.rglob("*")):
            out.mkdir(parents=True, exist_ok=True)
            import gdown
            try:
                gdown.download_folder(id=DEEPLURE_GDRIVE_ID, output=str(out), quiet=False, use_cookies=False)
            except Exception as e1:
                print("  folder download failed (", str(e1)[:80], ") -> trying single-file download")
                try:
                    gdown.download(id=DEEPLURE_GDRIVE_ID, output=str(out / "deeplure_download"), quiet=False)
                except Exception as e2:
                    print("[WARN] Google-Drive download failed:", str(e2)[:120],
                          "\n       Upload the corpus as a PRIVATE Kaggle dataset and attach it, or fix Drive permissions/quota.")
        _extract_archives(out)
        p = out if any(out.rglob("*")) else None
    if p is not None: roots["deeplure"] = Path(p)
    if not roots: raise RuntimeError("No dataset could be located. See the warnings above.")
    return roots

DATA_ROOTS = download_dataset(CFG)
print({k: str(v) for k, v in DATA_ROOTS.items()})

# %% [markdown]
# ### 1.2 Dataset inspection (no assumptions about directory structure)

# %%
def discover_images(root) -> List[str]:
    out = []
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".") and d != "__MACOSX"]
        for fn in fns:
            if Path(fn).suffix.lower() in IMG_EXT and not fn.startswith("._"):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def inspect_dataset(root, name, n_show=10):
    root = Path(root); paths = discover_images(root)
    print(f"\n===== {name}  ({root}) =====")
    print("images found:", len(paths), "| extensions:", dict(Counter(Path(p).suffix.lower() for p in paths)))
    folders = Counter(os.path.relpath(os.path.dirname(p), root) for p in paths)
    sizes = np.array(list(folders.values())) if folders else np.array([0])
    print(f"leaf folders: {len(folders)} | images/folder min/median/mean/max = {sizes.min()}/{np.median(sizes):.0f}/{sizes.mean():.1f}/{sizes.max()}")
    print("largest folders:", folders.most_common(5))
    depth = Counter(len(Path(os.path.relpath(p, root)).parts) for p in paths); print("path depth histogram:", dict(depth))
    meta = [str(f.relative_to(root)) for f in root.rglob("*") if f.is_file() and f.suffix.lower() in {".csv", ".tsv", ".json", ".xlsx", ".txt"}]
    print("metadata-like files:", meta[:10])
    rng = random.Random(0)
    for p in rng.sample(paths, min(n_show, len(paths))): print("  e.g.", os.path.relpath(p, root))
    return paths

for _n, _r in DATA_ROOTS.items(): inspect_dataset(_r, _n)

# %% [markdown]
# ### 1.3 Label / design identification

# %%
COLOR_WORDS = ("red maroon crimson pink magenta orange yellow gold golden green olive teal blue navy turquoise cyan purple violet "
               "lavender brown beige cream white black grey gray silver peach rust mustard wine coral").split()
# a trailing number is only stripped when it carries an explicit colourway/view marker ("_c2", "-var3", " (2)"),
# otherwise "saree12" and "saree13" would wrongly collapse into one design
_SUFFIX_RE = re.compile(r"(?:[\s_\-\.]*(?:color|colour|col|cw|colorway|variant|var|v|c|view|img|image|pic|photo)[\s_\-\.]*\d+|\s*\(\d+\))$", re.I)

def _stem_group_key(path, root):
    stem = Path(path).stem.lower()
    for _ in range(3):
        s2 = _SUFFIX_RE.sub("", stem)
        for w in COLOR_WORDS:
            s2 = re.sub(rf"[\s_\-\.]*{w}$", "", s2)
        if s2 == stem or not s2: break
        stem = s2
    return os.path.relpath(os.path.dirname(path), root) + "|" + stem

def _try_metadata(root, paths):
    base = defaultdict(list)
    for p in paths: base[os.path.basename(p)].append(p)
    best = None
    for f in Path(root).rglob("*"):
        if not (f.is_file() and f.suffix.lower() in {".csv", ".tsv"}): continue
        try: df = pd.read_csv(f, sep=None, engine="python")
        except Exception: continue
        fn_col, fn_match = None, 0
        for c in df.columns:
            if df[c].dtype != object: continue
            m = df[c].astype(str).map(os.path.basename).isin(base).mean()
            if m > max(0.5, fn_match): fn_col, fn_match = c, m
        if fn_col is None: continue
        for c in df.columns:
            if c == fn_col: continue
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 0.9 * len(df):
                hint = any(h in str(c).lower() for h in ("design", "pattern", "style", "sku", "group", "id", "label", "class"))
                score = (hint, fn_match, -nun)
                if best is None or score > best[0]: best = (score, f, fn_col, c, df)
    if best is None: return None
    _, f, fn_col, c, df = best
    mp = {}
    for fn, lab in zip(df[fn_col].astype(str).map(os.path.basename), df[c].astype(str)):
        if len(base.get(fn, [])) == 1: mp[base[fn][0]] = lab
    labels = [mp.get(p) for p in paths]
    if sum(l is not None for l in labels) < 0.5 * len(paths): return None
    return labels, f"metadata:{f.name}:{c}"

def _try_folder(root, paths, level, cfg):
    def lab(p):
        parts = Path(os.path.relpath(os.path.dirname(p), root)).parts
        return "/".join(parts[-level:]) if len(parts) >= level else None
    labels = [lab(p) for p in paths]
    cnt = Counter(l for l in labels if l is not None)
    if len(cnt) < 5: return None
    avg = sum(cnt.values()) / len(cnt)
    multi = sum(1 for v in cnt.values() if v >= 2)
    return labels, avg, multi, cnt

def _try_filename(root, paths, cfg):
    keys = [_stem_group_key(p, root) for p in paths]
    cnt = Counter(keys)
    if max(cnt.values()) > cfg.max_colorways_filename: return None
    multi = [k for k, v in cnt.items() if v >= 2]
    if len(multi) < 10 or sum(cnt[k] for k in multi) < 0.2 * len(paths): return None
    return keys, cnt

def infer_labels(root, paths, name, cfg):
    """-> (labels list[Optional[str]], category list, method str, mode 'design'|'instance')"""
    forced = cfg.label_mode.get(name, "auto")
    category = [os.path.relpath(os.path.dirname(p), root) for p in paths]
    if forced in ("auto", "metadata"):
        r = _try_metadata(root, paths)
        if r: return r[0], category, r[1], "design"
        if forced == "metadata": raise RuntimeError(f"{name}: metadata labelling forced but not found")
    if forced in ("auto", "folder"):
        r = _try_folder(root, paths, cfg.folder_level.get(name, 1), cfg)
        if r:
            labels, avg, multi, cnt = r
            if forced == "folder" or (avg <= cfg.design_folder_max_avg and multi >= 10):
                return labels, category, f"folder(level={cfg.folder_level.get(name,1)}, {len(cnt)} folders, avg {avg:.1f} img/folder)", "design"
            print(f"  [{name}] folders look like CATEGORIES (avg {avg:.1f} img/folder > {cfg.design_folder_max_avg}) -> not used as design IDs")
    if forced in ("auto", "filename"):
        r = _try_filename(root, paths, cfg)
        if r:
            keys, cnt = r
            ex = [k for k, v in cnt.items() if v >= 2][:3]
            return keys, category, f"filename-colorway-grouping({len(cnt)} groups; e.g. {ex})", "design"
        if forced == "filename": raise RuntimeError(f"{name}: filename grouping forced but implausible")
    return [None] * len(paths), category, "instance (no design IDs found; each image/near-dup cluster = own identity)", "instance"

def build_dataframe(roots: Dict[str, Path], cfg: Config) -> pd.DataFrame:
    rows, report = [], {}
    rng = random.Random(cfg.seed)
    for name, root in roots.items():
        paths = discover_images(root)
        if not paths: print(f"[WARN] {name}: no images"); continue
        labels, cats, method, mode = infer_labels(root, paths, name, cfg)
        print(f"[{name}] label method -> {method}")
        idx = list(range(len(paths)))
        if cfg.max_images_per_source and len(idx) > cfg.max_images_per_source:   # subsample WHOLE designs where possible
            groups = defaultdict(list)
            for i in idx: groups[labels[i] if labels[i] is not None else f"__{i}"].append(i)
            keys = list(groups); rng.shuffle(keys); idx = []
            for k in keys:
                if len(idx) >= cfg.max_images_per_source: break
                idx += groups[k]
        for i in idx:
            rows.append(dict(path=paths[i], source=name, label_raw=labels[i], category=cats[i], label_method=method))
        report[name] = dict(method=method, n_images=len(idx))
    df = pd.DataFrame(rows)
    df.attrs["label_report"] = report
    return df

RAW_DF = build_dataframe(DATA_ROOTS, CFG)
print(RAW_DF.groupby("source").size())

# %% [markdown]
# ### 1.4 Image validation & cleaning
# Each image is opened twice (`verify()` then full decode), EXIF-rotated, alpha-flattened to white, rejected if corrupt / tiny / blank, resized (short side 256) and cached as JPEG. We also record an MD5 (exact duplicates are **dropped**), a 64-bit dHash (near-duplicates are **clustered**, not dropped — colourways of one design legitimately look alike in grayscale) and a colour descriptor (3-D HSV histogram, 12x3x3, stored as square-root so Euclid/√2 = Hellinger distance in [0,1]).

# %%
def color_descriptor(pil: Image.Image) -> np.ndarray:
    hsv = np.asarray(pil.convert("RGB").resize((64, 64), Image.BILINEAR).convert("HSV"))
    h = np.minimum((hsv[..., 0].astype(np.int32) * 12) // 256, 11)
    s = np.minimum((hsv[..., 1].astype(np.int32) * 3) // 256, 2)
    v = np.minimum((hsv[..., 2].astype(np.int32) * 3) // 256, 2)
    hist = np.bincount((h * 9 + s * 3 + v).ravel(), minlength=108).astype(np.float32)
    return np.sqrt(hist / hist.sum())

def hellinger_cdist(a, b):
    return torch.cdist(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)) / math.sqrt(2)

def dhash_bits(pil: Image.Image) -> np.ndarray:
    g = np.asarray(pil.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).astype(np.uint8).ravel()

def process_image(path, cache_path, side):
    try:
        with Image.open(path) as im: im.verify()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg
            im = im.convert("RGB"); w, h = im.size
            if min(w, h) < 32: return None, "too_small"
            if np.asarray(im.resize((32, 32))).std() < 2.0: return None, "blank"
            sc = side / min(w, h)
            im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True); im.save(cache_path, quality=95)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        return dict(md5=md5, bits=dhash_bits(im), desc=color_descriptor(im)), "ok"
    except Exception as e:
        return None, f"corrupt:{type(e).__name__}"

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra

def hash_clusters(bits: np.ndarray, thr: int) -> np.ndarray:
    n = len(bits); B = torch.from_numpy(bits).float(); uf = UF(n)
    for s in range(0, n, 1024):
        blk = B[s:s + 1024]
        ham = 64 - (blk @ B.T + (1 - blk) @ (1 - B).T)
        ii, jj = torch.nonzero(ham <= thr, as_tuple=True)
        for a, b in zip((ii + s).tolist(), jj.tolist()):
            if a < b: uf.union(a, b)
    return np.array([uf.find(i) for i in range(n)])

def clean_dataset(df: pd.DataFrame, cfg: Config):
    cache_paths = [str(CACHE / r.source / f"{i}.jpg") for i, r in enumerate(df.itertuples())]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda a: process_image(a[0], a[1], cfg.cache_short_side), zip(df.path, cache_paths)))
    stats = Counter(r[1] if r[0] is None else "ok" for r in res)
    keep = [i for i, r in enumerate(res) if r[0] is not None]
    df = df.iloc[keep].copy().reset_index(drop=True)
    info = [res[i][0] for i in keep]
    df["cache"] = [cache_paths[i] for i in keep]
    df["md5"] = [d["md5"] for d in info]
    bits = np.stack([d["bits"] for d in info]); desc = np.stack([d["desc"] for d in info])
    dup = df.duplicated("md5", keep="first").values
    stats["exact_duplicate_removed"] = int(dup.sum())
    df, bits, desc = df[~dup].reset_index(drop=True), bits[~dup], desc[~dup]
    df["cluster"] = hash_clusters(bits, cfg.hash_hamming)
    df["uid"] = np.arange(len(df))
    print("cleaning stats:", dict(stats), "| remaining:", len(df))
    return df, desc.astype(np.float32), dict(stats)

CLEAN_DF, DESC, CLEAN_STATS = clean_dataset(RAW_DF, CFG)
LABEL_REPORT = RAW_DF.attrs.get("label_report", {})

# %% [markdown]
# ### 1.5 Design identities, leakage-safe splits, colour buckets
# * **Design ID** – design mode: `source::label`; instance mode: `source::cluster` (near-dup cluster, if ≤ `max_cluster_identity`) else `source::img`.
# * **Split group** – union-find over (design ID ∪ near-duplicate cluster). Whole groups go to exactly one split, so neither an image, a near-duplicate, nor *any colourway of the same design* can cross splits. Evaluation is therefore **open-set** (test designs are never seen in training).
# * **Test = Gallery + Query.** Design mode: from each test design with ≥2 images, 1 image (2 if ≥4) becomes a *query*, the rest are *gallery*; singleton designs are gallery distractors. Instance mode: gallery = originals, query = deterministic synthetic recolor+crop+rotation of each (flagged `variant=1`). Validation gets the identical treatment and is used for model selection and the verification threshold.
# * **Colour buckets** (label-free): Hellinger distance between colour descriptors; *similar* ≤ q20 and *different* ≥ q50 of the distance between random image pairs.

# %%
def assign_design_ids(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy(); csize = df.groupby(["source", "cluster"]).uid.transform("size")
    des, mode = [], []
    for r, cs in zip(df.itertuples(), csize):
        if r.label_raw is not None and not (isinstance(r.label_raw, float) and np.isnan(r.label_raw)):
            des.append(f"{r.source}::{r.label_raw}"); mode.append("design")
        else:
            des.append(f"{r.source}::cluster{r.cluster}" if cs <= cfg.max_cluster_identity else f"{r.source}::img{r.uid}"); mode.append("instance")
    df["design"], df["mode"] = des, mode
    uf = UF(len(df))
    for _, idx in df.groupby("design").indices.items():
        for j in idx[1:]: uf.union(idx[0], j)
    for _, idx in df.groupby(["source", "cluster"]).indices.items():
        for j in idx[1:]: uf.union(idx[0], j)
    df["group"] = [uf.find(i) for i in range(len(df))]
    return df

def create_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy(); df["split"] = "train"; rng = np.random.default_rng(cfg.seed)
    for src, d in df.groupby("source"):
        groups = d.group.unique(); rng.shuffle(groups); n = len(groups)
        n_val = max(1, int(round(cfg.split_val * n))); n_test = max(1, int(round((1 - cfg.split_train - cfg.split_val) * n)))
        if n < 5: print(f"[WARN] {src}: only {n} groups; splits will be tiny"); n_val = n_test = 1 if n > 2 else 0
        va, te = set(groups[:n_val]), set(groups[n_val:n_val + n_test])
        df.loc[d.index[d.group.isin(va)], "split"] = "val"; df.loc[d.index[d.group.isin(te)], "split"] = "test"
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    for a, b, nm in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        for col in ("group", "design", "md5", "uid"):
            assert not (set(a[col]) & set(b[col])), f"LEAKAGE in {nm} via {col}"
    print("leakage checks passed (group/design/md5/uid disjoint across splits)")
    return df

def color_thresholds(desc, cfg, n=200000):
    rng = np.random.default_rng(cfg.seed); i = rng.integers(0, len(desc), n); j = rng.integers(0, len(desc), n); m = i != j
    d = (torch.as_tensor(desc[i[m]]) - torch.as_tensor(desc[j[m]])).norm(dim=1).numpy() / math.sqrt(2)
    return float(np.quantile(d, cfg.color_sim_q)), float(np.quantile(d, cfg.color_diff_q))

DF = create_splits(assign_design_ids(CLEAN_DF, CFG), CFG)
T_SIM, T_DIFF = color_thresholds(DESC, CFG)
print(f"colour thresholds: similar <= {T_SIM:.3f} | different >= {T_DIFF:.3f}")
summary = DF.groupby(["split", "mode"]).agg(images=("uid", "size"), designs=("design", "nunique"), groups=("group", "nunique"))
print(summary)
TRAIN_DF = DF[DF.split == "train"].reset_index(drop=True)
VAL_DF = DF[DF.split == "val"].reset_index(drop=True)
TEST_DF = DF[DF.split == "test"].reset_index(drop=True)
n_multi = DF[DF["mode"] == "design"].groupby("design").size().ge(2).sum()
print("designs with >=2 images (real colorway pairs available):", int(n_multi))
if n_multi == 0:
    print("[NOTE] no real multi-colorway designs found -> only the SYNTHETIC recolor protocol can be evaluated; the notebook reports it as such.")

# %% [markdown]
# ## 2. TRAINING
# ### 2.1 Colour-invariance augmentation
# | Augmentation | Purpose |
# |---|---|
# | `ColorJitter(brightness .5, contrast .5, saturation .7, hue .5)` p=.9 | Re-colours the saree: shifts hue anywhere on the colour wheel, so palette cannot identify a design. |
# | `RandomGrayscale(.2)` | Forces the net to also work from luminance structure alone (motif/texture) — but only 20 % of the time, so colour is **not** removed globally. |
# | Random channel permutation (p=.5) | Cheap, exact "new colourway" (red↔blue swap etc.). |
# | Random channel mixing (p=.3, Dirichlet matrix) | Smooth non-hue palette remaps (e.g. gold→green with different tonal relationship). |
# | `RandomResizedCrop(scale .35–1)` | Scale variation + partial views; teaches the net that a design is the *repeat structure*, not one exact framing. |
# | Horizontal flip, rotation ±12° | Draping/camera variation. No vertical flip (borders/pallu have a direction). |
# | Light Gaussian blur (p=.1) | Robustness to focus/compression. |
# The *baseline* gets only crop+flip+mild brightness jitter; the ablation shows what the colour pipeline adds.

# %%
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class RandomChannelPermute:
    def __init__(self, p): self.p = p
    def __call__(self, x): return x[torch.randperm(3)] if random.random() < self.p else x

class RandomColorMix:
    def __init__(self, p, strength=(0.2, 0.8)): self.p, self.s = p, strength
    def __call__(self, x):
        if random.random() >= self.p: return x
        M = torch.distributions.Dirichlet(torch.full((3,), 0.5)).sample((3,))
        a = random.uniform(*self.s)
        M = (1 - a) * torch.eye(3) + a * M
        return torch.einsum("ij,jhw->ihw", M, x).clamp(0, 1)

def build_transforms(mode: str, cfg: Config):
    S = cfg.img_size
    if mode == "minimal":
        return T.Compose([T.RandomResizedCrop(S, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.0)], p=0.5), T.ToTensor(), T.Normalize(MEAN, STD)])
    if mode == "color_strong":
        return T.Compose([
            T.RandomResizedCrop(S, scale=(0.35, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomRotation(12, interpolation=T.InterpolationMode.BILINEAR, fill=(128, 128, 128))], p=0.5),
            T.RandomApply([T.ColorJitter(0.5, 0.5, 0.7, 0.5)], p=0.9), T.RandomGrayscale(0.2), T.ToTensor(),
            RandomChannelPermute(0.5), RandomColorMix(0.3),
            T.RandomApply([T.GaussianBlur(3, (0.1, 1.0))], p=0.1), T.Normalize(MEAN, STD)])
    raise ValueError(mode)

EVAL_TF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

def synth_variant(im: Image.Image, seed: int) -> Image.Image:
    """Deterministic synthetic 'other colourway + other photo': crop, small rotation, hue/sat/brightness/channel change."""
    r = random.Random(seed); w, h = im.size
    s = r.uniform(0.7, 1.0); cw, ch = int(w * s), int(h * s); x0, y0 = r.randint(0, w - cw), r.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch)).rotate(r.uniform(-8, 8), resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    if r.random() < 0.5:
        perm = r.sample([0, 1, 2], 3); c = im.split(); im = Image.merge("RGB", [c[i] for i in perm])
    im = TF.adjust_hue(im, r.choice([-1, 1]) * r.uniform(0.2, 0.5))
    im = TF.adjust_saturation(im, r.uniform(0.7, 1.4)); im = TF.adjust_brightness(im, r.uniform(0.8, 1.2))
    return TF.adjust_contrast(im, r.uniform(0.85, 1.2))

def make_eval_pil(cache_path, variant, uid, cfg: Config) -> Image.Image:
    im = Image.open(cache_path).convert("RGB")
    if variant == 1: im = synth_variant(im, cfg.seed * 100003 + int(uid))
    S = cfg.img_size; w, h = im.size; sc = S / min(w, h)
    im = im.resize((max(S, round(w * sc)), max(S, round(h * sc))), Image.BICUBIC)
    return TF.center_crop(im, [S, S])

class TrainDataset(Dataset):
    def __init__(self, df, tf):
        self.cache = df.cache.tolist(); self.tf = tf
        self.codes = pd.factorize(df.design)[0].astype(np.int64)
    def __len__(self): return len(self.cache)
    def __getitem__(self, i):
        with Image.open(self.cache[i]) as im: x = self.tf(im.convert("RGB"))
        return x, int(self.codes[i])

class EvalDataset(Dataset):
    def __init__(self, df, cfg): self.df, self.cfg = df.reset_index(drop=True), cfg
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]; im = make_eval_pil(r.cache, r.variant, r.uid, self.cfg)
        return EVAL_TF(im), torch.from_numpy(color_descriptor(im))

# %% [markdown]
# ### 2.2 Colour-aware PK sampler (positives / hard positives / negatives / hard negatives)
# Each batch holds **P designs × K images**.
# * **Positive pair:** two images of the same design (or two augmented views of one image in instance mode).
# * **Hard positive (`hard_pos`):** for multi-colourway designs the other K−1 images are drawn with probability ∝ (colour distance to the anchor)^2 → the *most differently coloured* colourways are over-sampled.
# * **Hard negative (`hard_neg`):** batches are assembled from `hard_neg_clusters` *colour neighbourhoods*: a seed design plus designs whose mean colour histogram is among its `topm` nearest — so in-batch negatives are different designs with similar colours. The rest of the batch is random (easy negatives).
# The losses below use **every** in-batch pair, so these hard cases automatically dominate the gradient.

# %%
class ColorAwarePKSampler(Sampler):
    def __init__(self, df, desc, cfg: Config, steps, hard_neg=True, hard_pos=True, seed=0):
        self.cfg, self.steps, self.hard_neg, self.hard_pos, self.seed, self.epoch = cfg, steps, hard_neg, hard_pos, seed, 0
        self.desc = desc[df.uid.values]
        groups = df.groupby("design").indices; self.members = [np.asarray(v) for v in groups.values()]
        self.ND = len(self.members); self.P = min(cfg.P, self.ND)
        if hard_neg:
            dm = torch.tensor(np.stack([self.desc[m].mean(0) for m in self.members]))
            m = min(cfg.hard_neg_topm, self.ND - 1); nb = []
            for s in range(0, self.ND, 2048):
                d = torch.cdist(dm[s:s + 2048], dm); d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = 1e9
                nb.append(d.topk(m, largest=False).indices)
            self.nbrs = torch.cat(nb).numpy()
    def __len__(self): return self.steps
    def _pick(self, d, rng):
        idx = self.members[d]; K = self.cfg.K
        if len(idx) == 1: return [int(idx[0])] * K
        a = int(rng.choice(idx)); others = np.array([i for i in idx if i != a])
        if len(others) >= K - 1:
            if self.hard_pos:
                dist = (np.linalg.norm(self.desc[others] - self.desc[a], axis=1) / math.sqrt(2)) ** self.cfg.hard_pos_power + 1e-3
                pick = rng.choice(others, size=K - 1, replace=False, p=dist / dist.sum())
            else:
                pick = rng.choice(others, size=K - 1, replace=False)
            return [a] + [int(i) for i in pick]
        out = [a] + [int(i) for i in others]
        while len(out) < K: out.append(int(rng.choice(idx)))
        return out
    def __iter__(self):
        rng = np.random.default_rng(self.seed + 7919 * self.epoch); self.epoch += 1
        for _ in range(self.steps):
            if self.hard_neg and self.ND > self.P:
                chosen = set(); per = max(2, self.P // self.cfg.hard_neg_clusters)
                for _ in range(self.cfg.hard_neg_clusters):
                    s = int(rng.integers(self.ND)); chosen.add(s)
                    chosen.update(int(x) for x in rng.choice(self.nbrs[s], size=min(per - 1, self.nbrs.shape[1]), replace=False))
                while len(chosen) < self.P: chosen.add(int(rng.integers(self.ND)))
                chosen = list(chosen)[:self.P]
            else:
                chosen = rng.choice(self.ND, size=self.P, replace=False).tolist()
            batch = []
            for d in chosen: batch += self._pick(d, rng)
            yield batch

def build_dataloaders(train_df, desc, cfg: Config, aug_mode, hard_sampling, seed):
    steps = cfg.max_steps_per_epoch   # batches are re-sampled each epoch, so 'epoch' = this many P x K batches
    ds = TrainDataset(train_df, build_transforms(aug_mode, cfg))
    sampler = ColorAwarePKSampler(train_df, desc, cfg, steps, hard_neg=hard_sampling, hard_pos=hard_sampling, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers, pin_memory=DEVICE.type == "cuda",
                        worker_init_fn=worker_init, persistent_workers=cfg.num_workers > 0)
    return loader, steps

# %% [markdown]
# ### 2.3 Model

# %%
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(float(p))); self.eps = eps
    def forward(self, x):
        x = x.float().clamp(min=self.eps)
        return F.avg_pool2d(x.pow(self.p), x.shape[-2:]).pow(1.0 / self.p).flatten(1)

class DesignEmbeddingNet(nn.Module):
    """Shared-weight (Siamese-style) encoder: trunk -> GeM -> Linear -> BN -> L2-norm."""
    def __init__(self, backbone="mobilenet_v3_large", dim=256, pretrained=True):
        super().__init__()
        tv = torchvision.models
        def mk(fn, w):
            try:
                return fn(weights="DEFAULT" if pretrained else None)
            except Exception as e:
                print(f"[WARN] pretrained weights unavailable ({str(e)[:70]}) -> RANDOM INIT; results will be much worse. Turn Internet ON.")
                return fn(weights=None)
        if backbone == "mobilenet_v3_large":
            m = mk(tv.mobilenet_v3_large, None); self.features, c = m.features, 960
        elif backbone == "efficientnet_b0":
            m = mk(tv.efficientnet_b0, None); self.features, c = m.features, 1280
        elif backbone == "resnet18":
            m = mk(tv.resnet18, None); self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4); c = 512
        else: raise ValueError(backbone)
        self.pool = GeM(); self.head = nn.Sequential(nn.Linear(c, dim), nn.BatchNorm1d(dim)); self.dim = dim
    def forward(self, x):
        return F.normalize(self.head(self.pool(self.features(x))), dim=-1)

def build_model(cfg: Config, pretrained=None):
    return DesignEmbeddingNet(cfg.backbone, cfg.embed_dim, cfg.pretrained if pretrained is None else pretrained).to(DEVICE)

# %% [markdown]
# ### 2.4 Losses
# * **Supervised contrastive (SupCon, τ=0.1):** every same-design pair in the batch is a positive, every other image a negative. Because batches contain far-apart colourways (hard positives) and colour-neighbour designs (hard negatives), the softmax denominator is dominated by exactly the color-vs-design confusions we care about.
# * **Batch-hard triplet (margin 0.2, cosine distance):** hardest positive vs hardest negative per anchor — a direct margin on the worst colour/design case.
# No classification head: the number of identities is large/open-set and evaluation is on unseen designs.

# %%
def supcon_loss(z, y, tau):
    n = len(y); eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    npos = pos.sum(1); ok = npos > 0
    if not ok.any(): return z.sum() * 0
    return (-(logp * pos).sum(1)[ok] / npos[ok]).mean()

def batch_hard_triplet(z, y, margin):
    d = 1 - z @ z.T; n = len(y); eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = y[:, None] == y[None, :]; pm, nm = same & ~eye, ~same
    dp = d.masked_fill(~pm, -1.0).max(1).values; dn = d.masked_fill(~nm, 9.0).min(1).values
    ok = pm.any(1) & nm.any(1)
    if not ok.any(): return z.sum() * 0
    return F.relu(dp - dn + margin)[ok].mean()

# %% [markdown]
# ### 2.5 Embedding / retrieval / identification-metric utilities (needed by validation)

# %%
@torch.no_grad()
def generate_embeddings(model, df_items, cfg: Config, bs=128):
    """df_items needs columns cache, uid, variant. Returns (emb [N,D] cpu float32, colour_desc [N,108] np)."""
    model.eval(); loader = DataLoader(EvalDataset(df_items, cfg), batch_size=bs, num_workers=cfg.num_workers, worker_init_fn=worker_init)
    E, C = [], []
    for x, c in loader:
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            z = model(x.to(DEVICE, non_blocking=True))
        E.append(z.float().cpu()); C.append(c)
    return torch.cat(E), torch.cat(C).numpy()

def build_gallery(split_df: pd.DataFrame, cfg: Config, seed=0):
    """Gallery / query construction (see section 1.5)."""
    rng = np.random.default_rng(cfg.seed + seed); g_rows, q_rows = [], []
    for design, d in split_df.groupby("design"):
        if d["mode"].iloc[0] == "instance":
            g_rows += [(i, 0) for i in d.index]; q_rows += [(i, 1) for i in d.index]
        else:
            idx = list(d.index); rng.shuffle(idx)
            nq = 0 if len(idx) < 2 else (1 if len(idx) < 4 else 2)
            q_rows += [(i, 0) for i in idx[:nq]]; g_rows += [(i, 0) for i in idx[nq:]]
    def mk(rows):
        out = split_df.loc[[i for i, _ in rows]].copy(); out["variant"] = [v for _, v in rows]; return out.reset_index(drop=True)
    return mk(g_rows), mk(q_rows)

def retrieve(q_emb, g_emb, k=10):
    sim = q_emb @ g_emb.T; v, i = sim.topk(min(k, g_emb.shape[0]), dim=1); return v, i

def _rank_metrics(sim, rel, ignore, rows, ks=(1, 5, 10)):
    s = sim[rows].clone(); r = rel[rows]; ig = ignore[rows]; s[ig] = -1e9
    valid = r.any(1)
    if valid.sum() == 0: return dict(n=0, **{f"top{k}": float("nan") for k in ks}, mrr=float("nan"))
    s, r = s[valid], r[valid]; order = s.argsort(1, descending=True); rs = r.gather(1, order)
    first = rs.float().argmax(1)
    out = dict(n=int(valid.sum()), **{f"top{k}": float((first < k).float().mean()) for k in ks}, mrr=float((1.0 / (first + 1)).mean()))
    return out

def evaluate_identification(q_emb, g_emb, q_df, g_df, q_desc, g_desc, t_sim=None, t_diff=None):
    t_sim = T_SIM if t_sim is None else t_sim; t_diff = T_DIFF if t_diff is None else t_diff
    codes = pd.factorize(pd.concat([g_df.design, q_df.design]))[0]
    gl, ql = torch.as_tensor(codes[:len(g_df)]), torch.as_tensor(codes[len(g_df):])
    sim = q_emb @ g_emb.T; same = ql[:, None] == gl[None, :]; cd = hellinger_cdist(q_desc, g_desc)
    none = torch.zeros_like(same)
    protos = {"all": torch.ones(len(q_df), dtype=torch.bool),
              "real_colorway": torch.as_tensor((q_df["mode"] == "design").values & (q_df.variant == 0).values),
              "synthetic_recolor": torch.as_tensor((q_df.variant == 1).values)}
    res = {}
    for pn, rows in protos.items():
        if rows.sum() == 0: continue
        diff_rel = same & (cd >= t_diff); sim_rel = same & (cd <= t_sim)
        res[pn] = {"standard": _rank_metrics(sim, same, none, rows),
                   "same_design_diff_color": _rank_metrics(sim, diff_rel, same & ~diff_rel, rows),   # PRIMARY challenge metric
                   "same_design_same_color": _rank_metrics(sim, sim_rel, same & ~sim_rel, rows)}
        # colour-lure error: top-1 is a DIFFERENT design whose colour is similar to the query's
        has = same.any(1) & rows; top1 = sim.argmax(1)
        wrong = ~same[torch.arange(len(q_df)), top1]
        lure = wrong & (cd[torch.arange(len(q_df)), top1] <= t_sim)
        res[pn]["top1_color_lure_error"] = float((lure & has).sum() / max(1, has.sum()))
    return res

def validate(model, val_pack, cfg: Config):
    g_emb, g_desc = generate_embeddings(model, val_pack["gallery"], cfg)
    q_emb, q_desc = generate_embeddings(model, val_pack["query"], cfg)
    r = evaluate_identification(q_emb, g_emb, val_pack["query"], val_pack["gallery"], q_desc, g_desc)
    return r["all"]["standard"] if "all" in r else dict(top1=float("nan"), mrr=float("nan"))

VAL_G, VAL_Q = build_gallery(VAL_DF, CFG, seed=1)
TEST_G, TEST_Q = build_gallery(TEST_DF, CFG, seed=2)
VAL_PACK = dict(gallery=VAL_G, query=VAL_Q); TEST_PACK = dict(gallery=TEST_G, query=TEST_Q)
print(f"val: gallery {len(VAL_G)} / query {len(VAL_Q)} | test: gallery {len(TEST_G)} / query {len(TEST_Q)}")
print("test queries by protocol:", dict(Counter(np.where(TEST_Q.variant == 1, "synthetic_recolor", "real_colorway"))))

# %% [markdown]
# ### 2.6 Training loop (AMP, AdamW, warm-up + cosine, best-checkpoint on validation MRR)

# %%
def train_one_epoch(model, loader, opt, sched, scaler, exp, epoch):
    model.train(); tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            z = model(x)
        z = z.float()
        loss = CFG.w_supcon * supcon_loss(z, y, CFG.supcon_temp)
        if exp["triplet"]: loss = loss + CFG.w_triplet * batch_hard_triplet(z, y, CFG.triplet_margin)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item(); n += 1
    return tot / max(1, n)

EXPERIMENTS = {   # name -> (augmentation, hard colour sampling, triplet term)
    "A_baseline":       dict(aug="minimal",      hard=False, triplet=False),
    "B_baseline+color": dict(aug="color_strong", hard=False, triplet=False),
    "C_final":          dict(aug="color_strong", hard=True,  triplet=True),
}

def train_model(name, exp):
    set_seed(CFG.seed); model = build_model(CFG)
    loader, steps = build_dataloaders(TRAIN_DF, DESC, CFG, exp["aug"], exp["hard"], CFG.seed)
    head = [p for n, p in model.named_parameters() if not n.startswith("features")]
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": CFG.lr_backbone},
                             {"params": head, "lr": CFG.lr_head}], weight_decay=CFG.weight_decay)
    total, warm = CFG.epochs * steps, CFG.warmup_epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    ckpt = WORK / f"ckpt_{name}.pth"; best, hist = -1.0, []
    t0 = time.time()
    for ep in range(1, CFG.epochs + 1):
        loss = train_one_epoch(model, loader, opt, sched, scaler, exp, ep)
        v = validate(model, VAL_PACK, CFG)
        hist.append(dict(epoch=ep, train_loss=loss, val_top1=v["top1"], val_mrr=v["mrr"]))
        score = v["mrr"] if not math.isnan(v["mrr"]) else -loss
        if score > best:
            best = score; torch.save({"model": model.state_dict(), "cfg": asdict(CFG), "exp": exp, "epoch": ep}, ckpt)
        print(f"[{name}] ep {ep:02d}/{CFG.epochs} loss {loss:.4f} | val top1 {v['top1']:.3f} mrr {v['mrr']:.3f} | {time.time()-t0:.0f}s")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
    return model, ckpt, hist

# %% [markdown]
# ### 2.7 Train baseline, baseline + colour augmentation, and the final colour-invariant model
# A zero-shot (ImageNet, untrained head) reference is evaluated too. All variants share the same backbone, seed, data, epochs and steps; only the listed ingredients differ.

# %%
MODELS, HISTORY, CKPTS = {}, {}, {}
set_seed(CFG.seed); MODELS["0_pretrained_no_finetune"] = build_model(CFG).eval()
for _name, _exp in EXPERIMENTS.items():
    print(f"\n=== training {_name}: {_exp} ===")
    MODELS[_name], CKPTS[_name], HISTORY[_name] = train_model(_name, _exp)
FINAL_NAME = "C_final"; FINAL = MODELS[FINAL_NAME]
shutil.copy(CKPTS[FINAL_NAME], WORK / "best_model.pth")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for n, h in HISTORY.items():
    ax[0].plot([e["train_loss"] for e in h], label=n); ax[1].plot([e["val_mrr"] for e in h], label=n)
ax[0].set_title("train loss (not comparable across loss types)"); ax[1].set_title("validation MRR"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(WORK / "training_curves.png", dpi=120); plt.show()

# %% [markdown]
# ## 3. EVALUATION
# ### 3.1 Verification pairs & metrics
# Pairs are built *within a split* from unseen designs. Instance-mode identities contribute (original, synthetic recolor) positives. Every pair is bucketed by the label-free colour distance into
# `same_design/same_color`, `same_design/diff_color`, `diff_design/similar_color`, `diff_design/diff_color` (+ `mid_color` leftovers). Besides random negatives we **mine, for every image, the colour-nearest image of a different design** so the critical `diff_design/similar_color` bucket is well populated. The threshold is chosen on **validation** pairs (Youden J) and applied unchanged to **test** pairs.

# %%
def make_items(split_df, max_items=8000, seed=0):
    a = split_df.copy(); a["variant"] = 0
    b = split_df[split_df["mode"] == "instance"].copy(); b["variant"] = 1
    items = pd.concat([a, b]).reset_index(drop=True)
    if len(items) > max_items:
        rng = np.random.default_rng(seed); ds = items.design.unique(); rng.shuffle(ds); keep, n = set(), 0
        cnt = items.groupby("design").size()
        for d in ds:
            if n >= max_items: break
            keep.add(d); n += cnt[d]
        items = items[items.design.isin(keep)].reset_index(drop=True)
    return items

def pair_category(y, cd):
    if y: return "same_design/same_color" if cd <= T_SIM else ("same_design/diff_color" if cd >= T_DIFF else "same_design/mid_color")
    return "diff_design/similar_color" if cd <= T_SIM else ("diff_design/diff_color" if cd >= T_DIFF else "diff_design/mid_color")

def build_pairs(emb, desc, labels, cfg: Config, seed=0):
    rng = np.random.default_rng(cfg.seed + seed); M = len(labels); L = torch.as_tensor(labels)
    same = L[:, None] == L[None, :]; cd = hellinger_cdist(desc, desc)
    pos = torch.nonzero(same & torch.triu(torch.ones(M, M, dtype=torch.bool), 1)).numpy()
    if len(pos) > cfg.max_pos_pairs: pos = pos[rng.choice(len(pos), cfg.max_pos_pairs, replace=False)]
    n_rand = min(cfg.max_neg_pairs, max(len(pos), 1000)); i = rng.integers(0, M, 4 * n_rand); j = rng.integers(0, M, 4 * n_rand)
    ok = (L[i] != L[j]).numpy(); neg_rand = np.stack([i[ok], j[ok]], 1)[:n_rand]
    d = cd.masked_fill(same, 9.0); mj = d.argmin(1); mok = (d.gather(1, mj[:, None])[:, 0] < 9).numpy()
    neg_mined = np.stack([np.arange(M)[mok], mj.numpy()[mok]], 1)[: cfg.max_neg_pairs // 2]
    P = np.concatenate([pos, neg_rand, neg_mined]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg_rand) + len(neg_mined))].astype(int)
    c = cd[P[:, 0], P[:, 1]].numpy(); sims = (emb[P[:, 0]] * emb[P[:, 1]]).sum(1).numpy()
    return pd.DataFrame(dict(i=P[:, 0], j=P[:, 1], y=y, color_dist=c, sim=sims, cat=[pair_category(a, b) for a, b in zip(y, c)]))

def pick_threshold(y, s):
    if len(set(y)) < 2: print("[WARN] validation pairs lack a class -> threshold 0.5"); return 0.5
    fpr, tpr, th = roc_curve(y, s); return float(th[np.argmax(tpr - fpr)])

def evaluate_verification(pairs: pd.DataFrame, thr: float):
    y, s = pairs.y.values, pairs.sim.values; pred = (s >= thr).astype(int)
    out = dict(threshold=thr, n_pairs=int(len(y)), n_pos=int(y.sum()))
    if len(set(y)) == 2:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; k = int(np.argmin(np.abs(fnr - fpr)))
        out.update(accuracy=float(accuracy_score(y, pred)), precision=float(p), recall=float(r), f1=float(f),
                   roc_auc=float(roc_auc_score(y, s)), eer=float((fpr[k] + fnr[k]) / 2))
        hard = pairs.cat.isin(["same_design/diff_color", "diff_design/similar_color"]).values
        hy = y[hard]
        out["auc_samedesign_diffcolor_vs_diffdesign_similarcolor"] = float(roc_auc_score(hy, s[hard])) if len(set(hy)) == 2 else float("nan")
    out["by_category"] = {c: dict(n=int(len(g)), mean_sim=float(g.sim.mean()),
                                  accuracy=float(((g.sim.values >= thr).astype(int) == g.y.values).mean()))
                          for c, g in pairs.groupby("cat")}
    return out

def verify_pair(model, path_a, path_b, thr, cfg: Config = CFG):
    """Public API: are the two saree images the same design? Returns cosine similarity + decision."""
    model.eval(); xs = []
    for p in (path_a, path_b):
        im = Image.open(p).convert("RGB"); S = cfg.img_size; w, h = im.size; sc = S / min(w, h)
        im = TF.center_crop(im.resize((max(S, round(w * sc)), max(S, round(h * sc))), Image.BICUBIC), [S, S]); xs.append(EVAL_TF(im))
    with torch.no_grad(): za, zb = model(torch.stack(xs).to(DEVICE)).float()
    s = float((za * zb).sum()); return dict(cosine=s, same_design=bool(s >= thr))

# %%
VAL_ITEMS, TEST_ITEMS = make_items(VAL_DF, seed=1), make_items(TEST_DF, seed=2)
CACHE_EVAL = {}

def evaluate_model(model, tag):
    t0 = time.time(); out = {}
    g_emb, g_desc = generate_embeddings(model, TEST_G, CFG); q_emb, q_desc = generate_embeddings(model, TEST_Q, CFG)
    out["identification"] = evaluate_identification(q_emb, g_emb, TEST_Q, TEST_G, q_desc, g_desc)
    ve, vd = generate_embeddings(model, VAL_ITEMS, CFG); te, td = generate_embeddings(model, TEST_ITEMS, CFG)
    vp = build_pairs(ve, vd, pd.factorize(VAL_ITEMS.design)[0], CFG, 1); tp = build_pairs(te, td, pd.factorize(TEST_ITEMS.design)[0], CFG, 2)
    thr = pick_threshold(vp.y.values, vp.sim.values)
    out["verification"] = evaluate_verification(tp, thr)
    out["verification_val_threshold_pairs"] = int(len(vp))
    CACHE_EVAL[tag] = dict(g_emb=g_emb, g_desc=g_desc, q_emb=q_emb, q_desc=q_desc, test_pairs=tp, val_pairs=vp, thr=thr)
    print(f"evaluated {tag} in {time.time()-t0:.0f}s (threshold {thr:.3f})"); return out

ALL_METRICS = {n: evaluate_model(m, n) for n, m in MODELS.items()}

# %% [markdown]
# ### 3.2 Ablation (test set, unseen designs)
# `real` = genuine different-colourway queries (only if the data contains multi-colourway designs); `synthetic` = deterministic synthetic recolor of the gallery image (a *proxy*: it is closer to the training augmentations, so it flatters colour-augmented models — read the `real` rows first when available).

# %%
def ablation_table(metrics):
    rows = []
    for name, m in metrics.items():
        for proto in ("real_colorway", "synthetic_recolor"):
            if proto not in m["identification"]: continue
            r = m["identification"][proto]; v = m["verification"]
            rows.append({"model": name, "protocol": proto, "n_queries": r["standard"]["n"],
                         "Top1": r["standard"]["top1"], "Top5": r["standard"]["top5"], "MRR": r["standard"]["mrr"],
                         "Top1 same-design/diff-color": r["same_design_diff_color"]["top1"],
                         "color-lure err": r["top1_color_lure_error"],
                         "verif ROC-AUC": v.get("roc_auc", float("nan")),
                         "AUC SD-vs-DS": v.get("auc_samedesign_diffcolor_vs_diffdesign_similarcolor", float("nan"))})
    return pd.DataFrame(rows)
ABLATION = ablation_table(ALL_METRICS)
print(ABLATION.round(3).to_string(index=False))

# %% [markdown]
# ## 4. IDENTIFICATION (final model, test gallery/query)

# %%
FINAL_ID = ALL_METRICS[FINAL_NAME]["identification"]
for proto, r in FINAL_ID.items():
    print(f"\n--- protocol: {proto} ---")
    print(pd.DataFrame({k: v for k, v in r.items() if isinstance(v, dict)}).T.round(4).to_string())
    print("top-1 colour-lure error rate (wrong top-1 that is a similar-colour different design):", round(r["top1_color_lure_error"], 4))
print("\nPRIMARY challenge metric = 'same_design_diff_color' rows of protocol 'real_colorway' (or 'synthetic_recolor' if no real colourways exist).")

def retrieval_demo(model, query_row_idx=0, k=5):
    e = CACHE_EVAL[FINAL_NAME]; v, i = retrieve(e["q_emb"][query_row_idx:query_row_idx + 1], e["g_emb"], k)
    return [(TEST_G.design[j], round(float(s), 3)) for s, j in zip(v[0], i[0])]
if len(TEST_Q): print("demo retrieval for query 0:", TEST_Q.design[0], "->", retrieval_demo(FINAL))

# %% [markdown]
# ### 4.1 Retrieval visualisation
# Border colours: **green** = correct design, **red** = wrong design. Tags: `same design / diff color` (the win we want), `same design / similar color`, `DIFF design / SIMILAR color` (colour lure — the failure we fear), `diff design`.

# %%
def match_tag(correct, cd):
    if correct: return "same design / diff color" if cd >= T_DIFF else ("same design / similar color" if cd <= T_SIM else "same design / mid color")
    return "DIFF design / SIMILAR color" if cd <= T_SIM else "diff design"

def visualize_results(tag=FINAL_NAME, n_each=4, k=5, path=WORK / "retrieval_examples.png"):
    e = CACHE_EVAL[tag]; sim = e["q_emb"] @ e["g_emb"].T; vals, idx = sim.topk(k, dim=1)
    cd = hellinger_cdist(e["q_desc"], e["g_desc"])
    codes = pd.factorize(pd.concat([TEST_G.design, TEST_Q.design]))[0]; gl, ql = codes[:len(TEST_G)], codes[len(TEST_G):]
    ok1 = np.array([gl[idx[q, 0]] == ql[q] for q in range(len(TEST_Q))]); has = np.array([(gl == ql[q]).any() for q in range(len(TEST_Q))])
    rng = np.random.default_rng(0)
    cross = [q for q in np.where(ok1 & has)[0] if cd[q, idx[q, 0]] >= T_DIFF]; fail = list(np.where(~ok1 & has)[0])
    sel = list(rng.permutation(cross)[:n_each]) + list(rng.permutation(fail)[:n_each])
    if len(sel) < 2 * n_each: sel += [q for q in rng.permutation(len(TEST_Q)) if q not in sel][: 2 * n_each - len(sel)]
    fig, axes = plt.subplots(len(sel), k + 1, figsize=(2.3 * (k + 1), 2.6 * len(sel)), squeeze=False)
    for r, q in enumerate(sel):
        qr = TEST_Q.iloc[q]; a = axes[r, 0]; a.imshow(make_eval_pil(qr.cache, qr.variant, qr.uid, CFG)); a.set_title("QUERY" + (" (synthetic recolor)" if qr.variant else ""), fontsize=8); a.axis("off")
        for c in range(k):
            g = TEST_G.iloc[int(idx[q, c])]; correct = gl[idx[q, c]] == ql[q]; a = axes[r, c + 1]
            a.imshow(make_eval_pil(g.cache, 0, g.uid, CFG)); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_edgecolor("limegreen" if correct else "red"); sp.set_linewidth(4)
            a.set_title(f"Top-{c+1} cos={vals[q, c]:.2f}\n{'✓' if correct else '✗'} {match_tag(correct, float(cd[q, idx[q, c]]))}", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.show()
if len(TEST_Q): visualize_results()

# %% [markdown]
# ## 5. VERIFICATION (final model)

# %%
FINAL_VER = ALL_METRICS[FINAL_NAME]["verification"]
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in FINAL_VER.items() if k != "by_category"})
print(pd.DataFrame(FINAL_VER["by_category"]).T.round(4).to_string())
print("\ncritical cases: 'same_design/diff_color' should be accepted; 'diff_design/similar_color' should be rejected (accuracy column = fraction correct at the validation threshold).")

tp, thr = CACHE_EVAL[FINAL_NAME]["test_pairs"], CACHE_EVAL[FINAL_NAME]["thr"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4)); bins = np.linspace(-0.2, 1, 60)
ax[0].hist(tp.sim[tp.y == 1], bins, alpha=.6, label="positive pairs", density=True); ax[0].hist(tp.sim[tp.y == 0], bins, alpha=.6, label="negative pairs", density=True)
ax[0].axvline(thr, color="k", ls="--", label=f"val threshold {thr:.2f}"); ax[0].set_xlabel("cosine similarity"); ax[0].legend(); ax[0].set_title("pair similarity (test)")
for c, g in tp.groupby("cat"):
    if len(g) > 20: ax[1].hist(g.sim, bins, alpha=.45, label=f"{c} (n={len(g)})", density=True)
ax[1].axvline(thr, color="k", ls="--"); ax[1].legend(fontsize=6); ax[1].set_title("by colour/design category"); ax[1].set_xlabel("cosine similarity")
plt.tight_layout(); plt.savefig(WORK / "similarity_distribution.png", dpi=120); plt.show()

# demo of the two-image API on real test pairs
for cat in ("same_design/diff_color", "diff_design/similar_color"):
    g = tp[tp.cat == cat]
    if len(g):
        r = g.iloc[0]; a, b = TEST_ITEMS.iloc[int(r.i)], TEST_ITEMS.iloc[int(r.j)]
        if a.variant == 0 and b.variant == 0:
            print(cat, "->", verify_pair(FINAL, a.cache, b.cache, thr))

# %% [markdown]
# ## 6. EFFICIENCY ANALYSIS

# %%
def measure_efficiency(model, cfg: Config, ckpt_path):
    import copy
    out = {}
    out["params_total"] = sum(p.numel() for p in model.parameters()); out["params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out["embedding_dim"] = cfg.embed_dim; out["checkpoint_MB"] = os.path.getsize(ckpt_path) / 1e6
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = copy.deepcopy(model).cpu().eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad(): m(torch.randn(1, 3, cfg.img_size, cfg.img_size))
        out["GFLOPs_per_image(2*MAC)"] = fc.get_total_flops() / 1e9
    except Exception as e:
        out["GFLOPs_per_image(2*MAC)"] = None; print("FLOP counter unavailable:", e)
    def lat(dev, n=100):
        m = copy.deepcopy(model).to(dev).eval(); x = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=dev); ts = []
        with torch.no_grad():
            for _ in range(15): m(x)
            for _ in range(n):
                if dev.type == "cuda": torch.cuda.synchronize()
                t = time.perf_counter(); m(x)
                if dev.type == "cuda": torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1e3)
        return dict(mean_ms=float(np.mean(ts)), p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)))
    out["latency_batch1_cpu"] = lat(torch.device("cpu"), 50)
    if DEVICE.type == "cuda": out["latency_batch1_gpu"] = lat(DEVICE)
    t = time.perf_counter(); generate_embeddings(model, TEST_G, cfg); dt = time.perf_counter() - t
    out["gallery_embedding_time_s"] = dt; out["gallery_size"] = int(len(TEST_G)); out["gallery_ms_per_image_end_to_end"] = 1e3 * dt / max(1, len(TEST_G))
    return out

EFFICIENCY = measure_efficiency(FINAL, CFG, WORK / "best_model.pth")
print(json.dumps(EFFICIENCY, indent=2))

# %% [markdown]
# ## 7. Save artefacts (`best_model.pth`, `config.json`, `metrics.json`)

# %%
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and math.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o

torch.save({"model": FINAL.state_dict(), "cfg": asdict(CFG), "verification_threshold": CACHE_EVAL[FINAL_NAME]["thr"],
            "color_thresholds": {"similar": T_SIM, "different": T_DIFF}}, WORK / "best_model.pth")
json.dump(_clean(dict(asdict(CFG), device=str(DEVICE), color_threshold_similar=T_SIM, color_threshold_different=T_DIFF,
                      data_roots={k: str(v) for k, v in DATA_ROOTS.items()}, label_report=LABEL_REPORT,
                      final_model=FINAL_NAME, experiments=EXPERIMENTS)), open(WORK / "config.json", "w"), indent=2)
json.dump(_clean(dict(models=ALL_METRICS, ablation=ABLATION.to_dict("records"), efficiency=EFFICIENCY, cleaning=CLEAN_STATS,
                      split_summary=summary.reset_index().to_dict("records"), history=HISTORY)), open(WORK / "metrics.json", "w"), indent=2)
print("saved:", [p.name for p in WORK.glob("*") if p.suffix in {".pth", ".json", ".png"}])

# %% [markdown]
# ## 8. Approach note (≤ 500 characters)

# %%
APPROACH_NOTE = ("MobileNetV3-L (ImageNet)+GeM+256-d L2-normed head, shared-weight Siamese, cosine similarity. Colour invariance: "
 "heavy hue/sat/brightness jitter, grayscale, channel permute/mix + crop/rotation/flip. SupCon+batch-hard triplet on PxK batches "
 "with far-apart colourways as hard positives and colour-neighbour designs as hard negatives. Design-disjoint, near-dup-safe splits; "
 "eval = cross-colour retrieval (Top-K, MRR) and verification (ROC-AUC, EER) vs baseline ablation.")
print(len(APPROACH_NOTE), "chars"); assert len(APPROACH_NOTE) <= 500
print(APPROACH_NOTE)

# %% [markdown]
# ## 9. Technical defence notes
# **Why this works.** Pretrained conv features give edges/texture/motif primitives; GeM pooling preserves salient repeated motifs. The head maps to a 256-d hypersphere where cosine similarity is the only decision variable, for both retrieval and verification (one model, shared weights = Siamese). Colour invariance is *learned*, not hard-coded: the same design is shown under random hue/saturation/brightness/grayscale/channel changes while SupCon and triplet losses demand identical embeddings; colour-neighbour batches make "similar colour, different design" the dominant negative, so the cheapest solution for the network is to ignore palette and encode structure.
# **Honest limitations.** (i) Label quality bounds everything: in instance mode "same design" means "same image/near-duplicate", and the synthetic-recolor test is a proxy that shares augmentation families with training. (ii) Hue-shift augmentations can create colourways that never occur in real catalogues; (iii) some designs differ *only* by colour placement, which a colour-invariant model cannot separate by construction; (iv) dHash clustering at Hamming ≤ 4 may merge visually plain sarees into one split-group (safe for leakage, slightly reduces data diversity).
# **Extension to other garments.** Nothing is saree-specific: replace the data source/label inference, keep the sampler + losses; add per-category galleries or a category-prefix token if mixing garment types.
# **Ablations to be reported from the table above** (zero-shot, baseline, +colour aug, final): quote only the numbers measured on your run.
