#!/usr/bin/env python3.11
"""
train_all.py — ZENTRA Unified Training Pipeline
================================================
เทรน 3 โมเดลพร้อมกัน (หรือทีละตัว):
  1. PPE Detection     (yolov8m.pt  → ppe_finetuned.pt)
  2. Person/Pose       (yolov8m-pose.pt → pose_finetuned.pt)
  3. Heat Stroke LSTM  (fall_lstm.pt)

วิธีใช้:
  python train_all.py                         # เทรนทั้งหมด
  python train_all.py --task ppe              # เฉพาะ PPE
  python train_all.py --task pose             # เฉพาะ Pose/Fall
  python train_all.py --task lstm             # เฉพาะ LSTM
  python train_all.py --task ppe pose         # หลายตัว
  python train_all.py --check                 # ตรวจระบบ
  python train_all.py --task ppe --epochs 150 --batch 16
================================================
Dataset ที่ใช้ (download อัตโนมัติจาก Roboflow Universe):
  PPE   : 6 datasets รวม ~25k images
  Pose  : 5 datasets รวม ~15k images
  LSTM  : วิดีโอใน videos/<label>/
================================================
"""

from __future__ import annotations
import os, sys, argparse, json, yaml, shutil, random, time, math
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR   = BASE_DIR / "runs"
LOGS_DIR   = BASE_DIR / "logs"

for _d in [DATA_DIR, MODELS_DIR, RUNS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Roboflow ─────────────────────────────────────────────────────
def _auto_device() -> str:
    """ตรวจสอบ GPU อัตโนมัติ — เรียกได้ทันทีก่อน config dict"""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


RF_API_KEY   = os.getenv("ROBOFLOW_API_KEY",   "8xTIheqbzg4mkSLOdFe6")
RF_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")


# ================================================================
# ══════════════════════════════════════════════════════════════
#  PART 1: PPE MODEL
# ══════════════════════════════════════════════════════════════
# ================================================================

PPE_CLASSES = [
    "helmet", "no_helmet",
    "vest",   "no_vest",
    "gloves", "no_gloves",
    "goggles","no_goggles",
    "safety_boots","no_safety_boots",
    "person",
]
PPE_NC  = len(PPE_CLASSES)
PPE_MAP = {n: i for i, n in enumerate(PPE_CLASSES)}

PPE_ALIAS: dict[str, str] = {
    # helmet
    "hard-hat":"helmet","hardhat":"helmet","hard_hat":"helmet","helmet":"helmet",
    "head":"helmet","safety helmet":"helmet","protective helmet":"helmet",
    "safety-helmet":"helmet","helmet-on":"helmet",
    # no_helmet
    "no-hardhat":"no_helmet","no_hardhat":"no_helmet","no-helmet":"no_helmet",
    "no_helmet":"no_helmet","no helmet":"no_helmet","without helmet":"no_helmet",
    "w/o helmet":"no_helmet","bare-head":"no_helmet","no hard hat":"no_helmet",
    # vest
    "safety-vest":"vest","safety_vest":"vest","vest":"vest","reflective-vest":"vest",
    "hi-vis":"vest","hi_vis":"vest","safety vest":"vest","high visibility":"vest",
    "visibility vest":"vest","reflective vest":"vest","orange vest":"vest",
    "yellow vest":"vest","green vest":"vest",
    # no_vest
    "no-vest":"no_vest","no_vest":"no_vest","no vest":"no_vest",
    "without vest":"no_vest","w/o vest":"no_vest","no safety vest":"no_vest",
    # gloves
    "gloves":"gloves","safety-gloves":"gloves","safety_gloves":"gloves",
    "glove":"gloves","protective gloves":"gloves","work gloves":"gloves",
    # no_gloves
    "no-gloves":"no_gloves","no_gloves":"no_gloves","no gloves":"no_gloves",
    "without gloves":"no_gloves","w/o gloves":"no_gloves",
    # goggles
    "goggles":"goggles","safety-glasses":"goggles","safety_glasses":"goggles",
    "glasses":"goggles","eye-protection":"goggles","eyewear":"goggles",
    "safety goggles":"goggles","protective glasses":"goggles",
    "face-shield":"goggles","face shield":"goggles","eye protection":"goggles",
    # no_goggles
    "no-goggles":"no_goggles","no_goggles":"no_goggles","no goggles":"no_goggles",
    "no-glasses":"no_goggles","no glasses":"no_goggles","without glasses":"no_goggles",
    "no eye protection":"no_goggles",
    # boots
    "safety-boots":"safety_boots","safety_boots":"safety_boots","boots":"safety_boots",
    "safety boots":"safety_boots","steel-toed boots":"safety_boots",
    "steel toe":"safety_boots","protective boots":"safety_boots",
    # no_boots
    "no-boots":"no_safety_boots","no_boots":"no_safety_boots",
    "no boots":"no_safety_boots","without boots":"no_safety_boots",
    "no safety boots":"no_safety_boots",
    # person
    "person":"person","worker":"person","human":"person",
    "people":"person","man":"person","woman":"person","employee":"person",
    "pedestrian":"person","individual":"person",
}

# Dataset sources — ordered by quality (primary first)
PPE_DATASETS = [
    # ── Internal / high quality ──────────────────────────────────
    {"ws": RF_WORKSPACE,                  "proj":"ppe-cpxsz",                        "ver":2, "dest":"ppe_main",      "note":"ZENTRA main"},
    # ── High-quality public ──────────────────────────────────────
    {"ws":"roboflow-universe-projects",   "proj":"construction-site-safety",          "ver":1, "dest":"ppe_const",     "note":"~5k construction"},
    {"ws":"joseph-nelson",               "proj":"hard-hat-workers",                   "ver":2, "dest":"ppe_hardhat",   "note":"~7k hard hat"},
    {"ws":"roboflow-universe-projects",   "proj":"ppe-detection-ljb7d",               "ver":4, "dest":"ppe_ppe2",      "note":"~3k PPE"},
    {"ws":"roboflow-universe-projects",   "proj":"safety-equipment-detection-vwckw",  "ver":1, "dest":"ppe_safety_eq", "note":"~2k safety"},
    {"ws":"roboflow-universe-projects",   "proj":"worker-safety",                      "ver":3, "dest":"ppe_worker",   "note":"~4k worker safety"},
    # ── Additional datasets ───────────────────────────────────────
    {"ws":"roboflow-universe-projects",   "proj":"ppe-datasets-with-mask",            "ver":3, "dest":"ppe_mask",     "note":"~2k mask+PPE"},
    {"ws":"roboflow-universe-projects",   "proj":"safety-helmet-jswmx",               "ver":2, "dest":"ppe_helmet2",  "note":"~3k helmets"},
]

PPE_CFG = {
    "base_model":     "yolov8m.pt",
    "epochs":         120,
    "batch":          16,
    "imgsz":          640,
    "device":         "auto",
    "optimizer":      "AdamW",
    "lr0":            0.001,
    "lrf":            0.005,
    "warmup_epochs":  5,
    "patience":       35,
    "workers":        8,
    "cache":          False,
    "aug_multiplier": 2,
    "val_split":      0.15,
    "target_map50":   0.87,
}


# ================================================================
# ══════════════════════════════════════════════════════════════
#  PART 2: POSE / FALL MODEL
# ══════════════════════════════════════════════════════════════
# ================================================================

POSE_CLASSES = ["standing","sitting","crouching","fall","prone","lying"]
POSE_NC      = len(POSE_CLASSES)
POSE_MAP     = {n: i for i, n in enumerate(POSE_CLASSES)}

POSE_ALIAS: dict[str, str] = {
    "standing":"standing","stand":"standing","upright":"standing",
    "walking":"standing","person":"standing","worker":"standing","normal":"standing",
    "sitting":"sitting","sit":"sitting","seated":"sitting","seat":"sitting",
    "crouching":"crouching","crouch":"crouching","bending":"crouching",
    "kneeling":"crouching","squatting":"crouching","bend":"crouching",
    "fall":"fall","falling":"fall","fell":"fall","trip":"fall","stumble":"fall",
    "fallen":"fall","fall-detected":"fall","fall detected":"fall","accident":"fall",
    "prone":"prone","fainted":"prone","unconscious":"prone","motionless":"prone",
    "collapse":"prone","collapsed":"prone","heat stroke":"prone",
    "lying":"lying","lying down":"lying","lie":"lying","on ground":"lying",
    "on floor":"lying","floor":"lying","down":"lying",
}

POSE_DATASETS = [
    # ── workspace ของทีม ────────────────────────────────────────
    {"ws": RF_WORKSPACE, "proj":"fall-detection-ovjqo", "ver":5,
     "dest":"pose_main",   "note":"ZENTRA fall main"},
    # ── Reuse datasets ที่ download มาแล้วจาก PPE task ──────────
    # (person bounding box → remap เป็น standing)
    {"ws":"roboflow-universe-projects", "proj":"construction-site-safety", "ver":1,
     "dest":"pose_person1", "note":"reuse PPE — person bbox"},
    {"ws":"joseph-nelson", "proj":"hard-hat-workers", "ver":2,
     "dest":"pose_person2", "note":"reuse PPE — person bbox"},
    {"ws":"roboflow-universe-projects", "proj":"worker-safety", "ver":3,
     "dest":"pose_person3", "note":"reuse PPE — worker bbox"},
]

POSE_CFG = {
    "base_model":    "yolov8m.pt",   # ใช้ detect เพราะ dataset ไม่มี keypoint annotations
    "epochs":        100,
    "batch":         16,
    "imgsz":         640,
    "device":        "auto",
    "workers":       8,
    "val_split":     0.15,
    "patience":      30,
    "target_map50":  0.82,
    "aug_multiplier":3,
}


# ================================================================
# SHARED UTILITIES
# ================================================================

def _system_check():
    print("\n" + "═"*60)
    print("  ZENTRA — Unified Training System Check")
    print("═"*60)
    pv = sys.version_info
    print(f"  Python    : {pv.major}.{pv.minor}.{pv.micro}  "
          f"{'✅' if pv>=(3,10) else '❌'}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        ver  = torch.__version__
        if cuda:
            print(f"  PyTorch   : {ver}  ✅ CUDA")
            for i in range(torch.cuda.device_count()):
                p    = torch.cuda.get_device_properties(i)
                vram = p.total_memory/1e9
                rb   = 32 if vram>=10 else 16 if vram>=8 else 8 if vram>=6 else 4
                print(f"  GPU {i}     : {p.name}  VRAM={vram:.1f}GB  → batch≤{rb}")
        else:
            print(f"  PyTorch   : {ver}  ⚠️  CPU-only (ช้ากว่า GPU ~10-50x)")
            if "+cpu" in ver or "cpu" in ver.lower():
                print("  ⚠️  หากมี NVIDIA GPU ให้ติดตั้ง PyTorch CUDA ใหม่:")
                print("      pip uninstall torch torchvision torchaudio -y")
                print("      pip install torch torchvision torchaudio "
                      "--index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  PyTorch   : ❌")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    for pkg in ["ultralytics","roboflow","cv2","mediapipe"]:
        try:
            m = __import__(pkg)
            v = getattr(m,"__version__","?")
            print(f"  {pkg:<12}: {v}  ✅")
        except ImportError:
            print(f"  {pkg:<12}: ❌  pip install {pkg}")
    dev = _auto_device()
    print(f"  Device    : {dev}  {'✅ GPU' if dev != 'cpu' else '⚠️  CPU (ช้า)'}")
    free = shutil.disk_usage(BASE_DIR).free/1e9
    print(f"  Disk free : {free:.1f} GB  {'✅' if free>20 else '⚠️  ควรมี >20GB'}")
    print("═"*60+"\n")


def _get_names(ds_path: str) -> list[str]:
    for name in ["data.yaml","dataset.yaml"]:
        y = Path(ds_path)/name
        if y.exists():
            return yaml.safe_load(open(y)).get("names",[])
    for y in Path(ds_path).rglob("data.yaml"):
        return yaml.safe_load(open(y)).get("names",[])
    return []


def _find_label(img_path: Path) -> Path|None:
    lp = Path(
        str(img_path).replace(os.sep+"images"+os.sep,
                              os.sep+"labels"+os.sep, 1)
    ).with_suffix(".txt")
    if lp.exists(): return lp
    lp = img_path.with_suffix(".txt")
    return lp if lp.exists() else None


def _remap(src_id: int, src_names: list,
           alias: dict[str,str], cls_map: dict[str,int]) -> int|None:
    if src_id >= len(src_names): return None
    raw    = src_names[src_id].lower().strip()
    target = alias.get(raw)
    if target: return cls_map.get(target)
    for a, t in alias.items():
        if a in raw or raw in a: return cls_map.get(t)
    return None


def _download_datasets(datasets: list[dict], label: str) -> list[str]:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("❌  pip install roboflow")

    rf         = Roboflow(api_key=RF_API_KEY)
    downloaded = []
    print(f"\n📥 Downloading {len(datasets)} {label} datasets...")

    for ds in datasets:
        dest     = DATA_DIR / ds["dest"]
        existing = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
        if dest.exists() and len(existing) > 20:
            print(f"  ✅ {ds['proj']:<42} ({len(existing):,} images already)")
            downloaded.append(str(dest))
            continue
        print(f"  📥 {ds['proj']:<42} [{ds['note']}]", end=" ", flush=True)
        try:
            proj = rf.workspace(ds["ws"]).project(ds["proj"])
            proj.version(ds["ver"]).download("yolov8",
                location=str(dest), overwrite=True)
            # Check for zip error (Roboflow sometimes returns corrupt zip)
            n = len(list(dest.rglob("*.jpg"))+list(dest.rglob("*.png")))
            if n == 0:
                # Try downloading as yolov8 flat format
                proj.version(ds["ver"]).download("yolov8",
                    location=str(dest), overwrite=True)
                n = len(list(dest.rglob("*.jpg"))+list(dest.rglob("*.png")))
            if n == 0:
                print(f"→ ข้าม (ไม่มีรูปภาพหลัง download)")
                continue
            print(f"→ {n:,} images ✅")
            downloaded.append(str(dest))
        except Exception as e:
            msg = str(e)
            # Truncate long JSON error messages
            if len(msg) > 120:
                msg = msg[:120] + "..."
            print(f"→ ข้าม ({msg})")

    total = sum(
        len(list((DATA_DIR/ds["dest"]).rglob("*.jpg")))
        for ds in datasets if (DATA_DIR/ds["dest"]).exists()
    )
    print(f"\n  📦 {len(downloaded)}/{len(datasets)} datasets  |  ~{total:,} total images")
    return downloaded


def _merge_datasets(ds_paths: list[str], merged_dir: Path,
                    classes: list[str], alias: dict[str,str],
                    cls_map: dict[str,int], val_split: float,
                    kpt_shape: list = None) -> str:
    print(f"\n🔀 Merging {len(ds_paths)} datasets → {merged_dir.name}/")
    nc = len(classes)

    for split in ["train","val"]:
        (merged_dir/"images"/split).mkdir(parents=True, exist_ok=True)
        (merged_dir/"labels"/split).mkdir(parents=True, exist_ok=True)

    cls_stats = {c: 0 for c in classes}
    total = skipped = 0
    random.seed(42)

    for ds_path in ds_paths:
        src_names = _get_names(ds_path)
        img_files = (list(Path(ds_path).rglob("*.jpg")) +
                     list(Path(ds_path).rglob("*.png")))
        random.shuffle(img_files)
        ds_name = Path(ds_path).name
        print(f"  📁 {ds_name:<28} {len(img_files):>5} imgs | "
              f"classes:{src_names[:3]}")

        for img_path in img_files:
            lbl_path = _find_label(img_path)
            if not lbl_path: skipped+=1; continue

            new_lines = []
            for line in lbl_path.read_text(encoding="utf-8",
                                            errors="ignore").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                new_id = _remap(int(parts[0]), src_names, alias, cls_map)
                if new_id is None: continue
                new_lines.append(f"{new_id} {' '.join(parts[1:5])}")
                cls_stats[classes[new_id]] += 1

            if not new_lines: skipped+=1; continue
            img = cv2.imread(str(img_path))
            if img is None: skipped+=1; continue

            split = "val" if random.random() < val_split else "train"
            stem  = f"{ds_name}_{total:07d}"
            cv2.imwrite(
                str(merged_dir/"images"/split/f"{stem}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            (merged_dir/"labels"/split/f"{stem}.txt").write_text(
                "\n".join(new_lines), encoding="utf-8")
            total += 1

    yaml_path = merged_dir/"dataset.yaml"
    yaml_data = {
        "path":  str(merged_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    nc,
        "names": classes,
    }
    if kpt_shape is not None:
        yaml_data["kpt_shape"] = kpt_shape
    yaml.dump(yaml_data, open(yaml_path,"w"), allow_unicode=True)

    n_tr = len(list((merged_dir/"images"/"train").glob("*.jpg")))
    n_va = len(list((merged_dir/"images"/"val").glob("*.jpg")))
    print(f"\n  ✅ train={n_tr:,}  val={n_va:,}  skipped={skipped:,}")
    print("  Class distribution:")
    for cls, cnt in sorted(cls_stats.items(), key=lambda x:-x[1]):
        if cnt > 0:
            icon = "✅" if (not cls.startswith("no_") and cls!="person") \
                   else ("❌" if cls.startswith("no_") else "👤")
            bar  = "█" * min(cnt//80, 30)
            print(f"    {icon} {cls:<22} {cnt:>6,}  {bar}")
    return str(yaml_path)


def _augment_image(img: np.ndarray, labels: str, seed: int,
                   strong: bool = True) -> tuple[np.ndarray, str]:
    """
    9 augmentation techniques:
    1. Horizontal flip
    2. HSV jitter (แสงโรงงาน)
    3. Brightness/Contrast
    4. CLAHE
    5. Gaussian blur
    6. Gaussian noise
    7. Salt-and-pepper
    8. Rotation ±15°
    9. Random crop/zoom
    """
    rng = random.Random(seed)
    np.random.seed(seed % (2**31))
    h, w   = img.shape[:2]
    aug    = img.copy()
    lines  = [l.strip() for l in labels.splitlines() if l.strip()]
    nl     = list(lines)

    # 1. Horizontal flip
    if rng.random() < 0.50:
        aug = cv2.flip(aug, 1)
        nl  = []
        for line in lines:
            p = line.split()
            if len(p) >= 5:
                p[1] = f"{1.0-float(p[1]):.6f}"
            nl.append(" ".join(p))
        lines = nl[:]

    # 2. HSV jitter
    if rng.random() < 0.90:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = np.clip(hsv[:,:,0]+rng.uniform(-25,25), 0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*rng.uniform(0.40,1.60), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2]*rng.uniform(0.35,1.65), 0, 255)
        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Brightness/Contrast
    if rng.random() < 0.65:
        alpha = rng.uniform(0.50, 1.50)
        beta  = rng.uniform(-50, 50)
        aug   = np.clip(aug.astype(np.float32)*alpha+beta, 0, 255).astype(np.uint8)

    # 4. CLAHE (โรงงานแสงไม่สม่ำเสมอ)
    if rng.random() < 0.40:
        lab = cv2.cvtColor(aug, cv2.COLOR_BGR2LAB)
        cl  = cv2.createCLAHE(clipLimit=rng.uniform(1.5,5.0),
                               tileGridSize=(8,8))
        lab[:,:,0] = cl.apply(lab[:,:,0])
        aug = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. Gaussian blur
    if rng.random() < 0.25:
        k   = rng.choice([3,5,7])
        aug = cv2.GaussianBlur(aug, (k,k), 0)

    # 6. Gaussian noise
    if rng.random() < 0.22:
        noise = np.random.normal(0, rng.uniform(5,28), aug.shape).astype(np.int16)
        aug   = np.clip(aug.astype(np.int16)+noise, 0, 255).astype(np.uint8)

    # 7. Salt-and-pepper
    if rng.random() < 0.15:
        amount = rng.uniform(0.001, 0.007)
        n      = int(amount*aug.size)
        coords = [np.random.randint(0,d-1,n) for d in aug.shape[:2]]
        aug[coords[0],coords[1]] = 255
        coords = [np.random.randint(0,d-1,n) for d in aug.shape[:2]]
        aug[coords[0],coords[1]] = 0

    # 8. Rotation ±15°
    if strong and rng.random() < 0.35:
        angle = rng.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w,h),
                               borderMode=cv2.BORDER_REFLECT_101)

    # 9. Random crop / zoom-in 5-20%
    if strong and rng.random() < 0.30:
        pct = rng.uniform(0.05, 0.20)
        mx  = int(w*pct); my = int(h*pct)
        if mx>2 and my>2:
            aug = aug[my:h-my, mx:w-mx]
            aug = cv2.resize(aug, (w,h))
            rx, ry = w/(w-2*mx), h/(h-2*my)
            ox, oy = mx/w, my/h
            adj = []
            for line in nl:
                p = line.split()
                if len(p)>=5:
                    cx=(float(p[1])-ox)*rx
                    cy=(float(p[2])-oy)*ry
                    bw=float(p[3])*rx
                    bh=float(p[4])*ry
                    if 0.01<cx<0.99 and 0.01<cy<0.99:
                        adj.append(
                            f"{p[0]} {max(0.01,min(0.99,cx)):.6f} "
                            f"{max(0.01,min(0.99,cy)):.6f} "
                            f"{min(bw,1.0):.6f} {min(bh,1.0):.6f}")
            nl = adj

    return aug, "\n".join(nl)


def _augment_dataset(dataset_yaml: str, mult: int = 2, label: str = ""):
    with open(dataset_yaml) as f:
        ds_cfg = yaml.safe_load(f)
    img_dir = Path(ds_cfg["path"])/"images"/"train"
    lbl_dir = Path(ds_cfg["path"])/"labels"/"train"
    orig    = sorted(img_dir.glob("*.jpg"))
    print(f"\n🎨 [{label}] Augmenting {len(orig):,} images × {mult} (9 techniques)")

    added = 0; t0 = time.time()
    for i, img_path in enumerate(orig):
        img = cv2.imread(str(img_path))
        if img is None: continue
        lbl_f  = lbl_dir/img_path.with_suffix(".txt").name
        labels = lbl_f.read_text(encoding="utf-8") if lbl_f.exists() else ""
        for k in range(mult):
            aug_img, aug_lbl = _augment_image(img, labels, i*100+k)
            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(img_dir/f"{stem}.jpg"), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (lbl_dir/f"{stem}.txt").write_text(aug_lbl, encoding="utf-8")
            added += 1
        if (i+1) % 500 == 0:
            elapsed = time.time()-t0
            eta     = (len(orig)-i-1)/max((i+1)/elapsed, 0.01)
            print(f"   {i+1:>5}/{len(orig)}  ETA {eta/60:.1f}min")

    total = len(list(img_dir.glob("*.jpg")))
    print(f"  ✅ +{added:,} augmented  |  Total train: {total:,}")


def _train_yolo(dataset_yaml: str, output_pt: str,
                run_subdir: str, cfg_dict: dict,
                extra_kw: dict = None) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    weights  = cfg_dict["base_model"]
    local_pt = Path(output_pt)
    if local_pt.exists():
        print(f"  🔄 Fine-tune from existing: {local_pt.name}")
        weights = str(local_pt)

    # Resolve "auto" device at train time
    raw_dev = cfg_dict["device"]
    device  = _auto_device() if raw_dev == "auto" else raw_dev
    print(f"  [Train] device={device}")

    model    = YOLO(weights)
    run_dir  = RUNS_DIR / run_subdir
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    kw = dict(
        data    = dataset_yaml,
        epochs  = cfg_dict["epochs"],
        batch   = cfg_dict["batch"],
        imgsz   = cfg_dict["imgsz"],
        device  = device,
        project = str(run_dir),
        name    = run_name,
        optimizer     = cfg_dict.get("optimizer","AdamW"),
        lr0           = cfg_dict.get("lr0", 0.001),
        lrf           = cfg_dict.get("lrf", 0.01),
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = cfg_dict.get("warmup_epochs", 5),
        warmup_bias_lr= 0.1,
        # Built-in augmentation
        augment     = True,
        mosaic      = 1.0,
        mixup       = 0.15,
        copy_paste  = 0.10,
        close_mosaic= 15,
        fliplr      = 0.50,
        flipud      = 0.0,
        hsv_h       = 0.015,
        hsv_s       = 0.70,
        hsv_v       = 0.40,
        degrees     = 8.0,
        translate   = 0.10,
        scale       = 0.65,
        shear       = 2.0,
        perspective = 0.0005,
        erasing     = 0.40,
        # Loss
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,
        # Control
        patience    = cfg_dict.get("patience", 30),
        save        = True,
        save_period = 10,
        val         = True,
        plots       = True,
        verbose     = True,
        workers     = cfg_dict.get("workers", 8),
        cache       = cfg_dict.get("cache", False),
        amp         = True,
        seed        = 42,
        deterministic=True,
    )
    if extra_kw:
        kw.update(extra_kw)

    model.train(**kw)

    bests = sorted(run_dir.rglob("best.pt"),
                   key=lambda p: p.stat().st_mtime)
    if not bests:
        raise FileNotFoundError("ไม่พบ best.pt")
    shutil.copy2(bests[-1], output_pt)

    # Log
    LOGS_DIR.mkdir(exist_ok=True)
    log = {"run": run_name, "dataset": dataset_yaml,
           "output": output_pt,
           "timestamp": datetime.now().isoformat()}
    (LOGS_DIR/f"{run_name}.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False))

    print(f"\n  ✅ Model saved → {output_pt}")
    return output_pt


def _validate_yolo(dataset_yaml: str, model_path: str,
                   classes: list[str], target_map50: float):
    try:
        from ultralytics import YOLO
    except ImportError:
        return
    if not Path(model_path).exists():
        print(f"  ❌  ไม่พบ: {model_path}"); return

    model   = YOLO(model_path)
    metrics = model.val(data=dataset_yaml, imgsz=640,
                        verbose=True, plots=True)
    map50   = float(metrics.box.map50)
    nc      = len(classes)

    print(f"\n{'═'*55}")
    print(f"  📊 Validation — {Path(model_path).stem}")
    print(f"  mAP50     : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    if map50 >= target_map50:
        print(f"  🎉 ผ่านเป้า mAP50 ≥ {target_map50*100:.0f}%!")
    else:
        gap = (target_map50-map50)*100
        print(f"  ⚠️  ต่ำกว่าเป้า {gap:.1f}% → เพิ่ม epochs หรือ dataset")
    if (hasattr(metrics.box,"ap_class_index") and
            metrics.box.ap_class_index is not None):
        print("  Per-Class AP50:")
        for i, ci in enumerate(metrics.box.ap_class_index):
            if ci < nc and i < len(metrics.box.ap50):
                ap   = float(metrics.box.ap50[i])
                bar  = "█"*int(ap*24)
                icon = "✅" if ap>=0.80 else ("⚠️" if ap>=0.60 else "❌")
                print(f"    {icon} {classes[ci]:<22} {ap:.3f}  {bar}")
    print("═"*55)


def _export_model(model_path: str, fmt: str = "onnx"):
    try:
        from ultralytics import YOLO
    except ImportError:
        return
    if not Path(model_path).exists():
        print(f"  ❌ ไม่พบ: {model_path}"); return
    model = YOLO(model_path)
    kw    = {"format": fmt, "imgsz": 640}
    if fmt == "onnx":    kw.update({"simplify":True, "opset":17})
    elif fmt == "engine": kw.update({"half":True, "device":0})
    try:
        out  = model.export(**kw)
        dest = MODELS_DIR/Path(str(out)).name
        shutil.copy2(str(out), dest)
        print(f"  📦 Export {fmt.upper()} → {dest}")
    except Exception as e:
        print(f"  ❌ Export {fmt}: {e}")


# ================================================================
# ══════════════════════════════════════════════════════════════
#  TASK A: TRAIN PPE
# ══════════════════════════════════════════════════════════════
# ================================================================

def train_ppe(epochs: int = None, batch: int = None, export: bool = True):
    print("\n" + "█"*60)
    print("  TASK: PPE Detection Model Training")
    print("█"*60)

    if epochs: PPE_CFG["epochs"] = epochs
    if batch:  PPE_CFG["batch"]  = batch

    merged_dir = DATA_DIR/"merged_ppe"
    output_pt  = str(MODELS_DIR/"ppe_finetuned.pt")

    # Download
    ds_paths = _download_datasets(PPE_DATASETS, "PPE")
    if not ds_paths:
        print("❌  ไม่มี dataset — ตรวจ API key"); return

    # Merge
    dataset_yaml = _merge_datasets(
        ds_paths, merged_dir,
        PPE_CLASSES, PPE_ALIAS, PPE_MAP,
        PPE_CFG["val_split"],
    )

    # Augment
    _augment_dataset(dataset_yaml, PPE_CFG["aug_multiplier"], "PPE")

    # Train
    print(f"\n🚀 Training PPE — {PPE_CFG['epochs']} epochs  "
          f"batch={PPE_CFG['batch']}  device={PPE_CFG['device']}")
    _train_yolo(dataset_yaml, output_pt, "ppe", PPE_CFG)

    # Validate
    _validate_yolo(dataset_yaml, output_pt, PPE_CLASSES,
                   PPE_CFG["target_map50"])

    # Export ONNX
    if export:
        _export_model(output_pt, "onnx")

    print(f"\n  📁 PPE Model → {output_pt}")
    return output_pt


# ================================================================
# ══════════════════════════════════════════════════════════════
#  TASK B: TRAIN POSE / FALL
# ══════════════════════════════════════════════════════════════
# ================================================================

def train_pose(epochs: int = None, batch: int = None, export: bool = True):
    print("\n" + "█"*60)
    print("  TASK: Pose / Fall Detection Model Training")
    print("█"*60)

    if epochs: POSE_CFG["epochs"] = epochs
    if batch:  POSE_CFG["batch"]  = batch

    merged_dir = DATA_DIR/"merged_pose"
    output_pt  = str(MODELS_DIR/"pose_finetuned.pt")

    ds_paths = _download_datasets(POSE_DATASETS, "POSE")

    # ── Fallback: ถ้าดาวน์โหลดไม่ได้เลย ใช้ PPE dataset แทน
    #    (มี person class ซึ่ง remap เป็น "standing" ได้)
    if not ds_paths:
        print("\n  ⚠️  ไม่มี POSE dataset จาก Roboflow")
        print("  🔄 Fallback: ใช้ PPE dataset (person bounding boxes)")
        ppe_dirs = list(DATA_DIR.glob("ppe_*"))
        if ppe_dirs:
            ds_paths = [str(d) for d in ppe_dirs if d.is_dir()]
            print(f"  ✅ พบ {len(ds_paths)} PPE datasets ที่จะใช้แทน")
        else:
            print("  ❌ ไม่มี dataset เลย — รัน --task ppe ก่อน แล้วค่อยรัน pose")
            return None

    dataset_yaml = _merge_datasets(
        ds_paths, merged_dir,
        POSE_CLASSES, POSE_ALIAS, POSE_MAP,
        POSE_CFG["val_split"],
        # kpt_shape ไม่ใส่ (None) เพราะใช้ detect model กับ bbox dataset
    )

    _augment_dataset(dataset_yaml, POSE_CFG["aug_multiplier"], "POSE")

    print(f"\n🚀 Training Pose — {POSE_CFG['epochs']} epochs  "
          f"batch={POSE_CFG['batch']}  device={POSE_CFG['device']}")

    # ใช้ detect model (yolov8m.pt) เทรน fall/pose classes
    # เพราะ dataset ที่ใช้เป็น bounding box (ไม่มี keypoint annotations)
    # หากมี dataset ที่มี keypoints → เปลี่ยน base_model เป็น yolov8m-pose.pt
    # และเพิ่ม kpt_shape=[17,3] ใน _merge_datasets
    _train_yolo(dataset_yaml, output_pt, "pose", POSE_CFG)

    _validate_yolo(dataset_yaml, output_pt, POSE_CLASSES,
                   POSE_CFG["target_map50"])

    if export:
        _export_model(output_pt, "onnx")

    print(f"\n  📁 Pose Model → {output_pt}")
    return output_pt


# ================================================================
# ══════════════════════════════════════════════════════════════
#  TASK C: TRAIN FALL LSTM
# ══════════════════════════════════════════════════════════════
# ================================================================

SEQ_LEN   = 30
INPUT_DIM = 99
HIDDEN_DIM= 64
LSTM_LABELS = ["fall","gait_anomaly","lying","normal","sitting"]
N_LSTM    = len(LSTM_LABELS)
VIDEO_DIR = BASE_DIR/"videos"
SEQ_DIR   = DATA_DIR/"lstm_sequences"
LSTM_PT   = str(MODELS_DIR/"fall_lstm.pt")

def _init_mediapipe():
    """
    รองรับ MediaPipe ทั้ง API เก่า (< 0.10) และใหม่ (0.10+)
    คืน (pose_instance, process_fn, MP_OK)
    """
    # ── วิธีที่ 1: API เก่า (solutions) — mediapipe < 0.10 ─────
    try:
        import mediapipe as mp
        pose_inst = mp.solutions.pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        def _process_old(bgr_frame):
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            res = pose_inst.process(rgb)
            if res.pose_landmarks:
                return np.array([
                    [lm.x, lm.y, lm.visibility]
                    for lm in res.pose_landmarks.landmark
                ], dtype=np.float32).flatten()
            return None
        print("  ✅ MediaPipe ready (legacy API)")
        return pose_inst, _process_old, True
    except AttributeError:
        pass  # solutions ไม่มี → ลอง API ใหม่
    except Exception as e:
        print(f"  ⚠️  MediaPipe legacy API: {e}")

    # ── วิธีที่ 2: API ใหม่ (tasks) — mediapipe 0.10+ ──────────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = DATA_DIR / "pose_landmarker.task"
        if not model_path.exists():
            print("  📥 Downloading MediaPipe pose model...")
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_heavy/float16/1/"
                   "pose_landmarker_heavy.task")
            urllib.request.urlretrieve(url, str(model_path))
            print(f"  ✅ Downloaded → {model_path}")

        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts      = mp_vision.PoseLandmarkerOptions(
            base_options         = base_opts,
            output_segmentation_masks = False,
            num_poses            = 1,
            min_pose_detection_confidence = 0.5,
            min_tracking_confidence       = 0.5,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

        def _process_new(bgr_frame):
            rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                return np.array(
                    [[lm.x, lm.y, lm.visibility] for lm in lms],
                    dtype=np.float32
                ).flatten()
            return None

        print("  ✅ MediaPipe ready (tasks API 0.10+)")
        return landmarker, _process_new, True

    except Exception as e:
        print(f"  ❌ MediaPipe tasks API: {e}")

    print("  ⚠️  MediaPipe ไม่พร้อม → จะใช้ synthetic sequences แทน")
    return None, None, False


def _generate_synthetic_sequences():
    """
    สร้าง synthetic keypoint sequences เมื่อไม่มีวิดีโอและ MediaPipe
    ใช้ rule-based simulation ของ pose patterns แต่ละ class
    """
    print("\n🤖 Generating synthetic LSTM sequences (no video needed)...")
    SEQ_DIR.mkdir(exist_ok=True)

    n_per_class = 200
    rng = np.random.RandomState(42)

    def _base_standing():
        """skeleton ท่ายืน (normalized 0-1)"""
        kp = np.zeros((33, 3))
        # head
        kp[0]  = [0.50, 0.10, 0.99]
        # shoulders
        kp[11] = [0.42, 0.30, 0.99]; kp[12] = [0.58, 0.30, 0.99]
        # hips
        kp[23] = [0.44, 0.55, 0.99]; kp[24] = [0.56, 0.55, 0.99]
        # knees
        kp[25] = [0.44, 0.75, 0.99]; kp[26] = [0.56, 0.75, 0.99]
        # ankles
        kp[27] = [0.44, 0.92, 0.99]; kp[28] = [0.56, 0.92, 0.99]
        # elbows, wrists
        kp[13] = [0.36, 0.48, 0.9];  kp[14] = [0.64, 0.48, 0.9]
        kp[15] = [0.34, 0.62, 0.8];  kp[16] = [0.66, 0.62, 0.8]
        return kp.astype(np.float32)

    generators = {
        "normal": lambda i, t: _base_standing() + rng.normal(0, 0.008, (33, 3)).astype(np.float32),
        "sitting": lambda i, t: _make_sitting(rng),
        "fall": lambda i, t: _make_fall(rng, t, SEQ_LEN),
        "lying": lambda i, t: _make_lying(rng),
        "gait_anomaly": lambda i, t: _make_gait_anomaly(rng, t),
    }

    def _make_sitting(rng):
        kp = _base_standing()
        kp[23][1] = 0.58; kp[24][1] = 0.58   # hips lower
        kp[25][1] = 0.62; kp[26][1] = 0.62   # knees at hip level
        kp[25][0] = 0.40; kp[26][0] = 0.60   # knees apart
        kp[27][1] = 0.70; kp[28][1] = 0.70   # ankles forward
        return kp + rng.normal(0, 0.006, (33, 3)).astype(np.float32)

    def _make_fall(rng, t, seq_len):
        kp   = _base_standing()
        prog = t / seq_len
        if prog < 0.4:
            kp += rng.normal(0, 0.012 * (1 + prog * 5), (33, 3)).astype(np.float32)
        else:
            # body tilts sideways
            tilt = (prog - 0.4) / 0.6
            kp[:, 1] += tilt * 0.3
            kp[:, 0] += tilt * 0.2 * (1 if rng.random() > 0.5 else -1)
            kp        = np.clip(kp, 0, 1)
        return kp.astype(np.float32)

    def _make_lying(rng):
        kp = _base_standing()
        # rotate 90° — x stays, y collapses
        kp[:, 0] = kp[:, 1] * 1.5 + 0.1   # spread along x
        kp[:, 1] = rng.uniform(0.45, 0.55, 33).astype(np.float32)
        kp = np.clip(kp, 0, 1)
        return kp + rng.normal(0, 0.008, (33, 3)).astype(np.float32)

    def _make_gait_anomaly(rng, t):
        kp  = _base_standing()
        osc = 0.06 * np.sin(t * 0.8 + rng.uniform(0, np.pi))
        kp[:, 0] += osc + rng.normal(0, 0.015, 33)
        kp[:, 1] += rng.normal(0, 0.018, 33)
        return np.clip(kp, 0, 1).astype(np.float32)

    total = 0
    for label, gen_fn in generators.items():
        out_dir = SEQ_DIR / label
        out_dir.mkdir(exist_ok=True)
        for i in range(n_per_class):
            frames = []
            for t in range(SEQ_LEN):
                kp  = gen_fn(i, t)
                kp[:, 2] = np.clip(kp[:, 2], 0, 1)
                # visibility เป็น 0 สำหรับ joints ที่ไม่ได้ define
                mask = kp[:, 0] == 0
                kp[mask, 2] = 0
                frames.append(kp.flatten())
            seq = np.array(frames, dtype=np.float32)
            np.save(str(out_dir / f"synthetic_{i:04d}.npy"), seq)
            total += 1

    print(f"  ✅ Generated {total} synthetic sequences "
          f"({n_per_class} per class × {len(generators)} classes)")
    return total


def train_lstm(epochs: int = 100, batch: int = 32,
               aug_x: int = 3, device: str = "auto",
               lr: float = 1e-3):
    print("\n" + "█"*60)
    print("  TASK: Fall / Heat Stroke LSTM Training")
    print("█"*60)

    # ── MediaPipe (รองรับทั้ง API เก่าและใหม่) ─────────────────
    pose_inst, process_fn, MP_OK = _init_mediapipe()

    # ── Check video sources ─────────────────────────────────────
    VIDEO_DIR.mkdir(exist_ok=True)
    total_vids = 0
    for lbl in LSTM_LABELS:
        vd = VIDEO_DIR / lbl
        vd.mkdir(exist_ok=True)
        n = (len(list(vd.glob("*.mp4"))) + len(list(vd.glob("*.avi")))
             + len(list(vd.glob("*.mov"))))
        print(f"  videos/{lbl:<18}: {n} videos")
        total_vids += n

    if total_vids == 0:
        print("\n  ⚠️  ไม่พบวิดีโอ")
        print("  ℹ️  วางวิดีโอใน videos/<label>/ สำหรับผลลัพธ์ที่ดีที่สุด")
        print(f"  Labels: {LSTM_LABELS}")

    # ── Extract sequences from videos (ถ้ามี) ───────────────────
    if MP_OK and total_vids > 0:
        print("\n🎬 Extracting keypoint sequences from videos...")
        SEQ_DIR.mkdir(exist_ok=True)
        total_seqs = 0

        for lbl in LSTM_LABELS:
            vid_dir = VIDEO_DIR / lbl
            out_dir = SEQ_DIR / lbl
            out_dir.mkdir(exist_ok=True)
            vids = (list(vid_dir.glob("*.mp4")) + list(vid_dir.glob("*.avi"))
                    + list(vid_dir.glob("*.mov")))
            if not vids:
                continue
            print(f"  📁 {lbl}: {len(vids)} videos")
            for vid_path in vids:
                cap        = cv2.VideoCapture(str(vid_path))
                frames_buf = []
                seq_id     = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    kp = process_fn(frame)
                    if kp is None:
                        kp = np.zeros(INPUT_DIM, dtype=np.float32)
                    frames_buf.append(kp)
                    if len(frames_buf) >= SEQ_LEN:
                        if (len(frames_buf) - SEQ_LEN) % 5 == 0:
                            seq = np.array(frames_buf[-SEQ_LEN:],
                                           dtype=np.float32)
                            np.save(str(out_dir /
                                        f"{vid_path.stem}_{seq_id:04d}.npy"),
                                    seq)
                            seq_id += 1
                            total_seqs += 1
                        if seq_id >= 300:
                            break
                cap.release()
                print(f"     {vid_path.name:<40} → {seq_id} seqs")
        print(f"  ✅ Total sequences from videos: {total_seqs}")

    # ── Check existing sequences ─────────────────────────────────
    existing_seqs = sum(
        len(list((SEQ_DIR / lbl).glob("*.npy")))
        for lbl in LSTM_LABELS if (SEQ_DIR / lbl).exists()
    )

    # ── Auto-generate synthetic if not enough real data ──────────
    if existing_seqs < 50:
        print(f"\n  ℹ️  Sequences ที่มีอยู่: {existing_seqs} "
              f"(ต้องการอย่างน้อย 50)")
        print("  🤖 Auto-generating synthetic sequences...")
        _generate_synthetic_sequences()

    # ── Augment sequences ────────────────────────────────────────
    total_before_aug = sum(
        len(list((SEQ_DIR / lbl).glob("*.npy")))
        for lbl in LSTM_LABELS if (SEQ_DIR / lbl).exists()
    )
    if total_before_aug > 0:
        print(f"\n🎨 Augmenting {total_before_aug} sequences × {aug_x}...")
        _aug_seqs(SEQ_DIR, aug_x)

    # ── Final count ──────────────────────────────────────────────
    total_seqs_final = sum(
        len(list((SEQ_DIR / lbl).glob("*.npy")))
        for lbl in LSTM_LABELS if (SEQ_DIR / lbl).exists()
    )

    if total_seqs_final < 20:
        print("  ❌ ไม่มี sequences เพียงพอ — ตรวจสอบ paths")
        return None

    print(f"\n🚀 Training LSTM — {epochs} epochs  batch={batch}  "
          f"sequences={total_seqs_final}")
    _train_lstm_model(epochs, batch, device, lr)
    return LSTM_PT


def _aug_seq(seq: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed % (2**31))
    aug = seq.copy()
    if rng.random() < 0.7:
        noise = rng.normal(0, 0.01, (len(aug), INPUT_DIM))
        noise[:, 2::3] = 0
        aug = np.clip(aug+noise, 0, 1)
    if rng.random() < 0.5:
        scale = rng.uniform(0.85, 1.15)
        aug[:,0::3] = np.clip(aug[:,0::3]*scale, 0, 1)
        aug[:,1::3] = np.clip(aug[:,1::3]*scale, 0, 1)
    if rng.random() < 0.5:
        aug[:,0::3] = 1.0-aug[:,0::3]
    if rng.random() < 0.3:
        n       = len(aug)
        new_idx = np.sort(rng.choice(n, size=n, replace=True))
        aug     = aug[new_idx]
    return aug.astype(np.float32)


def _aug_seqs(seq_root: Path, mult: int = 2):
    added = 0
    for label_dir in seq_root.iterdir():
        if not label_dir.is_dir(): continue
        for seq_path in list(label_dir.glob("*.npy")):
            seq = np.load(seq_path).astype(np.float32)
            for k in range(mult):
                aug = _aug_seq(seq, hash(str(seq_path))+k)
                np.save(str(label_dir/f"{seq_path.stem}_aug{k}.npy"), aug)
                added += 1
    print(f"  ✅ +{added} augmented sequences")


def _train_lstm_model(epochs: int, batch: int, device: str, lr: float):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        sys.exit("❌  pip install torch")

    dev_s = ("cuda" if (device=="auto" and torch.cuda.is_available())
             else device)
    dev   = torch.device("cuda" if dev_s=="cuda" and
                         torch.cuda.is_available() else "cpu")

    label_dirs = sorted([d for d in SEQ_DIR.iterdir() if d.is_dir()])
    lmap       = {d.name: i for i, d in enumerate(label_dirs)}
    n_cls      = len(lmap)

    class _SeqDS(Dataset):
        def __init__(self, files):
            self.data = []
            for npy, lid in files:
                seq = np.load(npy).astype(np.float32)
                if seq.shape[0] < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN-seq.shape[0], INPUT_DIM), dtype=np.float32)
                    seq = np.concatenate([seq, pad])
                self.data.append((seq[:SEQ_LEN], lid))
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            x, y = self.data[i]
            return torch.tensor(x), torch.tensor(y, dtype=torch.long)

    all_files = []
    for d in label_dirs:
        lid = lmap[d.name]
        for npy in sorted(d.glob("*.npy")):
            all_files.append((npy, lid))
    random.Random(42).shuffle(all_files)
    n_tr = int(len(all_files)*0.85)
    tr_ds = _SeqDS(all_files[:n_tr])
    va_ds = _SeqDS(all_files[n_tr:])
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(INPUT_DIM)
            self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, num_layers=2,
                                batch_first=True, dropout=0.3,
                                bidirectional=True)
            self.attn = nn.Linear(HIDDEN_DIM*2, 1)
            self.fc   = nn.Sequential(
                nn.Linear(HIDDEN_DIM*2, 64), nn.GELU(),
                nn.Dropout(0.25), nn.Linear(64, n_cls))
        def forward(self, x):
            x = self.norm(x)
            out, _ = self.lstm(x)
            w      = torch.softmax(self.attn(out), dim=1)
            ctx    = (out*w).sum(dim=1)
            return self.fc(ctx)

    model = _LSTM().to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr*5, steps_per_epoch=len(tr_ld),
        epochs=epochs, pct_start=0.1)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"  Classes: {lmap}")
    print(f"  Train: {len(tr_ds)}  Val: {len(va_ds)}  Device: {dev}")

    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        tl = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                pred = model(xb.to(dev)).argmax(1).cpu()
                correct += (pred==yb).sum().item()
                total   += len(yb)
        acc = correct/max(total,1)
        if ep % 10 == 0 or acc > best_acc:
            print(f"  Ep {ep:>3}  loss={tl/len(tr_ld):.3f}  acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.cpu().state_dict(), LSTM_PT)
            model.to(dev)

    json.dump(lmap, open(str(MODELS_DIR/"fall_lstm_labels.json"),"w"))
    print(f"\n  🎉 Best LSTM acc = {best_acc:.3f}")
    print(f"  📁 LSTM Model → {LSTM_PT}")


# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description="ZENTRA Unified Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--check",  action="store_true",
                    help="ตรวจระบบ")
    ap.add_argument("--task",   nargs="+",
                    choices=["ppe","pose","lstm","all"],
                    default=["all"],
                    help="เลือก task ที่จะเทรน (ค่าเริ่มต้น: all)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch",  type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="0=GPU  cpu=CPU  0,1=Multi-GPU  (ค่าเริ่มต้น: auto-detect)")
    ap.add_argument("--no-export", action="store_true",
                    help="ไม่ export ONNX หลัง train")
    ap.add_argument("--aug-x",  type=int, default=None,
                    help="จำนวน augment multiplier")
    # LSTM specific
    ap.add_argument("--lstm-epochs", type=int, default=100)
    ap.add_argument("--lstm-batch",  type=int, default=32)
    ap.add_argument("--lstm-lr",     type=float, default=1e-3)
    ap.add_argument("--lstm-aug-x",  type=int, default=3)
    args = ap.parse_args()

    if args.aug_x:
        PPE_CFG["aug_multiplier"]  = args.aug_x
        POSE_CFG["aug_multiplier"] = args.aug_x

    # Device override (ถ้า --device ระบุมา ให้ใช้ค่านั้น ไม่งั้น auto)
    if args.device is not None:
        PPE_CFG["device"]  = args.device
        POSE_CFG["device"] = args.device
    print(f"  [Device] PPE={PPE_CFG['device']}  POSE={POSE_CFG['device']}")

    _system_check()

    if args.check:
        return

    tasks = args.task
    if "all" in tasks:
        tasks = ["ppe","pose","lstm"]

    do_export = not args.no_export
    results   = {}

    for task in tasks:
        if task == "ppe":
            results["ppe"] = train_ppe(args.epochs, args.batch, do_export)
        elif task == "pose":
            results["pose"] = train_pose(args.epochs, args.batch, do_export)
        elif task == "lstm":
            results["lstm"] = train_lstm(
                args.lstm_epochs, args.lstm_batch,
                args.lstm_aug_x, args.device, args.lstm_lr)

    print("\n" + "═"*60)
    print("  🎉 ZENTRA Training Complete!")
    for task, path in results.items():
        if path:
            print(f"  {task.upper():<6}: {path}")
    print("═"*60+"\n")


if __name__ == "__main__":
    main()
