#!/usr/bin/env python3.11
"""
train_pose_local.py — ZENTRA Pose / Fall Detection Training
============================================================
เทรน YOLOv8-pose สำหรับ Heat Stroke module
ตรวจจับ: fall, prone, standing, sitting, crouching

วิธีใช้:
  python train_pose_local.py --check
  python train_pose_local.py --mode all
  python train_pose_local.py --mode download
  python train_pose_local.py --mode train
  python train_pose_local.py --mode validate
  python train_pose_local.py --mode test --source 1
============================================================
"""

import os, sys, json, yaml, shutil, random, time, argparse
import cv2, numpy as np
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY",   "8xTIheqbzg4mkSLOdFe6")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")

CFG = {
    "base_model":  "yolov8m-pose.pt",   # pose model
    "epochs":      80,
    "batch":       16,
    "imgsz":       640,
    "device":      "0",
    "workers":     8,
    "val_split":   0.15,
    "patience":    25,
    "target_map50": 0.80,
}

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
RUNS_DIR    = BASE_DIR / "runs" / "pose"
MERGED_DIR  = DATA_DIR / "merged_pose"

for _d in [DATA_DIR, MODELS_DIR, RUNS_DIR, MERGED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Pose / Fall Classes ──────────────────────────────────────
POSE_CLASSES = [
    "standing",   # 0 ✅ ปกติ
    "sitting",    # 1 ✅ นั่งปกติ
    "crouching",  # 2 ✅ ก้มทำงาน
    "fall",       # 3 ❌ ล้ม
    "prone",      # 4 ❌ นอนคว่ำ/หงาย (หมดสติ)
    "lying",      # 5 ❌ นอนบนพื้น
]
NC       = len(POSE_CLASSES)
CLS_MAP  = {n: i for i, n in enumerate(POSE_CLASSES)}

ALIAS: dict[str, str] = {
    # standing
    "standing":"standing","stand":"standing","upright":"standing","walking":"standing",
    "person":"standing","worker":"standing",
    # sitting
    "sitting":"sitting","sit":"sitting","seated":"sitting",
    # crouching
    "crouching":"crouching","crouch":"crouching","bending":"crouching",
    "kneeling":"crouching","squatting":"crouching",
    # fall
    "fall":"fall","falling":"fall","fell":"fall","trip":"fall","stumble":"fall",
    "fallen":"fall","fall-detected":"fall","fall detected":"fall",
    # prone
    "prone":"prone","fainted":"prone","unconscious":"prone","motionless":"prone",
    "collapse":"prone","collapsed":"prone",
    # lying
    "lying":"lying","lying down":"lying","lie":"lying","on ground":"lying",
    "on floor":"lying","floor":"lying",
}

def remap(src_id: int, src_names: list) -> int | None:
    if src_id >= len(src_names): return None
    raw    = src_names[src_id].lower().strip()
    target = ALIAS.get(raw)
    if target: return CLS_MAP.get(target)
    for a, t in ALIAS.items():
        if a in raw or raw in a: return CLS_MAP.get(t)
    return None


# ================================================================
# SYSTEM CHECK
# ================================================================
def check_system():
    print("\n" + "═"*55)
    print("  ZENTRA Pose Training — System Check")
    print("═"*55)
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  PyTorch: {torch.__version__}  {'✅ CUDA' if cuda else '⚠️ CPU'}")
        if cuda:
            p    = torch.cuda.get_device_properties(0)
            vram = p.total_memory/1e9
            rb   = 16 if vram>=8 else 8 if vram>=6 else 4
            print(f"  GPU    : {p.name}  VRAM={vram:.1f}GB  → batch={rb}")
    except ImportError:
        print("  ❌ pip install torch --index-url https://download.pytorch.org/whl/cu121")
    try:
        import ultralytics as ul
        print(f"  Ultralytics: {ul.__version__} ✅")
    except ImportError:
        print("  ❌ pip install ultralytics")
    print("═"*55 + "\n")


# ================================================================
# DATASETS (fall / pose detection)
# ================================================================
DATASETS = [
    # Fall detection datasets
    {"ws": ROBOFLOW_WORKSPACE,            "proj": "fall-detection-ovjqo", "ver": 5,  "dest": "pose_ds_fall_main",  "note": "ZENTRA fall model"},
    {"ws": "roboflow-universe-projects",  "proj": "fall-detection-tlgol", "ver": 1,  "dest": "pose_ds_fall1",      "note": "Fall detection ~3k"},
    {"ws": "roboflow-universe-projects",  "proj": "fall-down-detect",     "ver": 1,  "dest": "pose_ds_fall2",      "note": "Fall down ~2k"},
    {"ws": "roboflow-universe-projects",  "proj": "human-action-recognition-dataset", "ver": 1, "dest": "pose_ds_action", "note": "Action recognition"},
    # Pose datasets (person bounding box for YOLO-pose)
    {"ws": "roboflow-universe-projects",  "proj": "person-detection-9a8mk","ver": 1, "dest": "pose_ds_person",     "note": "Person ~5k"},
]


def download_datasets() -> list[str]:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("❌  pip install roboflow")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    downloaded = []
    print(f"\n📥 Downloading {len(DATASETS)} pose datasets...")

    for ds in DATASETS:
        dest     = DATA_DIR / ds["dest"]
        existing = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
        if dest.exists() and len(existing) > 20:
            print(f"  ✅ {ds['proj']:<38} ({len(existing):,} images)")
            downloaded.append(str(dest))
            continue
        print(f"  📥 {ds['proj']:<38} [{ds['note']}]", end=" ", flush=True)
        try:
            proj = rf.workspace(ds["ws"]).project(ds["proj"])
            proj.version(ds["ver"]).download("yolov8", location=str(dest), overwrite=True)
            n = len(list(dest.rglob("*.jpg"))+list(dest.rglob("*.png")))
            print(f"→ {n:,} ✅")
            downloaded.append(str(dest))
        except Exception as e:
            print(f"→ ข้าม ({e})")

    return downloaded


# ================================================================
# MERGE
# ================================================================
def _get_names(ds_path):
    for name in ["data.yaml","dataset.yaml"]:
        y = Path(ds_path)/name
        if y.exists(): return yaml.safe_load(open(y)).get("names",[])
    for y in Path(ds_path).rglob("data.yaml"):
        return yaml.safe_load(open(y)).get("names",[])
    return []

def _find_label(img_path):
    lp = Path(str(img_path).replace(os.sep+"images"+os.sep, os.sep+"labels"+os.sep, 1)).with_suffix(".txt")
    if lp.exists(): return lp
    lp = img_path.with_suffix(".txt")
    return lp if lp.exists() else None


def merge_datasets(ds_paths: list[str]) -> str:
    print(f"\n🔀 Merging {len(ds_paths)} pose datasets...")
    for split in ["train","val"]:
        (MERGED_DIR/"images"/split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR/"labels"/split).mkdir(parents=True, exist_ok=True)

    cls_stats = {c:0 for c in POSE_CLASSES}
    total = skipped = 0
    random.seed(42)

    for ds_path in ds_paths:
        src_names = _get_names(ds_path)
        img_files = list(Path(ds_path).rglob("*.jpg")) + list(Path(ds_path).rglob("*.png"))
        random.shuffle(img_files)
        ds_name = Path(ds_path).name
        print(f"  📁 {ds_name:<25} {len(img_files):>5} imgs")

        for img_path in img_files:
            lbl_path = _find_label(img_path)
            if not lbl_path: skipped+=1; continue

            new_lines = []
            for line in lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                new_id = remap(int(parts[0]), src_names)
                if new_id is None: continue
                new_lines.append(f"{new_id} {' '.join(parts[1:5])}")
                cls_stats[POSE_CLASSES[new_id]] += 1

            if not new_lines: skipped+=1; continue
            img = cv2.imread(str(img_path))
            if img is None: skipped+=1; continue

            split = "val" if random.random() < CFG["val_split"] else "train"
            stem  = f"{ds_name}_{total:06d}"
            cv2.imwrite(str(MERGED_DIR/"images"/split/f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (MERGED_DIR/"labels"/split/f"{stem}.txt").write_text("\n".join(new_lines), encoding="utf-8")
            total += 1

    yaml_path = MERGED_DIR/"dataset.yaml"
    yaml.dump({"path": str(MERGED_DIR.resolve()), "train": "images/train", "val": "images/val",
               "nc": NC, "names": POSE_CLASSES}, open(yaml_path,"w"), allow_unicode=True)

    n_tr = len(list((MERGED_DIR/"images"/"train").glob("*.jpg")))
    n_va = len(list((MERGED_DIR/"images"/"val").glob("*.jpg")))
    print(f"\n  ✅ train={n_tr:,} val={n_va:,} skipped={skipped:,}")
    print("  Class distribution:")
    for cls, cnt in sorted(cls_stats.items(), key=lambda x:-x[1]):
        if cnt > 0:
            icon = "❌" if cls in ("fall","prone","lying") else "✅"
            bar  = "█"*min(cnt//30, 35)
            print(f"    {icon} {cls:<15} {cnt:>5,} {bar}")
    return str(yaml_path)


# ================================================================
# AUGMENTATION (สำหรับ pose — เน้น lighting + occlusion)
# ================================================================
def augment_dataset(dataset_yaml: str):
    with open(dataset_yaml) as f: ds_cfg = yaml.safe_load(f)
    img_dir = Path(ds_cfg["path"])/"images"/"train"
    lbl_dir = Path(ds_cfg["path"])/"labels"/"train"
    orig    = sorted(img_dir.glob("*.jpg"))
    mult    = 3   # pose dataset มักน้อย → augment 3x
    print(f"\n🎨 Augmenting {len(orig):,} pose images × {mult}...")

    added = 0
    for i, img_path in enumerate(orig):
        img    = cv2.imread(str(img_path))
        if img is None: continue
        lbl    = lbl_dir/img_path.with_suffix(".txt").name
        labels = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        h,w    = img.shape[:2]

        for k in range(mult):
            aug = img.copy()
            rng = random.Random(i*100+k)
            np.random.seed(i*100+k)

            # HSV
            if rng.random() < 0.85:
                hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,0] = np.clip(hsv[:,:,0]+rng.uniform(-25,25), 0, 179)
                hsv[:,:,1] = np.clip(hsv[:,:,1]*rng.uniform(0.4,1.6), 0, 255)
                hsv[:,:,2] = np.clip(hsv[:,:,2]*rng.uniform(0.35,1.65), 0, 255)
                aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # CLAHE (โรงงาน)
            if rng.random() < 0.40:
                lab = cv2.cvtColor(aug, cv2.COLOR_BGR2LAB)
                cl  = cv2.createCLAHE(clipLimit=rng.uniform(1.5,5.0), tileGridSize=(8,8))
                lab[:,:,0] = cl.apply(lab[:,:,0])
                aug = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Flip
            new_labels = labels
            if rng.random() < 0.5:
                aug = cv2.flip(aug, 1)
                flipped = []
                for line in labels.splitlines():
                    p = line.strip().split()
                    if len(p)>=5: p[1] = f"{1.0-float(p[1]):.6f}"
                    flipped.append(" ".join(p))
                new_labels = "\n".join(flipped)

            # Random dark patch (simulate shadow)
            if rng.random() < 0.30:
                px,py = rng.randint(0,w-1), rng.randint(0,h-1)
                pw = rng.randint(w//8, w//3)
                ph = rng.randint(h//8, h//3)
                x1,y1 = max(0,px-pw//2), max(0,py-ph//2)
                x2,y2 = min(w,px+pw//2), min(h,py+ph//2)
                aug[y1:y2,x1:x2] = (aug[y1:y2,x1:x2]*rng.uniform(0.2,0.6)).astype(np.uint8)

            # Blur
            if rng.random() < 0.20:
                aug = cv2.GaussianBlur(aug, (rng.choice([3,5]),)*2, 0)

            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(img_dir/f"{stem}.jpg"), aug, [cv2.IMWRITE_JPEG_QUALITY, 88])
            (lbl_dir/f"{stem}.txt").write_text(new_labels, encoding="utf-8")
            added += 1

    print(f"  ✅ +{added:,} augmented  |  Total: {len(list(img_dir.glob('*.jpg'))):,}")


# ================================================================
# TRAIN
# ================================================================
def train(dataset_yaml: str, epochs: int=None, batch: int=None,
          device: str=None, resume: bool=False) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    epochs = epochs or CFG["epochs"]
    batch  = batch  or CFG["batch"]
    device = device or CFG["device"]

    local = MODELS_DIR/"pose_finetuned.pt"
    weights = str(local) if (local.exists() and resume) else CFG["base_model"]
    model    = YOLO(weights)
    run_name = f"zentra_pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'═'*55}")
    print(f"  🚀 ZENTRA Pose Training")
    print(f"  Weights  : {weights}")
    print(f"  Dataset  : {dataset_yaml}")
    print(f"  Epochs   : {epochs}  Batch: {batch}  Device: {device}")
    print(f"{'═'*55}\n")

    model.train(
        data=dataset_yaml, epochs=epochs, batch=batch,
        imgsz=CFG["imgsz"], device=device,
        project=str(RUNS_DIR), name=run_name,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        momentum=0.937, weight_decay=0.0005, warmup_epochs=4,
        # augmentation
        mosaic=1.0, mixup=0.10, copy_paste=0.10, close_mosaic=10,
        fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, erasing=0.4,
        # loss
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=2.0,
        patience=CFG["patience"], save=True, save_period=10,
        val=True, plots=True, verbose=True,
        workers=CFG["workers"], amp=True, seed=42, resume=resume,
    )

    bests = sorted(RUNS_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not bests: raise FileNotFoundError("ไม่พบ best.pt")
    target = MODELS_DIR/"pose_finetuned.pt"
    shutil.copy2(bests[-1], target)
    print(f"\n✅ Pose model → {target}")
    return str(target)


# ================================================================
# VALIDATE
# ================================================================
def validate(dataset_yaml: str, model_path: str=None) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    m_path = model_path or str(MODELS_DIR/"pose_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌  ไม่พบ: {m_path}"); return {}

    model   = YOLO(m_path)
    metrics = model.val(data=dataset_yaml, imgsz=640, verbose=True, plots=True)
    map50   = float(metrics.box.map50)

    print(f"\n{'═'*50}")
    print(f"  📊 Pose Validation Results")
    print(f"  mAP50    : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    target = CFG["target_map50"]
    print(f"  {'🎉 ผ่านเป้า!' if map50>=target else '⚠️  ยังไม่ถึงเป้า → รัน --mode continue'}")
    if hasattr(metrics.box,"ap_class_index") and metrics.box.ap_class_index is not None:
        print("  Per-Class:")
        for i, ci in enumerate(metrics.box.ap_class_index):
            if ci < NC and i < len(metrics.box.ap50):
                ap = float(metrics.box.ap50[i])
                icon = "✅" if ap>=0.75 else ("⚠️" if ap>=0.55 else "❌")
                print(f"    {icon} {POSE_CLASSES[ci]:<14} {ap:.3f}  {'█'*int(ap*24)}")
    print("═"*50 + "\n")
    return {"mAP50": map50}


# ================================================================
# TEST LIVE
# ================================================================
def test_live(source: str="0", model_path: str=None, conf: float=0.40):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    m_path = model_path or str(MODELS_DIR/"pose_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌  ไม่พบ: {m_path}"); return

    src   = int(source) if source.isdigit() else source
    model = YOLO(m_path)
    cap   = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src,int) else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"❌  เปิด source ไม่ได้: {source}"); return

    DANGER = {"fall","prone","lying"}
    print(f"\n🎥 Pose Live Test | conf={conf} | Q=quit\n")
    while True:
        ret, frame = cap.read()
        if not ret: break
        results   = model.predict(frame, conf=conf, iou=0.45, verbose=False)
        annotated = results[0].plot()
        if len(results[0].boxes) > 0:
            detected = [POSE_CLASSES[int(c)] for c in results[0].boxes.cls.cpu().numpy()
                        if int(c)<NC]
            dangers  = [c for c in detected if c in DANGER]
            status   = f"🆘 {', '.join(set(dangers)).upper()}" if dangers else "✅ Normal"
            color    = (0,0,220) if dangers else (0,200,50)
            cv2.putText(annotated, status, (10,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imshow("ZENTRA Pose Test — Q=Quit", annotated)
        if cv2.waitKey(1)&0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="ZENTRA Pose/Fall Training")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mode", default="all",
        choices=["download","merge","augment","train","all","validate","test"])
    ap.add_argument("--dataset",  default=None)
    ap.add_argument("--epochs",   type=int, default=None)
    ap.add_argument("--batch",    type=int, default=None)
    ap.add_argument("--device",   default=None)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--model",    default=None)
    ap.add_argument("--source",   default="0")
    ap.add_argument("--conf",     type=float, default=0.40)
    args = ap.parse_args()

    if args.check: check_system(); return
    check_system()

    dataset_yaml = args.dataset
    paths = []

    if args.mode in ("download","all"):
        paths = download_datasets()
    else:
        paths = [str(d) for d in DATA_DIR.iterdir()
                 if d.is_dir() and d.name.startswith("pose_ds_")]

    if args.mode in ("merge","all"):
        if not paths: sys.exit("❌  รัน --mode download ก่อน")
        dataset_yaml = merge_datasets(paths)
    elif not dataset_yaml:
        y = MERGED_DIR/"dataset.yaml"
        dataset_yaml = str(y) if y.exists() else ""

    if args.mode in ("augment","all") and dataset_yaml:
        augment_dataset(dataset_yaml)

    if args.mode in ("train","all"):
        if not dataset_yaml or not Path(dataset_yaml).exists():
            sys.exit("❌  ไม่พบ dataset.yaml")
        train(dataset_yaml, args.epochs, args.batch, args.device, args.resume)

    if args.mode == "validate":
        if dataset_yaml: validate(dataset_yaml, args.model)

    if args.mode == "test":
        test_live(args.source, args.model, args.conf)

if __name__ == "__main__":
    main()
