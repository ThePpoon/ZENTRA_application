#!/usr/bin/env python3
"""
use_existing_data.py -- ZENTRA: Use Pre-Downloaded Data for Training
=====================================================================
Uses your existing datasets in D:\\ZENTRA_V1\\Data\\
instead of downloading from Roboflow again.

Steps:
  1. Copies/links existing data into the project
  2. Merges datasets if needed
  3. Trains PPE + Pose models using local data

Usage:
  python use_existing_data.py --check              # show what data is available
  python use_existing_data.py --task ppe           # train PPE only
  python use_existing_data.py --task pose          # train Pose only
  python use_existing_data.py --task ppe pose      # train both
  python use_existing_data.py --task ppe --epochs 150
=====================================================================
"""

from __future__ import annotations
import sys
import io
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import shutil
import yaml
import json
import random
from pathlib import Path
from datetime import datetime

# ================================================================
# PATHS
# ================================================================
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "models"
RUNS_DIR      = BASE_DIR / "runs"
LOGS_DIR      = BASE_DIR / "logs"

# Your existing data folder (change if different drive/path)
EXISTING_DATA = Path(r"D:\ZENTRA_V1\Data")

# Where merged training data will be placed
MERGED_PPE    = BASE_DIR / "data" / "merged_ppe_local"
MERGED_POSE   = BASE_DIR / "data" / "merged_pose_local"

for d in [MODELS_DIR, RUNS_DIR, LOGS_DIR, MERGED_PPE, MERGED_POSE]:
    d.mkdir(parents=True, exist_ok=True)

# ================================================================
# PPE CONFIG
# ================================================================
PPE_CLASSES = [
    "helmet", "no_helmet",
    "vest",   "no_vest",
    "gloves", "no_gloves",
    "goggles","no_goggles",
    "safety_boots","no_safety_boots",
    "person",
]
PPE_NC = len(PPE_CLASSES)

# Map from dataset folder names to our class names
PPE_KNOWN_FOLDERS = [
    "ppe_main", "merged_ppe", "ppe_const", "ppe_hardhat",
    "train_dataset", "annotations",
]

# ================================================================
# POSE CONFIG
# ================================================================
POSE_CLASSES = ["standing", "sitting", "crouching", "fall", "prone", "lying"]
POSE_NC      = len(POSE_CLASSES)

POSE_KNOWN_FOLDERS = [
    "pose_main", "merged_pose", "pose_person1", "pose_person2",
    "pose_ds_fall_main",
]

# ================================================================
# HELPERS
# ================================================================
def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            print(f"[Device] GPU detected: {name} (count={gpus})")
            return "0"
    except ImportError:
        pass
    print("[Device] No GPU -- using CPU (training will be slow)")
    return "cpu"


def _count_images(folder: Path) -> int:
    """Count image files in a folder tree"""
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def _count_labels(folder: Path) -> int:
    """Count YOLO label files"""
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*.txt") if f.name != "classes.txt")


# ================================================================
# CHECK
# ================================================================
def check_existing_data():
    print(f"\n======== Existing Data at {EXISTING_DATA} ========")
    if not EXISTING_DATA.exists():
        print(f"  ERROR: Folder not found: {EXISTING_DATA}")
        print(f"  Edit EXISTING_DATA path in this script")
        return

    total_images = 0
    for folder in sorted(EXISTING_DATA.iterdir()):
        if not folder.is_dir():
            continue
        n_img = _count_images(folder)
        n_lbl = _count_labels(folder)
        total_images += n_img

        if n_img > 0 or n_lbl > 0:
            tag = ""
            fname = folder.name.lower()
            if any(x in fname for x in ["ppe", "helmet", "vest", "hardhat"]):
                tag = "[PPE]"
            elif any(x in fname for x in ["pose", "fall", "person"]):
                tag = "[POSE]"
            elif "lstm" in fname:
                tag = "[LSTM]"
            print(f"  {tag:<7} {folder.name:<25} images={n_img:>5}  labels={n_lbl:>5}")
        else:
            # Check for yaml/json
            yamls = list(folder.glob("*.yaml")) + list(folder.glob("*.yml"))
            jsons = list(folder.glob("*.json"))
            if yamls or jsons:
                print(f"  [META]  {folder.name:<25} (config files only)")

    print(f"\n  Total images found: {total_images:,}")
    print("====================================================\n")


# ================================================================
# COLLECT YOLO IMAGES + LABELS
# ================================================================
def _collect_yolo_pairs(src_dir: Path, dest_img: Path, dest_lbl: Path,
                         split: str = "train") -> int:
    """
    Find all image+label pairs in src_dir and copy to dest dirs.
    Handles both flat and train/valid/test subdirectory structures.
    Returns number of pairs found.
    """
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)
    count = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Search paths: direct + common YOLO subdirs
    search_dirs = [src_dir]
    for sub in ["images", "train", "valid", "test", split,
                 f"images/{split}", f"images/train"]:
        candidate = src_dir / sub
        if candidate.exists():
            search_dirs.append(candidate)

    for img_dir in search_dirs:
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in exts:
                continue
            # Find corresponding label
            lbl_candidates = [
                img_file.with_suffix(".txt"),
                img_dir.parent / "labels" / img_file.with_suffix(".txt").name,
                src_dir / "labels" / img_file.with_suffix(".txt").name,
                src_dir / "labels" / split / img_file.with_suffix(".txt").name,
            ]
            lbl_file = next((l for l in lbl_candidates if l.exists()), None)
            if lbl_file is None:
                continue

            dest_i = dest_img / img_file.name
            dest_l = dest_lbl / lbl_file.name
            if not dest_i.exists():
                shutil.copy2(img_file, dest_i)
            if not dest_l.exists():
                shutil.copy2(lbl_file, dest_l)
            count += 1

    return count


# ================================================================
# BUILD DATASET YAML
# ================================================================
def _make_yaml(dest_dir: Path, class_names: list[str], yaml_path: Path):
    """Create YOLO dataset.yaml"""
    nc = len(class_names)
    data = {
        "path":  str(dest_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    nc,
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[Build] Created: {yaml_path}")


def _split_dataset(src_img: Path, src_lbl: Path,
                   dst_root: Path, val_ratio: float = 0.15):
    """Split flat dataset into train/val structure"""
    images = sorted(src_img.glob("*"))
    random.Random(42).shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))
    splits = {
        "val":   images[:n_val],
        "train": images[n_val:],
    }
    for split_name, split_imgs in splits.items():
        (dst_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            lbl = src_lbl / img.with_suffix(".txt").name
            dst_img = dst_root / "images" / split_name / img.name
            dst_lbl = dst_root / "labels" / split_name / lbl.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if lbl.exists() and not dst_lbl.exists():
                shutil.copy2(lbl, dst_lbl)

    n_train = len(splits["train"])
    n_val2  = len(splits["val"])
    print(f"[Split] train={n_train}  val={n_val2}")
    return n_train, n_val2


# ================================================================
# PREPARE PPE DATASET
# ================================================================
def prepare_ppe_dataset() -> Path:
    print("\n[PPE] Preparing dataset from existing data...")
    tmp_img = MERGED_PPE / "_tmp_images"
    tmp_lbl = MERGED_PPE / "_tmp_labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    for folder_name in PPE_KNOWN_FOLDERS:
        src = EXISTING_DATA / folder_name
        if src.exists():
            n = _collect_yolo_pairs(src, tmp_img, tmp_lbl)
            print(f"[PPE]   {folder_name}: +{n} pairs")
            total += n
        else:
            print(f"[PPE]   {folder_name}: not found (skip)")

    # Also scan for any other PPE-like folders
    if EXISTING_DATA.exists():
        for d in EXISTING_DATA.iterdir():
            if d.is_dir() and d.name.lower() not in [f.lower() for f in PPE_KNOWN_FOLDERS]:
                n_img = _count_images(d)
                if n_img > 50:
                    n = _collect_yolo_pairs(d, tmp_img, tmp_lbl)
                    if n > 0:
                        print(f"[PPE]   {d.name} (auto): +{n} pairs")
                        total += n

    print(f"[PPE] Total pairs collected: {total}")
    if total == 0:
        print("[PPE] ERROR: No training data found!")
        return None

    # Split into train/val
    n_train, n_val = _split_dataset(tmp_img, tmp_lbl, MERGED_PPE)

    # Create yaml
    yaml_path = MERGED_PPE / "dataset.yaml"
    _make_yaml(MERGED_PPE, PPE_CLASSES, yaml_path)

    # Cleanup tmp
    shutil.rmtree(tmp_img, ignore_errors=True)
    shutil.rmtree(tmp_lbl, ignore_errors=True)

    print(f"[PPE] Dataset ready: train={n_train}  val={n_val}")
    return yaml_path


# ================================================================
# PREPARE POSE DATASET
# ================================================================
def prepare_pose_dataset() -> Path:
    print("\n[Pose] Preparing dataset from existing data...")
    tmp_img = MERGED_POSE / "_tmp_images"
    tmp_lbl = MERGED_POSE / "_tmp_labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    for folder_name in POSE_KNOWN_FOLDERS:
        src = EXISTING_DATA / folder_name
        if src.exists():
            n = _collect_yolo_pairs(src, tmp_img, tmp_lbl)
            print(f"[Pose]  {folder_name}: +{n} pairs")
            total += n
        else:
            print(f"[Pose]  {folder_name}: not found (skip)")

    print(f"[Pose] Total pairs collected: {total}")
    if total == 0:
        print("[Pose] No dedicated pose data found. Using person boxes from PPE datasets.")
        # Fallback: use person class from PPE data
        for folder_name in PPE_KNOWN_FOLDERS:
            src = EXISTING_DATA / folder_name
            if src.exists():
                n = _collect_yolo_pairs(src, tmp_img, tmp_lbl)
                total += n
        print(f"[Pose] Fallback pairs: {total}")

    if total == 0:
        print("[Pose] ERROR: No training data found!")
        return None

    n_train, n_val = _split_dataset(tmp_img, tmp_lbl, MERGED_POSE)

    yaml_path = MERGED_POSE / "dataset.yaml"
    _make_yaml(MERGED_POSE, POSE_CLASSES, yaml_path)

    shutil.rmtree(tmp_img, ignore_errors=True)
    shutil.rmtree(tmp_lbl, ignore_errors=True)

    print(f"[Pose] Dataset ready: train={n_train}  val={n_val}")
    return yaml_path


# ================================================================
# TRAIN PPE
# ================================================================
def train_ppe(epochs: int = 120, batch: int = 16, device: str = "auto"):
    print("\n" + "="*60)
    print("  Training PPE Model")
    print("="*60)

    yaml_path = prepare_ppe_dataset()
    if yaml_path is None:
        return None

    if device == "auto":
        device = _auto_device()

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8m.pt")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name  = f"ppe_local_{timestamp}"

        print(f"\n[PPE] Starting training...")
        print(f"      Epochs  : {epochs}")
        print(f"      Batch   : {batch}")
        print(f"      Device  : {device}")
        print(f"      Data    : {yaml_path}")
        print(f"      Run     : {run_name}\n")

        results = model.train(
            data        = str(yaml_path),
            epochs      = epochs,
            batch       = batch,
            imgsz       = 640,
            device      = device,
            optimizer   = "AdamW",
            lr0         = 0.001,
            lrf         = 0.005,
            warmup_epochs = 5,
            patience    = 30,
            name        = run_name,
            project     = str(RUNS_DIR / "ppe"),
            exist_ok    = True,
            augment     = True,
            hsv_h       = 0.015,
            hsv_s       = 0.7,
            hsv_v       = 0.4,
            degrees     = 5.0,
            translate   = 0.1,
            scale       = 0.5,
            flipud      = 0.0,
            fliplr      = 0.5,
            mosaic      = 1.0,
        )

        # Copy best model to models/
        best_pt = RUNS_DIR / "ppe" / run_name / "weights" / "best.pt"
        if best_pt.exists():
            dest = MODELS_DIR / "ppe_finetuned.pt"
            shutil.copy2(best_pt, dest)
            print(f"\n[PPE] Best model saved: {dest}")

            # Export ONNX
            try:
                model_best = YOLO(str(dest))
                model_best.export(format="onnx", imgsz=640)
                print(f"[PPE] ONNX exported")
            except Exception as e:
                print(f"[PPE] ONNX export failed (optional): {e}")

            return str(dest)
        else:
            print("[PPE] WARNING: best.pt not found")
            return None

    except ImportError:
        print("[PPE] ERROR: ultralytics not installed")
        print("  --> pip install ultralytics")
        return None
    except Exception as e:
        print(f"[PPE] Training error: {e}")
        import traceback; traceback.print_exc()
        return None


# ================================================================
# TRAIN POSE
# ================================================================
def train_pose(epochs: int = 100, batch: int = 16, device: str = "auto"):
    print("\n" + "="*60)
    print("  Training Pose/Fall Model")
    print("="*60)

    yaml_path = prepare_pose_dataset()
    if yaml_path is None:
        return None

    if device == "auto":
        device = _auto_device()

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8m-pose.pt")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name  = f"pose_local_{timestamp}"

        print(f"\n[Pose] Starting training...")
        print(f"       Epochs : {epochs}")
        print(f"       Batch  : {batch}")
        print(f"       Device : {device}\n")

        results = model.train(
            data        = str(yaml_path),
            epochs      = epochs,
            batch       = batch,
            imgsz       = 640,
            device      = device,
            optimizer   = "AdamW",
            lr0         = 0.001,
            patience    = 25,
            name        = run_name,
            project     = str(RUNS_DIR / "pose"),
            exist_ok    = True,
        )

        best_pt = RUNS_DIR / "pose" / run_name / "weights" / "best.pt"
        if best_pt.exists():
            dest = MODELS_DIR / "pose_finetuned.pt"
            shutil.copy2(best_pt, dest)
            print(f"\n[Pose] Best model saved: {dest}")
            return str(dest)
        return None

    except Exception as e:
        print(f"[Pose] Training error: {e}")
        import traceback; traceback.print_exc()
        return None


# ================================================================
# CLI
# ================================================================
def main():
    global EXISTING_DATA
    ap = argparse.ArgumentParser(
        description="ZENTRA Training with Existing Local Data",
    )
    ap.add_argument("--check",   action="store_true",
                    help="Show available data only")
    ap.add_argument("--task",    nargs="+",
                    choices=["ppe", "pose", "all"],
                    default=["all"])
    ap.add_argument("--epochs",  type=int, default=None)
    ap.add_argument("--batch",   type=int, default=16)
    ap.add_argument("--device",  default="auto")
    ap.add_argument("--data",    default=None,
                    help=f"Override data path (default: {EXISTING_DATA})")
    args = ap.parse_args()

    # global EXISTING_DATA
    if args.data:
        EXISTING_DATA = Path(args.data)

    print("\n" + "="*60)
    print("  ZENTRA Training (Local Data Mode)")
    print(f"  Data source: {EXISTING_DATA}")
    print("="*60)

    check_existing_data()

    if args.check:
        return

    tasks = args.task
    if "all" in tasks:
        tasks = ["ppe", "pose"]

    results = {}
    for task in tasks:
        if task == "ppe":
            epochs = args.epochs or 120
            results["ppe"] = train_ppe(epochs, args.batch, args.device)
        elif task == "pose":
            epochs = args.epochs or 100
            results["pose"] = train_pose(epochs, args.batch, args.device)

    print("\n" + "="*60)
    print("  Training Complete!")
    for task, path in results.items():
        if path:
            print(f"  {task.upper()}: {path}")
        else:
            print(f"  {task.upper()}: FAILED")
    print("="*60)

    if any(results.values()):
        print("\n  To use the trained model, set in .env:")
        print("  USE_LOCAL_MODEL=true")
        print("\n  Then restart: python main.py")


if __name__ == "__main__":
    main()
