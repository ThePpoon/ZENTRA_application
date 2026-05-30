#!/usr/bin/env python3.11
"""
train_fall_lstm.py — ZENTRA Fall/Heat Stroke LSTM Training
==========================================================
ขั้นตอน:

  1. เตรียมวิดีโอ (ดาวน์โหลด dataset หรือถ่ายเอง)
     วางไว้ใน:
       videos/fall/      ← วิดีโอคนล้ม
       videos/lying/     ← วิดีโอคนนอน/หมดสติ
       videos/normal/    ← วิดีโอเดินปกติ
       videos/sitting/   ← วิดีโอนั่งทำงาน (ลด false positive)
       videos/gait_anomaly/ ← เดินโซเซ/ผิดปกติ

  2. แยก keypoint sequences จากวิดีโอ
     python train_fall_lstm.py --mode extract

  3. เทรน LSTM
     python train_fall_lstm.py --mode train

  4. ทดสอบ
     python train_fall_lstm.py --mode test --source 1

  # หรือรันทีเดียวจบ
  python train_fall_lstm.py --mode all
==========================================================
Dataset สาธารณะแนะนำ:
  • UR Fall Detection Dataset: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
  • URFD (RGB videos)
  • Multiple Cameras Fall Dataset
  • Le2i Fall Detection Dataset
==========================================================
"""

import os, sys, yaml, json, shutil, random, argparse, time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VIDEO_DIR = BASE_DIR / "videos"
SEQ_DIR   = DATA_DIR / "lstm_sequences"
RUNS_DIR  = BASE_DIR / "runs" / "lstm"

for _d in [DATA_DIR, MODELS_DIR, VIDEO_DIR, SEQ_DIR, RUNS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# LSTM constants
SEQ_LEN    = 30
INPUT_DIM  = 99      # 33 keypoints × 3 (x,y,visibility)
HIDDEN_DIM = 64
LABELS     = ["fall", "gait_anomaly", "lying", "normal", "sitting"]
N_LABELS   = len(LABELS)
OUTPUT_PT  = str(MODELS_DIR / "fall_lstm.pt")

# MediaPipe
MP_OK = False
try:
    import mediapipe as mp
    _mp_pose   = mp.solutions.pose
    _pose_inst = _mp_pose.Pose(
        model_complexity         = 1,
        smooth_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )
    MP_OK = True
    print("✅ MediaPipe Pose loaded")
except Exception as e:
    print(f"❌ MediaPipe not available: {e}")
    print("   pip install mediapipe")


# ================================================================
# SYSTEM CHECK
# ================================================================
def check():
    print("\n═"*55)
    print("  ZENTRA Fall LSTM — System Check")
    print("═"*55)
    try:
        import torch
        print(f"  PyTorch  : {torch.__version__}  GPU={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  GPU      : {p.name}  VRAM={p.total_memory/1e9:.1f}GB")
    except ImportError:
        print("  PyTorch  : ❌  pip install torch")
    print(f"  MediaPipe: {'✅' if MP_OK else '❌'}")

    # วิดีโอ
    for label in LABELS:
        vids = list((VIDEO_DIR/label).glob("*.mp4")) + \
               list((VIDEO_DIR/label).glob("*.avi")) + \
               list((VIDEO_DIR/label).glob("*.mov"))
        seqs = list((SEQ_DIR/label).glob("*.npy")) if (SEQ_DIR/label).exists() else []
        print(f"  {label:<15}: {len(vids):>3} videos  {len(seqs):>4} sequences")
    print("═"*55 + "\n")


# ================================================================
# STEP 1: DOWNLOAD PUBLIC DATASETS
# ================================================================
def download_datasets():
    """
    แนะนำ dataset สาธารณะและวิธีใช้
    (ไม่ auto-download เพราะบาง dataset ต้องลงทะเบียน)
    """
    print("\n📥 Fall Detection Dataset Sources:")
    print("─"*60)
    print("""
  1. UR Fall Detection Dataset (แนะนำ)
     URL: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
     ดาวน์โหลด RGB videos → วางใน:
       videos/fall/     (camera adl-fall-* sequences)
       videos/normal/   (camera adl-* non-fall sequences)

  2. Le2i Fall Detection Dataset
     URL: http://le2i.cnrs.fr/Fall-detection-Dataset
     วางใน videos/fall/ และ videos/normal/

  3. URFD (University of Rochester Fall Dataset)
     URL: https://www.cs.rochester.edu/~jluo/URFD/
     ใช้ RGB camera sequences

  4. ถ่ายวิดีโอเอง (วิธีที่ดีสุดสำหรับโรงงาน)
     - fall/:       จำลองล้ม ล้มจริง
     - lying/:      นอนบนพื้น หมดสติ
     - normal/:     เดินปกติ ยืน นั่งทำงาน
     - sitting/:    นั่งเก้าอี้ นั่งพื้น
     - gait_anomaly/: เดินโซเซ เดินช้าผิดปกติ

  วางไฟล์วิดีโอที่: videos/<label>/video_name.mp4
""")
    print("─"*60)


# ================================================================
# STEP 2: EXTRACT KEYPOINT SEQUENCES FROM VIDEOS
# ================================================================
def extract_sequences(
    video_root: str = None,
    output_dir: str = None,
    stride:     int = 5,
    max_per_vid: int = 300,
):
    """
    วิ่งผ่านวิดีโอทุกไฟล์ใน videos/<label>/
    → แยก keypoints ทุก frame ด้วย MediaPipe
    → บันทึกเป็น .npy (shape: SEQ_LEN × 99)
    """
    if not MP_OK:
        sys.exit("❌  ต้องการ MediaPipe: pip install mediapipe")

    vid_root = Path(video_root or VIDEO_DIR)
    out_root = Path(output_dir or SEQ_DIR)

    print(f"\n🎬 Extracting sequences: {vid_root}")
    print(f"   stride={stride}  max_per_video={max_per_vid}")

    total_seqs = 0
    for label_dir in sorted(vid_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        out_label = out_root / label
        out_label.mkdir(parents=True, exist_ok=True)

        videos = (list(label_dir.glob("*.mp4")) +
                  list(label_dir.glob("*.avi")) +
                  list(label_dir.glob("*.mov")) +
                  list(label_dir.glob("*.mkv")))

        if not videos:
            print(f"  ⚠️  ไม่พบวิดีโอใน {label_dir}")
            continue

        print(f"\n  📁 {label}: {len(videos)} videos")
        label_seqs = 0

        for vid_path in videos:
            cap    = cv2.VideoCapture(str(vid_path))
            frames = []
            seq_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = _pose_inst.process(rgb)

                if results.pose_landmarks:
                    kp = np.array([
                        [lm.x, lm.y, lm.visibility]
                        for lm in results.pose_landmarks.landmark
                    ], dtype=np.float32).flatten()
                else:
                    kp = np.zeros(INPUT_DIM, dtype=np.float32)

                frames.append(kp)

                # ตัด sequence ทุก stride frames
                if len(frames) >= SEQ_LEN:
                    if (len(frames) - SEQ_LEN) % stride == 0:
                        seq_arr = np.array(frames[-SEQ_LEN:], dtype=np.float32)
                        np.save(str(out_label / f"{vid_path.stem}_{seq_id:04d}.npy"), seq_arr)
                        seq_id    += 1
                        label_seqs += 1
                        total_seqs += 1
                        if seq_id >= max_per_vid:
                            break

            cap.release()
            print(f"     {vid_path.name:<40} → {seq_id} sequences")

        print(f"     {label}: {label_seqs} sequences total")

    print(f"\n  ✅ Total sequences: {total_seqs}")
    _print_seq_stats(out_root)
    return total_seqs


def _print_seq_stats(root: Path):
    print(f"\n  📊 Sequence distribution:")
    for label_dir in sorted(root.iterdir()):
        if label_dir.is_dir():
            n = len(list(label_dir.glob("*.npy")))
            bar = "█" * min(n//10, 40)
            print(f"    {label_dir.name:<15} {n:>5} seqs  {bar}")


# ================================================================
# STEP 3: AUGMENT SEQUENCES
# ================================================================
def augment_sequences(seq_dir: str = None, multiplier: int = 2):
    """
    เพิ่ม sequence ด้วย augmentation
    - Random noise บน keypoints
    - Random scale
    - Time warping (stretch/compress)
    """
    root = Path(seq_dir or SEQ_DIR)
    print(f"\n🎨 Augmenting sequences × {multiplier}...")

    added = 0
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        orig = list(label_dir.glob("*.npy"))
        for seq_path in orig:
            seq = np.load(seq_path).astype(np.float32)   # (SEQ_LEN, 99)
            for k in range(multiplier):
                aug = _augment_seq(seq, seed=hash(str(seq_path)) + k)
                np.save(str(label_dir / f"{seq_path.stem}_aug{k}.npy"), aug)
                added += 1

    print(f"  ✅ Added {added} augmented sequences")
    _print_seq_stats(root)


def _augment_seq(seq: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed % (2**31))
    aug = seq.copy()

    # 1. Gaussian noise on keypoints (x, y) not visibility
    if rng.random() < 0.7:
        noise = rng.normal(0, 0.01, (len(aug), 99))
        noise[:, 2::3] = 0   # visibility ไม่ใส่ noise
        aug = np.clip(aug + noise, 0, 1)

    # 2. Random scale (ย่อ/ขยาย keypoints)
    if rng.random() < 0.5:
        scale = rng.uniform(0.85, 1.15)
        aug[:, 0::3] = np.clip(aug[:, 0::3] * scale, 0, 1)
        aug[:, 1::3] = np.clip(aug[:, 1::3] * scale, 0, 1)

    # 3. Horizontal flip
    if rng.random() < 0.5:
        aug[:, 0::3] = 1.0 - aug[:, 0::3]

    # 4. Time warp (drop หรือ repeat บาง frames)
    if rng.random() < 0.3:
        n = len(aug)
        new_idx = np.sort(rng.choice(n, size=n, replace=True))
        aug = aug[new_idx]

    # 5. Speed up/slow down (sub-sample)
    if rng.random() < 0.3:
        factor = rng.uniform(0.7, 1.3)
        new_len = max(8, int(len(aug) * factor))
        x_old = np.linspace(0, 1, len(aug))
        x_new = np.linspace(0, 1, new_len)
        aug = np.array([np.interp(x_new, x_old, aug[:, i])
                        for i in range(aug.shape[1])]).T.astype(np.float32)
        # trim/pad
        if len(aug) > SEQ_LEN:
            aug = aug[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN-len(aug), aug.shape[1]), dtype=np.float32)
            aug = np.concatenate([aug, pad])

    return aug.astype(np.float32)


# ================================================================
# STEP 4: TRAIN LSTM
# ================================================================
def train(
    seq_dir:  str  = None,
    epochs:   int  = 80,
    batch:    int  = 32,
    device:   str  = "auto",
    lr:       float = 1e-3,
    aug:      bool = True,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        sys.exit("❌  pip install torch")

    root   = Path(seq_dir or SEQ_DIR)
    dev_s  = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
    dev    = torch.device(dev_s if dev_s == "cuda" and torch.cuda.is_available() else "cpu")

    # augment ก่อนเทรน
    if aug:
        augment_sequences(seq_dir, multiplier=2)

    # Dataset
    class _SeqDS(Dataset):
        def __init__(self, root, splits=None):
            self.samples = []
            label_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
            self.lmap  = {d.name: i for i, d in enumerate(label_dirs)}
            all_files  = []
            for d in label_dirs:
                lid = self.lmap[d.name]
                for npy in sorted(d.glob("*.npy")):
                    all_files.append((npy, lid))
            random.Random(42).shuffle(all_files)
            if splits:
                n  = len(all_files)
                lo = int(n * splits[0])
                hi = int(n * splits[1])
                all_files = all_files[lo:hi]
            for npy, lid in all_files:
                seq = np.load(npy).astype(np.float32)
                if seq.shape[0] < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN-seq.shape[0], INPUT_DIM), dtype=np.float32)
                    seq = np.concatenate([seq, pad])
                self.samples.append((seq[:SEQ_LEN], lid))

        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            x, y = self.samples[i]
            return torch.tensor(x), torch.tensor(y, dtype=torch.long)

    full_ds = _SeqDS(root)
    n_cls   = len(full_ds.lmap)
    n_total = len(full_ds)

    if n_total < 10:
        print(f"❌  ไม่มีข้อมูลพอ ({n_total} sequences) — รัน --mode extract ก่อน")
        return

    tr_ds = _SeqDS(root, splits=(0.0, 0.85))
    va_ds = _SeqDS(root, splits=(0.85, 1.0))
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=(dev_s=="cuda"))
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)

    # Model
    class _FallLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm  = nn.LayerNorm(INPUT_DIM)
            self.lstm  = nn.LSTM(INPUT_DIM, HIDDEN_DIM, num_layers=2,
                                 batch_first=True, dropout=0.3, bidirectional=True)
            self.attn  = nn.Linear(HIDDEN_DIM*2, 1)
            self.fc    = nn.Sequential(
                nn.Linear(HIDDEN_DIM*2, 64), nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(64, n_cls),
            )
        def forward(self, x):
            x        = self.norm(x)
            out, _   = self.lstm(x)                         # (B, T, H*2)
            weights  = torch.softmax(self.attn(out), dim=1) # (B, T, 1)
            ctx      = (out * weights).sum(dim=1)            # (B, H*2)
            return self.fc(ctx)

    model = _FallLSTM().to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr*5, steps_per_epoch=len(tr_ld), epochs=epochs, pct_start=0.1)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n{'═'*55}")
    print(f"  ZENTRA LSTM Fall Classifier Training")
    print(f"  Classes : {full_ds.lmap}")
    print(f"  Train   : {len(tr_ds)}  Val: {len(va_ds)}")
    print(f"  Epochs  : {epochs}  Batch: {batch}  Device: {dev}")
    print(f"{'═'*55}\n")

    best_acc = 0.0
    t0       = time.time()

    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = total = 0
        cls_correct = [0]*n_cls
        cls_total   = [0]*n_cls
        with torch.no_grad():
            for xb, yb in va_ld:
                pred = model(xb.to(dev)).argmax(1).cpu()
                for p, t in zip(pred, yb):
                    cls_total[t]   += 1
                    cls_correct[t] += int(p == t)
                correct += (pred == yb).sum().item()
                total   += len(yb)

        acc = correct / max(total, 1)

        if ep % 5 == 0 or acc > best_acc:
            elapsed = time.time()-t0
            bar     = "█"*int(acc*20)
            per_cls = " ".join(
                f"{list(full_ds.lmap.keys())[i][:4]}={cls_correct[i]/max(cls_total[i],1):.0%}"
                for i in range(n_cls)
            )
            print(f"  Ep {ep:>3}  loss={train_loss/len(tr_ld):.3f}  acc={acc:.3f} {bar}")
            print(f"         [{per_cls}]  ({elapsed:.0f}s)")

        if acc > best_acc:
            best_acc = acc
            Path(OUTPUT_PT).parent.mkdir(parents=True, exist_ok=True)
            # บันทึก state dict เท่านั้น
            torch.save(model.cpu().state_dict(), OUTPUT_PT)
            model.to(dev)

    print(f"\n  🎉 Best val_acc = {best_acc:.3f}")
    print(f"  📁 Model → {OUTPUT_PT}")

    # บันทึก label map
    lmap_path = str(MODELS_DIR / "fall_lstm_labels.json")
    json.dump(full_ds.lmap, open(lmap_path,"w"))
    print(f"  📄 Labels → {lmap_path}")
    return OUTPUT_PT


# ================================================================
# STEP 5: VALIDATE
# ================================================================
def validate(seq_dir: str = None):
    try:
        import torch
    except ImportError:
        sys.exit("❌  pip install torch")

    if not Path(OUTPUT_PT).exists():
        print(f"❌  ไม่พบ {OUTPUT_PT} — รัน --mode train ก่อน")
        return

    from modules.heat_stroke import _load_lstm, _lstm_model, LSTM_OK
    _load_lstm()

    # Quick validation on sequences
    root = Path(seq_dir or SEQ_DIR)
    print(f"\n🧪 Validating: {OUTPUT_PT}")

    results_per_cls: dict[str, list[bool]] = {}
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir(): continue
        label = label_dir.name
        results_per_cls[label] = []
        seqs = list(label_dir.glob("*.npy"))[:50]
        for seq_path in seqs:
            seq = np.load(seq_path).astype(np.float32)
            if len(seq) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN-len(seq), INPUT_DIM), dtype=np.float32)
                seq = np.concatenate([seq, pad])
            import torch
            x    = torch.tensor([seq[:SEQ_LEN]])
            with torch.no_grad():
                pred = torch.softmax(_lstm_model(x), 1).argmax(1).item()
            pred_label = LABELS[pred] if pred < len(LABELS) else "?"
            results_per_cls[label].append(pred_label == label)

    print(f"\n  Per-Class Accuracy:")
    print(f"{'─'*40}")
    total_c = total_t = 0
    for label, corrects in results_per_cls.items():
        acc  = sum(corrects) / max(len(corrects), 1)
        bar  = "█"*int(acc*25)
        icon = "✅" if acc >= 0.75 else ("⚠️" if acc >= 0.50 else "❌")
        print(f"  {icon} {label:<15} {acc:.3f} {bar} (n={len(corrects)})")
        total_c += sum(corrects)
        total_t += len(corrects)
    if total_t:
        print(f"{'─'*40}")
        print(f"  Overall: {total_c/total_t:.3f}")
    print()


# ================================================================
# STEP 6: LIVE TEST
# ================================================================
def test_live(source: str = "0"):
    if not MP_OK:
        sys.exit("❌  MediaPipe ไม่พร้อม")
    if not Path(OUTPUT_PT).exists():
        print(f"⚠️  ไม่พบ LSTM model — จะใช้ rule-based fallback")

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"❌  เปิด source ไม่ได้: {source}")
        return

    from modules.heat_stroke import _pose_model as pm, _analyze_pose, _analyze_lstm
    from modules.heat_stroke import _PersonState

    state = _PersonState(0)
    print(f"\n🎥 Live test | Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pm.process(rgb)

        status_color = (0, 200, 80)
        status_text  = "Normal"

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            from modules.heat_stroke import _mp_draw, _mp_styles, _mp_pose as mpp
            _mp_draw.draw_landmarks(frame, results.pose_landmarks, mpp.POSE_CONNECTIONS,
                                    landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style())

            pose_res = _analyze_pose(results.pose_landmarks.landmark, w, h, state)
            lstm_res = _analyze_lstm(state)

            ps = pose_res["score"]
            ls = lstm_res["score"]
            ll = lstm_res.get("label", "?")

            if ps > 0.5 or ls > 0.5:
                status_color = (0, 0, 220)
                status_text  = f"RISK! pose={ps:.0%} lstm={ls:.0%}({ll})"
            elif ps > 0.3 or ls > 0.35:
                status_color = (0, 120, 220)
                status_text  = f"Warning pose={ps:.0%} lstm={ll}"
            else:
                status_text  = f"Normal pose={ps:.0%} lstm={ll}"

        cv2.putText(frame, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
        cv2.imshow("ZENTRA Fall/HeatStroke Live Test — Q=quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description="ZENTRA Fall LSTM Training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--mode", default="all",
        choices=["check","info","extract","augment","train","all","validate","test"])
    ap.add_argument("--video-dir",  default=None)
    ap.add_argument("--seq-dir",    default=None)
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--stride",     type=int,   default=5)
    ap.add_argument("--max-per-vid",type=int,   default=300)
    ap.add_argument("--aug-x",      type=int,   default=2)
    ap.add_argument("--no-aug",     action="store_true")
    ap.add_argument("--source",     default="0")
    args = ap.parse_args()

    if args.mode == "check":
        check(); return
    if args.mode == "info":
        download_datasets(); return

    check()

    if args.mode in ("extract", "all"):
        n = extract_sequences(args.video_dir, args.seq_dir, args.stride, args.max_per_vid)
        if n == 0 and args.mode == "all":
            print("\n⚠️  ไม่มี sequences — วางวิดีโอใน videos/<label>/ ก่อน")
            print("   รัน: python train_fall_lstm.py --mode info  เพื่อดูรายละเอียด")
            return

    if args.mode in ("augment",):
        augment_sequences(args.seq_dir, args.aug_x)

    if args.mode in ("train", "all"):
        train(
            seq_dir = args.seq_dir,
            epochs  = args.epochs,
            batch   = args.batch,
            device  = args.device,
            lr      = args.lr,
            aug     = not args.no_aug,
        )

    if args.mode == "validate":
        validate(args.seq_dir)

    if args.mode == "test":
        test_live(args.source)


if __name__ == "__main__":
    main()
