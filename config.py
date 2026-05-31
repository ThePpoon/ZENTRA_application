# config.py — ZENTRA System Configuration
# Zone Environment Network Thermal Risk Analysis
# Windows 11 + NVIDIA GPU | Python 3.11
# อ้างอิง Slide: CEDT Innovation Summit 2026 (อันดับ 1)
# ================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load this project's .env explicitly so it works no matter what the
# current working directory is (the desktop app imports config from a
# different cwd). override=False keeps real environment vars authoritative.
load_dotenv(Path(__file__).parent / ".env", override=False)

# ================================================================
# BASE PATHS
# ================================================================
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
LOGS_DIR      = BASE_DIR / "logs"
COLLECTED_DIR = DATA_DIR / "collected"

for _d in [
    DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    COLLECTED_DIR / "ppe_violations",
    COLLECTED_DIR / "zone_intrusions",
    COLLECTED_DIR / "fall_events",
    COLLECTED_DIR / "normal",
]:
    _d.mkdir(parents=True, exist_ok=True)

# ================================================================
# ROBOFLOW / INFERENCE SERVER
# ================================================================
ROBOFLOW_API_KEY     = os.getenv("ROBOFLOW_API_KEY",  "8xTIheqbzg4mkSLOdFe6")
ROBOFLOW_WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:9001")

PPE_MODEL_ID  = os.getenv("PPE_MODEL_ID",  "ppe-cpxsz/2")
FALL_MODEL_ID = os.getenv("FALL_MODEL_ID", "fall-detection-ovjqo/5")

USE_LOCAL_MODEL  = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
PPE_LOCAL_MODEL  = str(MODELS_DIR / "ppe_finetuned.pt")
FALL_LOCAL_MODEL = str(MODELS_DIR / "fall_finetuned.pt")

ROBOFLOW_PPE_PROJECT  = os.getenv("ROBOFLOW_PPE_PROJECT",  "zentra-ppe")
ROBOFLOW_FALL_PROJECT = os.getenv("ROBOFLOW_FALL_PROJECT", "zentra-fall")

# ================================================================
# CAMERA  (Windows 11: USE_DSHOW=true เร็วกว่า CAP_FFMPEG)
# ================================================================
CAMERA_SOURCE   = os.getenv("CAMERA_SOURCE", "webcam")
WEBCAM_INDEX    = int(os.getenv("WEBCAM_INDEX", "0"))
RTSP_URL        = os.getenv("RTSP_URL", "rtsp://admin:password@192.168.1.100:554/stream1")
VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH", "")
USE_DSHOW       = os.getenv("USE_DSHOW", "true").lower() == "true"

# ================================================================
# INFERENCE
# ================================================================
INFERENCE_CONFIDENCE = float(os.getenv("INFERENCE_CONFIDENCE", "0.45"))
INFERENCE_IOU        = float(os.getenv("INFERENCE_IOU",        "0.45"))
INFER_EVERY_N_FRAMES = int(os.getenv("INFER_EVERY_N_FRAMES",   "3"))

# ================================================================
# PPE CLASSES
# Slide Module 1: Helmet / Vest / Goggles / Gloves / Safety Boots
# ================================================================
PPE_CLASSES: dict[str, dict] = {
    "helmet":          {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "vest":            {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    "goggles":         {"label": "Goggles",   "label_th": "สวมแว่นตานิรภัย",   "color": (0, 210, 0),   "violation": False},
    "gloves":          {"label": "Gloves",    "label_th": "สวมถุงมือ",          "color": (0, 210, 0),   "violation": False},
    "safety_boots":    {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
    "glasses":         {"label": "Glasses",   "label_th": "สวมแว่นตา",         "color": (0, 210, 0),   "violation": False},
    "boots":           {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
    "no_helmet":       {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no_vest":         {"label": "No Vest",   "label_th": "ไม่สวมเสื้อกั๊ก",   "color": (0, 0, 220),   "violation": True},
    "no_goggles":      {"label": "No Goggles","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    "no_gloves":       {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "no_safety_boots": {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    "no helmet":       {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no vest":         {"label": "No Vest",   "label_th": "ไม่สวมเสื้อกั๊ก",   "color": (0, 0, 220),   "violation": True},
    "no glasses":      {"label": "No Glasses","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    "no gloves":       {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "no boots":        {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    "person":          {"label": "Person",    "label_th": "บุคคล",              "color": (255, 190, 0), "violation": False},
}
REQUIRED_PPE = {"helmet", "vest"}

# ================================================================
# MEDIAPIPE — Slide Module 3: 33 Keypoints, 3 Detection Methods
# ================================================================
MEDIAPIPE_MODEL_COMPLEXITY    = 1
FALL_KEYPOINT_VELOCITY_THRESH = 0.30
FALL_BBOX_RATIO_THRESH        = 0.72
FALL_CONFIRM_FRAMES           = 6
GAIT_HISTORY_FRAMES           = 30
GAIT_ANOMALY_THRESH           = 0.20

# ================================================================
# BYTETRACK — Slide Module 2: Multi-Object Tracking
# ================================================================
BYTETRACK_TRACK_THRESH = 0.50
BYTETRACK_TRACK_BUFFER = 30
BYTETRACK_MATCH_THRESH = 0.80

# ================================================================
# SAFETY ZONE — Slide Module 2
# ================================================================
ZONE_POLYGON_FILE = str(DATA_DIR / "zones.json")
MAX_ZONES         = 10
ZONE_USE_FOOT_POINT = os.getenv("ZONE_USE_FOOT_POINT", "true").lower() == "true"  # test feet, not bbox centre
ZONE_TRACK_MIN_HITS = int(os.getenv("ZONE_TRACK_MIN_HITS", "3"))   # ignore unstable tracks
ZONE_CONFIRM_FRAMES = int(os.getenv("ZONE_CONFIRM_FRAMES", "3"))   # debounce intrusion

# ================================================================
# PPE — accuracy / debounce
# ================================================================
PPE_CONFIRM_FRAMES = int(os.getenv("PPE_CONFIRM_FRAMES", "3"))     # consecutive frames before alerting

# ================================================================
# ALERT COOLDOWN — Slide: 3 ระดับ
# ================================================================
VIOLATION_COOLDOWN_SECONDS = 30
ZONE_COOLDOWN_SECONDS      = 20
FALL_COOLDOWN_SECONDS      = 15

ALERT_LEVEL_WARNING   = "warning"
ALERT_LEVEL_ALERT     = "alert"
ALERT_LEVEL_EMERGENCY = "emergency"

# ================================================================
# LINE OA — Slide: ส่งถึง หัวหน้างาน / Safety / Emergency
# ================================================================
LINE_OA_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_OA_CHANNEL_ACCESS_TOKEN", "")
LINE_OA_GROUP_SUPERVISOR     = os.getenv("LINE_OA_GROUP_SUPERVISOR", "")
LINE_OA_GROUP_SAFETY         = os.getenv("LINE_OA_GROUP_SAFETY",     "")
LINE_OA_GROUP_EMERGENCY      = os.getenv("LINE_OA_GROUP_EMERGENCY",  "")

_FB = os.getenv("LINE_OA_GROUP_ID", "")
if not LINE_OA_GROUP_SUPERVISOR: LINE_OA_GROUP_SUPERVISOR = _FB
if not LINE_OA_GROUP_SAFETY:     LINE_OA_GROUP_SAFETY     = _FB
if not LINE_OA_GROUP_EMERGENCY:  LINE_OA_GROUP_EMERGENCY  = _FB

ALERT_RECIPIENTS: dict[str, list[str]] = {
    ALERT_LEVEL_WARNING:   [LINE_OA_GROUP_SUPERVISOR],
    ALERT_LEVEL_ALERT:     [LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_SUPERVISOR],
    ALERT_LEVEL_EMERGENCY: [LINE_OA_GROUP_EMERGENCY, LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_SUPERVISOR],
}

DAILY_REPORT_TIME = os.getenv("DAILY_REPORT_TIME", "20:00")
IMAGE_UPLOAD_URL  = "https://catbox.moe/user/api.php"

# ================================================================
# DATA COLLECTION
# ================================================================
AUTO_COLLECT_FRAMES      = True
COLLECT_VIOLATION_FRAMES = True
COLLECT_NORMAL_INTERVAL  = 300
COLLECT_MAX_PER_CLASS    = 2000
COLLECT_JPEG_QUALITY     = 90

# ================================================================
# TRAINING
# ================================================================
TRAIN_EPOCHS        = int(os.getenv("TRAIN_EPOCHS",     "100"))
TRAIN_BATCH_SIZE    = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
TRAIN_IMG_SIZE      = 640
TRAIN_DEVICE        = os.getenv("TRAIN_DEVICE", "0")
TRAIN_WORKERS       = int(os.getenv("TRAIN_WORKERS", "8"))
TRAIN_LR0           = 0.001
TRAIN_LRF           = 0.01
TRAIN_MOMENTUM      = 0.937
TRAIN_WEIGHT_DECAY  = 0.0005
TRAIN_WARMUP_EPOCHS = 5
TRAIN_VAL_SPLIT     = 0.15
TRAIN_AUG           = os.getenv("TRAIN_AUG", "true").lower() == "true"  # used by training/trainer.py
YOLO_BASE_MODEL     = os.getenv("YOLO_BASE_MODEL", "yolov8m.pt")

# ================================================================
# DISPLAY — Windows 11
# ================================================================
WINDOW_TITLE   = "ZENTRA Smart Detection"
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
FONT_SCALE     = 0.5
FONT_THICKNESS = 1
OSD_COLOR      = (255, 255, 255)
OSD_BG_COLOR   = (20, 20, 20)

# ================================================================
# PERFORMANCE
# ================================================================
TARGET_FPS        = 60
FRAME_BUFFER_SIZE = 4
ENABLE_THREADING  = True
