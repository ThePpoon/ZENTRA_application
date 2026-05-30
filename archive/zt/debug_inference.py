#!/usr/bin/env python3
"""
debug_inference.py -- ZENTRA Detection Debug Tool
==================================================
Test if PPE/Fall detection is working correctly.

  python debug_inference.py          # test with webcam
  python debug_inference.py --image PATH  # test with image file
  python debug_inference.py --server      # check server only
==================================================
"""

from __future__ import annotations
import sys
import io
import os

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import cv2
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import config as cfg


def check_server():
    print(f"\n[Check] Inference server: {cfg.INFERENCE_SERVER_URL}")
    try:
        resp = requests.get(cfg.INFERENCE_SERVER_URL + "/", timeout=5)
        if resp.status_code == 200:
            print(f"[Check] OK - Server is running")
            return True
        print(f"[Check] FAILED - Status: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("[Check] FAILED - Cannot connect!")
        print("  --> Run: docker compose up inference -d")
        print("  --> Wait for 'Application startup complete'")
    except Exception as e:
        print(f"[Check] Error: {e}")
    return False


def run_inference_on_frame(frame: np.ndarray):
    """Run PPE + Fall inference and print results"""
    try:
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(
            api_url=cfg.INFERENCE_SERVER_URL,
            api_key=cfg.ROBOFLOW_API_KEY,
        )

        print(f"\n[Infer] Running PPE model ({cfg.PPE_MODEL_ID})...")
        try:
            r1 = client.infer(frame, model_id=cfg.PPE_MODEL_ID)
            ppe_preds = r1.get("predictions", [])
            print(f"[Infer] PPE detections: {len(ppe_preds)}")
            for p in ppe_preds:
                print(f"        class={p.get('class','?')}  conf={p.get('confidence',0):.2f}")
        except Exception as e:
            print(f"[Infer] PPE ERROR: {e}")
            ppe_preds = []

        print(f"\n[Infer] Running Fall model ({cfg.FALL_MODEL_ID})...")
        try:
            r2 = client.infer(frame, model_id=cfg.FALL_MODEL_ID)
            fall_preds = r2.get("predictions", [])
            print(f"[Infer] Fall detections: {len(fall_preds)}")
            for p in fall_preds:
                print(f"        class={p.get('class','?')}  conf={p.get('confidence',0):.2f}")
        except Exception as e:
            print(f"[Infer] Fall ERROR: {e}")
            fall_preds = []

        return ppe_preds, fall_preds

    except ImportError:
        print("[Infer] ERROR: inference_sdk not installed")
        print("  --> pip install inference-sdk")
        return [], []


def test_with_webcam():
    print("\n[Test] Opening webcam...")
    cap = cv2.VideoCapture(cfg.WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Test] ERROR: Cannot open webcam index={cfg.WEBCAM_INDEX}")
        return

    print("[Test] Webcam OK. Capturing test frame...")
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[Test] ERROR: Cannot read frame")
        return

    h, w = frame.shape[:2]
    print(f"[Test] Frame size: {w}x{h}")

    # Save test frame
    cv2.imwrite("debug_frame.jpg", frame)
    print("[Test] Saved: debug_frame.jpg")

    ppe_preds, fall_preds = run_inference_on_frame(frame)

    # Annotate and show
    annotated = frame.copy()
    for p in ppe_preds + fall_preds:
        x, y = int(p.get("x", 0)), int(p.get("y", 0))
        w2  = int(p.get("width",  0)) // 2
        h2  = int(p.get("height", 0)) // 2
        cls = p.get("class", "?")
        conf= p.get("confidence", 0)
        color = (0, 220, 0) if not p.get("violation", False) else (0, 0, 220)
        cv2.rectangle(annotated, (x-w2, y-h2), (x+w2, y+h2), color, 2)
        cv2.putText(annotated, f"{cls} {conf:.0%}", (x-w2, y-h2-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite("debug_annotated.jpg", annotated)
    print("[Test] Saved: debug_annotated.jpg")

    print("\n[Test] Showing result (press any key to close)...")
    cv2.imshow("ZENTRA Debug", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_with_image(image_path: str):
    print(f"\n[Test] Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Test] ERROR: Cannot load {image_path}")
        return
    h, w = frame.shape[:2]
    print(f"[Test] Image size: {w}x{h}")
    run_inference_on_frame(frame)


def main():
    ap = argparse.ArgumentParser(description="ZENTRA Inference Debug")
    ap.add_argument("--server", action="store_true", help="Check server only")
    ap.add_argument("--image",  default=None,        help="Test with image file")
    args = ap.parse_args()

    print("======== ZENTRA Inference Debug ========")
    print(f"  API Key  : {cfg.ROBOFLOW_API_KEY[:8]}...")
    print(f"  Server   : {cfg.INFERENCE_SERVER_URL}")
    print(f"  PPE Model: {cfg.PPE_MODEL_ID}")
    print(f"  Fall Mdl : {cfg.FALL_MODEL_ID}")
    print("========================================")

    server_ok = check_server()
    if not server_ok:
        print("\nFix inference server first!")
        return

    if args.server:
        print("\nServer OK!")
        return

    if args.image:
        test_with_image(args.image)
    else:
        test_with_webcam()


if __name__ == "__main__":
    main()
